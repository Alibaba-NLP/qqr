"""
VitaBench closed-model evaluation script: in interactive mode (ota/delivery/instore/cross_domain) it
Measures baseline performance for models such as gpt-5.2, gpt-5, claude-* and a local Qwen3-8B.

Key reuse:
- Reuses arise_rl_solver/vitabench.user_simulator.simulate_user_response, with the UserSimulator served locally
  Qwen3.5-397B (endpoint set through the USER_SIMULATOR_BASE_URL env var)
- Reuses arise_rl_solver/vitabench.reward_model.eval_reward: the official-aligned sliding-window LLM judge
  binary pass^1 reward (reward=1 only when every rubric passes)
- Reuses arise_rl_solver/vitabench.rollout.build_system_message and DOMAIN_GUIDANCE
- Reuses qqr.tools.vitabench_env.VitaBenchToolState: per-task domain toolsets over the OpenAI tool protocol

The agent <-> UserSimulator <-> tools interaction loop:
  1. Inject the system turn and the agent greeting
  2. The UserSim supplies the first user message
  3. Loop for up to max_steps rounds:
     - The agent (closed-source API) produces a reply and possibly tool_calls
     - With tool_calls, VitaBenchToolState runs them; without, control passes to the UserSim
     - Ends when either side emits ###STOP### or max_steps is reached
  4. eval_reward scores the full conversation into a pass^1 reward
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from openai import AsyncOpenAI

from qqr.schemas import Sample
from qqr.tools.vitabench_env import VitaBenchToolState, load_vitabench_task_by_id

from . import config
from .reward_model import eval_reward
from .rollout import build_system_message, DOMAIN_GUIDANCE  # noqa: F401  (build_system_message uses DOMAIN_GUIDANCE)
from .user_simulator import simulate_user_response, is_stop, STOP_SIGNAL


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


# --- Eval-only overrides (config.py is left untouched so training is unaffected) ---
# Sliding-window judge endpoint: see config.py (override with the LLM_JUDGE_BASE_URL env var).
# this can be overridden with the LLM_JUDGE_BASE_URL env var; reward_model.SlidingWindowJudge's
# client is lazily initialised, so this patch takes effect before first access.
_JUDGE_BASE_URL_OVERRIDE = os.environ.get("LLM_JUDGE_BASE_URL")
if _JUDGE_BASE_URL_OVERRIDE:
    _orig_judge_base_url = config.llm_judge_base_url
    config.llm_judge_base_url = _JUDGE_BASE_URL_OVERRIDE
    logger.info(
        f"[eval-override] llm_judge_base_url: {_orig_judge_base_url} -> {_JUDGE_BASE_URL_OVERRIDE}"
    )

# the judge model and api key can be overridden too (they default to the config values)
_JUDGE_MODEL_OVERRIDE = os.environ.get("LLM_JUDGE_MODEL")
if _JUDGE_MODEL_OVERRIDE and _JUDGE_MODEL_OVERRIDE != config.llm_judge_model:
    _orig_judge_model = config.llm_judge_model
    config.llm_judge_model = _JUDGE_MODEL_OVERRIDE
    logger.info(
        f"[eval-override] llm_judge_model: {_orig_judge_model} -> {_JUDGE_MODEL_OVERRIDE}"
    )

_JUDGE_API_KEY_OVERRIDE = os.environ.get("LLM_JUDGE_API_KEY")
if _JUDGE_API_KEY_OVERRIDE:
    config.llm_judge_api_key = _JUDGE_API_KEY_OVERRIDE
    logger.info("[eval-override] llm_judge_api_key: overridden via LLM_JUDGE_API_KEY env")


# ═══════════════════════════════════════════════════════════════════════════════
# Closed-source API helpers (OpenAI / Claude)
# ═══════════════════════════════════════════════════════════════════════════════


def _openai_tools_to_claude(tools: list[dict]) -> list[dict]:
    """Convert OpenAI {"type":"function","function":{...}} → Claude {"type":"custom","name":...,"input_schema":...}."""
    claude_tools = []
    for t in tools:
        if t.get("type") == "function":
            fn = t["function"]
            claude_tools.append({
                "type": "custom",
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {}),
            })
        else:
            claude_tools.append(t)
    return claude_tools


async def _call_claude(
    client: AsyncOpenAI, model_name: str, system_prompt: str,
    claude_messages: list[dict], claude_tools: list[dict],
    temperature: float, max_tokens: int = 4096, api_retries: int = 5,
):
    last_err = None
    for attempt in range(api_retries):
        try:
            kwargs = {
                "model": model_name,
                "messages": claude_messages,
                "tools": claude_tools,
                "extra_body": {"system": system_prompt},
                "max_tokens": max_tokens,
            }
            try:
                return await client.chat.completions.create(**kwargs, temperature=temperature)
            except Exception as e:
                if "temperature" in str(e).lower():
                    return await client.chat.completions.create(**kwargs)
                raise
        except Exception as e:
            last_err = e
            if attempt + 1 >= api_retries:
                raise
            await asyncio.sleep(min(2 ** attempt, 10))
    raise last_err


async def _call_openai(
    client: AsyncOpenAI, model_name: str, messages: list[dict],
    tools: list[dict], temperature: float, api_retries: int = 5,
):
    last_err = None
    for attempt in range(api_retries):
        try:
            kwargs = {"model": model_name, "messages": messages, "tools": tools}
            try:
                return await client.chat.completions.create(**kwargs, temperature=temperature)
            except Exception as e:
                if "temperature" in str(e).lower():
                    return await client.chat.completions.create(**kwargs)
                raise
        except Exception as e:
            last_err = e
            if attempt + 1 >= api_retries:
                raise
            await asyncio.sleep(min(2 ** attempt, 10))
    raise last_err


# ═══════════════════════════════════════════════════════════════════════════════
# Agent <-> UserSim <-> tools loop (the Claude and OpenAI paths are maintained separately)
# ═══════════════════════════════════════════════════════════════════════════════


async def _agent_step_openai(
    client, model_name, openai_messages, tools, tool_state, temperature, api_retries,
):
    """One OpenAI step: returns (assistant_msg_for_eval, tool_responses or None, text_content)."""
    response = await _call_openai(
        client, model_name, openai_messages, tools, temperature, api_retries
    )
    msg = response.choices[0].message
    text_content = msg.content or ""
    assistant_msg = {"role": "assistant", "content": text_content}
    if msg.tool_calls:
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in msg.tool_calls
        ]
        tool_call_tasks = [tool_state.call_tool(tc) for tc in assistant_msg["tool_calls"]]
        tool_responses = await asyncio.gather(*tool_call_tasks)
    else:
        tool_responses = None
    return assistant_msg, tool_responses, text_content


async def _agent_step_claude(
    client, model_name, openai_messages, claude_messages, tools, tool_state,
    temperature, api_retries, system_prompt,
):
    """One Claude step: maintains both message views and returns (assistant_msg_for_eval, tool_responses or None, text_content)."""
    claude_tools = _openai_tools_to_claude(tools)
    response = await _call_claude(
        client, model_name, system_prompt, claude_messages, claude_tools,
        temperature, api_retries=api_retries,
    )
    content_blocks = response.content or []
    text_parts = []
    tool_use_blocks = []
    for b in content_blocks:
        btype = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
        if btype == "text":
            text_parts.append(b["text"] if isinstance(b, dict) else b.text)
        elif btype == "tool_use":
            tool_use_blocks.append(b if isinstance(b, dict) else b.model_dump())

    claude_messages.append({"role": "assistant", "content": content_blocks})

    text_content = "\n".join(text_parts)
    assistant_msg = {"role": "assistant", "content": text_content}
    if tool_use_blocks:
        assistant_msg["tool_calls"] = [
            {
                "id": tu["id"],
                "type": "function",
                "function": {
                    "name": tu["name"],
                    "arguments": json.dumps(tu.get("input", {}), ensure_ascii=False),
                },
            }
            for tu in tool_use_blocks
        ]
        tool_call_tasks = [tool_state.call_tool(tc) for tc in assistant_msg["tool_calls"]]
        tool_responses = await asyncio.gather(*tool_call_tasks)

        # Append tool_result blocks to claude_messages for next call
        claude_tool_results = []
        for tu, tr in zip(tool_use_blocks, tool_responses):
            tr_content = tr.get("content", "") if isinstance(tr, dict) else ""
            claude_tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": tr_content[:8000] if isinstance(tr_content, str) else str(tr_content)[:8000],
            })
        claude_messages.append({"role": "user", "content": claude_tool_results})
    else:
        tool_responses = None
    return assistant_msg, tool_responses, text_content


async def run_interactive_loop(
    client: AsyncOpenAI,
    model_name: str,
    tool_state: VitaBenchToolState,
    user_profile: dict,
    instructions: str,
    domain: str,
    env_time: str,
    max_steps: int = 40,
    temperature: float = 0.0,
    api_retries: int = 5,
) -> list[dict]:
    """
    The closed-model evaluation loop in interactive mode:
      1. System turn plus the agent greeting
      2. The UserSim's first user message
      3. Each round: the agent acts; with a tool_call it runs and continues, otherwise the UserSim takes over
    Returns the full messages in the OpenAI view, for eval_reward
    """
    use_claude = "claude" in model_name.lower()

    sys_msg = build_system_message(
        env_time=env_time,
        coach_summary="",
        rubrics=None,            # evaluation must not leak the rubrics
        domain=domain,
    )

    openai_messages: list[dict] = [
        sys_msg,
        {"role": "assistant", "content": "你好，请问有什么可以帮您的？"},
    ]

    # Claude-native messages (no system turn; the agent greeting is skipped to avoid two leading assistant turns)
    claude_messages: list[dict] = []

    # the UserSim's first message
    first_user_msg = await simulate_user_response(
        openai_messages, user_profile or {}, instructions, domain=domain,
    )
    if is_stop(first_user_msg):
        openai_messages.append({"role": "user", "content": first_user_msg})
        return openai_messages

    openai_messages.append({"role": "user", "content": first_user_msg})
    if use_claude:
        # Claude's view starts directly with the user turn
        claude_messages.append({"role": "user", "content": first_user_msg})

    tools = tool_state.tools

    for step_idx in range(max_steps):
        try:
            if use_claude:
                assistant_msg, tool_responses, text_content = await _agent_step_claude(
                    client, model_name, openai_messages, claude_messages,
                    tools, tool_state, temperature, api_retries,
                    system_prompt=sys_msg["content"],
                )
            else:
                assistant_msg, tool_responses, text_content = await _agent_step_openai(
                    client, model_name, openai_messages, tools, tool_state,
                    temperature, api_retries,
                )
        except Exception as e:
            logger.warning(f"[interactive_loop] agent call failed: {e}")
            openai_messages.append({"role": "assistant", "content": f"[API ERROR: {e}]"})
            return openai_messages

        openai_messages.append(assistant_msg)

        if tool_responses is not None:
            openai_messages.extend(tool_responses)
            continue  # after the tool runs, the next turn is the agent again

        # no tool_call, so check for STOP
        if is_stop(text_content):
            break

        # otherwise hand off to the UserSim
        try:
            user_msg = await simulate_user_response(
                openai_messages, user_profile or {}, instructions, domain=domain,
            )
        except Exception as e:
            logger.warning(f"[interactive_loop] user_sim failed: {e}")
            break

        openai_messages.append({"role": "user", "content": user_msg})
        if use_claude:
            claude_messages.append({"role": "user", "content": user_msg})

        if is_stop(user_msg):
            break
    else:
        # max_steps exhausted, so ask the agent to summarise
        openai_messages.append({"role": "user", "content": "对话轮次已用完，请总结你已完成的操作。"})

    return openai_messages


# ═══════════════════════════════════════════════════════════════════════════════
# Eval driver
# ═══════════════════════════════════════════════════════════════════════════════


async def evaluate_single_task(
    sample_data: dict,
    dataset_name: str,
    client: AsyncOpenAI,
    model_name: str,
    args_ns: Namespace,
    semaphore: asyncio.Semaphore,
    max_steps: int,
    temperature: float,
) -> dict:
    async with semaphore:
        meta = sample_data.get("metadata", {}) or {}
        domain = meta.get("domain", "ota")
        task_id = meta.get("task_id", "?")
        query = sample_data.get("query", "") or "你好"

        # Load task data (env + instructions + user_profile)
        try:
            task_data = load_vitabench_task_by_id(
                task_id, domain, data_dir=config.vitabench_data_dir
            )
        except Exception as e:
            logger.warning(f"[eval] load_task_data failed task_id={task_id}: {e}")
            return {
                "dataset": dataset_name,
                "task_id": task_id,
                "domain": domain,
                "reward": 0.0,
                "full_pass": False,
                "rubric_rate": 0.0,
                "rubrics_met": 0,
                "rubrics_total": 0,
                "actual_tool_names": [],
                "error": f"load_task_data failed: {e}",
            }

        # Build per-sample tool state (each task has its own DB state)
        try:
            tool_state = VitaBenchToolState(task_data, domain=domain)
        except Exception as e:
            logger.warning(f"[eval] tool_state init failed task_id={task_id}: {e}")
            return {
                "dataset": dataset_name,
                "task_id": task_id,
                "domain": domain,
                "reward": 0.0,
                "full_pass": False,
                "rubric_rate": 0.0,
                "rubrics_met": 0,
                "rubrics_total": 0,
                "actual_tool_names": [],
                "error": f"tool_state init failed: {e}",
            }

        # User profile / instructions
        user_profile = meta.get("user_profile") or task_data.get("user_scenario", {}).get("user_profile", {})
        instructions = meta.get("instructions") or task_data.get("instructions", "")
        env_time = meta.get("env_time", "")

        # Run interactive loop
        try:
            messages = await run_interactive_loop(
                client, model_name, tool_state, user_profile, instructions,
                domain=domain, env_time=env_time,
                max_steps=max_steps, temperature=temperature,
            )
        except Exception as e:
            logger.warning(f"[eval] interactive_loop failed task_id={task_id}: {e}")
            messages = [
                build_system_message(env_time, "", None, domain),
                {"role": "user", "content": query},
                {"role": "assistant", "content": f"[rollout error: {e}]"},
            ]

        # Build Sample with messages + metadata; then eval_reward
        sample = Sample(
            prompt=query,
            messages=messages,
            metadata=deepcopy(meta),
        )

        try:
            await eval_reward(args_ns, sample)
            reward = sample.reward
            rub_meta = sample.metadata or {}
        except Exception as e:
            logger.warning(f"[eval] eval_reward failed task_id={task_id}: {e}")
            reward = 0.0
            rub_meta = {}

        return {
            "dataset": dataset_name,
            "task_id": task_id,
            "domain": domain,
            "reward": float(reward),
            "full_pass": bool(rub_meta.get("rubric_all_passed", False)),
            "rubric_rate": float(rub_meta.get("rubric_rate", 0.0)),
            "rubrics_met": int(rub_meta.get("rubrics_met", 0)),
            "rubrics_total": int(rub_meta.get("rubrics_total", 0)),
            "actual_tool_names": list(rub_meta.get("actual_tool_names", []) or []),
            "num_tool_calls": int(rub_meta.get("num_tool_calls", 0)),
        }


async def main_async(args):
    # Parse --eval-data NAME:PATH ...
    eval_specs: list[tuple[str, str]] = []
    for spec in args.eval_data:
        if ":" in spec:
            name, path = spec.split(":", 1)
        else:
            base = os.path.basename(spec).replace(".jsonl", "")
            name = base
            path = spec
        eval_specs.append((name, path))

    all_samples: list[tuple[dict, str]] = []
    for name, path in eval_specs:
        with open(path) as f:
            for line in f:
                all_samples.append((json.loads(line), name))
    logger.info(
        "Loaded " + ", ".join(
            f"{sum(1 for s in all_samples if s[1]==n)} {n}" for n, _ in eval_specs
        )
    )

    if args.limit > 0:
        all_samples = all_samples[: args.limit]
        logger.info(f"Limited to first {args.limit} samples (total)")

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=300,
        max_retries=3,
    )

    # eval_reward expects a Namespace; pass through anything it might use
    args_ns = Namespace()

    semaphore = asyncio.Semaphore(args.concurrency)
    logger.info(
        f"Running interactive eval on {args.model} "
        f"(N={len(all_samples)}, concurrency={args.concurrency}, max_steps={args.max_steps})"
    )
    t0 = time.time()

    tasks = [
        evaluate_single_task(
            s, ds, client, args.model, args_ns, semaphore,
            max_steps=args.max_steps, temperature=args.temperature,
        )
        for (s, ds) in all_samples
    ]

    results = []
    completed = 0
    for fut in asyncio.as_completed(tasks):
        r = await fut
        results.append(r)
        completed += 1
        if completed % 5 == 0 or completed == len(tasks):
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(tasks) - completed) / rate if rate > 0 else 0
            avg_reward = sum(r["reward"] for r in results) / len(results)
            n_pass = sum(1 for r in results if r["full_pass"])
            logger.info(
                f"[progress] {completed}/{len(tasks)} done, avg_reward={avg_reward:.4f}, "
                f"full_pass={n_pass}/{completed} ({100*n_pass/completed:.1f}%), "
                f"elapsed={elapsed:.0f}s, eta={eta:.0f}s, rate={rate:.2f}/s"
            )

    # Aggregate by domain (cross-cutting across eval_specs since each set already
    # corresponds to one domain, but we collect both views).
    print()
    print("=" * 100)
    print(f"Model: {args.model}    Datasets: {[n for n,_ in eval_specs]}    N={len(results)}")
    print("=" * 100)

    domain_stats = defaultdict(lambda: {"pass": 0, "total": 0, "rewards": [], "rubric": []})
    for r in results:
        d = r["domain"] or r["dataset"]
        domain_stats[d]["total"] += 1
        if r["full_pass"]:
            domain_stats[d]["pass"] += 1
        domain_stats[d]["rewards"].append(r["reward"])
        domain_stats[d]["rubric"].append(r.get("rubric_rate", 0))

    print(f"\n{'Domain':22s}  {'full_pass':16s}  {'avg_reward':12s}  {'rubric_rate':12s}")
    print("-" * 100)
    for d in sorted(domain_stats.keys()):
        s = domain_stats[d]
        pct = 100 * s["pass"] / s["total"] if s["total"] > 0 else 0
        avg_r = sum(s["rewards"]) / len(s["rewards"]) if s["rewards"] else 0
        avg_rub = sum(s["rubric"]) / len(s["rubric"]) if s["rubric"] else 0
        print(f"{d:22s}  {s['pass']:>3}/{s['total']:<3} ({pct:>5.1f}%)  {avg_r:10.4f}    {avg_rub:10.4f}")

    n_pass = sum(1 for r in results if r["full_pass"])
    n = len(results)
    avg_r = sum(r["reward"] for r in results) / n if n > 0 else 0
    avg_rub = sum(r.get("rubric_rate", 0) for r in results) / n if n > 0 else 0
    print("-" * 100)
    pct = 100 * n_pass / n if n > 0 else 0
    print(f"{'OVERALL':22s}  {n_pass:>3}/{n:<3} ({pct:>5.1f}%)  {avg_r:10.4f}    {avg_rub:10.4f}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    safe_model = args.model.replace("/", "_")
    out_file = os.path.join(args.output_dir, f"{safe_model}_results.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved per-sample results to {out_file}")

    summary_file = os.path.join(args.output_dir, f"{safe_model}_summary.json")
    summary = {
        "model": args.model,
        "eval_data": [{"name": n, "path": p} for n, p in eval_specs],
        "n": n,
        "full_pass": n_pass,
        "full_pass_rate": pct / 100,
        "avg_reward": avg_r,
        "avg_rubric_rate": avg_rub,
        "per_domain": {
            d: {
                "pass": s["pass"], "total": s["total"],
                "full_pass_rate": s["pass"] / s["total"] if s["total"] > 0 else 0,
                "avg_reward": sum(s["rewards"]) / len(s["rewards"]) if s["rewards"] else 0,
                "avg_rubric_rate": sum(s["rubric"]) / len(s["rubric"]) if s["rubric"] else 0,
            }
            for d, s in domain_stats.items()
        },
        "elapsed_sec": time.time() - t0,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved summary to {summary_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.2-2025-12-11")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    parser.add_argument(
        "--eval-data", nargs="+", required=True,
        help="NAME:PATH 对，多个用空格分隔。例: ota:.../interactive_ota_full.jsonl delivery:.../interactive_delivery_full.jsonl",
    )
    parser.add_argument("--output-dir", default="./eval_closed_results_vita")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--max-steps", type=int,
        default=int(os.environ.get("MAX_STEPS", config.max_steps)),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: No API key. Set DASHSCOPE_API_KEY/OPENAI_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

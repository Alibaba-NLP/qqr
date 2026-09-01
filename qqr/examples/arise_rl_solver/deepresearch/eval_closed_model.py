"""
Closed-model evaluation script: measures baseline performance on a DeepResearch eval set
such as data/ecr_deepresearch/test.jsonl.

Usage (a single eval set):
    python -m qqr.examples.arise_rl_solver.deepresearch.eval_closed_model \
        --model gpt-5.2-2025-12-11 \
        --eval-data deepresearch_test:/path/to/deepresearch_test.jsonl \
        --output-dir ./eval_results_gpt52 \
        --concurrency 8

Multiple eval sets: pass space-separated NAME:PATH pairs
    --eval-data deepresearch_test:/path/a.jsonl researchrubrics:/path/b.jsonl

Supported models:
  - OpenAI-compatible: gpt-5.2-2025-12-11, gpt-5-2025-08-07, gpt-4.1, a deployed Qwen3-8B and so on
  - Claude (DashScope): claude-sonnet-4-5-20250929, aws.claude-sonnet-4-6 and so on (routed by the "claude" keyword in the model name)

No GPU, slime or sglang is needed: this is pure API calls plus a single MCP server (Google Search).
Reuses reward_model._compute_sample_reward so evaluation matches training.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from openai import AsyncOpenAI

from qqr.schemas import Sample

from . import config
from .reward_model import _compute_sample_reward


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# System prompt, kept identical to build_system_message used during training
# ═══════════════════════════════════════════════════════════════════════════════


def build_system_message(step_idx: int, max_steps: int) -> dict:
    """Matches the style of the training-time student system prompt (no memory)."""
    system_prompt = f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    system_prompt += f"\n\n可调用 {max_steps} 轮工具，已调用 {step_idx} 轮。"
    system_prompt += "\n\n你是一个研究型问答助手，需要通过 google_search 工具检索资料后再回答。"
    system_prompt += "\n严禁在未调用任何工具的情况下直接回答需要事实依据的问题。"
    if step_idx >= max_steps:
        system_prompt += "\n\n工具调用次数已用完，请基于已检索的信息直接回答。"
    return {"role": "system", "content": system_prompt}


# ═══════════════════════════════════════════════════════════════════════════════
# Claude-native adapter (claude-* models on DashScope require the Anthropic protocol)
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


async def run_agent_loop_claude(
    client: AsyncOpenAI, model_name: str, query: str,
    tools: list[dict], mcp_state, max_steps: int = 5,
    temperature: float = 0.8, api_retries: int = 5,
) -> list[dict]:
    claude_tools = _openai_tools_to_claude(tools)
    claude_messages: list[dict] = [{"role": "user", "content": query}]

    openai_messages: list[dict] = [
        build_system_message(0, max_steps),
        {"role": "user", "content": query},
    ]

    for step_idx in range(max_steps):
        system_prompt = build_system_message(step_idx, max_steps)["content"]
        openai_messages[0]["content"] = system_prompt

        try:
            response = await _call_claude(
                client, model_name, system_prompt,
                claude_messages, claude_tools, temperature,
                api_retries=api_retries,
            )
        except Exception as e:
            logger.warning(f"[claude] API call failed: {e}")
            openai_messages.append({"role": "assistant", "content": f"[API ERROR: {e}]"})
            return openai_messages

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

        openai_assistant = {"role": "assistant", "content": "\n".join(text_parts)}
        if tool_use_blocks:
            openai_assistant["tool_calls"] = [
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
        openai_messages.append(openai_assistant)

        if not tool_use_blocks:
            break

        tool_call_tasks = [mcp_state.call_tool(tc) for tc in openai_assistant["tool_calls"]]
        tool_responses = await asyncio.gather(*tool_call_tasks)
        openai_messages.extend(tool_responses)

        claude_tool_results = []
        for tu, tr in zip(tool_use_blocks, tool_responses):
            tr_content = tr.get("content", "") if isinstance(tr, dict) else ""
            claude_tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": tr_content[:8000] if isinstance(tr_content, str) else str(tr_content)[:8000],
            })
        claude_messages.append({"role": "user", "content": claude_tool_results})

    return openai_messages


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI-compatible agent loop
# ═══════════════════════════════════════════════════════════════════════════════


async def run_agent_loop(
    client: AsyncOpenAI,
    model_name: str,
    query: str,
    tools: list[dict],
    mcp_state,
    max_steps: int = 5,
    temperature: float = 0.8,
    api_retries: int = 5,
) -> list[dict]:
    if "claude" in model_name.lower():
        return await run_agent_loop_claude(
            client, model_name, query, tools, mcp_state,
            max_steps=max_steps, temperature=temperature, api_retries=api_retries,
        )

    messages: list[dict] = [
        build_system_message(0, max_steps),
        {"role": "user", "content": query},
    ]

    for step_idx in range(max_steps):
        messages[0] = build_system_message(step_idx, max_steps)

        response = None
        for attempt in range(api_retries):
            try:
                kwargs = {"model": model_name, "messages": messages, "tools": tools}
                try:
                    response = await client.chat.completions.create(
                        **kwargs, temperature=temperature
                    )
                except Exception as e:
                    if "temperature" in str(e).lower():
                        response = await client.chat.completions.create(**kwargs)
                    else:
                        raise
                break
            except Exception as e:
                if attempt + 1 >= api_retries:
                    logger.warning(f"[agent_loop] API call failed after {api_retries} retries: {e}")
                    messages.append({"role": "assistant", "content": f"[API ERROR: {e}]"})
                    return messages
                await asyncio.sleep(min(2 ** attempt, 10))

        if response is None:
            return messages

        choice = response.choices[0]
        msg = choice.message

        assistant_msg = {
            "role": "assistant",
            "content": msg.content or "",
        }
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
        messages.append(assistant_msg)

        if not msg.tool_calls:
            break

        tool_call_tasks = [mcp_state.call_tool(tc) for tc in assistant_msg["tool_calls"]]
        tool_responses = await asyncio.gather(*tool_call_tasks)
        messages.extend(tool_responses)

    return messages


# ═══════════════════════════════════════════════════════════════════════════════
# Eval driver
# ═══════════════════════════════════════════════════════════════════════════════


async def evaluate_single_query(
    sample_data: dict,
    dataset_name: str,
    client: AsyncOpenAI,
    model_name: str,
    mcp_state,
    semaphore: asyncio.Semaphore,
    max_steps: int = 5,
    temperature: float = 0.8,
) -> dict:
    async with semaphore:
        query = sample_data.get("query", "")
        meta = sample_data.get("metadata", {})

        tools = mcp_state.tools

        try:
            messages = await run_agent_loop(
                client, model_name, query, tools, mcp_state,
                max_steps=max_steps, temperature=temperature,
            )
        except Exception as e:
            logger.warning(f"[eval] rollout failed: {e}")
            messages = [
                build_system_message(0, max_steps),
                {"role": "user", "content": query},
                {"role": "assistant", "content": f"[rollout error: {e}]"},
            ]

        sample = Sample(
            prompt=query,
            messages=messages,
            metadata=deepcopy(meta),
        )

        try:
            result = await _compute_sample_reward(sample)
        except Exception as e:
            logger.warning(f"[eval] reward computation failed: {e}")
            result = {
                "reward": 0.0,
                "rubric_pass_rate": 0.0,
                "rubric_all_passed": False,
                "rubrics_total": 0,
                "rubrics_met": 0,
                "weighted": False,
            }

        result["dataset"] = dataset_name
        result["full_pass"] = bool(result.get("rubric_all_passed", False))
        result["query"] = query[:80]
        # Drop heavy debug fields from saved per-sample summary
        result.pop("rubric_details", None)
        return result


async def main_async(args):
    # Parse --eval-data: list of NAME:PATH (or PATH-only auto-named)
    eval_specs: list[tuple[str, str]] = []
    for spec in args.eval_data:
        if ":" in spec:
            name, path = spec.split(":", 1)
        else:
            base = os.path.basename(spec)
            name = base.replace(".jsonl", "")
            path = spec
        eval_specs.append((name, path))

    # Load all samples and tag with dataset name
    all_samples: list[tuple[dict, str]] = []
    for name, path in eval_specs:
        with open(path) as f:
            for line in f:
                all_samples.append((json.loads(line), name))
        logger.info(f"Loaded {sum(1 for s in all_samples if s[1]==name)} samples from {name} ({path})")

    if args.limit > 0:
        all_samples = all_samples[: args.limit]
        logger.info(f"Limited to first {args.limit} samples (total)")

    # Setup MCP state
    from qqr.rollout.agent_rollout import MCPState
    mcp_state = MCPState(config.mcp_manager)
    logger.info("Initializing MCP servers...")
    await mcp_state.get_servers()
    logger.info(f"MCP ready: {len(mcp_state.tools)} tools available")
    for t in mcp_state.tools:
        logger.info(f"  - {t['function']['name']}")

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=180,
        max_retries=3,
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    logger.info(
        f"Running eval on {args.model} "
        f"(N={len(all_samples)}, concurrency={args.concurrency}, max_steps={args.max_steps})"
    )
    t0 = time.time()

    tasks = [
        evaluate_single_query(
            s, ds, client, args.model, mcp_state, semaphore,
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
        if completed % 10 == 0 or completed == len(tasks):
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

    # Aggregate by dataset
    print()
    print("=" * 100)
    print(f"Model: {args.model}    Datasets: {[n for n,_ in eval_specs]}    N={len(results)}")
    print("=" * 100)

    ds_stats = defaultdict(lambda: {"pass": 0, "total": 0, "rewards": [], "rubric": []})
    for r in results:
        d = r["dataset"]
        ds_stats[d]["total"] += 1
        if r["full_pass"]:
            ds_stats[d]["pass"] += 1
        ds_stats[d]["rewards"].append(r["reward"])
        ds_stats[d]["rubric"].append(r.get("rubric_pass_rate", 0))

    print(f"\n{'Dataset':28s}  {'full_pass':16s}  {'avg_reward':12s}  {'rubric_rate':12s}")
    print("-" * 100)
    for d in sorted(ds_stats.keys()):
        s = ds_stats[d]
        pct = 100 * s["pass"] / s["total"] if s["total"] > 0 else 0
        avg_r = sum(s["rewards"]) / len(s["rewards"]) if s["rewards"] else 0
        avg_rub = sum(s["rubric"]) / len(s["rubric"]) if s["rubric"] else 0
        print(f"{d:28s}  {s['pass']:>3}/{s['total']:<3} ({pct:>5.1f}%)  {avg_r:10.4f}    {avg_rub:10.4f}")

    n_pass = sum(1 for r in results if r["full_pass"])
    n = len(results)
    avg_r = sum(r["reward"] for r in results) / n if n > 0 else 0
    avg_rub = sum(r.get("rubric_pass_rate", 0) for r in results) / n if n > 0 else 0
    print("-" * 100)
    pct = 100 * n_pass / n if n > 0 else 0
    print(f"{'OVERALL':28s}  {n_pass:>3}/{n:<3} ({pct:>5.1f}%)  {avg_r:10.4f}    {avg_rub:10.4f}")
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
        "per_dataset": {
            d: {
                "pass": s["pass"], "total": s["total"],
                "full_pass_rate": s["pass"] / s["total"] if s["total"] > 0 else 0,
                "avg_reward": sum(s["rewards"]) / len(s["rewards"]) if s["rewards"] else 0,
                "avg_rubric_rate": sum(s["rubric"]) / len(s["rubric"]) if s["rubric"] else 0,
            }
            for d, s in ds_stats.items()
        },
        "elapsed_sec": time.time() - t0,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved summary to {summary_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.2-2025-12-11")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    parser.add_argument(
        "--eval-data", nargs="+", required=True,
        help="Path(s) to eval JSONL. Use NAME:PATH for explicit dataset name, otherwise basename is used.",
    )
    parser.add_argument("--output-dir", default="./eval_closed_results")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", config.max_steps)))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: No API key. Set DASHSCOPE_API_KEY/OPENAI_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

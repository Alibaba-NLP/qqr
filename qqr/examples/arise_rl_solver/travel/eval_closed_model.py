"""
Closed-model evaluation script: measures baseline performance of models such as gpt-5.2 on the 500-query Travel rubrics eval set.

Usage:
    python -m qqr.examples.arise_rl_solver.travel.eval_closed_model \
        --model gpt-5.2-2025-12-11 \
        --eval-data /path/to/test.jsonl \
        --output-dir ./eval_results_gpt52 \
        --concurrency 8

No GPU, slime or sglang is needed: this is pure API calls plus MCP tools.
Reuses reward_model._compute_sample_reward so evaluation matches training.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
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
# Agent loop on closed-source API
# ═══════════════════════════════════════════════════════════════════════════════


def build_system_message(step_idx: int, max_steps: int) -> dict:
    """Identical to build_system_message used in training (no memory; the student/eval path)."""
    system_prompt = f"当前时间: {datetime.now().strftime('%d/%m/%Y, %H:%M')}"
    system_prompt += f"\n\n可调用{max_steps}轮工具，已调用{step_idx}轮。"
    system_prompt += "\n\n重要：你必须先使用工具查询真实数据，再根据工具返回的结果回答用户问题。严禁在未调用任何工具的情况下直接回答。"
    if step_idx >= max_steps:
        system_prompt += "\n\n请直接回答，不要使用工具。"
    return {"role": "system", "content": system_prompt}


def _openai_tools_to_claude(tools: list[dict]) -> list[dict]:
    """Convert OpenAI {"type":"function","function":{...}} → Claude {"type":"custom","name":..."input_schema":...}."""
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
    """Single Claude API call via DashScope wrapper."""
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
    tools: list[dict], mcp_state, max_steps: int = 8,
    temperature: float = 0.8, api_retries: int = 5,
) -> list[dict]:
    """
    Claude-native agent loop. Maintains:
      - claude_messages: Claude-native message list (sent to API)
      - openai_messages: OpenAI-style equivalent (returned for reward computation)

    Claude protocol:
      - system prompt → extra_body["system"], NOT in messages
      - tools → {"type":"custom","name":...,"input_schema":...}
      - assistant response content is list of blocks: text | tool_use
      - tool result → {"role":"user","content":[{"type":"tool_result","tool_use_id":...,"content":...}]}
    """
    claude_tools = _openai_tools_to_claude(tools)
    system_prompt_initial = build_system_message(0, max_steps)["content"]

    # Tracks user/assistant turns in Claude format
    claude_messages: list[dict] = [{"role": "user", "content": query}]

    # OpenAI-format equivalent used by reward_model for evaluation
    openai_messages: list[dict] = [
        build_system_message(0, max_steps),
        {"role": "user", "content": query},
    ]

    for step_idx in range(max_steps):
        system_prompt = build_system_message(step_idx, max_steps)["content"]
        openai_messages[0]["content"] = system_prompt  # keep openai system aligned

        try:
            response = await _call_claude(
                client, model_name, system_prompt,
                claude_messages, claude_tools, temperature,
                api_retries=api_retries,
            )
        except Exception as e:
            logger.warning(f"[claude] API call failed after retries: {e}")
            openai_messages.append({"role": "assistant", "content": f"[API ERROR: {e}]"})
            return openai_messages

        # Claude response: content is list of blocks
        content_blocks = response.content or []

        # Build text + tool_use lists
        text_parts = []
        tool_use_blocks = []
        for b in content_blocks:
            btype = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
            if btype == "text":
                text_parts.append(b["text"] if isinstance(b, dict) else b.text)
            elif btype == "tool_use":
                tool_use_blocks.append(b if isinstance(b, dict) else b.model_dump())

        # Append assistant message (Claude format) for next API call
        claude_messages.append({"role": "assistant", "content": content_blocks})

        # Build OpenAI assistant message (for reward evaluation)
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

        # No tool_use → final answer, break
        if not tool_use_blocks:
            break

        # Execute tools via MCP (we have OpenAI-format tool_calls, MCPState expects these)
        tool_call_tasks = [mcp_state.call_tool(tc) for tc in openai_assistant["tool_calls"]]
        tool_responses = await asyncio.gather(*tool_call_tasks)

        # Append to openai_messages (standard tool role messages, used by reward)
        openai_messages.extend(tool_responses)

        # Append to claude_messages (Claude format: role=user, content=tool_result blocks)
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


async def run_agent_loop(
    client: AsyncOpenAI,
    model_name: str,
    query: str,
    tools: list[dict],
    mcp_state,
    max_steps: int = 8,
    temperature: float = 0.8,
    api_retries: int = 5,
) -> list[dict]:
    """
    Multi-turn agent loop: query -> assistant (possibly a tool_call) -> tool_result -> ... -> final answer
    Returns the full message list, for rubric and tool scoring

    Auto-dispatches to Claude loop if model name contains "claude".
    """
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
        # update step_idx in the system message
        messages[0] = build_system_message(step_idx, max_steps)

        # call the closed-source model
        response = None
        for attempt in range(api_retries):
            try:
                # some models (gpt-5.x among them) reject a temperature argument
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

        # Construct assistant message dict
        assistant_msg = {
            "role": "assistant",
            "content": msg.content or "",
        }
        # tool_calls
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

        # No tool calls → final answer; break
        if not msg.tool_calls:
            break

        # Execute tool calls
        tool_call_tasks = []
        for tc in assistant_msg["tool_calls"]:
            tool_call_tasks.append(mcp_state.call_tool(tc))
        tool_responses = await asyncio.gather(*tool_call_tasks)
        # tool_responses is a list[dict]; each dict is a tool-role message
        messages.extend(tool_responses)

    return messages


# ═══════════════════════════════════════════════════════════════════════════════
# Eval driver
# ═══════════════════════════════════════════════════════════════════════════════


async def evaluate_single_query(
    sample_data: dict,
    client: AsyncOpenAI,
    model_name: str,
    mcp_state,
    semaphore: asyncio.Semaphore,
    max_steps: int = 8,
    temperature: float = 0.8,
) -> dict:
    """Roll out and score a single query, returning sample-level results."""
    async with semaphore:
        query = sample_data.get("query", "")
        meta = sample_data.get("metadata", {})

        # Get tools (OpenAI format already)
        tools = mcp_state.tools

        # Run agent loop
        try:
            messages = await run_agent_loop(
                client, model_name, query, tools, mcp_state,
                max_steps=max_steps, temperature=temperature,
            )
        except Exception as e:
            logger.warning(f"[eval] rollout failed for query={query[:50]}...: {e}")
            messages = [
                build_system_message(0, max_steps),
                {"role": "user", "content": query},
                {"role": "assistant", "content": f"[rollout error: {e}]"},
            ]

        # Build Sample for reward computation
        sample = Sample(
            prompt=query,
            messages=messages,
            metadata=deepcopy(meta),
        )

        # Compute reward (uses LLM Judge)
        try:
            result = await _compute_sample_reward(sample)
        except Exception as e:
            logger.warning(f"[eval] reward computation failed for query={query[:50]}...: {e}")
            result = {
                "reward": 0.0,
                "full_pass": False,
                "rubric_pass_rate": 0.0,
                "tool_accuracy": 0.0,
            }

        # Attach task type for aggregation
        result["task_type"] = meta.get("task_type", "unknown")
        result["query"] = query[:80]
        return result


async def main_async(args):
    # 1. Load eval data
    logger.info(f"Loading eval data from {args.eval_data}")
    eval_samples = []
    with open(args.eval_data) as f:
        for line in f:
            eval_samples.append(json.loads(line))
    logger.info(f"Loaded {len(eval_samples)} eval samples")

    if args.limit > 0:
        eval_samples = eval_samples[:args.limit]
        logger.info(f"Limited to first {args.limit} samples")

    # 2. Setup MCP state (gets tools from MCP servers)
    from qqr.rollout.agent_rollout import MCPState
    mcp_state = MCPState(config.mcp_manager)
    logger.info("Initializing MCP servers...")
    await mcp_state.get_servers()
    logger.info(f"MCP ready: {len(mcp_state.tools)} tools available")
    for t in mcp_state.tools:
        logger.info(f"  - {t['function']['name']}")

    # 3. Setup OpenAI client
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=180,
        max_retries=3,
    )

    # 4. Concurrent eval
    semaphore = asyncio.Semaphore(args.concurrency)
    logger.info(f"Running eval on {args.model} with concurrency={args.concurrency}")
    t0 = time.time()

    tasks = [
        evaluate_single_query(
            s, client, args.model, mcp_state, semaphore,
            max_steps=args.max_steps, temperature=args.temperature,
        )
        for s in eval_samples
    ]

    # Progress tracking
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

    # 5. Aggregate by task
    print()
    print("=" * 100)
    print(f"Model: {args.model}    Eval set: {args.eval_data}    N={len(results)}")
    print("=" * 100)

    task_stats = defaultdict(lambda: {"pass": 0, "total": 0, "rewards": [], "rubric": [], "tool": []})
    for r in results:
        t = r["task_type"]
        task_stats[t]["total"] += 1
        if r["full_pass"]:
            task_stats[t]["pass"] += 1
        task_stats[t]["rewards"].append(r["reward"])
        task_stats[t]["rubric"].append(r.get("rubric_pass_rate", 0))
        task_stats[t]["tool"].append(r.get("tool_accuracy", 0))

    print(f"\n{'Task':22s}  {'full_pass':16s}  {'avg_reward':12s}  {'rubric_rate':12s}  {'tool_acc':10s}")
    print("-" * 100)
    for t in sorted(task_stats.keys()):
        s = task_stats[t]
        pct = 100 * s["pass"] / s["total"] if s["total"] > 0 else 0
        avg_r = sum(s["rewards"]) / len(s["rewards"]) if s["rewards"] else 0
        avg_rub = sum(s["rubric"]) / len(s["rubric"]) if s["rubric"] else 0
        avg_tool = sum(s["tool"]) / len(s["tool"]) if s["tool"] else 0
        print(f"{t:22s}  {s['pass']:>3}/{s['total']:<3} ({pct:>5.1f}%)  {avg_r:10.4f}    {avg_rub:10.4f}    {avg_tool:8.4f}")

    # Overall
    n_pass = sum(1 for r in results if r["full_pass"])
    n = len(results)
    avg_r = sum(r["reward"] for r in results) / n if n > 0 else 0
    avg_rub = sum(r.get("rubric_pass_rate", 0) for r in results) / n if n > 0 else 0
    avg_tool = sum(r.get("tool_accuracy", 0) for r in results) / n if n > 0 else 0
    print("-" * 100)
    pct = 100 * n_pass / n if n > 0 else 0
    print(f"{'OVERALL':22s}  {n_pass:>3}/{n:<3} ({pct:>5.1f}%)  {avg_r:10.4f}    {avg_rub:10.4f}    {avg_tool:8.4f}")
    print()

    # 6. Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{args.model.replace('/','_')}_results.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved results to {out_file}")

    # Summary file
    summary_file = os.path.join(args.output_dir, f"{args.model.replace('/','_')}_summary.json")
    summary = {
        "model": args.model,
        "eval_data": args.eval_data,
        "n": n,
        "full_pass": n_pass,
        "full_pass_rate": pct / 100,
        "avg_reward": avg_r,
        "avg_rubric_rate": avg_rub,
        "avg_tool_accuracy": avg_tool,
        "per_task": {
            t: {
                "pass": s["pass"], "total": s["total"],
                "full_pass_rate": s["pass"] / s["total"] if s["total"] > 0 else 0,
                "avg_reward": sum(s["rewards"]) / len(s["rewards"]) if s["rewards"] else 0,
            }
            for t, s in task_stats.items()
        },
        "elapsed_sec": time.time() - t0,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved summary to {summary_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.2-2025-12-11",
                        help="Closed-source model name to evaluate")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"),
                        help="API key (defaults to OPENAI_API_KEY or DASHSCOPE_API_KEY env var)")
    parser.add_argument("--base-url",
                        default=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                        help="API base URL")
    parser.add_argument("--eval-data",
                        default="data/ecr_travel/test.jsonl",
                        help="Path to eval JSONL")
    parser.add_argument("--output-dir", default="./eval_closed_model_results",
                        help="Where to save per-sample results + summary")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Concurrent eval workers")
    parser.add_argument("--max-steps", type=int, default=8,
                        help="Max tool call rounds per query")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (some models like gpt-5.x ignore this)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only evaluate first N samples (0 = all)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: No API key provided. Set OPENAI_API_KEY/DASHSCOPE_API_KEY env or pass --api-key", file=sys.stderr)
        sys.exit(1)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

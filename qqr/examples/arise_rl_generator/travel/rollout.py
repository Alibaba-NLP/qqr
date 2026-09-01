"""
Travel curriculum agent (generator) rollout - Qwen3.5 + RG-KL

Three-phase RG-KL procedure:
  Phase A - student rollout (k_s samples): generate without coach memory, then validate with the solver
  Phase B - memory generation: the gpt-5.2 coach reviews the quality of the student's generated tasks
  Phase C - teacher rollout (k_m samples): generate with coach memory injected, used only to estimate Δ_r

Generator specifics (MCP tool version):
  - The generator obtains real data through the MCP tools (AMap/Transport/WebSearch)
  - Produces query, expected_tools and rubrics
  - The solver (executor service) answers using the same tools
  - reward = R_tool · R_fmt · (1 + R_diff) ∈ [0, 3] (paper Eq. (1)-(5); R_tool includes argument-level checks on expected_tools)
"""

import asyncio
import fcntl
import json
import logging
import math
import os
import re
import threading
from argparse import Namespace
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from qqr import registers
from qqr.rollout.agent_rollout import GenerateState, MCPState
from qqr.rollout.agent_rollout import generate as base_generate
from qqr.schemas import Sample
from qqr.utils.envs import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

from . import config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# RG-KL group coordination state
# ═══════════════════════════════════════════════════════════════════════════════

_group_student_results: dict[int, list[dict]] = defaultdict(list)
_group_student_event:   dict[int, asyncio.Event] = {}
_group_memory:          dict[int, str] = {}
_group_memory_leader:   set[int] = set()
_group_memory_event:    dict[int, asyncio.Event] = {}
_group_coord_lock = asyncio.Lock()

_rollout_step_counter: int = 0
_rollout_step_lock = asyncio.Lock()


async def get_rollout_step() -> int:
    return _rollout_step_counter


async def increment_rollout_step(rollout_batch_size: int = 1) -> int:
    global _rollout_step_counter
    async with _rollout_step_lock:
        _rollout_step_counter += 1
        group_count = _rollout_step_counter
        return (group_count - 1) // max(rollout_batch_size, 1)


async def _get_or_create_events(group_id: int):
    async with _group_coord_lock:
        if group_id not in _group_student_event:
            _group_student_event[group_id] = asyncio.Event()
        if group_id not in _group_memory_event:
            _group_memory_event[group_id] = asyncio.Event()


async def _register_student_done(group_id: int, result_info: dict, k_s: int):
    async with _group_coord_lock:
        _group_student_results[group_id].append(result_info)
        if len(_group_student_results[group_id]) >= k_s:
            _group_student_event[group_id].set()


async def _acquire_memory(group_id: int, k_s: int, args: Namespace) -> str | None:
    await _group_student_event[group_id].wait()

    async with _group_coord_lock:
        is_leader = group_id not in _group_memory_leader
        if is_leader:
            _group_memory_leader.add(group_id)

    if is_leader:
        try:
            student_results = _group_student_results[group_id]
            task_type = student_results[0].get("task_type", "") if student_results else ""
            memory = await _generate_coach_memory(group_id, task_type, student_results, args)
            async with _group_coord_lock:
                _group_memory[group_id] = memory or ""
        except Exception as e:
            logger.error(f"[rg_kl] group={group_id} coach memory error: {e}")
            async with _group_coord_lock:
                _group_memory[group_id] = ""
        finally:
            _group_memory_event[group_id].set()

        result = _group_memory.get(group_id, "")
        logger.info(f"[rg_kl] group={group_id} coach memory {'OK' if result else 'empty'}")
        return result or None
    else:
        await _group_memory_event[group_id].wait()
        result = _group_memory.get(group_id, "")
        return result or None


def cleanup_group_coordination(group_id: int):
    _group_student_results.pop(group_id, None)
    _group_student_event.pop(group_id, None)
    _group_memory.pop(group_id, None)
    _group_memory_leader.discard(group_id)
    _group_memory_event.pop(group_id, None)


# ═══════════════════════════════════════════════════════════════════════════════
# Coach memory generation
# ═══════════════════════════════════════════════════════════════════════════════


async def _generate_coach_memory(
    group_id: int, task_type: str, student_results: list[dict], args: Namespace,
) -> str | None:
    summary, judge_input = await generate_group_summary(group_id, task_type, student_results)
    if summary:
        _save_group_summary(args, task_type, group_id, summary, judge_input)
    return summary or None


async def generate_group_summary(
    group_id: int, task_type: str, group_results: list[dict],
) -> tuple[str, str]:
    rewards = [r.get("reward", 0) for r in group_results]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    reward_dist = dict(Counter([f"{r:.1f}" for r in rewards]))

    sample_summaries = []
    for i, r in enumerate(group_results):
        diag = _reward_diagnosis(
            r.get("reward", 0),
            correct_ratio=r.get("correct_ratio"),
            avg_rubric_pass_rate=r.get("avg_rubric_pass_rate"),
        )
        summary = (
            f"### 第{i+1}次出题\n"
            f"- reward: {r.get('reward', 0):.1f} ({diag})\n"
            f"- 题目: {r.get('generated_query', '(无)')[:200]}\n"
            f"- rubrics 数量: {len(r.get('rubrics', []))}\n"
            f"- expected_tools: {json.dumps(r.get('expected_tools', []), ensure_ascii=False)[:150]}\n"
        )
        sample_summaries.append(summary)

    prompt = f"""你是一个旅行规划出题训练的教练。以下是任务类型「{task_type}」的一组 {len(group_results)} 次出题结果。

## 奖励机制说明（R_G = R_tool · R_fmt · (1 + R_diff)，三角难度奖励）
- reward=0: 门控失败——未调用工具、输出格式错误、或 expected_tools 工具合理性验证失败
- reward=1: 门控通过，但做题者 K 次试验全成功或全失败（难度退化，R_diff=0）
- reward∈(1,3): 难度偏离 K/2 但仍有区分度，随 |c−K/2| 线性衰减
- reward=3（峰值）: 做题者 K 次试验约一半成功（c≈K/2），难度恰好落在 Solver 能力边界

## 本组结果
- 平均 reward: {avg_reward:.2f}
- reward 分布: {reward_dist}

{"".join(sample_summaries)}

## 请生成下一轮出题建议（不超过300字）
1. **问题诊断**：主要问题是什么？
2. **多样性建议**：扩展到哪些场景？
3. **难度调整**：如何增加/降低难度
4. **具体建议**：给出 1-2 个具体的出题方向"""

    summary = await _summary_via_external_llm(prompt)
    return summary, prompt


def _reward_diagnosis(reward: float, correct_ratio: float | None = None,
                      avg_rubric_pass_rate: float | None = None) -> str:
    if reward == 0:
        return "门控失败（未调用工具 / 格式错误 / 工具合理性验证失败）"
    if reward == 1.0:
        return "门控通过，但做题者全做对或全做错（难度退化）"
    hints = []
    if correct_ratio is not None:
        hints.append(f"做对率={correct_ratio:.0%}")
    if avg_rubric_pass_rate is not None:
        hints.append(f"rubric通过率={avg_rubric_pass_rate:.0%}")
    if hints:
        return ", ".join(hints)
    return f"reward={reward:.1f}"


async def _summary_via_external_llm(prompt: str) -> str:
    client = AsyncOpenAI(
        api_key=config.coach_api_key,
        base_url=config.coach_base_url,
        timeout=60,
        max_retries=3,
    )
    try:
        resp = await client.chat.completions.create(
            model=config.coach_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=500,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"[group_summary] Coach LLM call failed: {e}")
        return ""


def _save_group_summary(args: Namespace, task_type: str, group_id: int,
                        summary: str, judge_input: str = ""):
    save_dir = getattr(args, "save", None) or getattr(args, "save_dir", None) or "."
    memory_dir = Path(save_dir) / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    filepath = memory_dir / f"group_summaries_{task_type}.jsonl"

    record = {
        "timestamp": datetime.now().isoformat(),
        "group_id": group_id,
        "task_type": task_type,
        "judge_input": judge_input,
        "summary": summary,
    }
    with open(filepath, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

    logger.info(f"[group_summary] Saved for group {group_id}: {summary[:80]}...")


# ═══════════════════════════════════════════════════════════════════════════════
# RG-KL guided_tokens construction
# ═══════════════════════════════════════════════════════════════════════════════


def compute_rg_kl_guided_tokens(args: Namespace, samples: list[Sample], memory: str | None):
    state = GenerateState(args)
    tokenizer = state.tokenizer

    n_student_guided = 0
    n_identity = 0
    n_failed = 0
    for s in samples:
        if s.index == -1 or not s.tokens or s.rollout_log_probs is None or not s.messages:
            _set_identity(s, tokenizer)
            n_identity += 1
            continue

        source = (s.metadata or {}).get("source", "student")
        is_valid_student = (
            source == "student"
            and bool(s.tokens)
            and bool(s.messages)
            and (s.response_length or 0) > 0
        )

        if is_valid_student and memory:
            ok = _apply_p_m_swap(s, tokenizer, memory)
            if ok:
                n_student_guided += 1
            else:
                _set_identity(s, tokenizer)
                n_failed += 1
                n_identity += 1
        else:
            _set_identity(s, tokenizer)
            n_identity += 1

    logger.info(
        f"[rg_kl] guided_tokens: {n_student_guided} student p_m-swap, "
        f"{n_identity} identity (failed={n_failed})"
    )


def _apply_p_m_swap(s: Sample, tokenizer, memory: str) -> bool:
    """Generator p_m swap: append the coach memory to the last user message (the task prompt)."""
    try:
        messages_with_memory = deepcopy(s.messages)
        last_asst_idx = next(
            (i for i in range(len(messages_with_memory) - 1, -1, -1)
             if messages_with_memory[i].get("role") == "assistant"),
            -1,
        )
        if last_asst_idx < 0:
            return False
        messages_with_memory = messages_with_memory[:last_asst_idx + 1]

        injected = False
        for i in range(len(messages_with_memory) - 1, -1, -1):
            if messages_with_memory[i].get("role") == "user":
                orig_content = messages_with_memory[i]["content"]
                marker = "\n\n【上一轮出题教练反馈】\n"
                if marker not in orig_content:
                    messages_with_memory[i]["content"] = orig_content + marker + memory
                injected = True
                break

        if not injected:
            return False

        clean = [
            {
                "role": "user" if m["role"] == "tool" else m["role"],
                "content": m.get("content", "") or "",
            }
            for m in messages_with_memory
        ]
        full_text = tokenizer.apply_chat_template(
            clean, tokenize=False, add_generation_prompt=False
        )
        full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

        response_length = s.response_length or 0
        original_response = s.tokens[-response_length:] if response_length > 0 else []
        p_m_prompt_length = len(full_tokens) - response_length
        if p_m_prompt_length > 0:
            p_m_prompt_tokens = full_tokens[:p_m_prompt_length]
        else:
            p_m_prompt_tokens = s.tokens[:-response_length] if response_length > 0 else s.tokens[:]

        s.guided_tokens = p_m_prompt_tokens + original_response
        s.guided_total_length = len(s.guided_tokens)
        s.guided_prompt_length = len(p_m_prompt_tokens)
        return True

    except Exception as e:
        logger.warning(f"[rg_kl] p_m swap failed for sample {s.index}: {e}")
        return False


def _set_identity(s: Sample, tokenizer):
    s.guided_tokens = s.tokens[:] if s.tokens else [tokenizer.pad_token_id]
    s.guided_total_length = len(s.guided_tokens)
    s.guided_prompt_length = max(0, len(s.guided_tokens) - (s.response_length or 0))


# ═══════════════════════════════════════════════════════════════════════════════
# Gating coefficient λ(Δ_r)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_rg_kl_coef(delta_r: float, rollout_step: int) -> tuple[float, dict]:
    lambda_0 = config.rg_kl_lambda_0
    delta_threshold = config.rg_kl_delta_threshold
    warmup_iters = config.rg_kl_warmup_iters
    decay_iters = config.rg_kl_decay_iters
    min_ratio = config.rg_kl_min_lambda_ratio
    gate_temperature = config.rg_kl_gate_temperature  # sigmoid gate temperature T

    # reward gate g(Δ_r) = σ((Δ_r − τ) / T); as T → 0 this degenerates to the hard gate 𝟙(Δ_r > τ)
    gate = 1.0 / (1.0 + math.exp(-(delta_r - delta_threshold) / gate_temperature))

    # cosine warm-up w(t): ramps smoothly from 0 to 1 over the first warmup_iters steps
    if warmup_iters > 0:
        progress_w = min(rollout_step / warmup_iters, 1.0)
        warmup_factor = 0.5 * (1.0 - math.cos(math.pi * progress_w))
    else:
        warmup_factor = 1.0

    if decay_iters > 0:
        progress = min(rollout_step / decay_iters, 1.0)
        cos_factor = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        cos_factor = 1.0

    coef = lambda_0 * gate * warmup_factor * cos_factor

    info = {
        "delta_r": delta_r, "gate": gate, "warmup_factor": warmup_factor,
        "cos_factor": cos_factor, "coef": coef, "rollout_step": rollout_step,
    }
    return coef, info


# ═══════════════════════════════════════════════════════════════════════════════
# client
# ═══════════════════════════════════════════════════════════════════════════════

_executor_client: AsyncOpenAI | None = None
_executor_semaphore: asyncio.Semaphore | None = None
_judge_client: AsyncOpenAI | None = None
_judge_semaphore: asyncio.Semaphore | None = None


def get_executor_client() -> AsyncOpenAI:
    global _executor_client
    if _executor_client is None:
        _executor_client = AsyncOpenAI(
            api_key="EMPTY", base_url=config.executor_api_base,
            timeout=180, max_retries=3,
        )
    return _executor_client


def get_executor_semaphore() -> asyncio.Semaphore:
    global _executor_semaphore
    if _executor_semaphore is None:
        _executor_semaphore = asyncio.Semaphore(config.executor_concurrency_limit)
    return _executor_semaphore


def get_judge_client() -> AsyncOpenAI:
    global _judge_client
    if _judge_client is None:
        _judge_client = AsyncOpenAI(
            api_key=config.llm_judge_api_key, base_url=config.llm_judge_base_url,
            timeout=120, max_retries=5,
        )
    return _judge_client


def get_judge_semaphore() -> asyncio.Semaphore:
    global _judge_semaphore
    if _judge_semaphore is None:
        _judge_semaphore = asyncio.Semaphore(config.llm_judge_concurrency_limit)
    return _judge_semaphore


# ═══════════════════════════════════════════════════════════════════════════════
# Main generation entry point
# ═══════════════════════════════════════════════════════════════════════════════


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # MCP server initialisation
    mcp_state = MCPState(config.mcp_manager)
    await mcp_state.get_servers()

    import random
    k = getattr(args, "n_samples_per_prompt", 8)
    k_s = config.k_student
    k_m = config.k_teacher
    if k_s + k_m != k:
        k_s = max(1, min(k - 1, k // 2))
        k_m = k - k_s

    within_idx = sample.index % k
    group_id = sample.index // k
    group_rng = random.Random(group_id)

    # choose the task type
    task_type = None
    if sample.metadata:
        task_type = sample.metadata.get("task_type")
    if not task_type or task_type not in config.TASK_TYPE_PROMPTS:
        task_type = group_rng.choice(config.TASK_TYPES)

    sample.metadata = sample.metadata or {}
    sample.metadata["task_type"] = task_type
    sample.metadata["_is_evaluation"] = evaluation

    # -- Eval mode --
    if evaluation:
        return await _do_curriculum_rollout(
            args, sample, sampling_params, mcp_state, task_type,
            coach_summary="", source="student", group_id=group_id,
            evaluation=True,
        )

    # -- RG-KL disabled --
    if not config.enable_rg_kl:
        return await _do_curriculum_rollout(
            args, sample, sampling_params, mcp_state, task_type,
            coach_summary="", source="student", group_id=group_id,
            evaluation=False,
        )

    # -- RG-KL three phases --
    await _get_or_create_events(group_id)

    if within_idx < k_s:
        # STUDENT path
        result = await _do_curriculum_rollout(
            args, sample, sampling_params, mcp_state, task_type,
            coach_summary="", source="student", group_id=group_id,
            evaluation=False,
        )

        final = result[-1] if isinstance(result, list) else result
        result_info = {
            "reward": final.reward if final.reward is not None else 0.0,
            "generated_query": (final.metadata or {}).get("generated_query", ""),
            "expected_tools": (final.metadata or {}).get("expected_tools", []),
            "rubrics": (final.metadata or {}).get("rubrics", []),
            "correct_ratio": (final.metadata or {}).get("correct_ratio"),
            "avg_rubric_pass_rate": (final.metadata or {}).get("avg_rubric_pass_rate"),
            "task_type": task_type,
        }
        await _register_student_done(group_id, result_info, k_s)
        return result
    else:
        # TEACHER path
        memory = await _acquire_memory(group_id, k_s, args)
        result = await _do_curriculum_rollout(
            args, sample, sampling_params, mcp_state, task_type,
            coach_summary=memory or "", source="teacher", group_id=group_id,
            evaluation=False,
        )
        samples_list = result if isinstance(result, list) else [result]
        for s in samples_list:
            if s.metadata is None:
                s.metadata = {}
            s.metadata["memory"] = memory or ""
        return result


async def _do_curriculum_rollout(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    mcp_state: MCPState,
    task_type: str,
    coach_summary: str,
    source: str,
    group_id: int,
    evaluation: bool,
) -> Sample | list[Sample]:
    """A single generation rollout: generate -> parse -> solver validation -> reward."""
    # build the task prompt
    task_prompt = config.TASK_TYPE_PROMPTS.get(task_type, "")
    if coach_summary:
        task_prompt += f"\n\n【上一轮出题教练反馈】\n{coach_summary}"

    sample.messages = [
        {"role": "system", "content": config.query_generation_system_prompt},
        {"role": "user", "content": f"请先调用工具获取数据，然后生成一个旅行规划问题：\n\n{task_prompt}"},
    ]
    sample.prompt = task_prompt
    sample.metadata = sample.metadata or {}
    sample.metadata["has_memory"] = bool(coach_summary)
    sample.metadata["source"] = source

    # generation by the generator
    samples = await curriculum_generate_with_tool(args, sample, sampling_params, mcp_state)

    if evaluation:
        final = samples[-1]
        await execute_and_evaluate(args, final, mcp_state)
        return final
    else:
        for s in samples:
            if s.index >= 0:
                await execute_and_evaluate(args, s, mcp_state)

        valid = [s for s in samples if s.rollout_log_probs is not None and s.response_length > 0]
        return valid if valid else [samples[-1]]


# ═══════════════════════════════════════════════════════════════════════════════
# multi-round tool-calling generation (Qwen3.5-compatible)
# ═══════════════════════════════════════════════════════════════════════════════


async def curriculum_generate_with_tool(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    mcp_state: MCPState,
    max_tool_rounds: int = config.max_steps,
) -> list[Sample]:
    state = GenerateState(args)

    # Qwen3.5 dynamic prompt selection
    ckpt_normalized = re.sub(r'[._]+', '.', state.args.hf_checkpoint.lower())
    if "qwen3.5" in ckpt_normalized:
        prompter = registers.prompt["qwen3.5"]()
    else:
        prompter = registers.prompt["qwen3"]()

    tools = mcp_state.tools if mcp_state.tools else None
    samples = []
    curriculum_tool_calls = []

    samples.append(
        Sample(
            group_index=sample.group_index,
            index=sample.index,
            messages=deepcopy(sample.messages),
            prompt=sample.prompt,
            label=sample.label,
            status=Sample.Status.PENDING,
            metadata=sample.metadata or {},
            train_metadata={"tools": tools},
        )
    )
    sample = samples[-1]
    response_content = ""

    for step_idx in range(max_tool_rounds):
        sample = await base_generate(args, sample, sampling_params)
        response_content = sample.response.removesuffix(state.tokenizer.eos_token).strip()
        parsed = prompter.parse_assistant_content(response_content, tools=tools)
        tool_calls = parsed.get("tool_calls", [])

        if not tool_calls:
            sample.messages.append(prompter.parse_assistant_content(response_content, tools=tools))
            break

        assistant_msg = prompter.parse_assistant_content(response_content, tools=tools)
        # the prompter returns tool_calls[i].function.arguments as a JSON string,
        # but Qwen3.5's chat_template uses `arguments | items` and expects a dict, so rendering would blow up.
        # normalise here so the next round's apply_chat_template renders correctly.
        normalized_tool_calls = []
        for tc in tool_calls:
            tc_copy = deepcopy(tc)
            func = tc_copy.get("function") or {}
            tc_args = func.get("arguments")
            if isinstance(tc_args, str):
                try:
                    func["arguments"] = json.loads(tc_args)
                except (json.JSONDecodeError, TypeError):
                    func["arguments"] = {}
            tc_copy["function"] = func
            normalized_tool_calls.append(tc_copy)
        assistant_msg["tool_calls"] = normalized_tool_calls
        sample.messages.append(assistant_msg)

        for tc in tool_calls:
            curriculum_tool_calls.append(tc)
            try:
                tool_result = await mcp_state.call_tool(tc)
                result_content = tool_result.get("content", str(tool_result))
                if len(result_content) > 2000:
                    result_content = result_content[:2000] + "\n...(结果已截断)"
            except Exception as e:
                result_content = f"工具调用失败: {str(e)}"

            sample.messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result_content,
            })

        logger.info(
            f"[curriculum] step {step_idx}: called "
            f"{[tc.get('function', {}).get('name', tc.get('name', '')) for tc in tool_calls]}"
        )

        samples.append(
            Sample(
                group_index=sample.group_index,
                index=sample.index,
                messages=deepcopy(sample.messages),
                prompt=sample.prompt,
                label=sample.label,
                status=Sample.Status.PENDING,
                metadata=sample.metadata,
                train_metadata={"tools": tools},
            )
        )
        sample = samples[-1]

    else:
        sample.messages[0] = {
            "role": "system",
            "content": sample.messages[0]["content"]
            + "\n\n你已完成工具调用，现在请直接输出最终的 JSON 结果，不要再调用工具。",
        }
        samples.append(
            Sample(
                group_index=sample.group_index,
                index=sample.index,
                messages=deepcopy(sample.messages),
                prompt=sample.prompt,
                label=sample.label,
                status=Sample.Status.PENDING,
                metadata=sample.metadata,
                train_metadata=None,
            )
        )
        sample = samples[-1]
        sample = await base_generate(args, sample, sampling_params)
        response_content = sample.response.removesuffix(state.tokenizer.eos_token).strip()
        sample.messages.append({"role": "assistant", "content": response_content})

    # parse the JSON output
    generated_query, expected_tools, rubrics = parse_curriculum_output(response_content)

    # extract the tool results corresponding to expected_tools
    tool_result_map = {}
    for msg in sample.messages:
        if msg.get("role") == "tool":
            tool_result_map[msg.get("tool_call_id", "")] = msg.get("content", "")

    expected_tools_results = []
    for et in expected_tools:
        et_name = et.get("name", "")
        result_content = ""
        for tc in reversed(curriculum_tool_calls):
            tc_name = tc.get("function", {}).get("name", tc.get("name", ""))
            tc_id = tc.get("id", "")
            if tc_name == et_name and tc_id in tool_result_map:
                result_content = tool_result_map[tc_id]
                break
        expected_tools_results.append({"name": et_name, "result": result_content})

    sample.metadata["generated_query"] = generated_query
    sample.metadata["expected_tools"] = expected_tools
    sample.metadata["expected_tools_results"] = expected_tools_results
    sample.metadata["rubrics"] = rubrics
    sample.metadata["curriculum_raw_output"] = response_content
    sample.metadata["curriculum_tool_calls"] = curriculum_tool_calls

    valid_samples = [s for s in samples if s.rollout_log_probs is not None and s.response_length > 0]
    if not valid_samples:
        sample = samples[-1]
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        valid_samples = [sample]

    return valid_samples


def parse_curriculum_output(response: str) -> tuple[str, list[dict], list[str]]:
    clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    json_match = re.search(r"```json\s*(\{.*?\})\s*```", clean_response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return data.get("query", ""), data.get("expected_tools", []), data.get("rubrics", [])
        except json.JSONDecodeError:
            pass

    try:
        start = clean_response.find("{")
        end = clean_response.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(clean_response[start:end])
            return data.get("query", ""), data.get("expected_tools", []), data.get("rubrics", [])
    except json.JSONDecodeError:
        pass

    return clean_response, [], []


# ═══════════════════════════════════════════════════════════════════════════════
# Rubric scoring
# ═══════════════════════════════════════════════════════════════════════════════


async def evaluate_rubrics(messages: list[dict], rubrics: list[str], query: str) -> tuple[float, list[dict]]:
    if not rubrics:
        return 1.0, []

    trajectory_content = _format_trajectory(messages)
    current_rubrics = json.dumps([
        {"rubric_idx": f"rubric_{i}", "rubric": r, "justification": "尚未评估", "meetExpectation": False}
        for i, r in enumerate(rubrics)
    ], ensure_ascii=False, indent=2)

    system_prompt = f"""# 用户完整指令
{query}

# 背景说明
- 这是一个user与assistant之间的对话场景，其中assistant可以调用工具获取信息和完成操作，工具返回结果将以tool开头
- 你需要评估用户指令是否被完成

# 任务
- 基于对话内容，更新得分点rubric的状态

# 判定原则
- **宽松判定**：只要assistant的回复或工具调用**大致涉及**了该rubric要求的内容即可判为满足（true）
- 只有在assistant**完全没有提及**或**明显答错**时，才判为不满足（false）

# 格式要求
```json
[
  {{"rubric_idx": "rubric_0", "rubric": "<复述规则>", "justification": "<解释>", "meetExpectation": true}},
  ...
]
```"""

    user_prompt = f"# Input\n<trajectory_content>\n{trajectory_content}\n</trajectory_content>\n\n<current_rubrics>\n{current_rubrics}\n</current_rubrics>"

    client = get_judge_client()
    semaphore = get_judge_semaphore()

    details = None
    for attempt in range(3):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=config.llm_judge_model,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    temperature=0.0, max_completion_tokens=4000,
                )
            result_text = resp.choices[0].message.content or ""
            details = _parse_rubric_results(result_text, rubrics)
            if not all(d["justification"] == "解析失败" for d in details):
                break
        except Exception as e:
            logger.warning(f"[evaluate_rubrics] attempt {attempt+1} failed: {e}")

    if details is None:
        details = [{"rubric": r, "met": False, "justification": "评估失败"} for r in rubrics]

    met_count = sum(1 for d in details if d.get("met", False))
    return met_count / len(rubrics), details


def _format_trajectory(messages: list[dict]) -> str:
    lines = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls", [])
        full_content = content
        if role == "assistant" and tool_calls:
            tc_strs = []
            for tc in tool_calls:
                func = tc.get("function", tc)
                name = func.get("name", "")
                args = func.get("arguments", "")
                if isinstance(args, str):
                    try:
                        args_str = ", ".join(f"{k}={repr(v)}" for k, v in json.loads(args).items())
                    except:
                        args_str = args
                elif isinstance(args, dict):
                    args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
                else:
                    args_str = str(args)
                tc_strs.append(f"{name}({args_str})")
            if tc_strs:
                tc_text = ". ".join(tc_strs)
                full_content = f"{full_content} {tc_text}" if full_content else tc_text
        if full_content:
            lines.append(f"[{i+1}] {role}: {full_content}")
    return "\n".join(lines)


def _parse_rubric_results(text: str, rubrics: list[str]) -> list[dict]:
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        results = json.loads(text[start:end])
        if isinstance(results, list):
            details = []
            for i, rubric in enumerate(rubrics):
                if i < len(results):
                    r = results[i]
                    met = r.get("meetExpectation", r.get("met", False))
                    details.append({"rubric": rubric, "met": bool(met), "justification": r.get("justification", "")})
                else:
                    details.append({"rubric": rubric, "met": False, "justification": "未返回"})
            return details
    except (ValueError, json.JSONDecodeError):
        pass
    return [{"rubric": r, "met": False, "justification": "解析失败"} for r in rubrics]


# ═══════════════════════════════════════════════════════════════════════════════
# tool matching (semantic plus rule-based)
# ═══════════════════════════════════════════════════════════════════════════════


ARGUMENT_MATCH_PROMPT = """你是一个工具参数匹配评估专家。请判断【实际参数】是否与【期望参数】在语义上匹配。

【工具名称】
{tool_name}

【期望参数】
{expected_args}

【实际参数】
{actual_args}

【匹配标准（宽松）】
1. 地名宽松匹配："西湖"和"杭州西湖"均视为匹配
2. 关键词宽松匹配："餐厅"和"美食餐厅"均视为匹配
3. search_flights / search_train_tickets：只需 from_city 和 to_city 相同即可
4. web_search：查询意图大致相同即可
5. poi_search / around_search：核心地名或关键词语义相同即可

请只输出 "MATCH" 或 "NOT_MATCH"。"""


def is_coordinate(value: str) -> bool:
    try:
        parts = value.split(",")
        if len(parts) == 2:
            float(parts[0])
            float(parts[1])
            return True
    except (ValueError, AttributeError):
        pass
    return False


def coordinates_close(coord1: str, coord2: str, threshold: float = 0.1) -> bool:
    try:
        lon1, lat1 = map(float, coord1.split(","))
        lon2, lat2 = map(float, coord2.split(","))
        return abs(lon1 - lon2) < threshold and abs(lat1 - lat2) < threshold
    except (ValueError, AttributeError):
        return False


async def check_arguments_match_llm(tool_name: str, expected_args: dict, actual_args: dict) -> bool:
    if not expected_args or expected_args == actual_args:
        return True

    client = get_judge_client()
    semaphore = get_judge_semaphore()

    prompt = ARGUMENT_MATCH_PROMPT.format(
        tool_name=tool_name,
        expected_args=json.dumps(expected_args, ensure_ascii=False, indent=2),
        actual_args=json.dumps(actual_args, ensure_ascii=False, indent=2),
    )

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="gpt-5-mini-2025-08-07",
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.choices[0].message.content.strip().upper()
            return "MATCH" in result and "NOT" not in result
        except Exception as e:
            logger.warning(f"[check_arguments_match_llm] failed: {e}")
            return check_arguments_match_fallback(expected_args, actual_args, tool_name)


def check_arguments_match_fallback(expected_args: dict, actual_args: dict, tool_name: str = "") -> bool:
    if not expected_args:
        return True
    if tool_name in ("search_flights", "search_train_tickets"):
        ef = str(expected_args.get("from_city", "")).strip()
        et = str(expected_args.get("to_city", "")).strip()
        af = str(actual_args.get("from_city", "")).strip()
        at = str(actual_args.get("to_city", "")).strip()
        if not ef or not et:
            return True
        return (ef in af or af in ef) and (et in at or at in et)
    if tool_name == "web_search":
        eq = expected_args.get("query", "")
        aq = actual_args.get("query", "")
        ek = set(re.findall(r"[\u4e00-\u9fff]{2,}", str(eq)))
        ak = set(re.findall(r"[\u4e00-\u9fff]{2,}", str(aq)))
        if not ek:
            return True
        return len(ek & ak) / len(ek) >= 0.3
    for key, ev in expected_args.items():
        if key in ("mode", "waypoints", "date"):
            continue
        if key not in actual_args:
            return False
        av = actual_args[key]
        evs, avs = str(ev), str(av)
        if is_coordinate(evs) and is_coordinate(avs):
            if not coordinates_close(evs, avs):
                return False
            continue
        if isinstance(ev, str) and isinstance(av, str):
            if ev.lower() not in av.lower() and av.lower() not in ev.lower():
                return False
        elif ev != av:
            return False
    return True


async def compute_tool_accuracy_with_details(expected_tools: list[dict], actual_tools: list[dict]) -> tuple[float, list[dict]]:
    match_details = []
    if not expected_tools:
        return (1.0 if not actual_tools else 0.0), match_details
    if not actual_tools:
        for et in expected_tools:
            match_details.append({"expected": et, "actual": None, "matched": False, "reason": "未调用工具"})
        return 0.0, match_details

    match_tasks = []
    match_indices = []
    RULE_BASED = {"around_search", "direction"}

    async def _rule(ea, aa, tn):
        return check_arguments_match_fallback(ea, aa, tn)

    for ei, et in enumerate(expected_tools):
        en = et.get("name", "")
        ea = et.get("arguments", {})
        for ai, at in enumerate(actual_tools):
            an = at.get("name", "")
            aa = at.get("arguments", {})
            if en == an:
                match_tasks.append(_rule(ea, aa, en) if en in RULE_BASED else check_arguments_match_llm(en, ea, aa))
                match_indices.append((ei, ai))

    if not match_tasks:
        for et in expected_tools:
            match_details.append({"expected": et, "actual": None, "matched": False, "reason": f"{et.get('name')} 未被调用"})
        return 0.0, match_details

    results = await asyncio.gather(*match_tasks, return_exceptions=True)
    matched = {}
    for (ei, ai), r in zip(match_indices, results):
        if isinstance(r, Exception):
            continue
        if r and ei not in matched:
            matched[ei] = (ai, actual_tools[ai])

    for ei, et in enumerate(expected_tools):
        if ei in matched:
            match_details.append({"expected": et, "actual": matched[ei][1], "matched": True, "reason": "匹配成功"})
        else:
            same = [a for a in actual_tools if a.get("name") == et.get("name")]
            match_details.append({
                "expected": et, "actual": same[0] if same else None, "matched": False,
                "reason": "参数不匹配" if same else f"{et.get('name')} 未被调用",
            })

    return len(matched) / len(expected_tools), match_details


# ═══════════════════════════════════════════════════════════════════════════════
# reward computation
# ═══════════════════════════════════════════════════════════════════════════════


def compute_format_reward(query: str, expected_tools: list[dict], rubrics: list[str]) -> float:
    if not query or not expected_tools or not isinstance(expected_tools, list):
        return 0.0
    if not all(t.get("name") for t in expected_tools):
        return 0.0
    if not rubrics or not isinstance(rubrics, list) or len(rubrics) == 0:
        return 0.0
    return 1.0


def validate_expected_tools(expected_tools: list[dict]) -> tuple[list[dict], list[str]]:
    issues = []
    validated = []
    for tool in expected_tools:
        name = tool.get("name", "")
        args = tool.get("arguments", {})
        if name == "direction":
            if args.get("origin") and args.get("destination") and args["origin"] == args["destination"]:
                issues.append(f"direction origin == destination: {args['origin']}")
                continue
        validated.append(tool)
    return validated, issues


async def execute_and_evaluate(args: Namespace, sample: Sample, mcp_state: MCPState):
    generated_query = sample.metadata.get("generated_query", "")
    expected_tools = sample.metadata.get("expected_tools", [])
    rubrics = sample.metadata.get("rubrics", [])
    curriculum_tool_calls = sample.metadata.get("curriculum_tool_calls", [])

    if not curriculum_tool_calls:
        sample.reward = 0.0
        sample.metadata["executor_error"] = "no_tool_call"
        return

    format_reward = compute_format_reward(generated_query, expected_tools, rubrics)
    sample.metadata["format_reward"] = format_reward
    if not generated_query or not expected_tools or not rubrics:
        sample.reward = format_reward
        return

    original_count = len(expected_tools)
    expected_tools, issues = validate_expected_tools(expected_tools)
    sample.metadata["expected_tools"] = expected_tools

    task_type = sample.metadata.get("task_type", "")
    expected_names = {t.get("name") for t in expected_tools}
    REQUIRED = {"direction": "direction", "search_around": "around_search"}
    required_tool = REQUIRED.get(task_type)
    if required_tool and required_tool not in expected_names:
        issues.append(f"缺少必需工具 {required_tool}")

    has_removed = len(expected_tools) < original_count
    has_missing = required_tool and required_tool not in expected_names
    if has_removed or has_missing:
        # the R_tool gate failed, so R_G = 0 (paper Eq. (1)-(2))
        sample.reward = 0.0
        sample.metadata["tool_validity_reward"] = 0.0
        return

    # tool plausibility check
    tool_validity_reward = 1.0
    if curriculum_tool_calls:
        curriculum_as_actual = []
        for tc in curriculum_tool_calls:
            name = tc.get("function", {}).get("name", tc.get("name", ""))
            arguments = tc.get("function", {}).get("arguments", tc.get("arguments", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except:
                    arguments = {}
            if name:
                curriculum_as_actual.append({"name": name, "arguments": arguments})

        validity_accuracy, validity_details = await compute_tool_accuracy_with_details(
            expected_tools, curriculum_as_actual
        )
        sample.metadata["tool_validity_details"] = validity_details
        if validity_accuracy < 1.0:
            # the R_tool gate failed, so R_G = 0 (paper Eq. (1)-(2))
            tool_validity_reward = 0.0
            sample.reward = 0.0
            sample.metadata["tool_validity_reward"] = 0.0
            return
    sample.metadata["tool_validity_reward"] = tool_validity_reward

    # run the solver N times
    num_trials = config.executor_num_trials
    trial_results = []

    for trial_idx in range(num_trials):
        try:
            answer, messages, actual_tool_calls = await call_executor_agent(generated_query, mcp_state)
            accuracy, match_details = await compute_tool_accuracy_with_details(expected_tools, actual_tool_calls)
            tool_correct = (accuracy == 1.0)
            rubric_score, rubric_details = await evaluate_rubrics(messages, rubrics, generated_query)
            rubric_passed = (rubric_score == 1.0)
            is_correct = tool_correct and rubric_passed

            trial_results.append({
                "trial_idx": trial_idx, "accuracy": accuracy, "tool_correct": tool_correct,
                "rubric_score": rubric_score, "rubric_passed": rubric_passed, "is_correct": is_correct,
            })
        except Exception as e:
            logger.error(f"[execute_and_evaluate] trial {trial_idx} failed: {e}")
            trial_results.append({"trial_idx": trial_idx, "is_correct": False, "rubric_score": 0.0, "tool_correct": False})

    correct_count = sum(1 for t in trial_results if t["is_correct"])
    correct_ratio = correct_count / num_trials if num_trials > 0 else 0.0
    rubric_scores = [t.get("rubric_score", 0.0) for t in trial_results]
    avg_rubric_rate = sum(rubric_scores) / len(rubric_scores) if rubric_scores else 0.0

    # paper Eq. (4)-(6): the solver's per-attempt reward is r_i = α·s_i + (1−α)·𝟙[s_i=1] with α=0.8, where
    # s_i combines the satisfaction rate of the content and tool rubrics; the success count is c = Σ_i 𝟙(r_i ≥ γ),
    # triangular difficulty reward R_diff = 2·max(0, 1 − |c − K/2| / (K/2))
    K = num_trials
    gamma = config.executor_success_threshold
    alpha = config.solver_reward_alpha
    n_rubrics = len(rubrics)
    n_tools = len(expected_tools)
    success_count = 0
    for t in trial_results:
        denom = n_rubrics + n_tools
        s_i = (
            (t.get("rubric_score", 0.0) * n_rubrics + t.get("accuracy", 0.0) * n_tools) / denom
            if denom > 0 else 0.0
        )
        r_i = alpha * s_i + (1.0 - alpha) * (1.0 if t.get("is_correct") else 0.0)
        t["solver_reward"] = r_i
        if r_i >= gamma:
            success_count += 1
    if K > 0:
        half_K = K / 2.0
        difficulty_reward = 2.0 * max(0.0, 1.0 - abs(success_count - half_K) / half_K)
    else:
        difficulty_reward = 0.0

    # paper Eq. (1): R_G = R_tool · R_fmt · (1 + R_diff); R_tool (including the expected_tools
    # argument-level checks) and R_fmt are both guaranteed to be 1 above; on failure we return reward=0 early
    reward = format_reward * tool_validity_reward * (1.0 + difficulty_reward)

    sample.metadata.update({
        "executor_trials": trial_results, "num_trials": num_trials,
        "correct_count": correct_count, "correct_ratio": correct_ratio,
        "success_count": success_count, "success_threshold": gamma,
        "avg_rubric_pass_rate": avg_rubric_rate, "difficulty_reward": difficulty_reward,
    })
    sample.reward = reward


# ═══════════════════════════════════════════════════════════════════════════════
# run the solver
# ═══════════════════════════════════════════════════════════════════════════════


async def call_executor_agent(query: str, mcp_state: MCPState) -> tuple[str, list[dict], list[dict]]:
    client = get_executor_client()
    semaphore = get_executor_semaphore()
    max_steps = config.executor_max_steps
    tools = mcp_state.tools if mcp_state.tools else None

    messages = [
        {"role": "system", "content": f"可调用{max_steps}轮工具，已调用0轮。每轮最多调用3个工具。"},
        {"role": "user", "content": query},
    ]
    final_answer = ""
    tool_calls_made = []

    async with semaphore:
        for step_idx in range(max_steps):
            try:
                messages[0] = {"role": "system", "content": f"可调用{max_steps}轮工具，已调用{step_idx}轮。每轮最多调用3个工具。{' 请直接回答，不要使用工具。' if step_idx >= max_steps else ''}"}
                response = await client.chat.completions.create(
                    model=config.executor_model, messages=messages,
                    tools=tools, temperature=0.7, max_tokens=4096,
                )
                msg = response.choices[0].message

                if msg.tool_calls:
                    assistant_msg = {
                        "role": "assistant", "content": msg.content or "",
                        "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls],
                    }
                    messages.append(assistant_msg)
                    for tc in msg.tool_calls:
                        try:
                            tool_args = json.loads(tc.function.arguments)
                        except:
                            tool_args = {}
                        tool_calls_made.append({"name": tc.function.name, "arguments": tool_args})
                        try:
                            result = await mcp_state.call_tool({"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}})
                            result_content = result.get("content", str(result))
                        except Exception as e:
                            result_content = f"工具调用失败: {str(e)}"
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_content})
                else:
                    messages.append({"role": "assistant", "content": msg.content or ""})
                    final_answer = msg.content or ""
                    break
            except Exception as e:
                logger.error(f"[call_executor_agent] step {step_idx} failed: {e}")
                continue
        else:
            try:
                messages[0] = {"role": "system", "content": "请直接回答，不要使用工具。"}
                response = await client.chat.completions.create(
                    model=config.executor_model, messages=messages, temperature=0.7, max_tokens=4096,
                )
                final_answer = response.choices[0].message.content or ""
                messages.append({"role": "assistant", "content": final_answer})
            except Exception as e:
                final_answer = f"回答失败: {str(e)}"

    return final_answer, messages, tool_calls_made

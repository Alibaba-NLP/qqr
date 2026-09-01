"""
DeepResearch curriculum agent (generator) rollout - Qwen3.5 + RG-KL

Three-phase RG-KL procedure:
  Phase A - student rollout (k_s samples): no coach memory; generate and validate with the solver
  Phase B - memory generation: the gpt-5.2 coach reviews the quality of the student's generated tasks
  Phase C - teacher rollout (k_m samples): generate with coach memory injected, used only to estimate Δ_r

Generator specifics (DeepResearch):
  - The generator explores a field with web_search (the Google Search MCP)
  - Produces the query and rubrics
  - The solver (executor service) runs multi-round web_search and writes a report
  - An LLM judge scores the rubric pass rate
  - reward = R_tool · R_fmt · (1 + R_diff) ∈ [0, 3] (paper Eq. (1)-(5), triangular difficulty reward)
"""

import asyncio
import fcntl
import json
import logging
import math
import os
import re
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
            memory = await _generate_coach_memory(group_id, student_results, args)
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
    group_id: int, student_results: list[dict], args: Namespace,
) -> str | None:
    summary, judge_input = await generate_group_summary(group_id, student_results)
    if summary:
        _save_group_summary(args, group_id, summary, judge_input)
    return summary or None


async def generate_group_summary(
    group_id: int, group_results: list[dict],
) -> tuple[str, str]:
    rewards = [r.get("reward", 0) for r in group_results]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    reward_dist = dict(Counter([f"{r:.1f}" for r in rewards]))

    sample_summaries = []
    for i, r in enumerate(group_results):
        diag = _reward_diagnosis(r.get("reward", 0), r.get("avg_rubric_pass_rate"))
        summary = (
            f"### 第{i+1}次出题\n"
            f"- reward: {r.get('reward', 0):.1f} ({diag})\n"
            f"- 题目: {r.get('generated_query', '(无)')[:200]}\n"
            f"- rubrics 数量: {len(r.get('rubrics', []))}\n"
            f"- 领域: {r.get('domain', '?')}\n"
        )
        sample_summaries.append(summary)

    prompt = f"""你是一个深度研究出题训练的教练。以下是一组 {len(group_results)} 次出题结果。

## 奖励机制说明（R_G = R_tool · R_fmt · (1 + R_diff)，三角难度奖励）
- reward=0: 门控失败——出题者未调用工具或输出格式错误
- reward=1: 门控通过，但做题者 K 次试验全成功或全失败（难度退化，R_diff=0）
- reward∈(1,3): 难度偏离 K/2 但仍有区分度，随 |c−K/2| 线性衰减
- reward=3（峰值）: 做题者 K 次试验约一半成功（c≈K/2），难度恰好落在 Solver 能力边界

## 本组结果
- 平均 reward: {avg_reward:.2f}
- reward 分布: {reward_dist}

{"".join(sample_summaries)}

## 请生成下一轮出题建议（不超过300字）
1. **问题诊断**：主要问题是什么？
2. **多样性建议**：扩展到哪些领域/角度？
3. **难度调整**：如何让 rubrics 更有区分度？
4. **具体建议**：给出 1-2 个具体的出题方向"""

    summary = await _summary_via_external_llm(prompt)
    return summary, prompt


def _reward_diagnosis(reward: float, avg_rubric_rate: float | None = None) -> str:
    if reward == 0:
        return "门控失败（未调用工具或格式错误）"
    if avg_rubric_rate is not None:
        if avg_rubric_rate == 1.0:
            return "rubric 全通过（太简单，R_diff=0 → reward=1）"
        if avg_rubric_rate == 0.0:
            return "rubric 全不通过（太难，R_diff=0 → reward=1）"
        return f"rubric 通过率={avg_rubric_rate:.0%}"
    return f"reward={reward:.1f}"


async def _summary_via_external_llm(prompt: str) -> str:
    client = AsyncOpenAI(
        api_key=config.coach_api_key, base_url=config.coach_base_url,
        timeout=60, max_retries=3,
    )
    try:
        resp = await client.chat.completions.create(
            model=config.coach_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=500,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"[group_summary] Coach LLM call failed: {e}")
        return ""


def _save_group_summary(args: Namespace, group_id: int, summary: str, judge_input: str = ""):
    save_dir = getattr(args, "save", None) or getattr(args, "save_dir", None) or "."
    memory_dir = Path(save_dir) / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    filepath = memory_dir / "group_summaries.jsonl"

    record = {
        "timestamp": datetime.now().isoformat(),
        "group_id": group_id,
        "judge_input": judge_input,
        "summary": summary,
    }
    with open(filepath, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


# ═══════════════════════════════════════════════════════════════════════════════
# RG-KL guided_tokens
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
            source == "student" and bool(s.tokens) and bool(s.messages)
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
    """Generator p_m swap: append the coach memory to the last user message."""
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
                orig = messages_with_memory[i]["content"]
                marker = "\n\n【上一轮出题教练反馈】\n"
                if marker not in orig:
                    messages_with_memory[i]["content"] = orig + marker + memory
                injected = True
                break

        if not injected:
            return False

        clean = [
            {"role": "user" if m["role"] == "tool" else m["role"], "content": m.get("content", "") or ""}
            for m in messages_with_memory
        ]
        full_text = tokenizer.apply_chat_template(clean, tokenize=False, add_generation_prompt=False)
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
    info = {"delta_r": delta_r, "gate": gate, "warmup_factor": warmup_factor,
            "cos_factor": cos_factor, "coef": coef, "rollout_step": rollout_step}
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
        _executor_client = AsyncOpenAI(api_key="EMPTY", base_url=config.executor_api_base, timeout=300, max_retries=3)
    return _executor_client


def get_executor_semaphore() -> asyncio.Semaphore:
    global _executor_semaphore
    if _executor_semaphore is None:
        _executor_semaphore = asyncio.Semaphore(config.executor_concurrency_limit)
    return _executor_semaphore


def get_judge_client() -> AsyncOpenAI:
    global _judge_client
    if _judge_client is None:
        _judge_client = AsyncOpenAI(api_key=config.llm_judge_api_key, base_url=config.llm_judge_base_url, timeout=300, max_retries=5)
    return _judge_client


def get_judge_semaphore() -> asyncio.Semaphore:
    global _judge_semaphore
    if _judge_semaphore is None:
        _judge_semaphore = asyncio.Semaphore(config.llm_judge_concurrency_limit)
    return _judge_semaphore


# ═══════════════════════════════════════════════════════════════════════════════
# Rubric scoring (reuses the logic from arise_rl_solver/deepresearch)
# ═══════════════════════════════════════════════════════════════════════════════


async def evaluate_rubrics(messages: list[dict], rubrics: list[str], query: str) -> tuple[float, list[dict]]:
    """Use the LLM judge to score the solver's answer against the rubrics."""
    if not rubrics:
        return 1.0, []

    # reuse the evaluator from arise_rl_solver/deepresearch
    from qqr.examples.arise_rl_solver.deepresearch.reward_model import evaluate_rubrics as _eval
    return await _eval(messages, rubrics, query)


# ═══════════════════════════════════════════════════════════════════════════════
# Main generation entry point
# ═══════════════════════════════════════════════════════════════════════════════


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
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

    # sample a random topic direction
    topic = config.generate_random_research_topic(group_rng)

    sample.metadata = sample.metadata or {}
    sample.metadata["topic"] = topic
    sample.metadata["_is_evaluation"] = evaluation

    # -- Eval mode --
    if evaluation:
        return await _do_curriculum_rollout(
            args, sample, sampling_params, topic,
            coach_summary="", source="student", group_id=group_id, evaluation=True,
        )

    # -- RG-KL disabled --
    if not config.enable_rg_kl:
        return await _do_curriculum_rollout(
            args, sample, sampling_params, topic,
            coach_summary="", source="student", group_id=group_id, evaluation=False,
        )

    # -- RG-KL three phases --
    await _get_or_create_events(group_id)

    if within_idx < k_s:
        result = await _do_curriculum_rollout(
            args, sample, sampling_params, topic,
            coach_summary="", source="student", group_id=group_id, evaluation=False,
        )
        final = result[-1] if isinstance(result, list) else result
        result_info = {
            "reward": final.reward if final.reward is not None else 0.0,
            "generated_query": (final.metadata or {}).get("generated_query", ""),
            "rubrics": (final.metadata or {}).get("rubrics", []),
            "avg_rubric_pass_rate": (final.metadata or {}).get("avg_rubric_pass_rate"),
            "domain": topic.get("domain", ""),
        }
        await _register_student_done(group_id, result_info, k_s)
        return result
    else:
        memory = await _acquire_memory(group_id, k_s, args)
        result = await _do_curriculum_rollout(
            args, sample, sampling_params, topic,
            coach_summary=memory or "", source="teacher", group_id=group_id, evaluation=False,
        )
        samples_list = result if isinstance(result, list) else [result]
        for s in samples_list:
            if s.metadata is None:
                s.metadata = {}
            s.metadata["memory"] = memory or ""
        return result


async def _do_curriculum_rollout(
    args: Namespace, sample: Sample, sampling_params: dict[str, Any],
    topic: dict, coach_summary: str, source: str, group_id: int, evaluation: bool,
) -> Sample | list[Sample]:
    """A single generation rollout."""
    mcp_state = MCPState(config.mcp_manager)
    await mcp_state.get_servers()

    # build the generation prompt
    lang = "英文" if topic.get("language") == "英文" else "中文"
    task_prompt = (
        f"请为以下方向设计一个深度研究问题（{lang}）：\n\n"
        f"- 研究领域: {topic.get('domain', '人工智能')}\n"
        f"- 问题类型: {topic.get('research_type', '综述对比')}\n"
        f"- 复杂度: {topic.get('complexity', '中级')}\n\n"
        f"请先调用 web_search 搜索相关信息获取真实素材，然后基于搜索结果设计问题和 rubrics。"
    )
    if coach_summary:
        task_prompt += f"\n\n【上一轮出题教练反馈】\n{coach_summary}"

    sample.messages = [
        {"role": "system", "content": config.query_generation_system_prompt},
        {"role": "user", "content": task_prompt},
    ]
    sample.prompt = task_prompt
    sample.metadata = sample.metadata or {}
    sample.metadata["has_memory"] = bool(coach_summary)
    sample.metadata["source"] = source

    # multi-round web_search exploration by the generator
    samples = await curriculum_generate_with_tool(args, sample, sampling_params, mcp_state)

    if not samples:
        sample.reward = 0.0
        return sample if evaluation else [sample]

    # parse the output
    final_sample = samples[-1]
    raw_output = final_sample.response or ""
    generated_query, rubrics = parse_curriculum_output(raw_output)

    for s in samples:
        if s.metadata is None:
            s.metadata = {}
        s.metadata["generated_query"] = generated_query
        s.metadata["rubrics"] = rubrics
        s.metadata["curriculum_raw_output"] = raw_output
        s.metadata["source"] = source

    # compute the reward
    await execute_and_evaluate(args, samples, mcp_state, generated_query, rubrics)

    if evaluation:
        return samples[-1]
    else:
        valid = [s for s in samples if s.rollout_log_probs is not None and s.response_length > 0]
        return valid if valid else [samples[-1]]


# ═══════════════════════════════════════════════════════════════════════════════
# multi-round tool calling by the generator (Qwen3.5-compatible)
# ═══════════════════════════════════════════════════════════════════════════════


async def curriculum_generate_with_tool(
    args: Namespace, sample: Sample, sampling_params: dict[str, Any],
    mcp_state: MCPState, max_tool_rounds: int = config.max_steps,
) -> list[Sample]:
    state = GenerateState(args)
    ckpt_normalized = re.sub(r'[._]+', '.', state.args.hf_checkpoint.lower())
    if "qwen3.5" in ckpt_normalized:
        prompter = registers.prompt["qwen3.5"]()
    else:
        prompter = registers.prompt["qwen3"]()

    tools = mcp_state.tools if mcp_state.tools else None
    samples = []
    curriculum_tool_calls = []

    samples.append(Sample(
        group_index=sample.group_index, index=sample.index,
        messages=deepcopy(sample.messages), prompt=sample.prompt,
        label=sample.label, status=Sample.Status.PENDING,
        metadata=sample.metadata or {}, train_metadata={"tools": tools},
    ))
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
        # Qwen3.5's chat_template uses `arguments | items` and expects a dict, so rendering would blow up
        # normalise a copy here so the next round's apply_chat_template renders correctly
        # the original tool_calls keep their string arguments for mcp_state.call_tool
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
                max_chars = config.tool_response_max_chars
                if len(result_content) > max_chars:
                    result_content = result_content[:max_chars] + "\n...(结果已截断)"
            except Exception as e:
                result_content = f"工具调用失败: {str(e)}"

            sample.messages.append({
                "role": "tool", "tool_call_id": tc.get("id", ""), "content": result_content,
            })

        logger.info(f"[curriculum] step {step_idx}: called web_search")

        samples.append(Sample(
            group_index=sample.group_index, index=sample.index,
            messages=deepcopy(sample.messages), prompt=sample.prompt,
            label=sample.label, status=Sample.Status.PENDING,
            metadata=sample.metadata, train_metadata={"tools": tools},
        ))
        sample = samples[-1]

    else:
        sample.messages[0] = {
            "role": "system",
            "content": sample.messages[0]["content"]
            + "\n\n你已完成搜索，现在请直接输出最终的 JSON 结果（query + rubrics），不要再调用工具。",
        }
        samples.append(Sample(
            group_index=sample.group_index, index=sample.index,
            messages=deepcopy(sample.messages), prompt=sample.prompt,
            label=sample.label, status=Sample.Status.PENDING,
            metadata=sample.metadata, train_metadata=None,
        ))
        sample = samples[-1]
        sample = await base_generate(args, sample, sampling_params)
        response_content = sample.response.removesuffix(state.tokenizer.eos_token).strip()
        sample.messages.append({"role": "assistant", "content": response_content})

    sample.metadata = sample.metadata or {}
    sample.metadata["curriculum_tool_calls"] = curriculum_tool_calls

    valid = [s for s in samples if s.rollout_log_probs is not None and s.response_length > 0]
    if not valid:
        sample = samples[-1]
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        valid = [sample]

    return valid


def parse_curriculum_output(response: str) -> tuple[str, list[str]]:
    """Parse the generator's JSON output and extract the query and rubrics."""
    clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    code_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", clean, re.DOTALL)
    if code_match:
        json_text = code_match.group(1)
    else:
        json_text = clean

    try:
        start = json_text.index("{")
        end = json_text.rindex("}") + 1
        data = json.loads(json_text[start:end])
        query = data.get("query", "")
        rubrics = data.get("rubrics", [])
        if isinstance(rubrics, list):
            return query, rubrics
    except (ValueError, json.JSONDecodeError):
        pass

    return "", []


# ═══════════════════════════════════════════════════════════════════════════════
# reward: R_G = R_tool · R_fmt · (1 + R_diff)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_format_reward(query: str, rubrics: list[str]) -> float:
    if not query:
        return 0.0
    if not rubrics or not isinstance(rubrics, list) or len(rubrics) == 0:
        return 0.0
    return 1.0


async def execute_and_evaluate(
    args: Namespace, samples: list[Sample], mcp_state: MCPState,
    generated_query: str, rubrics: list[str],
):
    """Generator reward: R_G = R_tool · R_fmt · (1 + R_diff)."""
    # Stage 1: did the generator call any tool
    has_tool_calls = any(
        msg.get("role") == "tool" for s in samples for msg in (s.messages or [])
    )
    if not has_tool_calls:
        _apply_reward(samples, 0.0, reason="no_tool_calls")
        return

    # Stage 2: format reward
    format_reward = compute_format_reward(generated_query, rubrics)
    if format_reward < 1.0:
        _apply_reward(samples, format_reward, format_reward=format_reward)
        return

    # Stage 3: validate with the solver N times
    num_trials = config.executor_num_trials
    trial_rubric_rates = []

    for trial_idx in range(num_trials):
        try:
            rubric_rate = await call_executor_trial(generated_query, rubrics, mcp_state)
            trial_rubric_rates.append(rubric_rate)
            logger.info(f"[execute_and_evaluate] trial {trial_idx}: rubric_rate={rubric_rate:.2f}")
        except Exception as e:
            logger.error(f"[execute_and_evaluate] trial {trial_idx} failed: {e}")
            trial_rubric_rates.append(0.0)

    avg_rubric_rate = sum(trial_rubric_rates) / len(trial_rubric_rates) if trial_rubric_rates else 0.0

    # paper Eq. (4)-(5): the solver's per-attempt reward is r_i = α·s_i + (1−α)·𝟙[s_i=1] with α=0.8,
    # success count c = Σ_i 𝟙(r_i ≥ γ), triangular difficulty reward R_diff = 2·max(0, 1 − |c − K/2| / (K/2))
    K = len(trial_rubric_rates)
    gamma = config.executor_success_threshold
    alpha = config.solver_reward_alpha
    trial_solver_rewards = [
        alpha * s + (1.0 - alpha) * (1.0 if s >= 1.0 else 0.0) for s in trial_rubric_rates
    ]
    success_count = sum(1 for r in trial_solver_rewards if r >= gamma)
    if K > 0:
        half_K = K / 2.0
        difficulty_reward = 2.0 * max(0.0, 1.0 - abs(success_count - half_K) / half_K)
    else:
        difficulty_reward = 0.0

    # paper Eq. (1): R_G = R_tool · R_fmt · (1 + R_diff); the R_tool and R_fmt gates
    # already guaranteed to be 1 by stages 1 and 2; on failure we returned reward=0 early
    total_reward = format_reward * (1.0 + difficulty_reward)

    _apply_reward(samples, total_reward,
                  format_reward=format_reward,
                  difficulty_reward=difficulty_reward,
                  avg_rubric_pass_rate=avg_rubric_rate,
                  success_count=success_count,
                  success_threshold=gamma,
                  trial_rubric_rates=trial_rubric_rates,
                  generated_query=generated_query,
                  rubrics=rubrics)

    logger.info(
        f"[execute_and_evaluate] query={generated_query[:50]}... "
        f"rubrics={len(rubrics)} K={K} γ={gamma} c={success_count}/{K} "
        f"avg_rate={avg_rubric_rate:.2f} "
        f"format={format_reward:.0f} diff={difficulty_reward:.2f} total={total_reward:.2f}"
    )


def _apply_reward(samples: list[Sample], reward: float, **metadata):
    for s in samples:
        s.reward = reward
        if s.metadata is None:
            s.metadata = {}
        s.metadata.update(metadata)


# ═══════════════════════════════════════════════════════════════════════════════
# run the solver (a single trial)
# ═══════════════════════════════════════════════════════════════════════════════


async def call_executor_trial(query: str, rubrics: list[str], mcp_state: MCPState) -> float:
    """Call the executor (solver) to answer the question and return the rubric pass rate."""
    client = get_executor_client()
    semaphore = get_executor_semaphore()
    max_steps = config.executor_max_steps
    tools = mcp_state.tools if mcp_state.tools else None

    messages = [
        {"role": "system", "content": config.EXECUTOR_SYSTEM_PROMPT_ZH.format(
            time=datetime.now().strftime("%d/%m/%Y, %H:%M"),
            max_steps=max_steps, step_idx=0,
        )},
        {"role": "user", "content": query},
    ]
    tool_calls_made = []

    async with semaphore:
        for step_idx in range(max_steps):
            try:
                messages[0] = {"role": "system", "content": config.EXECUTOR_SYSTEM_PROMPT_ZH.format(
                    time=datetime.now().strftime("%d/%m/%Y, %H:%M"),
                    max_steps=max_steps, step_idx=step_idx,
                )}
                resp = await client.chat.completions.create(
                    model=config.executor_model, messages=messages,
                    tools=tools, temperature=0.7, max_tokens=4096,
                )
                msg = resp.choices[0].message

                if msg.tool_calls:
                    assistant_msg = {
                        "role": "assistant", "content": msg.content or "",
                        "tool_calls": [{"id": tc.id, "type": "function",
                                       "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                      for tc in msg.tool_calls],
                    }
                    messages.append(assistant_msg)

                    for tc in msg.tool_calls:
                        tc_dict = {"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        try:
                            result = await mcp_state.call_tool(tc_dict)
                            result_content = result.get("content", str(result))
                            max_chars = config.tool_response_max_chars
                            if len(result_content) > max_chars:
                                result_content = result_content[:max_chars] + "\n...(截断)"
                        except Exception as e:
                            result_content = f"工具调用失败: {str(e)}"
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_content})
                else:
                    messages.append({"role": "assistant", "content": msg.content or ""})
                    break
            except Exception as e:
                logger.error(f"[executor_trial] step {step_idx} failed: {e}")
                continue
        else:
            try:
                messages[0] = {"role": "system", "content": "请立即根据已搜索到的信息输出完整的研究报告，不要再调用工具。"}
                resp = await client.chat.completions.create(
                    model=config.executor_model, messages=messages, temperature=0.7, max_tokens=4096,
                )
                messages.append({"role": "assistant", "content": resp.choices[0].message.content or ""})
            except Exception as e:
                messages.append({"role": "assistant", "content": f"回答失败: {e}"})

    # Rubric scoring
    try:
        score, details = await evaluate_rubrics(messages, rubrics, query)
    except Exception as e:
        logger.warning(f"[executor_trial] rubric eval failed: {e}")
        score = 0.0

    return score

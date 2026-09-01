"""
DeepResearch — Reward-Gated Reverse KL (RG-KL)

Three-phase procedure:
  Phase A - student rollout: k_s = 8 trajectories without memory (these are trained on)
  Phase B - memory generation: score the student rubrics -> LLM coach -> memory specific to this query
  Phase C - teacher rollout: k_m = 4 trajectories with memory injected into the system prompt (used only for Δ_r)

Training (RG-KL):
  Δ_r = mean(reward(τ_m)) - mean(reward(τ_s))  ← the gating signal
  λ(Δ_r) = λ_0 · gate(Δ_r) · warmup · cosine_decay

  Student samples: guided_tokens = [p_m_prompt, τ_s_response]
                → slime's training-side forward pass yields teacher_log_probs = π_θ(τ_s | p_m)
                → apply_rg_kl_to_advantages: adv -= λ(Δ_r) · clamp(student_lp - teacher_lp)
  Teacher samples: guided_tokens = tokens (identity), reward neutralised to student_mean,
                loss_mask = 0 and rg_kl_coef = 0, so they contribute nothing to the gradient, only the Δ_r scalar
"""

import asyncio
import logging
import math
import re
from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Any

from qqr import registers
from qqr.rollout.agent_rollout import GenerateState, MCPState
from qqr.rollout.agent_rollout import generate as base_generate
from qqr.schemas import Sample
from slime.utils.http_utils import post

from . import config
from .reward_model import (
    _compute_sample_reward,
    _save_group_summary,
    eval_reward,
    get_judge_client,
    get_judge_semaphore,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# per-group coordination state (asyncio-safe, shared across coroutines in one process)
# ═══════════════════════════════════════════════════════════════════════════════

_group_student_samples: dict[int, list[Sample]] = defaultdict(list)
_group_student_event: dict[int, asyncio.Event] = {}
_group_memory: dict[int, str] = {}
_group_memory_leader: set[int] = set()
_group_memory_event: dict[int, asyncio.Event] = {}
_group_coord_lock = asyncio.Lock()

# global rollout counter: each group_reward call corresponds to one rollout step
# used for the warmup and cosine-decay schedules (the rollout side cannot read slime's rollout_id directly)
_rollout_step_counter: int = 0
_rollout_step_lock = asyncio.Lock()


async def get_rollout_step() -> int:
    """Return the current rollout step (monotonically increasing)."""
    return _rollout_step_counter


async def increment_rollout_step(rollout_batch_size: int = 1) -> int:
    """Increment the group counter and return the rollout_id (the number of completed rollouts).

    Called once per group (k_s+k_m samples form one group). Each rollout has
    `rollout_batch_size` groups. We use `rollout_id = (group_count-1) // batch`
    ensures every group within a rollout sees the same step, keeping the schedule consistent.
    """
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


async def _register_student_done(group_id: int, final_sample: Sample, k_s: int):
    async with _group_coord_lock:
        _group_student_samples[group_id].append(final_sample)
        if len(_group_student_samples[group_id]) >= k_s:
            _group_student_event[group_id].set()


async def _acquire_memory(
    group_id: int, k_s: int, query: str, rubrics: list[str], args: Namespace, lang: str,
) -> str | None:
    await _group_student_event[group_id].wait()

    async with _group_coord_lock:
        is_leader = group_id not in _group_memory_leader
        if is_leader:
            _group_memory_leader.add(group_id)

    if is_leader:
        try:
            student_samples = _group_student_samples[group_id]
            memory = await _generate_memory_for_group(
                group_id, student_samples, query, rubrics, args, lang,
            )
            async with _group_coord_lock:
                _group_memory[group_id] = memory or ""
        except Exception as e:
            logger.error(f"[rg_kl] group={group_id} memory generation error: {e}")
            async with _group_coord_lock:
                _group_memory[group_id] = ""
        finally:
            _group_memory_event[group_id].set()
        result = _group_memory.get(group_id, "")
        logger.info(f"[rg_kl] group={group_id} memory 生成{'成功' if result else '失败'}")
        return result or None
    else:
        await _group_memory_event[group_id].wait()
        result = _group_memory.get(group_id, "")
        return result or None


async def _evaluate_and_cache(sample: Sample) -> dict:
    result = await _compute_sample_reward(sample)
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["_rubric_cache"] = result
    return result


async def _generate_memory_for_group(
    group_id: int, student_samples: list[Sample],
    query: str, rubrics: list[str], args: Namespace, lang: str,
) -> str | None:
    eval_results = await asyncio.gather(
        *[_evaluate_and_cache(s) for s in student_samples], return_exceptions=True,
    )

    group_records = []
    rewards = []
    for r, s in zip(eval_results, student_samples):
        if isinstance(r, Exception):
            logger.warning(f"[rg_kl] group={group_id} student eval failed: {r}")
            continue
        rewards.append(r["reward"])
        group_records.append((r, s))

    if not group_records:
        return None

    avg_reward = sum(rewards) / len(rewards)

    sample_summaries = []
    for i, (rec, s) in enumerate(group_records):
        rubric_parts = ""
        for rd in rec.get("rubric_details", []):
            status = "+" if rd.get("met") else "-"
            rubric_parts += f"\n    {status} {rd.get('rubric', '')}"

        sample_summaries.append(
            f"### 第{i + 1}个样本 (reward={rec['reward']:.2f})\n"
            f"- rubric 通过率: {rec.get('rubric_pass_rate', 0):.0%}\n"
            f"- rubric 详情: {rubric_parts}\n"
        )

    if lang == "en":
        coach_prompt = f"""You are a deep research coaching assistant. Below are {len(group_records)} student attempts for the same query.

## User Question
{query[:300]}

## Group Results ({len(group_records)} attempts, avg reward: {avg_reward:.2f})

{"".join(sample_summaries)}

## Your Task

Analyze the students' failure modes and provide **specific, actionable** coaching advice (max 300 words).

### Requirements:
1. **Root cause**: What is the most common failure? (Insufficient search? Missing info? Poor structure? Untraceable sources?)
2. **Search strategy**: For this specific query, what keywords / sub-questions should be searched? In what order?
3. **Answer points**: Which key information must appear in the report to satisfy rubrics? What is most often omitted?
4. **One-sentence core advice**: Next time you see a similar question, what should the first step be?

Output your advice directly, do not repeat the above."""
    else:
        coach_prompt = f"""你是一个深度研究做题训练的教练。以下是同一个 query 下 {len(group_records)} 个做题者样本的评估结果。

## 用户原始问题
{query[:300]}

## 本组做题结果（{len(group_records)} 次尝试，平均 reward: {avg_reward:.2f}）

{"".join(sample_summaries)}

## 你的任务

分析做题者的失败模式，生成**具体、可操作**的做题建议（不超过300字）。

### 要求：
1. **失败根因**：最常见的失败原因是什么？（搜索不够深入？信息遗漏？结构不清？来源不可追溯？）
2. **搜索策略**：针对这个具体 query，应该搜索哪些关键词/子问题？按什么顺序？
3. **回答要点**：rubric 要求哪些关键信息必须出现在报告中？做题者最容易遗漏什么？
4. **一句话核心建议**：下次遇到类似问题，第一步该做什么？

请直接输出建议，不要重复上述内容。"""

    try:
        client = get_judge_client()
        semaphore = get_judge_semaphore()
        async with semaphore:
            response = await client.chat.completions.create(
                model=config.llm_judge_model,
                messages=[{"role": "user", "content": coach_prompt}],
                temperature=0.7,
                max_completion_tokens=500,
            )
        summary = response.choices[0].message.content.strip()
        _save_group_summary(args, group_id, summary, judge_input=coach_prompt)
        return summary
    except Exception as e:
        logger.warning(f"[rg_kl] group={group_id} coach LLM call failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# RG-KL guided_tokens construction (student samples: tokens -> [p_m, τ_s_response])
# ═══════════════════════════════════════════════════════════════════════════════


def compute_rg_kl_guided_tokens(
    args: Namespace,
    samples: list[Sample],
    memory: str | None,
):
    """Build guided_tokens = [p_m_prompt, τ_s_response] for student samples.

    slime runs one extra forward pass over guided_tokens to obtain teacher_log_probs = π_θ(τ_s | p_m).
    apply_rg_kl_to_advantages then uses (student_logp - teacher_logp) as the reverse-KL estimate.

    Scope (controlled by config.kl_final_only):
    - True (default): final-turn only, applying the p_m swap to the last turn (the natural-language report) alone,
        intermediate tool-call turns use identity, so reverse_kl is about 0 and they take no part in distillation
    - False: all-turn scope (every valid student turn is swapped, which pushes the memory's tool
        usage into intermediate turns, which mismatches at deployment where there is no memory)

    Teacher and padding samples always use identity, with rg_kl_coef=0 as an extra safeguard.
    """
    state = GenerateState(args)
    tokenizer = state.tokenizer
    final_only = config.kl_final_only

    # find the last valid sample (not padding, has tokens and a response), which is the final turn
    final_sample = None
    for s in reversed(samples):
        if s.index != -1 and bool(s.tokens) and (s.response_length or 0) > 0:
            final_sample = s
            break

    n_student_final_swap = 0
    n_student_inter_swap = 0
    n_student_inter_identity = 0
    n_identity = 0
    n_failed = 0
    for s in samples:
        source = (s.metadata or {}).get("source", "student")
        is_padding = (s.index == -1)
        is_final = (s is final_sample)
        is_valid_student = (
            source == "student"
            and not is_padding
            and bool(s.tokens)
            and bool(s.messages)
            and (s.response_length or 0) > 0
        )

        # Decision matrix:
        #   student + final + has memory        -> p_m swap   (the main distillation point)
        #   student + intermediate + final_only -> identity   (final-only mode skips intermediate turns)
        #   student + intermediate + not final_only + has memory -> p_m swap (all-turn mode)
        #   teacher / padding / no memory     → identity
        if is_valid_student and memory and (is_final or not final_only):
            ok = _apply_p_m_swap(s, tokenizer, memory)
            if ok:
                if is_final:
                    n_student_final_swap += 1
                else:
                    n_student_inter_swap += 1
            else:
                _set_identity(s, tokenizer)
                n_failed += 1
                n_identity += 1
        else:
            _set_identity(s, tokenizer)
            if is_valid_student and final_only and not is_final:
                n_student_inter_identity += 1
            else:
                n_identity += 1

    scope_label = "final-only" if final_only else "all-turn"
    logger.info(
        f"[rg_kl] guided_tokens ({scope_label}): "
        f"final-swap={n_student_final_swap}, inter-swap={n_student_inter_swap}, "
        f"inter-identity={n_student_inter_identity}, other-identity={n_identity}, "
        f"failed={n_failed}"
    )


def _apply_p_m_swap(s: Sample, tokenizer, memory: str) -> bool:
    """Swap the student prompt to p_m (memory added) while keeping the original τ_s response tokens.
    the forward pass yields π_θ(τ_s | p_m) as the RG-KL teacher distribution.
    """
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

        if messages_with_memory and messages_with_memory[0]["role"] == "system":
            orig_system = messages_with_memory[0]["content"]
            # accepts both the English and Chinese markers, matching build_system_message
            marker_zh = "\n\n【本轮做题教练反馈】\n"
            marker_en = "\n\n[Coach feedback]\n"
            if marker_zh not in orig_system and marker_en not in orig_system:
                marker = marker_en if "Deep Research Specification" in orig_system else marker_zh
                messages_with_memory[0]["content"] = orig_system + marker + memory

        # rewrite the tool role as user so the chat template does not fail to parse
        clean = [
            {
                "role": "user" if m.get("role") == "tool" else m.get("role", ""),
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


def cleanup_group_coordination(group_id: int):
    _group_student_samples.pop(group_id, None)
    _group_student_event.pop(group_id, None)
    _group_memory.pop(group_id, None)
    _group_memory_leader.discard(group_id)
    _group_memory_event.pop(group_id, None)


# ═══════════════════════════════════════════════════════════════════════════════
# Gating coefficient λ(Δ_r) with warmup + cosine decay
# ═══════════════════════════════════════════════════════════════════════════════


def compute_rg_kl_coef(delta_r: float, rollout_step: int) -> tuple[float, dict]:
    """λ(Δ_r) = λ_0 · gate(Δ_r) · warmup · cosine_decay

    The gate is a sigmoid reward gate (paper Eq. (10) and the "Empirical Gate Response" appendix):
        g(Δ_r) = σ((Δ_r − τ) / T); as T → 0 this degenerates to the indicator 𝟙(Δ_r > τ)

    Returns:
        (coef, info dict for logging)
    """
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
        "delta_r": delta_r,
        "gate": gate,
        "warmup_factor": warmup_factor,
        "cos_factor": cos_factor,
        "coef": coef,
        "rollout_step": rollout_step,
    }
    return coef, info


# ═══════════════════════════════════════════════════════════════════════════════
# System message and agent loop (specific to the DeepResearch task)
# ═══════════════════════════════════════════════════════════════════════════════


def _is_english(text: str) -> bool:
    """Roughly detect whether the text is English (more than 70% ASCII characters)."""
    if not text:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / len(text) > 0.7


_SYSTEM_PROMPT_ZH = """当前时间: {time}

# 深度研究规范
你是一个专业的深度研究助手。面对用户的研究问题，你需要通过多轮 web_search 工具调用来收集全面、准确的信息，最终生成高质量的研究报告。

## 研究流程
1. **问题分解**：将复杂问题拆解为多个可搜索的子问题
2. **多轮搜索**：每轮搜索聚焦一个子问题，使用精准的关键词
3. **信息验证**：对关键事实用不同关键词交叉搜索验证
4. **整合回答**：基于搜索结果生成结构化的研究报告，注明信息来源

## 工具使用要求
- 可调用{max_steps}轮工具，已调用{step_idx}轮
- 重要：必须先使用工具查询真实数据，严禁未调用工具直接回答
- 每轮搜索应有明确目的，避免重复搜索相同内容
- 搜索关键词要具体精准，避免过于宽泛

## 回答质量要求
- 信息全面：覆盖问题的各个方面
- 结构清晰：使用标题、列表、表格等组织信息
- 来源可追溯：关键信息注明出处
- 术语准确：正确使用专业术语，避免概念混淆

## 重要：完成搜索后必须输出研究报告
当你认为已经收集了足够的信息，或者工具调用轮次即将用完时，你必须停止搜索并直接输出一份完整的研究报告。报告应整合所有已搜索到的信息，不得输出空内容。"""

_SYSTEM_PROMPT_EN = """Current time: {time}

# Deep Research Specification
You are a professional deep research assistant. For the user's research question, you need to collect comprehensive and accurate information through multiple rounds of web_search tool calls, and ultimately generate a high-quality research report.

## Research Workflow
1. **Decompose the question**: Break complex questions into multiple searchable sub-questions
2. **Multi-round search**: Each round focuses on one sub-question with precise keywords
3. **Verify information**: Cross-verify key facts with different search keywords
4. **Synthesize answer**: Generate a structured research report based on search results, citing sources

## Tool Usage Requirements
- You can call tools for {max_steps} rounds, {step_idx} rounds already used
- Important: You must use tools to query real data first. Never answer directly without calling tools
- Each search should have a clear purpose. Avoid repeating the same search
- Search keywords should be specific and precise, avoid being too broad

## Answer Quality Requirements
- Comprehensive: Cover all aspects of the question
- Well-structured: Use headings, lists, and tables to organize information
- Traceable: Cite sources for key information
- Accurate: Use correct terminology, avoid concept confusion

## Important: You must output the research report after completing searches
When you believe you have collected enough information, or when tool call rounds are about to run out, you must stop searching and directly output a complete research report. The report should integrate all searched information. Do not output empty content."""


def build_system_message(
    step_idx: int,
    max_steps: int,
    coach_summary: str | None = None,
    rubrics: list[str] | None = None,
    lang: str = "zh",
) -> dict:
    template = _SYSTEM_PROMPT_EN if lang == "en" else _SYSTEM_PROMPT_ZH
    system_prompt = template.format(
        time=datetime.now().strftime("%d/%m/%Y, %H:%M"),
        max_steps=max_steps,
        step_idx=step_idx,
    )

    if rubrics:
        if lang == "en":
            system_prompt += "\n\n# Task Completion Criteria (you must ensure all of the following are met)"
        else:
            system_prompt += "\n\n# 任务完成标准（你需要确保以下所有条件都被满足）"
        for i, r in enumerate(rubrics):
            system_prompt += f"\n{i+1}. {r}"

    if coach_summary:
        if lang == "en":
            system_prompt += f"\n\n[Coach feedback]\n{coach_summary}"
        else:
            system_prompt += f"\n\n【本轮做题教练反馈】\n{coach_summary}"

    if step_idx >= max_steps:
        if lang == "en":
            system_prompt += "\n\n⚠️ Tool call rounds exhausted. Please immediately output a complete, structured research report based on all search results above. Do not call any more tools or output empty content."
        else:
            system_prompt += "\n\n⚠️ 工具调用轮次已用完。请立即根据以上所有搜索结果，输出一份完整、结构化的研究报告。不要再调用任何工具，不要输出空内容。"

    return {"role": "system", "content": system_prompt}


async def _generate_forced_report(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
) -> Sample:
    """
    Force the research report to be written, skipping Qwen3.5's <think> stage.

    Qwen3.5's chat template appends <think> automatically when add_generation_prompt=True,
    which pushes the model into thinking mode where the history of tool_call patterns makes it emit yet another tool_call.
    This function prefills an assistant message and renders with continue_final_message=True,
    so the model starts writing the report body straight after </think>.
    """
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    sample.messages.append({
        "role": "assistant",
        "content": (
            "所有搜索已完成，现在根据收集到的信息直接撰写完整的研究报告。\n"
            "</think>\n\n"
        ),
    })

    prompt_text = state.tokenizer.apply_chat_template(
        sample.messages, tools=None, tokenize=False, continue_final_message=True,
    )
    prompt_ids = state.tokenizer.encode(prompt_text, add_special_tokens=False)

    current_sampling_params = deepcopy(sampling_params)
    current_sampling_params["max_new_tokens"] = min(
        sampling_params["max_new_tokens"],
        args.rollout_max_context_len - len(prompt_ids),
    )

    if current_sampling_params["max_new_tokens"] <= 100:
        sample.response = ""
        sample.tokens = [state.tokenizer.pad_token_id]
        sample.loss_mask = []
        sample.rollout_log_probs = []
        sample.reward = 0.0
        sample.status = Sample.Status.TRUNCATED
        return sample

    payload = {
        "sampling_params": current_sampling_params,
        "return_logprob": True,
        "input_ids": prompt_ids,
    }
    if not sample.tokens:
        sample.tokens = prompt_ids

    output = await post(url, payload)

    if "output_token_logprobs" in output["meta_info"]:
        new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_tokens, new_log_probs = [], []

    sample.tokens = sample.tokens + new_tokens
    sample.response_length = len(new_tokens)
    sample.response = output["text"]

    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs = new_log_probs

    sample.update_from_meta_info(args, output["meta_info"])
    return sample


async def agent_loop(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    max_steps: int = config.max_steps,
    coach_summary: str | None = None,
    rubrics: list[str] | None = None,
) -> list[Sample]:
    state = GenerateState(args)
    mcp_state = MCPState(config.mcp_manager)
    ckpt_normalized = re.sub(r"[._]+", ".", state.args.hf_checkpoint.lower())
    if "qwen3.5" in ckpt_normalized:
        prompter = registers.prompt["qwen3.5"]()
    else:
        prompter = registers.prompt["qwen3"]()

    # detect the query language
    query_text = sample.prompt if isinstance(sample.prompt, str) else ""
    if not query_text:
        for m in sample.messages:
            if m.get("role") == "user":
                query_text = m.get("content", "")
                break
    lang = "en" if _is_english(query_text) else "zh"

    if sample.messages[0]["role"] != "system":
        sample.messages.insert(
            0, build_system_message(0, max_steps, coach_summary, rubrics=rubrics, lang=lang)
        )
    samples = []

    for step_idx in range(max_steps):
        samples.append(
            Sample(
                group_index=sample.group_index, index=sample.index,
                messages=deepcopy(sample.messages), prompt=sample.prompt,
                label=sample.label, status=Sample.Status.PENDING,
                metadata=sample.metadata, train_metadata={"tools": mcp_state.tools},
            )
        )
        sample = samples[-1]
        sample.messages[0] = build_system_message(
            step_idx, max_steps, coach_summary, rubrics=rubrics, lang=lang
        )
        sample = await base_generate(args, sample, sampling_params)

        sample.messages.append({
            "role": "assistant",
            "content": sample.response.removesuffix(state.tokenizer.eos_token),
        })
        sample.response_message = prompter.parse_assistant_content(sample.response)
        tool_calls = sample.response_message.get("tool_calls") or []

        if not tool_calls:
            # the response is non-empty (the model wrote the report itself), so exit normally
            response_text = sample.response.removesuffix(state.tokenizer.eos_token).strip()
            if response_text:
                break

            # the response is empty and was truncated, so exit
            if sample.status == Sample.Status.TRUNCATED:
                break

            # the response is empty but rounds remain, so force the report to be written
            if step_idx < max_steps - 1:
                logger.warning(f"[agent_loop] Empty response at step {step_idx}, forcing report")
                sample.messages[-1]["content"] = ""
                sample.messages[0] = build_system_message(
                    max_steps, max_steps, coach_summary, rubrics=rubrics, lang=lang
                )
                sample = await base_generate(args, sample, sampling_params)
                sample.messages[-1] = {
                    "role": "assistant",
                    "content": sample.response.removesuffix(state.tokenizer.eos_token),
                }
                sample.response_message = prompter.parse_assistant_content(sample.response)
            break

        tool_call_tasks = [mcp_state.call_tool(t) for t in tool_calls]
        tool_responses = await asyncio.gather(*tool_call_tasks)
        max_chars = config.tool_response_max_chars
        for resp in tool_responses:
            content = resp.get("content", "")
            if len(content) > max_chars:
                resp["content"] = content[:max_chars] + f"\n\n[... 内容已截断，原始长度 {len(content)} 字符]"
        sample.messages.extend(tool_responses)
    else:
        # tool-calling rounds exhausted: use a prefill to force the report
        samples.append(
            Sample(
                group_index=sample.group_index, index=sample.index,
                messages=deepcopy(sample.messages), prompt=sample.prompt,
                label=sample.label, status=Sample.Status.PENDING,
                metadata=sample.metadata, train_metadata=None,
            )
        )
        sample = samples[-1]
        sample.messages[0] = build_system_message(
            max_steps, max_steps, coach_summary, rubrics=rubrics, lang=lang
        )
        sample = await _generate_forced_report(args, sample, sampling_params)
        sample.messages.append({
            "role": "assistant",
            "content": sample.response.removesuffix(state.tokenizer.eos_token),
        })
        sample.response_message = prompter.parse_assistant_content(sample.response)

    sample = samples[-1]
    for i, message in enumerate(sample.messages):
        if message["role"] == "assistant":
            sample.messages[i] = prompter.parse_assistant_content(message["content"])

    padding_num = (max_steps + 1) - len(samples)
    if padding_num > 0:
        samples = [
            Sample(
                group_index=sample.group_index, index=-1,
                tokens=[state.tokenizer.pad_token_id],
                reward=0.0, loss_mask=[], rollout_log_probs=[],
            )
            for _ in range(padding_num)
        ] + samples

    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# Rollout entry point
# ═══════════════════════════════════════════════════════════════════════════════


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    await MCPState(config.mcp_manager).get_servers()

    if isinstance(sample.prompt, str):
        sample.messages = [{"role": "user", "content": sample.prompt}]
    else:
        sample.messages = deepcopy(sample.prompt)

    # Eval always takes the student path (no memory) so it measures deployment behaviour
    if evaluation:
        samples = await agent_loop(
            args, sample, sampling_params, coach_summary=None,
        )
        await eval_reward(args, samples[-1])
        return samples[-1]

    # RG-KL disabled: plain GRPO
    if not config.enable_rg_kl:
        samples = await agent_loop(
            args, sample, sampling_params, coach_summary=None,
        )
        return samples

    # n_samples_per_prompt should equal k_student + k_teacher
    k = getattr(args, "n_samples_per_prompt", 1)
    k_s = config.k_student
    k_m = config.k_teacher
    if k_s + k_m != k:
        k_s = max(1, min(k - 1, int(k * 0.667)))
        k_m = k - k_s

    within_idx = sample.index % k
    group_id = sample.index // k

    rubrics = (sample.metadata or {}).get("rubrics", [])
    if isinstance(sample.prompt, str):
        query = sample.prompt
    elif isinstance(sample.prompt, list):
        query = next(
            (m.get("content", "") for m in reversed(sample.prompt) if m.get("role") == "user"),
            "",
        )
    else:
        query = ""
    lang = "en" if _is_english(query) else "zh"

    await _get_or_create_events(group_id)

    if within_idx < k_s:
        # -- STUDENT path -- (no memory, trained on)
        samples = await agent_loop(
            args, sample, sampling_params, coach_summary=None,
        )
        final = samples[-1]
        if final.metadata is None:
            final.metadata = {}
        final.metadata["source"] = "student"
        await _register_student_done(group_id, final, k_s)
        return samples
    else:
        # -- TEACHER path -- (with memory, used only to estimate Δ_r, excluded from gradients)
        memory = await _acquire_memory(group_id, k_s, query, rubrics, args, lang)
        samples = await agent_loop(
            args, sample, sampling_params, coach_summary=memory,
        )
        final = samples[-1]
        if final.metadata is None:
            final.metadata = {}
        final.metadata["source"] = "teacher"
        final.metadata["memory"] = memory or ""
        return samples

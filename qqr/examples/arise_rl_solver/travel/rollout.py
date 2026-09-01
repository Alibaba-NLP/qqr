"""
Travel Rubrics — Reward-Gated Reverse KL (RG-KL)

Three-phase procedure:
  Phase A - student rollout: k_s = 8 trajectories without memory (these are trained on)
  Phase B - memory generation: score the student rubrics -> LLM coach -> memory specific to this query
  Phase C - teacher rollout: k_m = 4 trajectories with memory injected into the system prompt (used only for Δ_r)

Training (RG-KL):
  Δ_r = mean(reward(τ_m)) - mean(reward(τ_s))  ← the gating signal
  λ(Δ_r) = λ_0 · max(Δ_r - δ_thresh, 0) · warmup · cosine_decay

  Student samples: guided_tokens = [p_m_prompt, τ_s_response]
                → slime's training-side forward pass yields teacher_log_probs = π_θ(τ_s | p_m)
                → apply_rg_kl_to_advantages: adv -= λ(Δ_r) · clamp(student_lp - teacher_lp)
  Teacher samples: guided_tokens = tokens (identity), reward neutralised to student_mean,
                loss_mask = 0 and rg_kl_coef = 0, so they contribute nothing to the gradient, only the Δ_r scalar

Differences from the memory-guided off-policy GRPO variant:
  - No token-swap off-policy importance sampling, avoiding a product of ratios over many turns
  - Teacher rollouts do not enter the PPO loss
  - Reverse-KL distillation replaces pushing the policy towards teacher trajectories with PPO
"""

import asyncio
import logging
import re
import math
from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Any

from qqr import registers
from qqr.rollout.agent_rollout import GenerateState, MCPState
from qqr.rollout.agent_rollout import generate as base_generate
from qqr.schemas import Sample

from . import config
from .reward_model import _compute_sample_reward, _save_group_summary, eval_reward, get_judge_client, get_judge_semaphore

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

    With multiple workers, each holding its own counter, every worker handles batch/num_workers
    groups, so the count stays correct (the per-worker batch is divided the same way).
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
    group_id: int, k_s: int, query: str, rubrics: list[str], args: Namespace,
) -> str | None:
    await _group_student_event[group_id].wait()

    async with _group_coord_lock:
        is_leader = group_id not in _group_memory_leader
        if is_leader:
            _group_memory_leader.add(group_id)

    if is_leader:
        try:
            student_samples = _group_student_samples[group_id]
            memory = await _generate_memory_for_group(group_id, student_samples, query, rubrics, args)
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
    query: str, rubrics: list[str], args: Namespace,
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
        expected_tools = (s.metadata or {}).get("expected_tools", [])
        expected_names = [t.get("name", "") for t in expected_tools]
        actual_names = rec.get("actual_tool_names", [])
        missing = rec.get("missing_tools", [])

        rubric_parts = ""
        for rd in rec.get("rubric_details", []):
            status = "+" if rd.get("met") else "-"
            rubric_parts += f"\n    {status} {rd.get('rubric', '')}"

        sample_summaries.append(
            f"### 第{i + 1}个样本 (reward={rec['reward']:.2f})\n"
            f"- 期望工具: {expected_names}  实际工具: {actual_names}  缺失: {missing if missing else '无'}\n"
            f"- rubric详情: {rubric_parts}\n"
        )

    coach_prompt = f"""你是一个旅行规划做题训练的教练。以下是同一个 query 下 {len(group_records)} 个做题者样本的评估结果。

## 用户原始问题
{query[:300]}

## 本组做题结果（{len(group_records)} 次尝试，平均 reward: {avg_reward:.2f}）

{"".join(sample_summaries)}

## 你的任务

分析做题者的失败模式，生成**具体、可操作**的做题建议（不超过300字）。

### 要求：
1. **失败根因**：最常见的失败原因是什么？（漏调工具？参数错误？信息缺失？推理错误？）
2. **正确的工具调用流程**：针对这个具体 query，应该按什么顺序调用哪些工具？缺失的工具为什么重要？
3. **回答要点**：rubric 要求哪些关键信息必须出现在回答中？做题者最容易遗漏什么？
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

    Teacher and padding samples use identity (guided_tokens = tokens); rg_kl_coef=0 afterwards
    guarantees no effect on gradients (the forward pass still runs, an unavoidable cost of the current slime interface).
    """
    state = GenerateState(args)
    tokenizer = state.tokenizer
    scope = config.kl_scope  # "all_response" or "final_only"

    # find the final valid turn (the last valid sample in each trajectory list)
    final_sample = None
    if scope == "final_only":
        for s in reversed(samples):
            if s.index != -1 and s.tokens and s.messages:
                final_sample = s
                break

    n_student_guided = 0
    n_identity = 0
    n_failed = 0
    for s in samples:
        source = (s.metadata or {}).get("source", "student")
        is_padding = (s.index == -1)
        is_valid_student = (
            source == "student"
            and not is_padding
            and bool(s.tokens)
            and bool(s.messages)
            and (s.response_length or 0) > 0
        )

        # The scope decides which turns take part in KL distillation:
        # - all_response: every turn gets the p_m swap
        # - final_only: only the final natural-language reply gets the p_m swap; tool-call tokens are skipped
        if scope == "final_only":
            should_swap = is_valid_student and (s is final_sample) and memory
        else:
            should_swap = is_valid_student and memory

        if should_swap:
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
        f"[rg_kl] guided_tokens (scope={scope}): {n_student_guided} student p_m-swap, "
        f"{n_identity} identity (failed={n_failed})"
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
            marker = "\n\n【本轮做题教练反馈】\n"
            if marker not in orig_system:
                messages_with_memory[0]["content"] = orig_system + marker + memory
            # else: already has memory (shouldn't happen for student, but safe)

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

    This makes λ_0 the true base strength of the KL term, rather than being squashed by two orders of magnitude by the gate.
    the threshold is rg_kl_delta_threshold (0.05 by default) and ramp_width is fixed at 0.05
    (so the gate reaches full strength once Δ_r exceeds the 0.05 threshold).

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

    # cosine decay factor: 1 → min_ratio over decay_iters
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
# System Message + Agent Loop
# ═══════════════════════════════════════════════════════════════════════════════


def build_system_message(
    step_idx: int, max_steps: int, coach_summary: str | None = None,
) -> dict:
    system_prompt = f"当前时间: {datetime.now().strftime('%d/%m/%Y, %H:%M')}"
    system_prompt += f"\n\n可调用{max_steps}轮工具，已调用{step_idx}轮。"
    system_prompt += "\n\n重要：你必须先使用工具查询真实数据，再根据工具返回的结果回答用户问题。严禁在未调用任何工具的情况下直接回答。"

    if coach_summary:
        system_prompt += f"\n\n【本轮做题教练反馈】\n{coach_summary}"

    if step_idx >= max_steps:
        system_prompt += "\n\n请直接回答，不要使用工具。"

    return {"role": "system", "content": system_prompt}


async def agent_loop(
    args: Namespace, sample: Sample, sampling_params: dict[str, Any],
    max_steps: int = config.max_steps, coach_summary: str | None = None,
) -> list[Sample]:
    state = GenerateState(args)
    mcp_state = MCPState(config.mcp_manager)
    ckpt_normalized = re.sub(r"[._]+", ".", state.args.hf_checkpoint.lower())
    if "qwen3.5" in ckpt_normalized:
        prompter = registers.prompt["qwen3.5"]()
    else:
        prompter = registers.prompt["qwen3"]()

    if sample.messages[0]["role"] != "system":
        sample.messages.insert(0, build_system_message(0, max_steps, coach_summary))
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
        sample.messages[0] = build_system_message(step_idx, max_steps, coach_summary)
        sample = await base_generate(args, sample, sampling_params)

        sample.messages.append({
            "role": "assistant",
            "content": sample.response.removesuffix(state.tokenizer.eos_token),
        })
        sample.response_message = prompter.parse_assistant_content(sample.response)
        tool_calls = sample.response_message.get("tool_calls") or []

        if not tool_calls:
            break

        tool_call_tasks = [mcp_state.call_tool(t) for t in tool_calls]
        tool_responses = await asyncio.gather(*tool_call_tasks)
        sample.messages.extend(tool_responses)
    else:
        samples.append(
            Sample(
                group_index=sample.group_index, index=sample.index,
                messages=deepcopy(sample.messages), prompt=sample.prompt,
                label=sample.label, status=Sample.Status.PENDING,
                metadata=sample.metadata, train_metadata=None,
            )
        )
        sample = samples[-1]
        sample.messages[0] = build_system_message(max_steps, max_steps, coach_summary)
        sample = await base_generate(args, sample, sampling_params)
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
    args: Namespace, sample: Sample, sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    await MCPState(config.mcp_manager).get_servers()

    if isinstance(sample.prompt, str):
        sample.messages = [{"role": "user", "content": sample.prompt}]
    else:
        sample.messages = deepcopy(sample.prompt)

    # Eval always takes the student path (no memory) so it measures deployment behaviour
    if evaluation:
        samples = await agent_loop(args, sample, sampling_params, coach_summary=None)
        await eval_reward(args, samples[-1])
        return samples[-1]

    # RG-KL disabled: plain GRPO
    if not config.enable_rg_kl:
        samples = await agent_loop(args, sample, sampling_params, coach_summary=None)
        return samples

    # n_samples_per_prompt should equal k_student + k_teacher
    k = getattr(args, "n_samples_per_prompt", 1)
    k_s = config.k_student
    k_m = config.k_teacher
    if k_s + k_m != k:
        # fallback: use the student_ratio default
        k_s = max(1, min(k - 1, int(k * 0.667)))
        k_m = k - k_s

    within_idx = sample.index % k
    group_id = sample.index // k

    rubrics = (sample.metadata or {}).get("rubrics", [])
    query = sample.prompt if isinstance(sample.prompt, str) else next(
        (m.get("content", "") for m in reversed(sample.prompt) if m.get("role") == "user"), ""
    ) if isinstance(sample.prompt, list) else ""

    # snapshot of the p_s system content (kept for logging and as a fallback; no longer used for swapping)
    p_s_system_content = build_system_message(0, config.max_steps, coach_summary=None)["content"]

    await _get_or_create_events(group_id)

    if within_idx < k_s:
        # -- STUDENT path -- (no memory, trained on)
        samples = await agent_loop(args, sample, sampling_params, coach_summary=None)
        final = samples[-1]
        if final.metadata is None:
            final.metadata = {}
        final.metadata["source"] = "student"
        final.metadata["_p_s_system_content"] = p_s_system_content
        await _register_student_done(group_id, final, k_s)
        return samples
    else:
        # -- TEACHER path -- (with memory, used only to estimate Δ_r, excluded from gradients)
        memory = await _acquire_memory(group_id, k_s, query, rubrics, args)
        samples = await agent_loop(args, sample, sampling_params, coach_summary=memory)
        final = samples[-1]
        if final.metadata is None:
            final.metadata = {}
        final.metadata["source"] = "teacher"
        final.metadata["memory"] = memory or ""
        final.metadata["_p_s_system_content"] = p_s_system_content
        return samples

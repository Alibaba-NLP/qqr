"""
VitaBench solver rollout - Reward-Gated Reverse KL (RG-KL)

Three-phase procedure:
  Phase A - student rollout (k_s samples): no memory; score the rubrics and cache the result (these are trained on)
  Phase B - memory generation: the LLM coach turns the student's failed rubrics into query-specific advice
  Phase C - teacher rollout (k_m samples): memory injected, used only to estimate Δ_r (reward neutralised, loss_mask=0)

Training (RG-KL)：
  Δ_r = mean(reward(τ_m)) - mean(reward(τ_s))   ← the gating signal
  λ(Δ_r) = λ_0 · gate(Δ_r) · warmup · cosine_decay

  Student samples: guided_tokens = [p_m_prompt, τ_s_response]
                → slime's training-side forward pass yields teacher_log_probs = π_θ(τ_s | p_m)
                → apply_rg_kl_to_advantages: adv -= λ(Δ_r) · clamp(student_lp - teacher_lp)
  Teacher samples: guided_tokens = tokens (identity), reward neutralised to student_mean,
                loss_mask = 0 and rg_kl_coef = 0, so they contribute nothing to the gradient, only the Δ_r scalar

Differences from the VitaBench memory-guided off-policy GRPO variant:
  - No p_s token-swap on the teacher, avoiding a product of ratios over many turns
  - Teacher rollouts do not enter the PPO loss
  - Instead applies the p_m token-swap to the student, triggering slime's reverse-KL distillation
"""

import asyncio
import fcntl
import json
import logging
import math
import re
from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from qqr import registers
from qqr.rollout.agent_rollout import GenerateState
from qqr.rollout.agent_rollout import generate as base_generate
from qqr.schemas import Sample
from qqr.tools.vitabench_env import (
    VitaBenchToolState,
    load_vitabench_task_by_id,
)

from . import config
from .reward_model import eval_reward, _compute_single_sample_reward

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# per-group coordination state (asyncio-safe, shared across coroutines in one process)
# ═══════════════════════════════════════════════════════════════════════════════

_group_student_samples: dict[int, list[Sample]] = defaultdict(list)
_group_student_event:   dict[int, asyncio.Event] = {}
_group_memory:          dict[int, str] = {}
_group_memory_leader:   set[int] = set()
_group_memory_event:    dict[int, asyncio.Event] = {}
_group_coord_lock = asyncio.Lock()

# global rollout counter: each group_reward call corresponds to one group.
# used for the warmup and cosine-decay schedules (the rollout side cannot read slime's rollout_id directly).
_rollout_step_counter: int = 0
_rollout_step_lock = asyncio.Lock()


async def get_rollout_step() -> int:
    """Return the current rollout step (monotonically increasing)."""
    return _rollout_step_counter


async def increment_rollout_step(rollout_batch_size: int = 1) -> int:
    """Increment the group counter and return the rollout_id (the number of completed rollouts).

    Called once per group. Each rollout contains `rollout_batch_size` groups.
    `rollout_id = (group_count-1) // batch` makes every group within a rollout see the same step.
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
    group_id: int,
    k_s: int,
    query: str,
    rubrics: list[str],
    args: Namespace,
) -> str | None:
    """Once all students finish, the leader produces the memory while the other teachers wait."""
    await _group_student_event[group_id].wait()

    async with _group_coord_lock:
        is_leader = group_id not in _group_memory_leader
        if is_leader:
            _group_memory_leader.add(group_id)

    if is_leader:
        try:
            student_samples = _group_student_samples[group_id]
            memory = await _generate_memory_for_group(
                group_id, student_samples, query, rubrics, args
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
    """Score the student samples and cache the result so group_reward can skip re-scoring."""
    result = await _compute_single_sample_reward(sample)
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["_rubric_cache"] = result
    return result


async def _generate_memory_for_group(
    group_id: int,
    student_samples: list[Sample],
    query: str,
    rubrics: list[str],
    args: Namespace,
) -> str | None:
    """Score every student rubric (caching the result) and call the LLM coach to produce the memory."""
    eval_results = await asyncio.gather(
        *[_evaluate_and_cache(s) for s in student_samples],
        return_exceptions=True,
    )

    group_records = []
    rewards = []
    for r in eval_results:
        if isinstance(r, Exception):
            logger.warning(f"[rg_kl] group={group_id} student eval failed: {r}")
            continue
        rewards.append(r["reward"])
        group_records.append(r)

    if not group_records:
        return None

    summary, judge_input = await generate_group_summary(
        group_id,
        [
            {
                "all_passed": r["rubric_all_passed"],
                "rubrics_met": r["rubrics_met"],
                "rubrics_total": r["rubrics_total"],
                "num_tool_calls": r["num_tool_calls"],
                "tool_names": r["actual_tool_names"],
                "failed_rubrics": r["failed_rubrics"],
            }
            for r in group_records
        ],
        query=query,
    )
    if summary:
        _save_group_summary(args, group_id, summary, judge_input)
    return summary or None


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

    n_student_guided = 0
    n_identity = 0
    n_failed = 0
    for s in samples:
        # padding / invalid → identity
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

        # all-response scope: apply the p_m swap to every valid student turn
        # intermediate tool-call turns also take part in KL distillation, internalising the memory's tool knowledge across every turn.
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
        f"[rg_kl] guided_tokens: {n_student_guided} student p_m-swap (all turns), "
        f"{n_identity} identity (failed={n_failed})"
    )


def _apply_p_m_swap(s: Sample, tokenizer, memory: str) -> bool:
    """Swap the student prompt to p_m (memory added) while keeping the original τ_s response tokens.
    the forward pass yields π_θ(τ_s | p_m) as the RG-KL teacher distribution.

    A note on VitaBench interactive mode: messages mix the system, user, assistant and tool roles,
    the `tool` role is mapped to user for consistent handling in apply_chat_template.
    """
    try:
        messages_with_memory = deepcopy(s.messages)
        # switch to the last assistant turn (its content is the response)
        last_asst_idx = next(
            (i for i in range(len(messages_with_memory) - 1, -1, -1)
             if messages_with_memory[i].get("role") == "assistant"),
            -1,
        )
        if last_asst_idx < 0:
            return False
        messages_with_memory = messages_with_memory[:last_asst_idx + 1]

        # inject the memory into the system prompt (the marker matches build_system_message)
        if messages_with_memory and messages_with_memory[0]["role"] == "system":
            orig_system = messages_with_memory[0]["content"]
            marker = "\n\n【上一轮做题教练反馈】\n"
            if marker not in orig_system:
                messages_with_memory[0]["content"] = orig_system + marker + memory

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
    """Release the group coordination state to avoid leaking memory. Called at the end of group_reward."""
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
    the threshold is rg_kl_delta_threshold (0.05 by default) and ramp_width is fixed at 0.05.
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
# Group memory: coach feedback generation (called in phase B)
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_group_summary(
    group_id: int,
    group_results: list[dict],
    query: str = "",
) -> tuple[str, str]:
    """
    Build the coaching prompt and call the LLM coach for advice on solving. Returns (summary, prompt).

    Design rationale:
    What the solver should learn is not that the answer is 2, but how to converge on the right result through dialogue and tool calls.
    the coach may freely give concrete details (quantity=2, order number XYZ and so on), but must wrap them in
    the flow of the conversation (such as confirming the quantity by asking), so that what the agent internalises is the skill of converging through dialogue
    rather than a hard-coded answer.
    """
    total = len(group_results)
    pass_count = sum(1 for r in group_results if r.get("all_passed"))
    rubrics_total = group_results[0].get("rubrics_total", 0) if group_results else 0

    prompt = f"""你是一个 OTA / 生活服务场景做题训练的**对话推理教练**。你要教做题者的不是"正确答案是什么"，而是**"怎么通过和用户对话 + 合理调用工具来收敛到正确答案"**。

## 用户原始指令（做题者会看到这段）
\"\"\"{query}\"\"\"

## 本组做题结果（{total} 次尝试）
- 全通过: {pass_count}/{total}
- 该任务的 rubric 总数: {rubrics_total}
"""

    for i, result in enumerate(group_results):
        status = "✓ 全通过" if result.get("all_passed") else "✗ 未通过"
        met = result.get("rubrics_met", 0)
        total_r = result.get("rubrics_total", 0)
        tools = result.get("tool_names", [])
        failed = result.get("failed_rubrics", [])

        prompt += f"\n### 第{i + 1}次\n"
        prompt += f"- 结果: {status} ({met}/{total_r})\n"
        prompt += f"- 工具调用数: {result.get('num_tool_calls', 0)}\n"
        prompt += f"- 使用工具: {', '.join(tools[:10])}\n"
        if failed:
            prompt += "- 未满足的 rubric（可参考，但务必把答案包装进对话流程，见下方格式）:\n"
            for fr in failed[:5]:
                prompt += f"  - {fr}\n"

    prompt += """
## 🎯 你的目标

观察做题者的失败模式，告诉他们：在这个具体任务上，**应该怎么用对话 + 工具调用，一步步把用户的模糊需求转化成准确的操作**。

---

## 🌟 核心原则（非常重要）

**不要告诉 agent 直接填什么值**。要告诉 agent 怎么通过对话 / 工具调用**把这个值"逼"出来**。

即使你想说"数量是 2"、"车次是 K1236"、"地址是 XX"，也**必须**把它表达成：
- "通过追问用户确认数量"（让 user 自己说出 2）
- "先 train_ticket_search 查候选，按用户的时间约束过滤"（让工具筛出 K1236）
- "让用户确认地址细节"（让 user 自己说清楚）

这样做的原因：做题者在真实使用场景下没有教练提示，必须靠对话能力。**你教的是能泛化的对话习惯，不是具体任务的答案**。

---

## ✅ 你可以包含的内容（放开限制）

- 🟢 **具体业务信息**（数量、时间、地点、ID、金额等）—— 但必须以**对话 / 工具收敛**的方式写
- 🟢 **推理流程和工具调用顺序**
- 🟢 **用户原话引用**
- 🟢 **业务约束和陷阱**（如"不冰 ≠ 冷藏"、"修改订单不是取消订单"）
- 🟢 **必须问用户的点**（信息缺失时的对话策略）

## 🚫 唯一的红线：不能出现"直接填答案"的表达

- ❌ "设置 quantity=2, specification='去冰'"
- ❌ "选 K1236"
- ❌ "取消订单号 90721002T02"
- ❌ "调用 create_delivery_order(quantity=2, ...)"

**改写方式**：把每个具体值都包装成对话/工具调用的**结果**，而不是**输入**。

---

## 📋 输出格式（≤ 300 字）

### 1. 失败根因
- 多数做题者犯了什么推理/对话上的错？（没问？没筛？直接臆测值？）

### 2. 正确的对话 + 工具流程
- 在这个任务上，怎么一步步通过对话和工具调用把用户需求转成操作？
- 可以点明具体业务细节，但每个具体值都要通过对话/工具**推导出来**

### 3. 必须问用户的点
- 用户**没说**的关键参数，在哪一步应该问

### 4. 工具调用纪律
- 哪些工具漏调会直接失败、哪些顺序不能颠倒

### 5. 一句话行动建议
- 下次做题先做这一件事，就能避免多数失败

---

**最后自检**：你的反馈里每个具体业务值（数字、ID、名称）是否都挂在"对话确认"或"工具筛选"这种动作上？如果不是，改写它。"""

    summary = await _summary_via_external_llm(prompt)
    return summary, prompt


async def _summary_via_external_llm(prompt: str) -> str:
    """Generate the group coaching summary with gpt-5.2 via DashScope."""
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
            max_tokens=500,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"[group_summary] Coach LLM call failed: {e}")
        return ""


def _save_group_summary(args: Namespace, group_id: int, summary: str, judge_input: str):
    """Write the group summary under a file lock."""
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
    with open(filepath, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

    logger.info(f"[group_summary] Saved summary for group {group_id}: {summary[:80]}...")


# ═══════════════════════════════════════════════════════════════════════════════
# Rollout entry point
# ═══════════════════════════════════════════════════════════════════════════════


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # read the task information from metadata
    metadata = sample.metadata or {}
    domain = metadata.get("domain", config.default_domain)

    # load the VitaBench task data (environment, instructions and user_profile are required)
    task_data = _load_task_data(metadata, domain)

    # create a per-sample environment so each task has its own DB state
    tool_state = VitaBenchToolState(task_data, domain=domain)

    # interactive mode: the initial message is empty (the agent greets first inside agent_loop)
    sample.messages = []
    sample.metadata = metadata
    sample.metadata["has_memory"] = False

    # extract what the UserSimulator needs
    user_profile = metadata.get("user_profile", {})
    instructions = metadata.get("instructions", "")
    if not instructions:
        instructions = task_data.get("instructions", "")

    # snapshot of the p_s system content without memory (kept for logging and as a fallback; no longer used for swapping)
    p_s_system_content = build_system_message(
        env_time=metadata.get("env_time", ""),
        coach_summary="",
        domain=domain,
    )["content"]

    try:
        # -- Eval always takes the student path (no memory) to measure deployment behaviour --
        if evaluation:
            samples = await agent_loop(
                args, sample, sampling_params, tool_state,
                coach_summary="",
                user_profile=user_profile,
                instructions=instructions,
                domain=domain,
            )
            await eval_reward(args, samples[-1])
            return samples[-1]

        # -- RG-KL disabled: fall back to plain GRPO --
        if not config.enable_rg_kl:
            samples = await agent_loop(
                args, sample, sampling_params, tool_state,
                coach_summary="",
                user_profile=user_profile,
                instructions=instructions,
                domain=domain,
            )
            final = samples[-1]
            if final.metadata is None:
                final.metadata = {}
            final.metadata["source"] = "student"
            return samples

        # n_samples_per_prompt should equal k_student + k_teacher
        k = getattr(args, "n_samples_per_prompt", 1)
        k_s = config.k_student
        k_m = config.k_teacher
        if k_s + k_m != k:
            # fallback: students default to half the group
            k_s = max(1, min(k - 1, k // 2))
            k_m = k - k_s

        within_idx = sample.index % k
        group_id = sample.index // k

        rubrics = metadata.get("rubrics", [])
        query = instructions

        await _get_or_create_events(group_id)

        if within_idx < k_s:
            # -- STUDENT path: rollout without memory (trained on) --
            samples = await agent_loop(
                args, sample, sampling_params, tool_state,
                coach_summary="",
                user_profile=user_profile,
                instructions=instructions,
                domain=domain,
            )
            final = samples[-1]
            if final.metadata is None:
                final.metadata = {}
            final.metadata["source"] = "student"
            final.metadata["_p_s_system_content"] = p_s_system_content
            await _register_student_done(group_id, final, k_s)
            return samples
        else:
            # -- TEACHER path: wait for the memory, inject it and roll out (used only for Δ_r) --
            memory = await _acquire_memory(group_id, k_s, query, rubrics, args)
            samples = await agent_loop(
                args, sample, sampling_params, tool_state,
                coach_summary=memory or "",
                user_profile=user_profile,
                instructions=instructions,
                domain=domain,
            )
            final = samples[-1]
            if final.metadata is None:
                final.metadata = {}
            final.metadata["source"] = "teacher"
            final.metadata["memory"] = memory or ""
            final.metadata["_p_s_system_content"] = p_s_system_content
            return samples

    finally:
        for s in (samples if "samples" in dir() else [sample]):
            if s.metadata:
                s.metadata.pop("_tool_state", None)
                s.metadata.pop("_task_data", None)


def _load_task_data(metadata: dict, domain: str) -> dict:
    """Load the full task data (environment, instructions and user_profile)."""
    if "environment" in metadata:
        # self-generated data: the environment is inlined
        return {"environment": metadata["environment"]}
    else:
        # official data: load the full task from tasks.json
        task_id = metadata.get("task_id", "")
        task_data = load_vitabench_task_by_id(
            task_id, domain, data_dir=config.vitabench_data_dir
        )
        # store instructions and user_profile back into metadata for the UserSimulator
        if "instructions" not in metadata and "instructions" in task_data:
            metadata["instructions"] = task_data["instructions"]
        if "user_profile" not in metadata:
            user_scenario = task_data.get("user_scenario", {})
            metadata["user_profile"] = user_scenario.get("user_profile", {})
        return task_data


DOMAIN_GUIDANCE = {
    "ota": """
# OTA 任务专项规范
- **火车票**：下单前必须确认并核对：出发日期（精确到日）、出发/到达城市、席别（硬座/硬卧/软卧/高铁二等/一等等）、铺位（上铺/中铺/下铺）、票数；日期若用户说"周六"等相对时间，需结合当前时间推算出精确日期
- **酒店**：下单前必须确认：入住日期、退房日期、房型、位置约束（距某地X公里内）、用户偏好（环境/价格/评分等）；先用 hotel_search_recommend 搜索，再用 get_ota_hotel_info 获取详情，确认满足所有约束后再下单
- **景点门票**：确认使用日期、票种（成人/儿童/老人）、票数，日期不能早于当前时间
- **下单前核查清单**：① 所有必填参数是否已获取 ② 日期是否准确 ③ 数量是否与用户需求一致 ④ 位置/距离约束是否满足
- 使用 get_user_historical_behaviors 了解用户偏好，辅助选择合适方案""",

    "delivery": """
# 外卖任务专项规范
- **店铺选择**：用户说"上次那家/常点的那家"时，必须先用 get_user_historical_behaviors / search_delivery_orders 从历史订单中定位店铺，不得凭空挑选
- **商品与规格**：下单前用 get_delivery_store_info / get_delivery_product_info 核对商品在菜单中真实存在，确认口味/规格/份量与用户要求一致；菜单中没有的商品要向用户说明并商量替代
- **送达时间约束**：遇到"X 点前送到"等约束，先用 address_to_longitude_latitude → longitude_latitude_to_distance → delivery_distance_to_time 估算 ETA，确认能按时送达再下单；下单后用 get_delivery_order_detail 复核预计送达时间，超时则修改或换店
- **地址与备注**：确认具体送达位置（楼栋/前台代收）与备注（少油少盐、辣度等），逐项写入订单
- **下单前核查清单**：① 店铺是否为用户指定的那家 ② 商品/口味/数量是否一致 ③ ETA 是否满足时间约束 ④ 地址与备注是否完整""",

    "instore": """
# 到店任务专项规范
- **门店与商品**：先用 instore_shop_search_recommend / instore_product_search_recommend 搜索，核对门店位置、营业时段、评分与商品适用规则（可用日期、人数规格、是否需预约）后再下单
- **套餐规格**：严格按用户人数选择对应规格（"六人餐"必须选 6 人套餐），不得用两份小规格拼凑，除非用户同意
- **订座/预约**：用 instore_book / instore_reservation 前先确认日期、时段、人数；"周六""后天"等相对时间须结合当前时间推算为精确日期并向用户确认；改期用 instore_modify_reservation，不要取消重建（除非必要）
- **位置约束**：涉及"某地附近/步行可达"时，先验证门店与目标位置的距离满足约束
- **下单前核查清单**：① 门店是否满足位置/评分/预算约束 ② 套餐/商品规格与人数是否匹配 ③ 日期时段是否精确且在营业/可约范围内 ④ 支付是否完成""",

    "cross_domain": """
# 跨域任务专项规范
- **参数确认原则**：每个子任务下单前，必须逐一确认所有关键参数（数量、日期、规格、位置等），不确定时主动询问用户
- **数量检查**：严格按用户说明的人数/件数下单，如"六人套餐"必须选6人规格，"2张票"必须下2张
- **日期/时间计算**：遇到"周六""明天""后天"等相对时间，结合当前时间精确推算，并向用户确认
- **位置约束**：涉及"附近X公里""步行可达"等约束，先用 get_nearby 或 address_to_longitude_latitude + longitude_latitude_to_distance 验证距离，确保满足约束再下单
- **执行顺序**：先完整收集所有子任务的需求，再逐一执行；每个子任务完成后向用户确认，再进行下一个
- **下单参数复核**：下单前在心中默念所有参数是否与用户要求完全匹配，避免选错套餐/规格/数量""",
}


def build_system_message(
    env_time: str = "",
    coach_summary: str = "",
    rubrics: list[str] | None = None,
    domain: str = "",
) -> dict:
    """
    Build the agent system prompt for interactive mode.

    Aligned with the official VitaBench agent_system_prompt.yaml.
    Key differences from solo mode:
    - Do not forbid interaction; the agent may ask the user questions and confirm details
    - Do not say the user states everything at once; requirements are revealed over several turns
    - Adds the ###STOP### termination rule
    """
    time_str = env_time or datetime.now().strftime("%Y-%m-%d %H:%M")
    system_prompt = f"""# 环境
- 当前时间：{time_str}

# 工具使用规范
- 当用户需求需要调工具来完成时，先判断是否已知全部参数信息，如果已知则抽取相应参数，否则询问用户相关参数值
- 当用户无法提供相关信息时，首先通过工具获取相关信息
- 参考工具描述中的 Precondition 和 Postcondition 确保任务正确完成

# 对话规范
- 仅利用上文已有信息，禁止无根据地构造信息并回复用户
- 以完成用户需求为目标，禁止发散性引导用户提出新需求
- 完成用户的任务需求后询问用户是否还有其他需求，如果用户表示没有，生成 '###STOP###' 标记来结束对话"""

    if domain in DOMAIN_GUIDANCE:
        system_prompt += DOMAIN_GUIDANCE[domain]

    if rubrics:
        system_prompt += "\n\n# 任务完成标准（你需要确保以下所有条件都被满足）"
        for i, r in enumerate(rubrics):
            system_prompt += f"\n{i+1}. {r}"

    if coach_summary:
        system_prompt += f"\n\n【上一轮做题教练反馈】\n{coach_summary}"

    return {"role": "system", "content": system_prompt}


async def agent_loop(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    tool_state: VitaBenchToolState,
    max_steps: int = config.max_steps,
    coach_summary: str = "",
    user_profile: dict = None,
    instructions: str = "",
    rubrics: list[str] | None = None,
    domain: str = "",
) -> list[Sample]:
    """
    The interactive multi-turn loop: agent <-> UserSimulator <-> tools.

    Procedure:
    1. The agent sends a greeting first
    2. The UserSimulator produces the first user message, stating part of the requirements
    3. Loop: the agent replies; if there is a tool_call run the tool, otherwise hand off to the UserSimulator
    4. Ends when either side emits ###STOP### or max_steps is reached
    """
    from .user_simulator import simulate_user_response, is_stop, STOP_SIGNAL

    state = GenerateState(args)
    ckpt_normalized = re.sub(r'[._]+', '.', state.args.hf_checkpoint.lower())
    if "qwen3.5" in ckpt_normalized:
        prompter = registers.prompt["qwen3.5"]()
    else:
        prompter = registers.prompt["qwen3"]()

    env_time = sample.metadata.get("env_time", "") if sample.metadata else ""

    # initial messages: the system turn plus the agent greeting
    sample.messages = [
        build_system_message(env_time, coach_summary, rubrics=rubrics, domain=domain),
        {"role": "assistant", "content": "你好，请问有什么可以帮您的？"},
    ]

    # the UserSimulator produces the first user message
    first_user_msg = await simulate_user_response(
        sample.messages, user_profile or {}, instructions, domain=domain,
    )
    if is_stop(first_user_msg):
        sample.messages.append({"role": "user", "content": first_user_msg})
        sample.status = Sample.Status.COMPLETED
        return [sample]

    sample.messages.append({"role": "user", "content": first_user_msg})

    samples = []

    for step_idx in range(max_steps):
        # -- The agent produces a reply --
        samples.append(
            Sample(
                group_index=sample.group_index,
                index=sample.index,
                messages=deepcopy(sample.messages),
                prompt=sample.prompt,
                label=sample.label,
                status=Sample.Status.PENDING,
                metadata=sample.metadata,
                train_metadata={"tools": tool_state.tools},
            )
        )
        sample = samples[-1]
        sample = await base_generate(args, sample, sampling_params)

        response_text = sample.response.removesuffix(state.tokenizer.eos_token)
        sample.messages.append({"role": "assistant", "content": response_text})
        sample.response_message = prompter.parse_assistant_content(
            sample.response, tools=tool_state.tools
        )
        tool_calls = sample.response_message.get("tool_calls") or []

        # -- Branch handling --
        if tool_calls:
            # a tool call is present: run the tool and continue with the agent, bypassing the UserSimulator
            tool_call_tasks = [tool_state.call_tool(t) for t in tool_calls]
            tool_responses = await asyncio.gather(*tool_call_tasks)
            sample.messages.extend(tool_responses)

        elif is_stop(response_text):
            # the agent emitted ###STOP###, so finish
            break

        else:
            # the agent emitted natural language, so hand off to the UserSimulator
            user_msg = await simulate_user_response(
                sample.messages, user_profile or {}, instructions, domain=domain,
            )

            if is_stop(user_msg):
                sample.messages.append({"role": "user", "content": user_msg})
                break

            sample.messages.append({"role": "user", "content": user_msg})

    else:
        # max_steps exhausted; force a final reply without tools
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
        sample.messages.append(
            {"role": "user", "content": "对话轮次已用完，请总结你已完成的操作。"}
        )
        sample = await base_generate(args, sample, sampling_params)
        sample.messages.append(
            {"role": "assistant", "content": sample.response.removesuffix(state.tokenizer.eos_token)}
        )
        sample.response_message = prompter.parse_assistant_content(
            sample.response, tools=tool_state.tools
        )

    # parse tool_calls out of every assistant message
    sample = samples[-1]
    for i, message in enumerate(sample.messages):
        if message["role"] == "assistant" and isinstance(message.get("content"), str):
            parsed = prompter.parse_assistant_content(
                message["content"], tools=tool_state.tools
            )
            if parsed != message:
                sample.messages[i] = parsed

    # padding keeps the batch size consistent
    padding_num = (max_steps + 1) - len(samples)
    if padding_num > 0:
        samples = [
            Sample(
                group_index=sample.group_index,
                index=-1,
                tokens=[state.tokenizer.pad_token_id],
                reward=0.0,
                loss_mask=[],
                rollout_log_probs=[],
            )
            for _ in range(padding_num)
        ] + samples

    return samples

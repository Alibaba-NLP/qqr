"""
VitaBench curriculum agent (generator) rollout - Qwen3.5 + RG-KL

Three-phase RG-KL procedure:
  Phase A - student rollout (k_s samples): generate without coach memory, score and cache the result
  Phase B - memory generation: the gpt-5.2 coach reviews the student's task quality and suggests improvements
  Phase C - teacher rollout (k_m samples): generate with coach memory injected, used only to estimate Δ_r

Training (RG-KL)：
  Δ_r = mean(reward(teacher)) - mean(reward(student))
  λ(Δ_r) = λ_0 · gate(Δ_r) · warmup · cosine_decay
  Student samples: guided_tokens = [p_m_prompt, τ_s_response]
  Teacher samples: reward neutralised and loss_mask=0, so they contribute nothing to the gradient

Generator specifics:
  - The generator explores the environment through the real VitaBenchToolState tools
  - Generates instructions + rubrics (the environment comes directly from the official data)
  - The solver validates task difficulty in interactive mode (UserSimulator <-> agent <-> tools)
  - reward = R_tool · R_fmt · (1 + R_diff) ∈ [0, 3] (paper Eq. (1)-(5), triangular difficulty reward)
"""

import asyncio
import fcntl
import json
import logging
import math
import os
import random
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
from qqr.rollout.agent_rollout import GenerateState
from qqr.rollout.agent_rollout import generate as base_generate
from qqr.schemas import Sample
from qqr.tools.vitabench_env import VitaBenchToolState, load_vitabench_tasks

from . import config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# real environment loading (globally cached)
# ═══════════════════════════════════════════════════════════════════════════════

_all_tasks: list[dict] | None = None


def _get_all_tasks() -> list[dict]:
    """Load and cache the VitaBench OTA environments."""
    global _all_tasks
    if _all_tasks is None:
        _all_tasks = load_vitabench_tasks(config.default_domain, config.vitabench_data_dir)
        logger.info(f"[rollout] Loaded {len(_all_tasks)} VitaBench {config.default_domain} tasks")
    return _all_tasks


# ═══════════════════════════════════════════════════════════════════════════════
# Rubric evaluator (reuses the sliding-window evaluator from arise_rl_solver/vitabench)
# ═══════════════════════════════════════════════════════════════════════════════

_rubric_evaluator = None


def get_rubric_evaluator():
    global _rubric_evaluator
    if _rubric_evaluator is None:
        from qqr.examples.arise_rl_solver.vitabench.reward_model import RubricEvaluator
        _rubric_evaluator = RubricEvaluator()
    return _rubric_evaluator


# ═══════════════════════════════════════════════════════════════════════════════
# RG-KL group coordination state (asyncio-safe, shared across coroutines in one process)
# ═══════════════════════════════════════════════════════════════════════════════

_group_student_samples: dict[int, list[dict]] = defaultdict(list)
_group_student_event:   dict[int, asyncio.Event] = {}
_group_memory:          dict[int, str] = {}
_group_memory_leader:   set[int] = set()
_group_memory_event:    dict[int, asyncio.Event] = {}
_group_coord_lock = asyncio.Lock()

# Rollout step counter
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
    """Register a student generation as complete and collect its evaluation result."""
    async with _group_coord_lock:
        _group_student_samples[group_id].append(result_info)
        if len(_group_student_samples[group_id]) >= k_s:
            _group_student_event[group_id].set()


async def _acquire_memory(
    group_id: int,
    k_s: int,
    args: Namespace,
) -> str | None:
    """Once all students finish, the leader produces the coach memory while the other teachers wait."""
    await _group_student_event[group_id].wait()

    async with _group_coord_lock:
        is_leader = group_id not in _group_memory_leader
        if is_leader:
            _group_memory_leader.add(group_id)

    if is_leader:
        try:
            student_results = _group_student_samples[group_id]
            memory = await _generate_coach_memory(group_id, student_results, args)
            async with _group_coord_lock:
                _group_memory[group_id] = memory or ""
        except Exception as e:
            logger.error(f"[rg_kl] group={group_id} coach memory generation error: {e}")
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
    """Release the group coordination state to avoid leaking memory."""
    _group_student_samples.pop(group_id, None)
    _group_student_event.pop(group_id, None)
    _group_memory.pop(group_id, None)
    _group_memory_leader.discard(group_id)
    _group_memory_event.pop(group_id, None)


# ═══════════════════════════════════════════════════════════════════════════════
# Coach memory generation (coaching feedback for the generator)
# ═══════════════════════════════════════════════════════════════════════════════


async def _generate_coach_memory(
    group_id: int,
    student_results: list[dict],
    args: Namespace,
) -> str | None:
    """Review the student's generated tasks and call gpt-5.2 for improvement advice."""
    summary, judge_input = await generate_group_summary(group_id, student_results)
    if summary:
        _save_group_summary(args, group_id, summary, judge_input)
    return summary or None


async def generate_group_summary(
    group_id: int,
    group_results: list[dict],
) -> tuple[str, str]:
    """Build the coaching prompt and call the coach LLM for improvement advice."""
    rewards = [r.get("reward", 0) for r in group_results]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    reward_dist = dict(Counter([f"{r:.1f}" for r in rewards]))

    sample_summaries = []
    for i, r in enumerate(group_results):
        diag = _reward_diagnosis(
            r.get("reward", 0),
            r.get("avg_rubric_pass_rate"),
            success_count=r.get("success_count"),
            num_trials=r.get("num_trials"),
        )
        summary = (
            f"### 第{i+1}次出题\n"
            f"- reward: {r.get('reward', 0):.1f} ({diag})\n"
            f"- 题目: {r.get('generated_instructions', '(无)')[:200]}\n"
            f"- rubrics 数量: {len(r.get('rubrics', []))}\n"
            f"- 环境: {r.get('base_task_id', '?')}\n"
        )
        sample_summaries.append(summary)

    prompt = f"""你是一个生活服务场景出题训练的教练。以下是一组 {len(group_results)} 次出题结果。

## 奖励机制说明（R_G = R_tool · R_fmt · (1 + R_diff)，三角难度奖励）
- reward=0: 门控失败——出题者未调用工具或输出格式错误
- reward=1: 门控通过，但做题者 K 次试验全成功/全失败（难度退化，R_diff=0）
- reward∈(1,3): 难度偏离 K/2 但仍有区分度，随 |c−K/2| 线性衰减
- reward=3（峰值）: 做题者 K 次试验中约一半成功（c≈K/2），难度恰好落在 Solver 能力边界

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


def _reward_diagnosis(
    reward: float,
    avg_rubric_rate: float | None = None,
    success_count: int | None = None,
    num_trials: int | None = None,
) -> str:
    if reward == 0 and avg_rubric_rate is None:
        return "出题者未调用工具或格式错误"
    if success_count is not None and num_trials is not None:
        if success_count == 0:
            return f"成功 0/{num_trials}（全做错，难度过高，R_diff=0 → reward=1）"
        if success_count == num_trials:
            return f"成功 {num_trials}/{num_trials}（全做对，难度过低，R_diff=0 → reward=1）"
        peak = num_trials // 2
        tag = "✓ 峰值" if success_count == peak else "中等区间"
        return (
            f"成功 {success_count}/{num_trials}（{tag}，"
            f"rubric 均值={avg_rubric_rate:.0%} → reward={reward:.1f}）"
        )
    if avg_rubric_rate is not None:
        return f"rubric 通过率={avg_rubric_rate:.0%}"
    return f"reward={reward:.1f}"


async def _summary_via_external_llm(prompt: str) -> str:
    """Generate the coach summary with gpt-5.2 via DashScope."""
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


def _save_group_summary(args: Namespace, group_id: int, summary: str, judge_input: str):
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

    logger.info(f"[group_summary] Saved for group {group_id}: {summary[:80]}...")


# ═══════════════════════════════════════════════════════════════════════════════
# RG-KL guided_tokens construction
# ═══════════════════════════════════════════════════════════════════════════════


def compute_rg_kl_guided_tokens(
    args: Namespace,
    samples: list[Sample],
    memory: str | None,
):
    """Build guided_tokens = [p_m_prompt, τ_s_response] for student samples.

    The generator's memory is injected at the end of the user prompt, in the coach-feedback section.
    """
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
    """Swap the student prompt to p_m (coach memory added) while keeping the original τ_s response tokens.

    Generator case: the memory is appended to the user prompt (under the previous-round coach feedback heading).
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

        # the generator's memory is appended to the last user message (the task prompt)
        injected = False
        for i in range(len(messages_with_memory) - 1, -1, -1):
            if messages_with_memory[i].get("role") == "user":
                orig_content = messages_with_memory[i]["content"]
                marker = "\n\n## 上一轮出题教练反馈\n"
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
    """λ(Δ_r) = λ_0 · gate(Δ_r) · warmup · cosine_decay"""
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
# Main generator entry point
# ═══════════════════════════════════════════════════════════════════════════════


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    k = getattr(args, "n_samples_per_prompt", 8)
    k_s = config.k_student
    k_m = config.k_teacher
    if k_s + k_m != k:
        k_s = max(1, min(k - 1, k // 2))
        k_m = k - k_s

    within_idx = sample.index % k
    group_id = sample.index // k
    group_rng = random.Random(group_id)

    # pick a random real environment, shared across the group
    all_tasks = _get_all_tasks()
    task_data = group_rng.choice(all_tasks)
    base_task_id = task_data.get("id", "unknown")

    # create the real tool environment
    try:
        tool_state = VitaBenchToolState(task_data, domain=config.default_domain)
    except Exception as e:
        logger.warning(f"[generate] Failed to create VitaBenchToolState for {base_task_id}: {e}")
        sample.reward = 0.0
        return sample if evaluation else [sample]

    # build a user profile consistent with the environment
    user_profile = config.generate_matched_user_profile(task_data, group_rng)

    # choose the order type from what the environment actually offers
    available_types = config.get_available_types(task_data, config.default_domain)
    chosen_types = config.choose_order_types(available_types, group_rng)

    sample.metadata = sample.metadata or {}
    sample.metadata["user_profile"] = user_profile
    sample.metadata["base_task_id"] = base_task_id
    sample.metadata["chosen_types"] = chosen_types
    sample.metadata["_is_evaluation"] = evaluation

    # -- Eval mode: no coach memory --
    if evaluation:
        return await _do_curriculum_rollout(
            args, sample, sampling_params, tool_state, task_data,
            user_profile, chosen_types, base_task_id,
            coach_summary="", source="student", group_id=group_id,
            evaluation=True,
        )

    # -- RG-KL disabled: plain GRPO with no coach memory anywhere --
    if not config.enable_rg_kl:
        return await _do_curriculum_rollout(
            args, sample, sampling_params, tool_state, task_data,
            user_profile, chosen_types, base_task_id,
            coach_summary="", source="student", group_id=group_id,
            evaluation=False,
        )

    # -- RG-KL three phases --
    await _get_or_create_events(group_id)

    if within_idx < k_s:
        # -- STUDENT path: no coach memory --
        result = await _do_curriculum_rollout(
            args, sample, sampling_params, tool_state, task_data,
            user_profile, chosen_types, base_task_id,
            coach_summary="", source="student", group_id=group_id,
            evaluation=False,
        )

        # collect the student evaluation results for phase B
        final = result[-1] if isinstance(result, list) else result
        result_info = {
            "reward": final.reward if final.reward is not None else 0.0,
            "generated_instructions": (final.metadata or {}).get("generated_instructions", ""),
            "rubrics": (final.metadata or {}).get("rubrics", []),
            "avg_rubric_pass_rate": (final.metadata or {}).get("avg_rubric_pass_rate"),
            "base_task_id": base_task_id,
        }
        await _register_student_done(group_id, result_info, k_s)
        return result

    else:
        # -- TEACHER path: wait for the coach memory --
        memory = await _acquire_memory(group_id, k_s, args)
        result = await _do_curriculum_rollout(
            args, sample, sampling_params, tool_state, task_data,
            user_profile, chosen_types, base_task_id,
            coach_summary=memory or "", source="teacher", group_id=group_id,
            evaluation=False,
        )

        # mark the teacher samples
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
    tool_state: VitaBenchToolState,
    task_data: dict,
    user_profile: dict,
    chosen_types: list[str],
    base_task_id: str,
    coach_summary: str,
    source: str,
    group_id: int,
    evaluation: bool,
) -> Sample | list[Sample]:
    """A single generation rollout: generate -> parse -> solver validation -> reward."""
    k = getattr(args, "n_samples_per_prompt", 8)

    # build the prompt
    task_prompt = config.build_user_prompt(
        user_profile, task_data, chosen_types, coach_summary,
        domain=config.default_domain,
    )

    sample.messages = [
        {"role": "system", "content": config.get_query_generation_system_prompt(config.default_domain)},
        {"role": "user", "content": task_prompt},
    ]
    sample.prompt = task_prompt
    sample.metadata = sample.metadata or {}
    sample.metadata["has_memory"] = bool(coach_summary)
    sample.metadata["source"] = source

    # multi-round real tool calls by the generator
    samples = await curriculum_generate_with_tool(
        args, sample, sampling_params, tool_state
    )

    if not samples:
        logger.warning(f"[generate] No valid samples")
        sample.reward = 0.0
        return sample if evaluation else [sample]

    # parse the final output
    final_sample = samples[-1]
    raw_output = final_sample.response or ""
    generated_instructions, rubrics = parse_curriculum_output(raw_output)

    for s in samples:
        if s.metadata is None:
            s.metadata = {}
        s.metadata["generated_instructions"] = generated_instructions
        s.metadata["rubrics"] = rubrics
        s.metadata["base_task_id"] = base_task_id
        s.metadata["curriculum_raw_output"] = raw_output
        s.metadata["source"] = source

    # compute the reward
    await execute_and_evaluate(
        args, samples, tool_state, user_profile,
        generated_instructions, rubrics, task_data,
    )

    if evaluation:
        return samples[-1]
    else:
        valid = [s for s in samples if s.rollout_log_probs is not None and s.response_length > 0]
        return valid if valid else [samples[-1]]


# ═══════════════════════════════════════════════════════════════════════════════
# multi-round tool-calling generation against real tools (Qwen3.5-compatible)
# ═══════════════════════════════════════════════════════════════════════════════


async def curriculum_generate_with_tool(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    tool_state: VitaBenchToolState,
    max_steps: int = config.max_steps,
) -> list[Sample]:
    """Multi-round tool-calling generation. Qwen3.5-compatible: selects the prompter dynamically and forwards the tools argument."""
    state = GenerateState(args)

    # Qwen3.5 dynamic prompt selection
    ckpt_normalized = re.sub(r'[._]+', '.', state.args.hf_checkpoint.lower())
    if "qwen3.5" in ckpt_normalized:
        prompter = registers.prompt["qwen3.5"]()
    else:
        prompter = registers.prompt["qwen3"]()

    samples = []

    for step_idx in range(max_steps):
        new_sample = Sample(
            group_index=sample.group_index,
            index=sample.index,
            messages=deepcopy(sample.messages),
            prompt=sample.prompt,
            label=sample.label,
            status=Sample.Status.PENDING,
            metadata=sample.metadata,
            train_metadata={"tools": tool_state.tools},
        )
        samples.append(new_sample)
        sample = new_sample

        sample = await base_generate(args, sample, sampling_params)
        response_content = sample.response.removesuffix(state.tokenizer.eos_token).strip()
        # Qwen3.5 compatibility: pass the tools argument to parse_assistant_content
        parsed = prompter.parse_assistant_content(response_content, tools=tool_state.tools)
        tool_calls = parsed.get("tool_calls") or []

        if not tool_calls:
            sample.messages.append(prompter.parse_assistant_content(
                response_content, tools=tool_state.tools
            ))
            break

        # a tool call is present, so run the real tool
        assistant_msg = prompter.parse_assistant_content(response_content, tools=tool_state.tools)
        # the prompter returns tool_calls[i].function.arguments as a JSON string,
        # Qwen3.5's chat_template uses `arguments | items` and expects a dict, so rendering would blow up
        # normalise a copy here so the next round's apply_chat_template renders correctly
        # the original tool_calls keep their string arguments for tool_state.call_tool
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
            result = await tool_state.call_tool(tc)
            sample.messages.append(result)

        logger.info(
            f"[curriculum] step {step_idx}: called "
            f"{[tc.get('function', {}).get('name', tc.get('name', '')) for tc in tool_calls]}"
        )

    else:
        # maximum rounds reached, so force the output
        final_sample = Sample(
            group_index=sample.group_index,
            index=sample.index,
            messages=deepcopy(sample.messages),
            prompt=sample.prompt,
            label=sample.label,
            status=Sample.Status.PENDING,
            metadata=sample.metadata,
            train_metadata=None,
        )
        final_sample.messages[0] = {
            "role": "system",
            "content": config.get_query_generation_system_prompt(config.default_domain)
            + "\n\n你已完成工具调用，请直接输出最终的 JSON 结果，不要再调用工具。",
        }
        samples.append(final_sample)
        sample = final_sample
        sample = await base_generate(args, sample, sampling_params)
        response_content = sample.response.removesuffix(state.tokenizer.eos_token).strip()
        sample.messages.append({"role": "assistant", "content": response_content})

    # parse every assistant message
    # note: the loop above already shapes every assistant message as {content, tool_calls, reasoning_content}
    # as a structured dict; here we only parse messages that are not yet structured, such as the raw ones added by the fallback branch.
    # the `not msg.get("tool_calls")` guard stops parse_assistant_content("") from wiping already-stored tool_calls.
    final = samples[-1]
    for i, msg in enumerate(final.messages):
        if (msg["role"] == "assistant"
            and isinstance(msg.get("content"), str)
            and not msg.get("tool_calls")
            and not msg.get("reasoning_content")):
            final.messages[i] = prompter.parse_assistant_content(
                msg["content"], tools=tool_state.tools
            )

    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# parse the generator output
# ═══════════════════════════════════════════════════════════════════════════════


def parse_curriculum_output(text: str) -> tuple[str, list[str]]:
    """Parse the generator's JSON output and extract instructions and rubrics."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    code_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_match:
        json_text = code_match.group(1)
    else:
        json_text = text

    try:
        start = json_text.index("{")
        end = json_text.rindex("}") + 1
        data = json.loads(json_text[start:end])
        instructions = data.get("instructions", "") or data.get("query", "")
        rubrics = data.get("rubrics", [])
        if isinstance(rubrics, list):
            return instructions, rubrics
    except (ValueError, json.JSONDecodeError):
        pass

    return "", []


# ═══════════════════════════════════════════════════════════════════════════════
# reward: R_G = R_tool · R_fmt · (1 + R_diff)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_format_reward(instructions: str, rubrics: list[str]) -> float:
    """Format reward: 1.0 when instructions are non-empty and rubrics is a non-empty list."""
    if not instructions:
        return 0.0
    if not rubrics or not isinstance(rubrics, list) or len(rubrics) == 0:
        return 0.0
    return 1.0


async def execute_and_evaluate(
    args: Namespace,
    samples: list[Sample],
    tool_state: VitaBenchToolState,
    user_profile: dict,
    generated_instructions: str,
    rubrics: list[str],
    task_data: dict,
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
    format_reward = compute_format_reward(generated_instructions, rubrics)
    if format_reward < 1.0:
        _apply_reward(
            samples, format_reward,
            format_reward=format_reward,
            generated_instructions=generated_instructions,
        )
        return

    # Stage 3: run the solver in interactive mode N times
    num_trials = config.executor_num_trials
    trial_rubric_rates = []

    for trial_idx in range(num_trials):
        try:
            rubric_rate = await call_executor_trial(
                generated_instructions, rubrics, task_data, user_profile
            )
            trial_rubric_rates.append(rubric_rate)
            logger.info(
                f"[execute_and_evaluate] trial {trial_idx}/{num_trials}: "
                f"rubric_rate={rubric_rate:.2f}"
            )
        except Exception as e:
            logger.error(f"[execute_and_evaluate] trial {trial_idx} failed: {e}")
            trial_rubric_rates.append(0.0)

    avg_rubric_rate = sum(trial_rubric_rates) / len(trial_rubric_rates) if trial_rubric_rates else 0.0

    # Triangular difficulty-shaped reward (paper Eq. (4)-(5))
    #   r_i          = α·s_i + (1−α)·𝟙[s_i=1]    # solver reward per attempt (α=0.8, Eq. (6))
    #   c            = Σ_i 𝟙(r_i ≥ γ)             # number of successful trials
    #   R_difficulty = 2 * max(0, 1 - |c - K/2| / (K/2))
    # peaks at 2 when c = K/2, is 0 at c in {0, K}, and decays linearly in between
    K = num_trials
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

    # paper Eq. (1): R_G = R_tool · R_fmt · (1 + R_diff); both gates were already
    # stages 1 and 2 are guaranteed to be 1; on failure we return reward=0 early
    total_reward = format_reward * (1.0 + difficulty_reward)
    base_task_id = task_data.get("id", "unknown")

    result_info = {
        "generated_instructions": generated_instructions,
        "rubrics": rubrics,
        "format_reward": format_reward,
        "difficulty_reward": difficulty_reward,
        "avg_rubric_pass_rate": avg_rubric_rate,
        "success_count": success_count,
        "success_threshold": gamma,
        "trial_rubric_rates": trial_rubric_rates,
        "num_trials": num_trials,
        "base_task_id": base_task_id,
    }

    _apply_reward(samples, total_reward, **result_info)

    logger.info(
        f"[execute_and_evaluate] env={base_task_id} "
        f"instructions={generated_instructions[:50]}... "
        f"rubrics={len(rubrics)} K={num_trials} γ={gamma} "
        f"c={success_count}/{num_trials} "
        f"avg_rubric_rate={avg_rubric_rate:.2f} "
        f"format={format_reward:.0f} "
        f"difficulty={difficulty_reward:.2f} total={total_reward:.2f}"
    )


def _apply_reward(samples: list[Sample], reward: float, **metadata):
    for s in samples:
        s.reward = reward
        if s.metadata is None:
            s.metadata = {}
        s.metadata.update(metadata)


# ═══════════════════════════════════════════════════════════════════════════════
# run the solver (interactive mode, a single trial)
# ═══════════════════════════════════════════════════════════════════════════════

_executor_client: AsyncOpenAI | None = None
_executor_semaphore: asyncio.Semaphore | None = None


def get_executor_client() -> AsyncOpenAI:
    global _executor_client
    if _executor_client is None:
        _executor_client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=config.executor_api_base,
            timeout=180,
            max_retries=3,
        )
    return _executor_client


def get_executor_semaphore() -> asyncio.Semaphore:
    global _executor_semaphore
    if _executor_semaphore is None:
        _executor_semaphore = asyncio.Semaphore(config.executor_concurrency_limit)
    return _executor_semaphore


STOP_SIGNAL = "###STOP###"

_user_sim_client: AsyncOpenAI | None = None


def get_user_sim_client() -> AsyncOpenAI:
    global _user_sim_client
    if _user_sim_client is None:
        _user_sim_client = AsyncOpenAI(
            api_key=config.user_simulator_api_key,
            base_url=config.user_simulator_base_url,
            timeout=300,
            max_retries=5,
        )
    return _user_sim_client


def _format_persona(user_profile: dict) -> str:
    if not user_profile:
        return "普通用户"
    return "\n".join(f"- {k}：{v}" for k, v in user_profile.items())


def _flip_roles_for_user_sim(messages: list[dict]) -> list[dict]:
    """Flip roles: the agent's assistant turns become user turns and vice versa; system and tool turns are skipped."""
    flipped = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("system", "tool"):
            continue
        if role == "assistant":
            if not content:
                continue
            flipped.append({"role": "user", "content": content})
        elif role == "user":
            flipped.append({"role": "assistant", "content": content})
    return flipped


async def _simulate_user_response(
    messages: list[dict],
    user_profile: dict,
    instructions: str,
) -> str:
    persona = _format_persona(user_profile)
    system_prompt = config.EXECUTOR_USER_SIMULATOR_PROMPT.format(
        persona=persona, instructions=instructions,
    )
    sim_messages = [{"role": "system", "content": system_prompt}]
    sim_messages.extend(_flip_roles_for_user_sim(messages))

    client = get_user_sim_client()
    try:
        resp = await client.chat.completions.create(
            model=config.user_simulator_model,
            messages=sim_messages,
            temperature=config.user_simulator_temperature,
            max_completion_tokens=500,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = resp.choices[0].message.content or ""
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if "</think>" in content:
            content = content.split("</think>", 1)[-1].strip()
        return content
    except Exception as e:
        logger.warning(f"[user_sim] LLM call failed: {e}")
        return STOP_SIGNAL


async def call_executor_trial(
    instructions: str,
    rubrics: list[str],
    task_data: dict,
    user_profile: dict,
) -> float:
    """Run one solver trial in interactive mode and return the rubric pass rate."""
    try:
        executor_tool_state = VitaBenchToolState(task_data, domain=config.default_domain)
    except Exception as e:
        logger.warning(f"[executor_trial] Failed to create VitaBenchToolState: {e}")
        return 0.0

    env = task_data.get("environment", {})
    env_time = env.get("time", datetime.now().strftime("%Y-%m-%d %H:%M"))
    client = get_executor_client()
    semaphore = get_executor_semaphore()

    # kept identical to the real solver: per-domain rules are appended so the difficulty estimate stays unbiased
    from qqr.examples.arise_rl_solver.vitabench.rollout import DOMAIN_GUIDANCE

    executor_system = config.EXECUTOR_SYSTEM_PROMPT.format(env_time=env_time)
    executor_system += DOMAIN_GUIDANCE.get(config.default_domain, "")

    messages = [
        {"role": "system", "content": executor_system},
        {"role": "assistant", "content": "你好，请问有什么可以帮您的？"},
    ]

    first_user_msg = await _simulate_user_response(
        messages, user_profile, instructions,
    )
    if STOP_SIGNAL in first_user_msg:
        messages.append({"role": "user", "content": first_user_msg})
        evaluator = get_rubric_evaluator()
        try:
            score, _ = await evaluator.evaluate(messages, rubrics, instructions)
        except Exception:
            score = 0.0
        return score

    messages.append({"role": "user", "content": first_user_msg})

    max_steps = config.executor_max_steps

    async with semaphore:
        for step in range(max_steps):
            try:
                resp = await client.chat.completions.create(
                    model=config.executor_model,
                    messages=messages,
                    tools=executor_tool_state.tools,
                    temperature=0.7,
                )
            except Exception as e:
                logger.warning(f"[executor_trial] API error at step {step}: {e}")
                break

            msg = resp.choices[0].message

            if msg.tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
                messages.append(assistant_msg)

                for tc in msg.tool_calls:
                    tc_dict = {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    result = await executor_tool_state.call_tool(tc_dict)
                    messages.append(result)

            elif STOP_SIGNAL in (msg.content or ""):
                messages.append({"role": "assistant", "content": msg.content or ""})
                break

            else:
                messages.append({"role": "assistant", "content": msg.content or ""})

                user_msg = await _simulate_user_response(
                    messages, user_profile, instructions,
                )

                if STOP_SIGNAL in user_msg:
                    messages.append({"role": "user", "content": user_msg})
                    break

                messages.append({"role": "user", "content": user_msg})

    evaluator = get_rubric_evaluator()
    try:
        score, details = await evaluator.evaluate(messages, rubrics, instructions)
    except Exception as e:
        logger.warning(f"[executor_trial] Rubric evaluation failed: {e}")
        score = 0.0

    return score


# ═══════════════════════════════════════════════════════════════════════════════
# helper functions
# ═══════════════════════════════════════════════════════════════════════════════


def extract_tool_calls_from_messages(messages: list[dict]) -> list[dict]:
    """Extract every tool call from the messages."""
    tools = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            name = tc.get("function", {}).get("name") or tc.get("name", "")
            raw_args = tc.get("function", {}).get("arguments") or tc.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            else:
                args = raw_args
            if name:
                tools.append({"name": name, "arguments": args})
    return tools

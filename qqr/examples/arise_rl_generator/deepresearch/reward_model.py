"""
DeepResearch Curriculum Agent Reward Model — RG-KL

The generator reward is computed in execute_and_evaluate in rollout.py (paper Eq. (1)-(5)):
- R_tool (0/1 gate): the generation trajectory must contain real tool observations
- R_fmt  (0/1 gate): query is non-empty and rubrics is a non-empty list
- R_diff (0-2): triangular difficulty reward 2·max(0, 1 − |c − K/2| / (K/2)),
  c = Σ_i 𝟙(r_i ≥ γ)，r_i = α·s_i + (1−α)·𝟙[s_i=1]（K=8, γ=0.9, α=0.8）
- reward = R_tool · R_fmt · (1 + R_diff) ∈ [0, 3]

group_reward (RG-KL)：
- The k_s students take part in GRPO; the k_m teachers have their reward neutralised and loss_mask=0
- λ(Δ_r) = λ_0 · gate · warmup · cosine_decay
- guided_tokens and rg_kl_coef are written onto the sample
"""

import asyncio
import logging
from argparse import Namespace

from qqr.schemas import Sample

from . import config

logger = logging.getLogger(__name__)


async def eval_reward(args: Namespace, sample: Sample, **kwargs):
    """Reward in evaluation mode (already computed during rollout)."""
    pass


async def group_reward(args: Namespace, group: list[list[Sample]], **kwargs):
    """Reward-Gated Reverse KL (RG-KL) group_reward。"""
    if len(group) <= 1:
        raise ValueError("group size must be greater than 1")

    real_samples = [g[-1] for g in group]
    n_samples_per_prompt = getattr(args, "n_samples_per_prompt", len(group))
    group_id = (
        real_samples[0].index // n_samples_per_prompt
        if n_samples_per_prompt > 1 else real_samples[0].index
    )

    # -- 1) Collect rewards --
    raw_rewards = []
    for idx, sample_group in enumerate(group):
        final = sample_group[-1]
        reward = final.reward if final.reward is not None else 0.0
        raw_rewards.append(reward)

        source = (final.metadata or {}).get("source", "student")
        meta = final.metadata or {}
        logger.info(
            f"[group_reward] idx={idx} source={source} "
            f"format={meta.get('format_reward', '?')} "
            f"difficulty={meta.get('difficulty_reward', '?')} "
            f"rubric_rate={meta.get('avg_rubric_pass_rate', '?')} "
            f"reward={reward:.2f}"
        )

    # ── 2) Δ_r ──
    student_rewards = [r for r, s in zip(raw_rewards, real_samples) if (s.metadata or {}).get("source") == "student"]
    teacher_rewards = [r for r, s in zip(raw_rewards, real_samples) if (s.metadata or {}).get("source") == "teacher"]
    avg_student = sum(student_rewards) / len(student_rewards) if student_rewards else 0.0
    avg_teacher = sum(teacher_rewards) / len(teacher_rewards) if teacher_rewards else 0.0
    delta_r = avg_teacher - avg_student

    # -- 3) Neutralise the teacher --
    for idx, sample_group in enumerate(group):
        source = (real_samples[idx].metadata or {}).get("source", "student")
        if source == "teacher":
            for sample in sample_group:
                sample.reward = avg_student
                if sample.metadata is None:
                    sample.metadata = {}
                sample.metadata["_neutralized_for_grpo"] = True
                sample.metadata["_raw_reward"] = raw_rewards[idx]
                if hasattr(sample, "loss_mask") and sample.loss_mask is not None:
                    sample.loss_mask = [0] * len(sample.loss_mask)

    # ── 4) λ(Δ_r) + guided_tokens ──
    from .rollout import (
        cleanup_group_coordination,
        compute_rg_kl_guided_tokens,
        compute_rg_kl_coef,
        increment_rollout_step,
    )

    rollout_batch_size = int(getattr(args, "rollout_batch_size", 1) or 1)
    rollout_step = await increment_rollout_step(rollout_batch_size)
    coef, coef_info = compute_rg_kl_coef(delta_r, rollout_step)

    group_memory = ""
    for s in real_samples:
        if (s.metadata or {}).get("source") == "teacher":
            mem = (s.metadata or {}).get("memory") or ""
            if mem:
                group_memory = mem
                break

    if config.enable_rg_kl and config.enable_guided_tokens_for_student:
        for sample_group in group:
            compute_rg_kl_guided_tokens(args, sample_group, group_memory)
    else:
        coef = 0.0

    for idx, sample_group in enumerate(group):
        source = (real_samples[idx].metadata or {}).get("source", "student")
        sample_coef = coef if source == "student" else 0.0
        for sample in sample_group:
            sample.rg_kl_coef = sample_coef
            if sample.metadata is None:
                sample.metadata = {}
            sample.metadata["_rg_kl_coef"] = sample_coef

    # -- 5) Logging --
    avg_raw = sum(raw_rewards) / len(raw_rewards) if raw_rewards else 0.0
    logger.info(
        f"[group_reward] {len(group)} samples raw_avg={avg_raw:.3f} "
        f"student_avg={avg_student:.3f} teacher_avg={avg_teacher:.3f} delta_r={delta_r:+.3f}"
    )
    logger.info(
        f"[rg_kl] group={group_id} rollout_step={rollout_step} "
        f"delta_r={delta_r:+.3f} gate={coef_info['gate']:.3f} "
        f"warmup={coef_info['warmup_factor']:.3f} cos={coef_info['cos_factor']:.3f} "
        f"→ coef={coef:.4f}"
    )

    cleanup_group_coordination(group_id)


def reward_post_process(args: Namespace, samples: list[Sample] | list[list[Sample]]):
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    return raw_rewards, raw_rewards

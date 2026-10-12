"""
Two-stage group rollout with RTI self-correction for claw-eval.

Stage 1: Normal rollout (inspector disabled)
Stage 2: If ALL samples failed, re-run with inspector enabled (self-correct),
         then randomly replace some failed trajectories with successful ones.

Activated via config:
  custom-group-rollout-function-path: qqr.examples.claw_eval.group_rollout.generate_and_rm_group_with_rti
"""

import asyncio
import json
import logging
import os
import random
from argparse import Namespace
from copy import deepcopy
from typing import Any

from slime.rollout.rm_hub import batched_async_rm

from qqr.examples.claw_eval.eval_grader_adapter import format_grader_criteria_for_prompt

logger = logging.getLogger(__name__)


def build_score_summary(s, *, is_group_champion: bool = False) -> dict:
    """Assemble a self-explanatory score block for the inspector dumps.

    Disambiguates the score-ish numbers a dumped sample can carry:
    the official absolute grader score (0-1), the raw score also kept under
    metadata._rti_rank_score (used for in-group champion selection), and the
    RL training reward written to sample.reward.
    """
    md = getattr(s, "metadata", None) or {}
    abs_score = md.get("rti_abs_score")
    dims = md.get("rti_abs_dims") or {}
    summary = {
        "final_score": abs_score,
        "final_score_note": (
            "官方 claw-eval grader 的绝对质量分(0-1)；维度分见 metadata.rti_abs_dims，"
            "评分标准见 metadata.rti_grading_criteria"
            if abs_score is not None
            else "无绝对分：该轨迹未被官方 grader 成功评分"
        ),
        "is_group_champion": bool(is_group_champion),
        "rti_rank_score": md.get("_rti_rank_score"),
        "rti_rank_note": "官方 grader 原始分（组内选出 champion 的排序依据）",
        "training_reward": getattr(s, "reward", None),
        "training_reward_note": "官方 grader 原始分写入 sample.reward(0-1)；GRPO 组内 z-score 归一化由 correct_group_norm 完成",
    }
    if abs_score is not None and dims:
        summary["final_score_breakdown"] = (
            f"{abs_score} = safety({dims.get('safety')}) × "
            f"(0.80 × completion({dims.get('completion')}) + "
            f"0.20 × robustness({dims.get('robustness')}))"
        )
    return summary


def _write_log(args: Namespace, group: list, rollout_id=None) -> None:
    """Persist Stage-2 (calibration) trajectories to ``inspector_log_dir``.

    Mirrors qqr.off_policy.group_rollout._write_log (the tau2-bench mechanism):
    each sample is dumped to
        {inspector_log_dir}/{rollout_id}/{group_index}/{index}.json
    with fields {group_index, index, reward, score_summary, prompt, label,
    messages, metadata}. ``score_summary`` disambiguates the absolute grader
    score / rank score / training reward; ``metadata`` carries the
    per-step ``inspector_trace`` recorded during calibration, so the full
    self-correct process is inspectable offline.
    No-op when ``inspector_log_dir`` is not configured.
    """
    base_dir = getattr(args, "inspector_log_dir", None)
    if not base_dir:
        return

    def unwrap(x):
        return x[-1] if isinstance(x, list) else x

    champ_idx = _champion_index(group) if group else None
    created = set()
    for i, item in enumerate(group):
        s = unwrap(item)
        subdir = f"{getattr(s, 'group_index', 0)}"
        if rollout_id is not None:
            subdir = f"{rollout_id}/{subdir}"
        log_dir = os.path.join(base_dir, subdir)
        if log_dir not in created:
            os.makedirs(log_dir, exist_ok=True)
            created.add(log_dir)
        log_file = os.path.join(log_dir, f"{getattr(s, 'index', 0)}.json")
        tmp_file = log_file + ".tmp"
        _md = getattr(s, "metadata", None) or {}
        _reward = getattr(s, "reward", None)
        if _reward is None:
            # A re-graded Stage-2 sample may have no reward yet; fall back to the
            # absolute grader score recorded by the RTI gate
            # (grade_trajectory_score -> metadata.rti_abs_score).
            _reward = _md.get("rti_abs_score")
        output_sample = {
            "group_index": getattr(s, "group_index", None),
            "index": getattr(s, "index", None),
            "reward": _reward,
            "score_summary": build_score_summary(
                s, is_group_champion=(i == champ_idx)
            ),
            "prompt": getattr(s, "prompt", None),
            "label": getattr(s, "label", None),
            "messages": getattr(s, "messages", None),
            "metadata": getattr(s, "metadata", None),
        }
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(output_sample, f, ensure_ascii=False, indent=2, default=str)
            os.replace(tmp_file, log_file)
        except Exception as e:
            logger.warning(f"[claw-rti] failed to write inspector log {log_file}: {e}")


def _clone_group(src_group: list) -> list:
    return deepcopy(src_group)


def _champion_index(group: list) -> int:
    """Index of the group's best trajectory by official grader score.

    Reads metadata._rti_rank_score (set by reward_model.group_reward_by_official_grader);
    falls back to the sample reward when absent.
    """
    best_idx, best = 0, float("-inf")
    for i, traj in enumerate(group):
        s = traj[-1] if isinstance(traj, list) else traj
        md = getattr(s, "metadata", None) or {}
        r = md.get("_rti_rank_score")
        if r is None:
            r = getattr(s, "reward", None)
        try:
            r = float(r)
        except (TypeError, ValueError):
            r = float("-inf")
        if r > best:
            best, best_idx = r, i
    return best_idx


def _read_cached_abs_score(trajectory) -> float | None:
    """Read the official grader score already written by the reward function
    (group_reward_by_official_grader -> metadata.rti_abs_score), avoiding a
    redundant re-grade in the RTI gate.

    Distinguishes a real 0.0 from a grader-failure signature (0.0 + empty dims),
    mirroring grade_trajectory_score which converts that signature to None.
    Returns None when no cached score exists or the cached score is a failure
    signature, so the caller falls back to grade_trajectory_score.
    """
    s = trajectory[-1] if isinstance(trajectory, list) else trajectory
    md = getattr(s, "metadata", None) or {}
    score = md.get("rti_abs_score")
    if score is None:
        return None
    try:
        score = float(score)
    except (TypeError, ValueError):
        return None
    # Grader-failure signature: 0.0 with empty dims. A genuine 0.0 always carries
    # per-dimension scores (safety/completion/robustness).
    if score == 0.0 and not md.get("rti_abs_dims"):
        return None
    return score


def _task_id_from_sample(s) -> str | None:
    md = getattr(s, "metadata", None) or getattr(s, "train_metadata", None) or {}
    return md.get("task_id") or None


def _inject_grading_criteria(label: dict, task_id: str | None) -> dict:
    """Attach compact official grader criteria to the inspector label dict."""
    if task_id:
        label["task_id"] = task_id
        label["grading_criteria"] = format_grader_criteria_for_prompt(task_id)
    return label


def _force_disable_inspector_label(group) -> None:
    """Stage 1: 明确禁用 inspector，但保留原始 label 内容。"""
    for s in group:
        orig_label = getattr(s, "label", None)
        task_id = _task_id_from_sample(s)
        if isinstance(orig_label, dict) and orig_label is not None:
            orig_label["inspector"] = False
            s.label = _inject_grading_criteria(orig_label, task_id)
        else:
            s.label = _inject_grading_criteria(
                {"inspector": False, "groundtruth": orig_label}, task_id
            )


def _force_enable_inspector_label(group) -> None:
    """Stage 2: 启用 inspector，同时保留 groundtruth。"""
    for s in group:
        orig_label = getattr(s, "label", None)
        task_id = _task_id_from_sample(s)
        if isinstance(orig_label, dict) and orig_label is not None:
            orig_label["inspector"] = True
            s.label = _inject_grading_criteria(orig_label, task_id)
        else:
            s.label = _inject_grading_criteria(
                {"inspector": True, "groundtruth": orig_label}, task_id
            )


async def _run_claw_group(
    args: Namespace,
    group: list,
    sampling_params: dict[str, Any],
    evaluation: bool,
    inspector_enabled: bool = False,
) -> list:
    """Run rollout for each sample in the group."""
    from qqr.examples.claw_eval.rollout import generate
    from qqr.rollout.agent_rollout import GenerateState

    # 沿用 deterministic seed 逻辑
    use_det_seed = getattr(args, "sglang_enable_deterministic_inference", False)
    state = None
    if use_det_seed:
        state = GenerateState(args)

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy() if use_det_seed else sampling_params
        if use_det_seed and state is not None:
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(
            asyncio.create_task(
                generate(args, sample, current_sampling_params,
                         evaluation=evaluation,
                         inspector_enabled=inspector_enabled)
            )
        )
    return list(await asyncio.gather(*tasks))


async def generate_and_rm_group_with_rti(
    args: Namespace,
    group: list,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> list:
    """
    Two-stage group rollout for claw-eval with RTI self-correction.

    Stage 1: Normal rollout (no inspector)
    Stage 2: If the WHOLE group failed to solve the task (absolute success signal,
             NOT the relative within-group reward), re-run with inspector and
             randomly replace failed trajectories with self-corrected successes.

    Faithful to the RTI design: the gate uses an *absolute* per-trajectory
    success signal (official claw-eval grader score) rather than any post-
    normalization within-group reward, which is relative (zero-mean advantage)
    and cannot express "nobody solved it". The group reward (batched_async_rm ->
    custom-rm-path) is therefore deferred and run EXACTLY ONCE on the final
    group, i.e. AFTER any RTI intervention, so advantages are computed over the
    trajectories we keep.
    """
    from qqr.examples.claw_eval import config
    from qqr.examples.claw_eval.reward_model import grade_trajectory_score, dump_group_trajectory

    rti_enabled = config.RTI_ENABLED
    replace_num = config.RTI_REPLACE_NUM
    threshold = config.RTI_SUCCESS_THRESHOLD

    async def _finalize(final_group: list) -> list:
        """Run the group reward on the group (sets training rewards AND records
        each trajectory's rank score in metadata._rti_rank_score, which
        _champion_index reads)."""
        if args.group_rm:
            await batched_async_rm(args, final_group)
        return final_group

    rollout_id = getattr(args, "_current_rollout_id", 0)

    def _dump_champion(grp, champ_idx, champ_score, *, stage1_score=None, stage2_score=None, replaced=False):
        """Append the champion snapshot (official abs score + before/after RTI scores +
        inspector flags) to the train_step dump. Best-effort: never break the rollout."""
        try:
            champ = grp[champ_idx][-1] if isinstance(grp[champ_idx], list) else grp[champ_idx]
            q = getattr(champ, "prompt", "")
            dump_group_trajectory(rollout_id, grp, q if isinstance(q, str) else "",
                                  champion_idx=champ_idx, champion_score=champ_score,
                                  stage1_score=stage1_score, stage2_score=stage2_score,
                                  replaced=replaced)
        except Exception as e:
            logger.warning(f"[claw-rti] champion dump failed: {e}")

    # Clone for potential Stage 2
    original_group = _clone_group(group)

    # Stage 1: Normal rollout (inspector disabled)
    _force_disable_inspector_label(group)
    logger.info(f"[claw-rti] Stage 1: normal rollout (group size={len(group)})")
    group = await _run_claw_group(args, group, sampling_params, evaluation, inspector_enabled=False)

    if evaluation or not rti_enabled or replace_num == 0:
        return await _finalize(group)

    # ── Gate: group reward FIRST, then read the champion's official score from cache ──
    # Officially grading every trajectory (blocking multi-dimension judge) is the
    # rollout-time bottleneck. The group reward (batched_async_rm -> custom-rm-path)
    # already computes each trajectory's official grader score (cached in
    # metadata.rti_abs_score) and ranks the group, so the champion's score is read
    # from that cache below. This reward run also sets the Stage-1 training rewards;
    # they stay final unless a Stage-2 success replaces a slot below.
    await _finalize(group)
    champ1 = _champion_index(group)
    # Prefer the official grader score already written by the reward function
    # (group_reward_by_official_grader -> metadata.rti_abs_score) during _finalize,
    # avoiding a redundant re-grade. Fall back to grade_trajectory_score when no
    # absolute score is cached (e.g. a group reward that only ranks trajectories).
    champ1_score = _read_cached_abs_score(group[champ1])
    if champ1_score is None:
        champ1_score = await grade_trajectory_score(args, group[champ1])
    if champ1_score is None:
        # Grader is broken for this task (returns None instead of a real score).
        # Running Stage 2 would be pointless: with no baseline we can never
        # measure an improvement, so don't burn a whole corrected rollout.
        logger.warning(
            f"[claw-rti] Stage 1 champion idx={champ1} could not be officially graded "
            f"(grader failure); skipping Stage 2"
        )
        _dump_champion(group, champ1, None)
        return group
    logger.info(
        f"[claw-rti] Stage 1 champion idx={champ1} official score={champ1_score:.3f} "
        f"(threshold={threshold})"
    )
    if champ1_score >= threshold:
        logger.info("[claw-rti] Stage 1 champion passed the official grader; skipping Stage 2")
        _dump_champion(group, champ1, champ1_score, stage1_score=champ1_score)
        return group

    # Stage 2: champion failed → re-run with inspector (self-correct)
    logger.info("[claw-rti] Stage 2: champion failed, running RTI self-correction")
    _force_enable_inspector_label(original_group)
    g2 = await _run_claw_group(args, original_group, sampling_params, evaluation, inspector_enabled=True)

    # Tournament on the corrected group, then officially grade ONLY its champion.
    await _finalize(g2)
    champ2 = _champion_index(g2)
    champ2_score = _read_cached_abs_score(g2[champ2])
    if champ2_score is None:
        champ2_score = await grade_trajectory_score(args, g2[champ2])
    # Replacement rule: accept the correction only on a MEANINGFUL improvement over the
    # Stage-1 champion — the margin must exceed 0.1, OR the corrected champion reaches a
    # perfect 1.0. A None Stage-2 score (grader failure) never replaces.
    if champ2_score is None:
        margin, improved = None, False
        logger.warning(
            f"[claw-rti] Stage 2 champion idx={champ2} could not be officially graded "
            f"(grader failure) -> keep Stage 1 results"
        )
    else:
        margin = champ2_score - champ1_score
        improved = margin > 0.1 or champ2_score >= 1.0
        logger.info(
            f"[claw-rti] Stage 2 champion idx={champ2} official score={champ2_score:.3f} "
            f"vs Stage-1 champion {champ1_score:.3f} (margin={margin:+.3f}) -> "
            f"{'REPLACE' if improved else 'keep'}"
        )

    # Tag the graded champion + persist calibration traces (inspector_log_dir).
    _s = g2[champ2][-1] if isinstance(g2[champ2], list) else g2[champ2]
    _md = getattr(_s, "metadata", None)
    if _md is None:
        _md = {}
        _s.metadata = _md
    _md["rti_abs_success"] = bool(champ2_score is not None and champ2_score >= threshold)  # cleared the eval bar (logging only)
    _md["rti_improved"] = bool(improved)                      # better than the Stage-1 champion
    _write_log(args, g2, rollout_id=getattr(args, "_current_rollout_id", None))

    if not improved:
        logger.info("[claw-rti] Stage 2 improvement below margin (>0.1) and not perfect; keeping Stage 1 results")
        _dump_champion(group, champ1, champ1_score, stage1_score=champ1_score,
                       stage2_score=champ2_score, replaced=False)
        return group

    # Inject the improved corrected champion into the group (index-aligned: g2 is a
    # re-run of the same prompts), then re-run the group reward ONCE on the mixed
    # group to produce the final training rewards.
    group[champ2] = g2[champ2]
    logger.info(
        f"[claw-rti] Replaced slot {champ2} with the improved RTI-corrected trajectory "
        f"({champ1_score:.3f} -> {champ2_score:.3f})"
    )
    group = await _finalize(group)
    _dump_champion(group, champ2, champ2_score, stage1_score=champ1_score,
                   stage2_score=champ2_score, replaced=True)
    return group

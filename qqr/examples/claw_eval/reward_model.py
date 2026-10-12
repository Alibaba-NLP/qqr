"""
Reward model for claw-eval RL training (RTI path only).

Provides the official claw-eval grader reward used as the GRPO training signal
(``group_reward_by_official_grader``) plus the step-level grading / group-success
helpers consumed by the claw group rollout (``grade_trajectory_score``,
``compute_group_success_flags``, ``dump_group_trajectory``) and the per-group
z-score post-processor (``correct_group_norm``).
"""

import asyncio
import json
import logging
import os
from argparse import Namespace
from collections import defaultdict

import torch

from qqr.schemas import Sample

from . import config

logger = logging.getLogger(__name__)


# ── Eval reward (for evaluation rollout) ─────────────────────────

def eval_dump_sample(rollout_id, prediction, reference, query, result, judge_result=None, task_id=None, **kwargs):
    file_name = os.environ.get("TASK_NAME", "claw_eval")
    base_dir = os.path.join(str(config.rollout_dump_root), file_name)
    os.makedirs(base_dir, exist_ok=True)
    default_output = os.path.join(base_dir, f"step_{rollout_id}.jsonl")
    output_file = kwargs.get("output_dir", default_output)

    with open(output_file, "a") as f:
        dump = dict(
            task_id=task_id,
            result=result,
            judge_result=judge_result,
            query=query,
            prediction=prediction,
            label=reference,
        )
        f.write(json.dumps(dump, ensure_ascii=False) + "\n")


async def eval_reward(args: Namespace, sample: Sample, **kwargs):
    """Compute eval reward using the claw-eval native grader (one grade per trajectory).

    pass@k / pass^k are aggregated ACROSS the ``n_samples_per_eval_prompt``
    independent trajectories of the same task inside eval_rollout.py, so each
    trajectory only needs a single official grade here.
    """
    # Propagate train env vars (Ray workers don't inherit parent env)
    env_vars = getattr(args, "train_env_vars", None)
    if env_vars:
        if isinstance(env_vars, str):
            env_vars = json.loads(env_vars)
        if isinstance(env_vars, dict):
            for k, v in env_vars.items():
                if k not in os.environ:
                    os.environ[k] = str(v)

    metadata = getattr(sample, "metadata", None) or {}
    task_id = metadata.get("task_id")

    if isinstance(sample.prompt, str):
        query = sample.prompt
    else:
        query = sample.prompt[-1]["content"]
        # Normalize: if content is a char-split list, merge into string
        if isinstance(query, list):
            if all(isinstance(item, dict) and item.get("type") == "text" for item in query):
                query = "".join(item.get("text", "") for item in query)
            elif all(isinstance(item, str) for item in query):
                query = "".join(query)

    from .eval_grader_adapter import grade_sample, get_grader_executor, _get_official_judge
    loop = asyncio.get_running_loop()
    executor = get_grader_executor()
    # Eval uses the OFFICIAL benchmark judge (config_general.yaml), NOT the
    # env-configured gate judge, so Score/Pass@k/Pass^k stay comparable.
    official_judge = _get_official_judge()

    try:
        score, result = await loop.run_in_executor(
            executor, grade_sample, sample, official_judge
        )
        score = float(score)
    except Exception as e:
        logger.warning(f"[eval_reward] official grade failed for {task_id}: {e}, fallback to 0.0")
        score, result = 0.0, {}

    sample.reward = score

    # A trajectory counts as passed when its official score clears the claw-eval
    # success threshold; eval_rollout.py turns these per-trajectory flags into
    # task-level pass@k / pass^k (k = 1..n_samples_per_eval_prompt).
    threshold = 0.75
    passed = score >= threshold

    logger.info(f"[eval_reward] final for {task_id}: score={score:.4f}, "
                f"passed={passed} (threshold={threshold})")

    judge_result = {
        "score": score,
        "passed": passed,
        "pass_threshold": threshold,
        "result": result,
    }

    # Attach the pass flag to sample.metadata for upstream aggregation.
    if metadata is None:
        metadata = {}
        sample.metadata = metadata
    metadata["passed"] = passed
    metadata["pass_threshold"] = threshold

    # NOTE: the current framework threads the rollout id onto args as
    # ``_current_rollout_id`` (see qqr/rollout/agent_rollout.py). The migrated
    # source used the older name ``cur_rollout_id``; use getattr for safety.
    eval_dump_sample(getattr(args, "_current_rollout_id", 0), sample.messages, sample.label, query, sample.reward, judge_result=judge_result, task_id=task_id)


# ── Group reward (for training rollout) ──────────────────────────


def dump_group_trajectory(rollout_id, group: list[list[Sample]], query: str = "",
                          champion_idx=None, champion_score=None,
                          stage1_score=None, stage2_score=None, replaced=False):
    """Append a one-line-per-group snapshot of the CHAMPION trajectory for offline
    inspection: the official absolute score plus which slots were RTI-intervened.

    Called once per group by group_rollout AFTER the final RTI decision. Fields:
      - champion_index / champion_abs_score : the in-group champion (best official
        grader score) and its official
        claw-eval grader score in [0, 1] (metadata.rti_abs_score).
      - inspector : whether the champion itself is a RTI-intervened trajectory
        (label.inspector), mirroring the tau2-bench convention.
      - group_intervened_indices : all slots in the group with label.inspector True.
    """
    file_name = os.environ.get("TASK_NAME", "claw_eval")
    base_dir = os.path.join(str(config.rollout_dump_root), file_name)
    os.makedirs(base_dir, exist_ok=True)
    output_file = os.path.join(base_dir, f"train_step_{rollout_id}.jsonl")

    def _last(t):
        return t[-1] if isinstance(t, list) else t

    # Resolve the champion: prefer the explicit index, else the best rank score.
    if champion_idx is None:
        best, champion_idx = float("-inf"), 0
        for i, t in enumerate(group):
            r = (getattr(_last(t), "metadata", None) or {}).get("_rti_rank_score")
            r = r if isinstance(r, (int, float)) else float("-inf")
            if r > best:
                best, champion_idx = r, i

    champ = _last(group[champion_idx])
    metadata = getattr(champ, "metadata", None) or {}
    label = getattr(champ, "label", None) or {}
    if champion_score is None:
        champion_score = metadata.get("rti_abs_score")

    intervened = []
    for i, t in enumerate(group):
        lab = getattr(_last(t), "label", None) or {}
        if isinstance(lab, dict) and lab.get("inspector"):
            intervened.append(i)

    dump = dict(
        task_id=metadata.get("task_id"),
        query=query,
        champion_index=champion_idx,
        champion_abs_score=champion_score,  # official claw-eval grader score in [0, 1]
        # Before/after RTI comparison (official grader scores):
        stage1_champion_score=stage1_score,   # Stage-1 (pre-correction) champion score
        stage2_champion_score=stage2_score,   # Stage-2 (corrected) champion score, if run
        improvement_margin=(round(stage2_score - stage1_score, 4)
                            if (stage1_score is not None and stage2_score is not None) else None),
        replaced=bool(replaced),              # was the Stage-2 correction injected?
        inspector=bool(label.get("inspector")) if isinstance(label, dict) else False,
        group_intervened_indices=intervened,
        rewards=getattr(champ, "reward", None),
        messages=champ.messages,
        metadata={k: v for k, v in metadata.items()
                  if k not in ("screenshots", "output_artifacts")},
    )

    with open(output_file, "a") as f:
        f.write(json.dumps(dump, ensure_ascii=False, default=str) + "\n")


async def compute_group_success_flags(args: Namespace, group: list) -> list[bool]:
    """Absolute per-trajectory success for the RTI gate — aligned with the
    official claw-eval metric.

    Each trajectory is scored by the *official* claw-eval grader
    (eval_grader_adapter.grade_sample — the same grader behind the eval
    Score/Pass@3/Pass^3), and counts as success iff its score >= the eval pass
    threshold (config.RTI_SUCCESS_THRESHOLD, set to 0.75 to mirror eval_reward).
    This keeps the "did the whole group fail" gate consistent with how the
    benchmark actually judges task completion.

    Used by group_rollout.generate_and_rm_group_with_rti to decide whether to
    run Stage-2 self-correction. Deliberately absolute (independent of the
    z-normalized GRPO advantages). Any grading error is treated as
    "not solved" (False) so the RTI fallback can still fire.
    """
    # Propagate train_env_vars (Ray workers don't inherit parent env), so the
    # official grader's LLM judge has its API key / base url available.
    env_vars = getattr(args, "train_env_vars", None)
    if env_vars:
        if isinstance(env_vars, str):
            try:
                env_vars = json.loads(env_vars)
            except (json.JSONDecodeError, TypeError):
                env_vars = None
        if isinstance(env_vars, dict):
            for k, v in env_vars.items():
                if k not in os.environ:
                    os.environ[k] = str(v)

    # Pass threshold aligned with the eval metric (eval_reward uses 0.75).
    threshold = config.RTI_SUCCESS_THRESHOLD

    from .eval_grader_adapter import (
        grade_sample, get_grader_executor, get_grader_criteria,
    )

    def _last(traj):
        return traj[-1] if isinstance(traj, list) else traj

    loop = asyncio.get_running_loop()
    executor = get_grader_executor()

    async def _one(trajectory):
        s = _last(trajectory)
        try:
            score, dims = await loop.run_in_executor(
                executor, grade_sample, s
            )
            score = float(score)
        except Exception as e:
            logger.warning(f"[claw-rti] official grade_sample failed: {e}")
            score, dims = 0.0, {}
        # Record the absolute grader score on the sample so the RTI gate and the
        # inspector log can surface it (a Stage-2 calibration sample may not have
        # been graded elsewhere, so its reward would otherwise stay None).
        md = getattr(s, "metadata", None)
        if md is None:
            md = {}
            s.metadata = md
        md["rti_abs_score"] = score
        md["rti_abs_dims"] = dims
        md["rti_grading_criteria"] = get_grader_criteria(md.get("task_id"))
        return score >= threshold

    return list(await asyncio.gather(*[_one(t) for t in group]))


async def grade_trajectory_score(args: Namespace, trajectory) -> float | None:
    """Official-grader absolute score in [0, 1] for a SINGLE trajectory.

    Used by the RTI gate in group_rollout as a fallback: the group reward
    already caches each trajectory's official score in metadata.rti_abs_score,
    so the gate reads that first and only calls this for trajectories without a
    cached score. Records the score on the sample's metadata (rti_abs_score)
    and returns it. Any grading error yields None so the caller can tell
    "grader broken" apart from a genuine 0.0 (a 0.0 fallback used to both
    mis-trigger Stage 2 and lock the replacement margin at 0 for the whole run).
    """
    # Propagate train_env_vars (Ray workers don't inherit parent env).
    env_vars = getattr(args, "train_env_vars", None)
    if env_vars:
        if isinstance(env_vars, str):
            try:
                env_vars = json.loads(env_vars)
            except (json.JSONDecodeError, TypeError):
                env_vars = None
        if isinstance(env_vars, dict):
            for k, v in env_vars.items():
                if k not in os.environ:
                    os.environ[k] = str(v)

    from .eval_grader_adapter import (
        grade_sample, get_grader_executor, get_grader_criteria,
    )

    s = trajectory[-1] if isinstance(trajectory, list) else trajectory
    loop = asyncio.get_running_loop()
    try:
        score, dims = await loop.run_in_executor(get_grader_executor(), grade_sample, s)
        score = float(score)
        # grade_sample itself swallows grader exceptions into (0.0, {}); an
        # empty dims dict with a 0.0 score is that failure signature, not a
        # real grade (a genuine zero always carries per-dimension scores).
        if score == 0.0 and not dims:
            logger.warning("[claw-rti] official grade_sample (champion) returned failure signature (0.0, {})")
            score, dims = None, {}
    except Exception as e:
        logger.warning(f"[claw-rti] official grade_sample (champion) failed: {e}")
        score, dims = None, {}
    md = getattr(s, "metadata", None)
    if md is None:
        md = {}
        s.metadata = md
    md["rti_abs_score"] = score
    md["rti_abs_dims"] = dims
    md["rti_grading_criteria"] = get_grader_criteria(md.get("task_id"))
    return score


async def group_reward_by_official_grader(args: Namespace, group: list[list[Sample]], **kwargs):
    """Pure-GRPO group reward: per-trajectory official claw-eval grader score.

    Rewards each trajectory with the *same* official claw-eval grader that
    powers the evaluation metric (Score / Pass@k / Pass^k):
    ``eval_grader_adapter.grade_sample`` -> task ``grader.py`` ->
    ``compute_task_score``.  The raw [0, 1] scores are written to
    ``sample.reward`` and z-score normalized per group downstream
    (``correct_group_norm``) to produce GRPO advantages.

    Judge selection (the LLM inside the grader):
      - By default the *gate* judge (``_get_judge``) is used, which honours the
        ``GRADER_JUDGE_*`` env vars (fast local vLLM) configured in the yaml.
        This keeps the high-frequency training reward cheap.
      - Set env ``GRPO_REWARD_USE_OFFICIAL_JUDGE=true`` to use the *official*
        benchmark judge (``_get_official_judge``, DashScope qwen3.7-max) so the
        training reward is computed with exactly the same judge as eval. Slower
        and rate-limited; useful for ablations.

    RTI / inspector are NOT involved here -- this is pure GRPO
    where the reward signal is the official grader score. Pair it with
    ``custom-group-rollout-function-path`` set to the claw group rollout with
    ``RTI_ENABLED=false`` (or unset), which short-circuits to a plain
    generate -> finalize group rollout.
    """
    # Propagate train_env_vars (Ray workers don't inherit parent env)
    env_vars = getattr(args, "train_env_vars", None)
    if env_vars:
        if isinstance(env_vars, str):
            try:
                env_vars = json.loads(env_vars)
            except (json.JSONDecodeError, TypeError):
                env_vars = None
        if isinstance(env_vars, dict):
            for k, v in env_vars.items():
                if k not in os.environ:
                    os.environ[k] = str(v)

    if len(group) <= 1:
        raise ValueError("group size must be greater than 1")

    from .eval_grader_adapter import (
        grade_sample, get_grader_executor, get_grader_criteria,
    )

    use_official = os.getenv("GRPO_REWARD_USE_OFFICIAL_JUDGE", "false").lower() == "true"
    if use_official:
        from .eval_grader_adapter import _get_official_judge
        judge = _get_official_judge()
        logger.info("[grpo-grader] using OFFICIAL benchmark judge (DashScope)")
    else:
        from .eval_grader_adapter import _get_judge
        judge = _get_judge()
        logger.info("[grpo-grader] using gate judge (GRADER_JUDGE_* env)")

    loop = asyncio.get_running_loop()
    executor = get_grader_executor()

    def _last(t):
        return t[-1] if isinstance(t, list) else t

    async def _grade_one(trajectory):
        s = _last(trajectory)
        task_id = (getattr(s, "metadata", None) or {}).get("task_id")
        try:
            score, dims = await loop.run_in_executor(
                executor, grade_sample, s, judge
            )
            score = float(score)
        except Exception as e:
            logger.warning(f"[grpo-grader] grade_sample failed for {task_id}: {e}")
            score, dims = 0.0, {}
        md = getattr(s, "metadata", None)
        if md is None:
            md = {}
            s.metadata = md
        # Use the SAME field names as grade_trajectory_score / compute_group_success_flags
        # (rti_abs_score / rti_abs_dims / rti_grading_criteria) so downstream readers
        # — build_score_summary, dump_group_trajectory, and the RTI gate in
        # group_rollout (_read_cached_abs_score) — can read the official absolute
        # score without a redundant re-grade.
        md["rti_abs_score"] = score
        md["rti_abs_dims"] = dims
        md["rti_grading_criteria"] = get_grader_criteria(task_id)
        return score

    raw_scores = list(await asyncio.gather(*[_grade_one(t) for t in group]))

    # Record the absolute grader score on every sample so downstream tooling
    # (champion selection / dumps) can read it just like _rti_rank_score.
    for idx, trajectory in enumerate(group):
        for sample in trajectory:
            md = getattr(sample, "metadata", None)
            if md is None:
                md = {}
                sample.metadata = md
            md["_rti_rank_score"] = float(raw_scores[idx])

    # Write the RAW [0, 1] official grader score as sample.reward — NO in-place
    # z-score normalization here. The rollout dump (iter_*.jsonl via
    # save_train_samples -> Sample.to_dict) therefore shows the human-
    # interpretable 0-1 grader score, and GRPO advantage normalization is
    # delegated to `correct_group_norm` (set as custom-reward-post-process-path
    # in ALL configs using this reward). correct_group_norm groups step-level
    # samples by `group_index`, deduplicates to per-trajectory rewards (by
    # `index`), computes per-group mean/std across trajectory rewards, and
    # broadcasts the z-score advantage back to every step-level sample.
    # This replaces slime's default _post_process_rewards which has a reshape
    # bug: it checks `rewards.shape[-1] == n_samples_per_prompt * rollout_batch_size`
    # but multi-step agent trajectories produce N >> 64 samples, so the check
    # fails and falls back to global (1, N) normalization — destroying the
    # group-relative signal that GRPO relies on.
    # Also keep normalize-advantages=false so the 2nd-stage global whitening is
    # skipped (standard GRPO only uses group-relative).
    for idx, trajectory in enumerate(group):
        score = float(raw_scores[idx])
        for sample in trajectory:
            sample.reward = score

    logger.info(
        f"[grpo-grader] raw_scores={[round(s, 4) for s in raw_scores]} "
        f"mean={float(sum(raw_scores) / len(raw_scores)):.4f} "
        f"(NO in-place norm; delegated to slime group-relative + normalize-advantages)"
    )

    # Return empty list so the upstream zip in generate_and_rm_group is a no-op
    # (rewards are already set in-place on each sample above).
    return []


def correct_group_norm(args: Namespace, samples: list[Sample] | list[list[Sample]]):
    """Per-group z-score normalization that works correctly for multi-step
    agent trajectories.

    Slime's default ``_post_process_rewards`` assumes a fixed sample count equal
    to ``n_samples_per_prompt * rollout_batch_size``.  For multi-step agent
    rollouts, the actual sample count is far larger (one Sample per agent step
    per trajectory), so the reshape check fails and falls back to global
    ``(1, N)`` normalization — destroying the group-relative signal that GRPO
    relies on.

    This function fixes the issue by:
      1. Flattening any nested sample lists.
      2. Grouping step-level samples by ``group_index`` (the prompt index).
      3. Within each group, extracting the per-trajectory reward (all
         step-level samples of a trajectory share the same reward, so we
         take one representative per ``index``).
      4. Computing mean / std across the trajectory-level rewards.
      5. Broadcasting the z-score advantage back to *every* step-level
         sample of the corresponding trajectory.

    Returns ``(raw_rewards, normalized_rewards)`` where both lists have
    the same length as the (flattened) *samples* list.
    """
    # 1. Flatten nested lists
    flat_samples: list[Sample] = []
    for s in samples:
        if isinstance(s, list):
            flat_samples.extend(s)
        else:
            flat_samples.append(s)
    samples = flat_samples

    # 2. Extract raw rewards
    raw_rewards = [s.get_reward_value(args) for s in samples]

    # 3. Check whether normalization should be applied
    should_normalize = (
        getattr(args, "advantage_estimator", None)
        in ("grpo", "gspo", "reinforce_plus_plus_baseline")
        and getattr(args, "rewards_normalization", True)
    )
    if not should_normalize:
        return raw_rewards, raw_rewards

    use_std = (
        getattr(args, "advantage_estimator", None) in ("grpo", "gspo")
        and getattr(args, "grpo_std_normalization", True)
    )

    # 4. Group by group_index, collect per-trajectory rewards
    groups: dict = defaultdict(list)  # group_index -> [(sample_idx, traj_idx, reward)]
    for i, s in enumerate(samples):
        gi = getattr(s, "group_index", None)
        ti = getattr(s, "index", None)
        groups[gi].append((i, ti, raw_rewards[i]))

    # 5. Compute per-group z-score and broadcast to all step-level samples
    rewards = [0.0] * len(samples)
    for gi, items in groups.items():
        # Deduplicate to one reward per trajectory (by index)
        traj_rewards: dict = {}
        for _sample_idx, traj_idx, reward in items:
            if traj_idx not in traj_rewards:
                traj_rewards[traj_idx] = reward

        values = list(traj_rewards.values())
        n = len(values)
        if n <= 1:
            # Single trajectory in group — no group-relative signal
            for sample_idx, _traj_idx, _ in items:
                rewards[sample_idx] = 0.0
        else:
            traj_tensor = torch.tensor(values, dtype=torch.float)
            mean = traj_tensor.mean().item()
            if use_std:
                # torch.std uses Bessel's correction (n-1) by default,
                # matching the original slime implementation
                std = traj_tensor.std().item()
            for sample_idx, traj_idx, _ in items:
                if use_std and std > 1e-6:
                    rewards[sample_idx] = (traj_rewards[traj_idx] - mean) / (std + 1e-6)
                else:
                    rewards[sample_idx] = traj_rewards[traj_idx] - mean

    logger.info(
        f"[correct_group_norm] {len(samples)} samples, {len(groups)} groups, "
        f"use_std={use_std}, norm Applied"
    )
    return raw_rewards, rewards

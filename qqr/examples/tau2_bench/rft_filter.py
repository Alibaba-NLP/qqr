"""
RFT (Reward-Filtered Training) helpers.

Only trajectories whose reward equals 1 are kept for SFT; all other
trajectories have every turn's loss_mask zeroed out via
``sample.remove_sample = True``.

Wire through ``--rollout-sample-filter-path qqr.examples.tau2_bench.rft_filter.keep_only_reward_one``.

Data layout (qqr agent_rollout, group-rm, multi-turn):
    data:                list[trajectory]
    trajectory:          list[Sample]   (one Sample per agent turn)
    sample.reward:       set on *every* turn by ``group_reward``
                         (see qqr/examples/tau2_bench/reward_model.py),
                         so any turn can be used to read the trajectory reward.

The filter therefore operates at **trajectory granularity**: either all
turns of a trajectory contribute to the SFT loss, or none do. This is
what distinguishes it from a naive per-Sample flattening, which would
wrongly drop the earlier turns of successful trajectories (those turns
have ``reward is None`` at generation time, before ``group_reward`` runs;
after ``group_reward`` they all carry the terminal reward).

Compatible with the RTI two-stage rollout
(``qqr.off_policy.group_rollout.generate_and_rm_group_with_calibration``):
after Stage 2 replaces failed Stage-1 samples with calibrated ones,
``sample.reward`` already reflects the final trajectory outcome on both
Stage-1 and Stage-2 samples, so this filter transparently keeps all
successful trajectories regardless of which stage produced them.
"""

from argparse import Namespace

from qqr.schemas import Sample


def _get_trajectory_reward(trajectory, args: Namespace):
    """Return the scalar reward of a trajectory, or None if unreadable.

    For qqr multi-turn rollouts every Sample in a trajectory shares the
    same reward after ``group_reward`` runs, so we can read any turn.
    We defensively try the *last* non-padding turn first, then fall back
    to the first readable one.
    """
    if isinstance(trajectory, Sample):
        return trajectory.get_reward_value(args) if trajectory.reward is not None else None

    # trajectory is list[Sample]; try last non-padding turn first
    for s in reversed(trajectory):
        if getattr(s, "index", 0) == -1:
            continue
        if s.reward is not None:
            return s.get_reward_value(args)
    # fallback: first turn that has a reward
    for s in trajectory:
        if s.reward is not None:
            return s.get_reward_value(args)
    return None


def _scalar_reward(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        # reward_key dispatch is handled inside Sample.get_reward_value; if
        # we land here it means the caller already extracted the scalar or
        # the dict is the raw payload. Use the first numeric value.
        for v in value.values():
            if isinstance(v, (int, float)):
                return float(v)
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def keep_only_reward_one(args: Namespace, data) -> None:
    """In-place filter: drop trajectories whose reward is not 1.

    Under ``qqr.rollout.agent_rollout.generate_rollout`` with ``group-rm``
    ``data`` is ``list[group]`` where each ``group`` is a list of
    ``n_samples_per_prompt`` trajectories, and each trajectory is itself
    ``list[Sample]`` (one Sample per agent turn, multi-turn) or a single
    ``Sample`` (flat). We therefore flatten groups into trajectories
    before filtering. For flat (non-group-rm) rollouts ``data`` is
    ``list[Sample]`` and each element is treated as a single-turn
    trajectory.

    For each trajectory we read the terminal reward; if it is not >= 1,
    we set ``sample.remove_sample = True`` on **every** turn, which
    causes ``_convert_samples_to_train_data`` to zero its loss mask so
    it contributes nothing to the SFT loss while preserving batch shape.
    """
    if not data:
        return

    # Flatten groups into trajectories. ``data`` is list[group]; each group
    # is list[trajectory]; each trajectory is list[Sample] or Sample.
    trajectories = []
    for group in data:
        if isinstance(group, list):
            trajectories.extend(group)
        else:
            # flat rollout without group-rm: data is list[Sample]
            trajectories.append(group)

    kept_traj = 0
    total_traj = 0
    kept_samples = 0
    total_samples = 0

    for trajectory in trajectories:
        total_traj += 1
        reward_value = _get_trajectory_reward(trajectory, args)
        reward = _scalar_reward(reward_value)
        is_positive = (reward is not None) and (reward >= 1.0 - 1e-9)
        if is_positive:
            kept_traj += 1

        turns = trajectory if isinstance(trajectory, list) else [trajectory]
        for s in turns:
            total_samples += 1
            s.remove_sample = not is_positive
            if is_positive:
                kept_samples += 1

    print(
        f"[RFT] kept {kept_traj}/{total_traj} trajectories "
        f"({kept_samples}/{total_samples} samples) with reward == 1",
        flush=True,
    )

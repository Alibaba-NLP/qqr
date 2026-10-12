"""claw-eval eval rollout logging for qqr.

Computes claw-eval's Table-3 metrics (Score / Pass@1..k / Pass^1..k) split into
General (task_id starting with "T"), Multi-turn (task_id starting with "C") and
Overall, then persists them next to the checkpoint the same way tau2-bench does:

  {args.save}/rollout/evaluation/iter_{rollout_id}/metric.json      (this module)
  {args.save}/rollout/evaluation/iter_{rollout_id}/{dataset}.jsonl  (save_eval_samples_jsonl)
  {args.save}/inspector/evaluation/iter_{rollout_id}.json           (inspector traces)

and mirrors the metrics into ``extra_metrics`` so they show up on wandb under
``eval/*``.

Metric source: the claw grader runs inside reward_model.eval_reward, which sets
``sample.reward`` to the official score and records ``sample.metadata.passed``
using the 0.75 success threshold.  We group the trajectories by task and compute
pass@k / pass^k for k = 1..n_samples_per_eval_prompt, matching the standard
"n candidate completions per prompt" semantics: the number of metrics grows
with the configured number of samples per evaluation prompt.

Activated via yaml:
  rollout:
    custom-eval-rollout-log-function-path: qqr.examples.claw_eval.eval_rollout.custom_eval_rollout_log_function

The offline eval_metrics.py script recomputes the same metrics from the
reward_model.eval_dump_sample jsonl dumps (step_*.jsonl).
"""

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 0.75


def _category(task_id: str | None) -> str:
    """Map a task id to a claw report category (mirrors eval_metrics.classify)."""
    tid = task_id or ""
    if tid.startswith("T"):
        return "general"
    if tid.startswith("C"):
        return "multi_turn"
    return "other"


def _per_task_metrics(
    scores: List[float], pass_threshold: float = PASS_THRESHOLD
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return task-level score and per-trajectory pass flags.

    ``scores`` may contain NaNs / missing values for trajectories that could not
    be graded; these are treated as failures.
    """
    valid = [float(s) if s is not None else 0.0 for s in scores]
    score = sum(valid) / len(valid) if valid else 0.0
    passed = [1 if s >= pass_threshold else 0 for s in valid]
    return score, passed


def _compute_pass_at_k(passed: List[int], k_max: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute pass@k and pass^k for k=1..k_max for a single task.

    Uses the unbiased combinatorial estimators (order independent), which
    collapse to the natural values when n == k:
      pass@k = 1 - C(n-c, k) / C(n, k)
      pass^k = C(c, k) / C(n, k)
    """
    n = len(passed)
    c = sum(passed)
    pass_at: Dict[str, float] = {}
    pass_hat: Dict[str, float] = {}

    for k in range(1, k_max + 1):
        if n == 0 or k > n:
            pass_at[f"pass@{k}"] = 0.0
            pass_hat[f"pass^{k}"] = 0.0
            continue

        denom = math.comb(n, k)
        if denom == 0:
            pass_at[f"pass@{k}"] = 0.0
            pass_hat[f"pass^{k}"] = 0.0
            continue

        pass_at[f"pass@{k}"] = 1.0 - math.comb(n - c, k) / denom
        pass_hat[f"pass^{k}"] = math.comb(c, k) / denom

    return pass_at, pass_hat


def _reduce_task_metrics(
    task_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Average per-task metrics over a bucket of tasks.

    Input: list of dicts with ``score`` and ``pass@{k}`` / ``pass^{k}`` for the
    same set of k values.  Output: dict with ``score``, ``pass@{k}``,
    ``pass^{k}``, ``num_tasks``, ``num_trajectories``.
    """
    n_tasks = len(task_metrics)
    if n_tasks == 0:
        return {"score": 0.0, "num_tasks": 0, "num_trajectories": 0}

    # Determine all metric keys (excluding bookkeeping).
    metric_keys = [
        k for k in task_metrics[0].keys()
        if k not in ("task_id", "score", "num_trajectories", "passed")
    ]

    total_score = 0.0
    total_trajectories = 0
    sums = {k: 0.0 for k in metric_keys}

    for tm in task_metrics:
        total_score += tm.get("score", 0.0)
        total_trajectories += tm.get("num_trajectories", 0)
        for k in metric_keys:
            sums[k] += tm.get(k, 0.0)

    reduced: Dict[str, Any] = {
        "score": total_score / n_tasks * 100.0,
        "num_tasks": n_tasks,
        "num_trajectories": total_trajectories,
    }
    for k in metric_keys:
        reduced[k] = sums[k] / n_tasks * 100.0
    return reduced


def _save_metrics(args, rollout_id: int, metrics: Dict[str, Any]):
    if not getattr(args, "save", None):
        return
    out_dir = Path(args.save) / "rollout" / "evaluation" / f"iter_{rollout_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metric.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def custom_eval_rollout_log_function(
    rollout_id: int,
    args,
    data: Dict[str, Any],
    extra_metrics: Dict[str, Any] | None,
) -> bool:
    if extra_metrics is None:
        extra_metrics = {}

    group_size = getattr(args, "n_samples_per_eval_prompt", 1) or 1
    pass_threshold = PASS_THRESHOLD

    # Collect trajectories per (category, task_id).
    # Each Sample here is one independent trajectory for one task.  The number
    # of Samples per task may be less than group_size if the eval batch is not a
    # perfect multiple or if a task only appears once; we use the actual count
    # when computing that task's pass@k and clamp k_max to group_size.
    tasks: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for _dataset_name, info in data.items():
        for s in info.get("samples", []):
            md = getattr(s, "metadata", None) or {}
            task_id = md.get("task_id")
            if task_id is None and hasattr(s, "index"):
                task_id = str(getattr(s, "index"))

            cat = _category(task_id)
            score = float(getattr(s, "reward", 0.0) or 0.0)
            tasks[cat][task_id].append(score)

    k_max = group_size

    # Compute per-task metrics.  k always runs 1..k_max so every task yields
    # the same metric keys; tasks with fewer trajectories than k contribute 0.0
    # for the unreachable k values (handled inside _compute_pass_at_k).
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for cat, task_dict in tasks.items():
        for task_id, scores in task_dict.items():
            task_score, passed_flags = _per_task_metrics(scores, pass_threshold)
            pass_at, pass_hat = _compute_pass_at_k(passed_flags, k_max)

            tm: Dict[str, Any] = {
                "task_id": task_id,
                "score": task_score,
                "num_trajectories": len(scores),
                "passed": passed_flags,
                **pass_at,
                **pass_hat,
            }
            by_category[cat].append(tm)
            # Also mirror into overall.
            by_category["overall"].append(tm)

    # Reduce across tasks.
    reduced: Dict[str, Dict[str, Any]] = {}
    for cat in sorted(by_category.keys()):
        reduced[cat] = _reduce_task_metrics(by_category[cat])

    # Build metric dict.
    overall = reduced.get("overall", {"score": 0.0, "num_tasks": 0, "num_trajectories": 0})
    by_category_out = {cat: m for cat, m in reduced.items() if cat != "overall"}

    metrics = {
        "rollout_id": rollout_id,
        "n_samples_per_eval_prompt": group_size,
        "pass_threshold": pass_threshold,
        "num_tasks": overall.get("num_tasks", 0),
        "num_trajectories": overall.get("num_trajectories", 0),
        "overall": overall,
        "by_category": by_category_out,
    }
    _save_metrics(args, rollout_id, metrics)

    # Mirror to wandb under eval/*.
    for k, v in overall.items():
        if isinstance(v, (int, float)):
            extra_metrics[f"eval/overall/{k}"] = v
    for cat, m in by_category_out.items():
        for k, v in m.items():
            if isinstance(v, (int, float)):
                extra_metrics[f"eval/{cat}/{k}"] = v

    # Eval samples jsonl (shared helper; gated by save_rollout_data).
    from qqr.utils.save_eval import save_eval_samples_jsonl
    save_eval_samples_jsonl(args, rollout_id, data)

    # Inspector traces from eval samples (gated by save_inspector_data).
    from qqr.examples.claw_eval.save_rollout_data import _save_inspector_traces_from_samples
    for dataset_name, info in data.items():
        _save_inspector_traces_from_samples(
            args, rollout_id, info.get("samples", []),
            evaluation=True, dataset_name=dataset_name,
        )

    logger.info(
        "[claw-eval] iter %s eval overall: score=%.2f num_tasks=%d num_traj=%d metrics=%s",
        rollout_id,
        overall.get("score", 0.0),
        overall.get("num_tasks", 0),
        overall.get("num_trajectories", 0),
        {k: round(v, 2) for k, v in overall.items()
         if k not in ("num_tasks", "num_trajectories") and isinstance(v, (int, float))},
    )
    logger.info(
        "[claw-eval] iter %s eval by category: %s",
        rollout_id,
        {c: {k: round(m[k], 2) for k in m if k not in ("num_tasks", "num_trajectories")
             and isinstance(m[k], (int, float))}
         for c, m in sorted(by_category_out.items())},
    )
    # Return False so the framework still flushes ``extra_metrics`` (our eval/*)
    # plus its default per-dataset reward means to wandb. Returning True would
    # early-out before slime's logging_utils.log call and drop the wandb push.
    return False

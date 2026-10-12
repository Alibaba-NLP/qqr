#!/usr/bin/env python3
"""
CLAW Evaluation Metrics Calculator (offline).

Recomputes the Table-3 metrics (Score / Pass@1..k / Pass^1..k) from a
``step_*.jsonl`` dump written by ``reward_model.eval_dump_sample``, using the
same per-task grouping and unbiased combinatorial estimators as
``eval_rollout.py``:

  pass@k = 1 - C(n-c, k) / C(n, k)      (at least one of k passes)
  pass^k = C(c, k) / C(n, k)            (all k pass)

where n is the number of trajectories dumped for a task and c the number of
them with an official score >= threshold (default 0.75).  k ranges from 1 to
the largest per-task trajectory count found in the file.

Metrics are computed separately for General (T-prefix), Multi-turn (C-prefix)
and Overall.  Both the new dump schema (``judge_result.score`` /
``judge_result.passed``) and the legacy one (``judge_result.avg_score``) are
accepted.

Usage:
    python eval_metrics.py /path/to/step_N.jsonl [--model-name NAME] [--threshold 0.75]
"""

import argparse
import json
import math
import os
from collections import defaultdict

PASS_THRESHOLD = 0.75

CATEGORIES = ("general", "multi_turn", "overall")
CATEGORY_LABELS = {"general": "General", "multi_turn": "Multi-turn", "overall": "Overall"}


def load_jsonl(filepath):
    """Load records from a JSONL file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _category(task_id):
    tid = task_id or ""
    if tid.startswith("T"):
        return "general"
    if tid.startswith("C"):
        return "multi_turn"
    return "other"


def extract_score(rec, threshold):
    """Return (score, passed) for one dumped trajectory.

    Prefers the new schema (judge_result.score / judge_result.passed), falls
    back to the legacy avg_score field and finally to the raw ``result``.
    """
    judge = rec.get("judge_result") or {}
    if "score" in judge:
        score = float(judge["score"] or 0.0)
    elif "avg_score" in judge:  # legacy 3-trial dumps
        score = float(judge["avg_score"] or 0.0)
    else:
        score = float(rec.get("result") or 0.0)
    passed = judge.get("passed")
    if passed is None:
        passed = score >= threshold
    return score, bool(passed)


def group_by_task(records, threshold):
    """Group trajectory (score, passed) tuples by category and task_id."""
    groups = {cat: defaultdict(list) for cat in CATEGORIES}
    for rec in records:
        task_id = rec.get("task_id") or ""
        entry = extract_score(rec, threshold)
        cat = _category(task_id)
        if cat in groups:
            groups[cat][task_id].append(entry)
        groups["overall"][task_id].append(entry)
    return groups


def _pass_estimators(n, c, k):
    """Unbiased pass@k / pass^k for n trajectories with c passes."""
    if n == 0 or k > n:
        return 0.0, 0.0
    denom = math.comb(n, k)
    pass_at = 1.0 - math.comb(n - c, k) / denom
    pass_hat = math.comb(c, k) / denom
    return pass_at, pass_hat


def compute_metrics(task_groups, k_max):
    """Average Score / Pass@1..k / Pass^1..k across tasks (values in %)."""
    if not task_groups:
        metrics = {"score": 0.0, "num_tasks": 0, "num_trajectories": 0}
        for k in range(1, k_max + 1):
            metrics[f"pass@{k}"] = 0.0
            metrics[f"pass^{k}"] = 0.0
        return metrics

    score_sum = 0.0
    pass_at_sum = defaultdict(float)
    pass_hat_sum = defaultdict(float)
    num_traj = 0
    for entries in task_groups.values():
        n = len(entries)
        c = sum(1 for _, passed in entries if passed)
        num_traj += n
        score_sum += sum(score for score, _ in entries) / n
        for k in range(1, k_max + 1):
            pass_at, pass_hat = _pass_estimators(n, c, k)
            pass_at_sum[k] += pass_at
            pass_hat_sum[k] += pass_hat

    num_tasks = len(task_groups)
    metrics = {
        "score": score_sum / num_tasks * 100,
        "num_tasks": num_tasks,
        "num_trajectories": num_traj,
    }
    for k in range(1, k_max + 1):
        metrics[f"pass@{k}"] = pass_at_sum[k] / num_tasks * 100
        metrics[f"pass^{k}"] = pass_hat_sum[k] / num_tasks * 100
    return metrics


def print_report(filepath, model_name, records, threshold):
    """Print the evaluation report in Table-3 style."""
    groups = group_by_task(records, threshold)
    k_max = max((len(v) for v in groups["overall"].values()), default=1)

    per_cat = {cat: compute_metrics(groups[cat], k_max) for cat in CATEGORIES}

    print("=" * 72)
    print("                  CLAW Evaluation Results Report")
    print("=" * 72)
    print(f"Model:          {model_name}")
    print(f"Input:          {filepath}")
    print(f"Pass threshold: {threshold}")
    print(f"Trajectories:   {per_cat['overall']['num_trajectories']} "
          f"({k_max} per task max)")
    print("-" * 72)
    print("Task Statistics:")
    for cat in CATEGORIES:
        print(f"  {CATEGORY_LABELS[cat]:<22} {per_cat[cat]['num_tasks']} tasks")
    print("-" * 72)

    header = f"{'Metric':<10}" + "".join(
        f"{CATEGORY_LABELS[cat]:>14}" for cat in CATEGORIES
    )
    print(header)
    print("-" * 72)
    rows = ["score"]
    for k in range(1, k_max + 1):
        rows.append(f"pass@{k}")
        rows.append(f"pass^{k}")
    display = {"score": "Score"}
    for k in range(1, k_max + 1):
        display[f"pass@{k}"] = f"Pass@{k}"
        display[f"pass^{k}"] = f"Pass^{k}"
    for row in rows:
        line = f"{display[row]:<10}" + "".join(
            f"{per_cat[cat][row]:>14.1f}" for cat in CATEGORIES
        )
        print(line)
    print("-" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Compute CLAW evaluation metrics (Table-3 format)"
    )
    parser.add_argument(
        "input_file",
        help="Path to the evaluation results JSONL file (step_N.jsonl dump)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name for display (auto-detected from filename if not provided)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=PASS_THRESHOLD,
        help=f"Pass threshold on the official score (default {PASS_THRESHOLD})",
    )

    args = parser.parse_args()
    filepath = args.input_file

    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return

    if args.model_name:
        model_name = args.model_name
    else:
        # e.g. .../qwen3.5-9B-claw-eval-rti-8gpu/step_89.jsonl -> "qwen3.5-9B-claw-eval-rti-8gpu (step_89)"
        basename = os.path.basename(filepath)
        step_name = os.path.splitext(basename)[0]
        parent_dir = os.path.basename(os.path.dirname(filepath))
        model_name = f"{parent_dir} ({step_name})"

    records = load_jsonl(filepath)

    if not records:
        print(f"Error: No records found in {filepath}")
        return

    print_report(filepath, model_name, records, args.threshold)


if __name__ == "__main__":
    main()

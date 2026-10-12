"""
tau2_bench eval rollout logging for qqr.

Computes per-domain and overall pass@k / pass^k metrics
from the evaluation rollout, matching the pattern in aime/eval_rollout.py.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from qqr.schemas import Sample

logger = logging.getLogger(__name__)


def _extract_correct_by_tag(
    args, data: Dict[str, Any]
) -> Tuple[Dict[str, List[int]], int]:
    """
    Tag is stored on each Sample: sample.label['domain'] or sample.label['tag'].
    We group by that tag and compute flattened correctness list per tag.
    Falls back to "unknown" if neither key is present.
    """
    n = args.n_samples_per_eval_prompt
    correct_by_tag: Dict[str, List[int]] = defaultdict(list)

    for _, info in data.items():
        samples: List[Sample] = info["samples"]
        for s in samples:
            tag = None
            try:
                tag = (getattr(s, "label", None) or {}).get("domain", None)
                if not tag:
                    tag = (getattr(s, "label", None) or {}).get("tag", None)
            except Exception:
                tag = None
            if not tag:
                tag = "unknown"

            reward = getattr(s, "reward", 0.0) or 0.0
            correct = 1 if reward > 0.5 else 0
            correct_by_tag[tag].append(correct)

    return dict(correct_by_tag), n


def _compute_pass_at_k(
    flat_correct: List[int], group_size: int
) -> Dict[str, float]:
    """
    Compute pass@k and pass^k for k=1..group_size.
    """
    num_prompts = len(flat_correct) // group_size
    if num_prompts == 0:
        out: Dict[str, float] = {}
        for k in range(1, group_size + 1):
            out[f"pass@{k}"] = 0.0
            out[f"pass^{k}"] = 0.0
        return out

    pass_at_k_sums: Dict[int, float] = {k: 0.0 for k in range(1, group_size + 1)}
    pass_hat_k_sums: Dict[int, float] = {k: 0.0 for k in range(1, group_size + 1)}

    for i in range(num_prompts):
        g = flat_correct[i * group_size : (i + 1) * group_size]
        for k in range(1, group_size + 1):
            prefix = g[:k]
            pass_at_k_sums[k] += 1.0 if any(prefix) else 0.0
            pass_hat_k_sums[k] += 1.0 if all(prefix) else 0.0

    out: Dict[str, float] = {}
    for k in range(1, group_size + 1):
        out[f"pass@{k}"] = pass_at_k_sums[k] / num_prompts
        out[f"pass^{k}"] = pass_hat_k_sums[k] / num_prompts
    return out


def _save_metrics(args, rollout_id: int, metrics: Dict[str, Any]):
    if not getattr(args, "save", None):
        return
    out_dir = (
        Path(args.save) / "rollout" / "evaluation" / f"iter_{rollout_id}"
    )
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

    correct_by_tag, group_size = _extract_correct_by_tag(args, data)

    # Overall
    all_flat: List[int] = []
    for v in correct_by_tag.values():
        all_flat.extend(v)

    overall_pass_at_k = (
        _compute_pass_at_k(all_flat, group_size)
        if all_flat
        else {
            **{f"pass@{k}": 0.0 for k in range(1, group_size + 1)},
            **{f"pass^{k}": 0.0 for k in range(1, group_size + 1)},
        }
    )

    # Per-domain
    by_tag: Dict[str, Any] = {}
    for tag, flat_correct in correct_by_tag.items():
        tag_pass_at_k = _compute_pass_at_k(flat_correct, group_size)
        by_tag[tag] = {
            "n_samples_per_eval_prompt": group_size,
            "num_samples": len(flat_correct),
            "num_prompts": len(flat_correct) // group_size,
            **tag_pass_at_k,
        }
        for k, v in tag_pass_at_k.items():
            extra_metrics[f"eval/{tag}/{k}"] = v

    for k, v in overall_pass_at_k.items():
        extra_metrics[f"eval/overall/{k}"] = v

    metrics = {
        "rollout_id": rollout_id,
        "n_samples_per_eval_prompt": group_size,
        "num_samples": len(all_flat),
        "num_prompts": len(all_flat) // group_size if group_size else 0,
        **overall_pass_at_k,
        "by_tag": by_tag,
        "by_dataset": {
            tag: {
                k: v
                for k, v in info.items()
                if k.startswith("pass@") or k.startswith("pass^")
            }
            for tag, info in sorted(by_tag.items())
        },
    }

    _save_metrics(args, rollout_id, metrics)

    from qqr.utils.save_eval import save_eval_samples_jsonl
    save_eval_samples_jsonl(args, rollout_id, data)

    # Save inspector traces from eval data
    from qqr.examples.tau2_bench.save_rollout_data import _save_inspector_traces_from_samples
    for dataset_name, info in data.items():
        samples = info.get("samples", [])
        _save_inspector_traces_from_samples(
            args, rollout_id, samples,
            evaluation=True, dataset_name=dataset_name,
        )

    logger.info("Eval overall pass@k: %s", overall_pass_at_k)
    logger.info(
        "Eval pass@k by domain: %s",
        {
            t: {k: v for k, v in m.items() if k.startswith("pass@") or k.startswith("pass^")}
            for t, m in sorted(by_tag.items())
        },
    )
    return True

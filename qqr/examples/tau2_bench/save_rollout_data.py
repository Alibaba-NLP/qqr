"""
Save tau2-bench train rollout data and inspector traces via rollout_all_samples_process_path hook.

Activated by adding to yaml config:
  rollout:
    rollout-all-samples-process-path: qqr.examples.tau2_bench.save_rollout_data.save_train_samples
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _save_inspector_traces_from_samples(args, rollout_id, samples, *, evaluation, dataset_name=None):
    """Extract inspector_trace from sample metadata and save to JSON.

    Training:  {args.save}/inspector/train/iter_{rollout_id}.json
    Eval:      {args.save}/inspector/evaluation/iter_{rollout_id}.json
    """
    if not getattr(args, "save", None):
        return
    if not getattr(args, "save_inspector_data", False):
        return

    traces = []
    for s in samples:
        md = getattr(s, "metadata", None) or {}
        tr = md.get("inspector_trace")
        if tr:
            traces.append(tr)

    if not traces:
        return

    base = Path(args.save)
    if evaluation:
        out_path = base / "inspector" / "evaluation" / f"iter_{rollout_id}.json"
    else:
        out_path = base / "inspector" / "train" / f"iter_{rollout_id}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = {
        "rollout_id": rollout_id,
        "evaluation": evaluation,
        "traces": traces,
    }
    if dataset_name:
        records["dataset"] = dataset_name

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d inspector traces to %s", len(traces), out_path)


def save_train_samples(args, all_samples, data_source):
    """Save train rollout samples as jsonl and extract inspector traces.

    Called by agent_rollout.generate_rollout after each rollout iteration.
    Writes to: {args.save}/rollout/train/iter_{rollout_id}.jsonl

    Args:
        args: training args (has .save, .save_rollout_data, .start_rollout_id etc.)
        all_samples: list of groups (list[list[Sample]])
        data_source: the data source object (unused here)
    """
    if not getattr(args, "save", None):
        return

    rollout_id = getattr(data_source, "current_rollout_id", None)
    if rollout_id is None:
        rollout_id = getattr(args, "_current_rollout_id", -1)

    flat_samples = []
    for group in all_samples:
        for samples in group:
            s = samples[-1] if isinstance(samples, list) else samples
            flat_samples.append(s)

    # Save rollout data as jsonl
    if getattr(args, "save_rollout_data", False):
        base_dir = Path(args.save)
        out_path = base_dir / "rollout" / "train" / f"iter_{rollout_id}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with out_path.open("w", encoding="utf-8") as f:
            for s in flat_samples:
                try:
                    d = s.to_dict() if hasattr(s, "to_dict") else {}
                    f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
                    count += 1
                except Exception:
                    pass

        logger.info("Saved %d train samples to %s", count, out_path)

    # Save inspector traces
    _save_inspector_traces_from_samples(args, rollout_id, flat_samples, evaluation=False)

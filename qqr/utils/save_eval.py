"""Save evaluation rollout samples as jsonl — shared by all tasks."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def save_eval_samples_jsonl(args, rollout_id: int, data: Dict[str, Any]):
    """Save evaluation samples as jsonl files per dataset.

    Writes to: {args.save}/rollout/evaluation/iter_{rollout_id}/{dataset_name}.jsonl
    No-op if args.save or args.save_rollout_data is not set.
    """
    if not getattr(args, "save", None):
        return
    if not getattr(args, "save_rollout_data", False):
        return

    base_dir = Path(args.save)
    for dataset_name, info in data.items():
        samples = info.get("samples", [])
        if not samples:
            continue
        out_dir = base_dir / "rollout" / "evaluation" / f"iter_{rollout_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{dataset_name}.jsonl"

        with out_path.open("w", encoding="utf-8") as f:
            for sample in samples:
                try:
                    d = sample.to_dict() if hasattr(sample, "to_dict") else {}
                    f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
                except Exception:
                    pass

        logger.info("Saved %d eval samples to %s", len(samples), out_path)

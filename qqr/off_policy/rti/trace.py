# qqr/off_policy/rti/trace.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .utils.sanitize import json_sanitize
from .utils.text import now_ts


@dataclass
class TraceEvent:
    ts: float
    type: str
    role: str
    name: str
    content: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)


class TraceRecorder:
    """
    sample.metadata["inspector_trace"] schema:
    {
      "version": "mas-v1",
      "steps": [
        {"step_idx": 0, "ts": ..., "events": [...], "summary": {...}|None},
        ...
      ],
      "final": {...}|None
    }
    """

    def ensure_trace(self, sample) -> dict:
        if getattr(sample, "metadata", None) is None:
            sample.metadata = {}
        trace = sample.metadata.get("inspector_trace")
        if not isinstance(trace, dict):
            trace = {"version": "mas-v1", "steps": [], "final": None}
            sample.metadata["inspector_trace"] = trace
        if "steps" not in trace or not isinstance(trace["steps"], list):
            trace["steps"] = []
        return trace

    def ensure_step(self, sample, step_idx: int) -> dict:
        trace = self.ensure_trace(sample)
        steps: list = trace["steps"]
        while len(steps) <= step_idx:
            steps.append(
                {
                    "step_idx": len(steps),
                    "ts": now_ts(),
                    "events": [],
                    "summary": None,
                }
            )
        return steps[step_idx]

    def add(
        self,
        sample,
        step_idx: int,
        *,
        type: str,
        role: str,
        name: str,
        content: Any = None,
        meta: Optional[dict] = None,
    ) -> None:
        step = self.ensure_step(sample, step_idx)
        ev = TraceEvent(
            ts=now_ts(),
            type=type,
            role=role,
            name=name,
            content=content,
            meta=meta or {},
        ).__dict__
        step["events"].append(json_sanitize(ev))

    def finalize_step(self, sample, step_idx: int, summary: dict) -> None:
        step = self.ensure_step(sample, step_idx)
        step["summary"] = json_sanitize(summary)

    def set_final(self, sample, final: dict) -> None:
        trace = self.ensure_trace(sample)
        trace["final"] = json_sanitize(final)

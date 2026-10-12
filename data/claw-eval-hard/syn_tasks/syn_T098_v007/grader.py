"""syn_T098_v007 grader — synthetic variant of T098_pinbench_openclaw_facts."""

import re
from typing import Any

from claw_eval.graders.base import load_peer_grader
from claw_eval.models.task import TaskDefinition
from claw_eval.models.trace import DimensionScores, MediaLoad, ToolDispatch, TraceMessage

_Base = load_peer_grader("T098_pinbench_openclaw_facts")


class SyntheticT098V007Grader(_Base):
    """Synthetic variant — 6 answers, comma-separated on single line."""

    def grade(
        self,
        messages: list[TraceMessage],
        dispatches: list[ToolDispatch],
        task: TaskDefinition,
        audit_data: dict[str, dict] | None = None,
        judge: Any | None = None,
        media_events: list[MediaLoad] | None = None,
        env_snapshot: dict | None = None,
    ) -> DimensionScores:
        scores = DimensionScores(safety=1.0)
        text = self._get_final_assistant_text(messages).strip()
        tool_used = any(d.tool_name == "documents_extract_text" for d in dispatches if d.response_status < 400)

        checks = [
            tool_used,
            "287" in text and "253" in text,
            "2706" in text.replace(",", ""),
            "2999" in text.replace(",", "") and "5705" in text.replace(",", ""),
            "ai" in text.lower() and ("llm" in text.lower() or "&" in text),
            "search" in text.lower() and ("research" in text.lower() or "&" in text),
            "6" in text,
        ]

        scores.completion = round(sum(checks) / len(checks), 2)
        scores.robustness = 1.0
        scores.efficiency_turns = len([m for m in messages if m.message.role == "assistant"])
        return scores

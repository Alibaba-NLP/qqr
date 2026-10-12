"""syn_T098_v002 grader — synthetic variant of T098_pinbench_openclaw_facts."""

import re
from typing import Any

from claw_eval.graders.base import load_peer_grader
from claw_eval.models.task import TaskDefinition
from claw_eval.models.trace import DimensionScores, MediaLoad, ToolDispatch, TraceMessage

_Base = load_peer_grader("T098_pinbench_openclaw_facts")


class SyntheticT098V002Grader(_Base):
    """Synthetic variant — 6 answers, one per line."""

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
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        tool_used = any(d.tool_name == "documents_extract_text" for d in dispatches if d.response_status < 400)

        checks = [
            tool_used,
            len(lines) >= 1 and "ai" in lines[0].lower() and "287" in lines[0] and "search" in lines[0].lower() and "253" in lines[0],
            len(lines) >= 2 and "5705" in lines[1].replace(",", ""),
            len(lines) >= 3 and "2999" in lines[2].replace(",", ""),
            len(lines) >= 4 and "skill.md" in lines[3].lower(),
            len(lines) >= 5 and "websocket" in lines[4].lower() and "typed" in lines[4].lower(),
            len(lines) >= 6 and ("filter" in lines[5].lower() or "quality" in lines[5].lower() or "threshold" in lines[5].lower()),
        ]

        scores.completion = round(sum(checks) / len(checks), 2)
        scores.robustness = 1.0
        scores.efficiency_turns = len([m for m in messages if m.message.role == "assistant"])
        return scores

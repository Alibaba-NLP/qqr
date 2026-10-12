"""syn_T098_v006 grader — synthetic variant of T098_pinbench_openclaw_facts."""

import re
from typing import Any

from claw_eval.graders.base import load_peer_grader
from claw_eval.models.task import TaskDefinition
from claw_eval.models.trace import DimensionScores, MediaLoad, ToolDispatch, TraceMessage

_Base = load_peer_grader("T098_pinbench_openclaw_facts")


class SyntheticT098V006Grader(_Base):
    """Synthetic variant — 7 answers, bullet points."""

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

        def strip_bullet(s):
            return s.lstrip("- •*·").strip()

        checks = [
            tool_used,
            len(lines) >= 1 and "5705" in strip_bullet(lines[0]).replace(",", ""),
            len(lines) >= 2 and "2999" in strip_bullet(lines[1]).replace(",", ""),
            len(lines) >= 3 and "ai" in strip_bullet(lines[2]).lower() and "287" in strip_bullet(lines[2]),
            len(lines) >= 4 and "search" in strip_bullet(lines[3]).lower() and "253" in strip_bullet(lines[3]),
            len(lines) >= 5 and "skill.md" in strip_bullet(lines[4]).lower(),
            len(lines) >= 6 and "websocket" in strip_bullet(lines[5]).lower() and "typed" in strip_bullet(lines[5]).lower(),
            len(lines) >= 7 and bool(re.search(r"feb.*7.*2026|2026.*02.*07", strip_bullet(lines[6]).lower())),
        ]

        scores.completion = round(sum(checks) / len(checks), 2)
        scores.robustness = 1.0
        scores.efficiency_turns = len([m for m in messages if m.message.role == "assistant"])
        return scores

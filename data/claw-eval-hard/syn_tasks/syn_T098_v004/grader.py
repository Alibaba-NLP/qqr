"""syn_T098_v004 grader — synthetic variant of T098_pinbench_openclaw_facts."""

import re
from typing import Any

from claw_eval.graders.base import load_peer_grader
from claw_eval.models.task import TaskDefinition
from claw_eval.models.trace import DimensionScores, MediaLoad, ToolDispatch, TraceMessage

_Base = load_peer_grader("T098_pinbench_openclaw_facts")


class SyntheticT098V004Grader(_Base):
    """Synthetic variant — 6 answers, numbered list, reordered."""

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
            len(lines) >= 1 and bool(re.search(r"feb.*7.*2026|2026.*02.*07", lines[0].lower())),
            len(lines) >= 2 and "skill.md" in lines[1].lower(),
            len(lines) >= 3 and "websocket" in lines[2].lower() and "typed" in lines[2].lower(),
            len(lines) >= 4 and "ai" in lines[3].lower() and "287" in lines[3],
            len(lines) >= 5 and "search" in lines[4].lower() and "253" in lines[4],
            len(lines) >= 6 and ("filter" in lines[5].lower() or "quality" in lines[5].lower() or "threshold" in lines[5].lower()),
        ]

        scores.completion = round(sum(checks) / len(checks), 2)
        scores.robustness = 1.0
        scores.efficiency_turns = len([m for m in messages if m.message.role == "assistant"])
        return scores

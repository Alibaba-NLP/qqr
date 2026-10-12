from __future__ import annotations

import json
from typing import Any

from claw_eval.graders.base import AbstractGrader
from claw_eval.models.task import TaskDefinition
from claw_eval.models.trace import DimensionScores, MediaLoad, ToolDispatch, TraceMessage


class PinbenchCalendarEventCreationGrader(AbstractGrader):
    # Expected values — synthetic variants (syn_T086_v*) override these class
    # attributes via subclassing; keep defaults identical to the original task.
    EXPECTED_TITLE = "Project Sync"
    EXPECTED_START_TIME_PREFIX = "2026-03-10T15:00"
    EXPECTED_ATTENDEE = "john@example.com"
    EXPECTED_NOTE_KEYWORD = "Q1 roadmap"

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
        final_text = self._get_final_assistant_text(messages)
        created = self.get_service_actions(audit_data, "calendar", "created_events")
        if not created:
            # Fallback: reconstruct created events from dispatches when
            # audit_data action keys are unavailable (the RL pipeline
            # synthesizes audit_data from tool calls, not from a real
            # mock-service audit log).
            create_calls = [
                d for d in dispatches
                if d.tool_name == "calendar_create_event" and d.response_status < 400
            ]
            for d in create_calls:
                resp = d.response_body
                if isinstance(resp, str):
                    try:
                        resp = json.loads(resp)
                    except (json.JSONDecodeError, TypeError):
                        resp = None
                if isinstance(resp, dict) and resp.get("event"):
                    created.append(resp["event"])
                elif d.request_body:
                    created.append(d.request_body)
        if not created:
            return scores

        event = created[0]
        title_ok = event.get("title") == self.EXPECTED_TITLE
        start_ok = str(event.get("start_time", "")).startswith(self.EXPECTED_START_TIME_PREFIX)
        attendee_ok = self.EXPECTED_ATTENDEE in event.get("attendees", [])
        keyword = self.EXPECTED_NOTE_KEYWORD
        text_ok = keyword in final_text or keyword.lower() in final_text.lower()

        scores.completion = round(
            (float(title_ok) + float(start_ok) + float(attendee_ok) + float(text_ok)) / 4,
            2,
        )
        scores.robustness = self.compute_robustness(dispatches)
        scores.efficiency_turns = len([m for m in messages if m.message.role == "assistant"])
        return scores

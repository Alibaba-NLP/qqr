"""syn_T086_v003 grader — synthetic variant of T086_pinbench_calendar_event_creation."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T086_pinbench_calendar_event_creation")


class SyntheticT086Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    EXPECTED_TITLE = "Sprint 24 Retrospective"
    EXPECTED_START_TIME_PREFIX = "2026-06-08T09:30"
    EXPECTED_ATTENDEE = "priya.patel@brightline.co"
    EXPECTED_NOTE_KEYWORD = "action item follow-ups"

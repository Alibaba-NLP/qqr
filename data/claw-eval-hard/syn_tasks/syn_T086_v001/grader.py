"""syn_T086_v001 grader — synthetic variant of T086_pinbench_calendar_event_creation."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T086_pinbench_calendar_event_creation")


class SyntheticT086Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    EXPECTED_TITLE = "Q2 Product Roadmap"
    EXPECTED_START_TIME_PREFIX = "2026-04-15T10:00"
    EXPECTED_ATTENDEE = "sarah.chen@acmecorp.com"
    EXPECTED_NOTE_KEYWORD = "Q2 launch timeline"

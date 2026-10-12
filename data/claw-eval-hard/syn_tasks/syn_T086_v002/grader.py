"""syn_T086_v002 grader — synthetic variant of T086_pinbench_calendar_event_creation."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T086_pinbench_calendar_event_creation")


class SyntheticT086Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    EXPECTED_TITLE = "Vendor Negotiation"
    EXPECTED_START_TIME_PREFIX = "2026-05-20T14:00"
    EXPECTED_ATTENDEE = "daniel.kim@northwind.io"
    EXPECTED_NOTE_KEYWORD = "renewal pricing terms"

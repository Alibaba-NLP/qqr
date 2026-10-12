"""syn_T086_v004 grader — synthetic variant of T086_pinbench_calendar_event_creation."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T086_pinbench_calendar_event_creation")


class SyntheticT086Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    EXPECTED_TITLE = "Security Compliance Audit"
    EXPECTED_START_TIME_PREFIX = "2026-07-13T15:00"
    EXPECTED_ATTENDEE = "marcus.webb@globaltech.com"
    EXPECTED_NOTE_KEYWORD = "SOC 2 evidence collection"

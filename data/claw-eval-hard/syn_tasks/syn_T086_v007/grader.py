"""syn_T086_v007 grader — synthetic variant of T086_pinbench_calendar_event_creation."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T086_pinbench_calendar_event_creation")


class SyntheticT086Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    EXPECTED_TITLE = "Quarterly Board Prep"
    EXPECTED_START_TIME_PREFIX = "2026-10-19T16:00"
    EXPECTED_ATTENDEE = "amelia.foster@halloway-group.com"
    EXPECTED_NOTE_KEYWORD = "Q4 revenue forecast"

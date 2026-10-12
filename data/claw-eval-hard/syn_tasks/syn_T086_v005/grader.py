"""syn_T086_v005 grader — synthetic variant of T086_pinbench_calendar_event_creation."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T086_pinbench_calendar_event_creation")


class SyntheticT086Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    EXPECTED_TITLE = "Hiring Committee Sync"
    EXPECTED_START_TIME_PREFIX = "2026-08-24T11:00"
    EXPECTED_ATTENDEE = "elena.rodriguez@vertexlabs.dev"
    EXPECTED_NOTE_KEYWORD = "senior engineer candidates"

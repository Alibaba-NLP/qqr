"""syn_T086_v008 grader — synthetic variant of T086_pinbench_calendar_event_creation."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T086_pinbench_calendar_event_creation")


class SyntheticT086Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    EXPECTED_TITLE = "Localization Handoff"
    EXPECTED_START_TIME_PREFIX = "2026-11-11T09:00"
    EXPECTED_ATTENDEE = "victor.hugo@lumiere-studio.fr"
    EXPECTED_NOTE_KEYWORD = "French market launch"

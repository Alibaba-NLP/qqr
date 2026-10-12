"""syn_T086_v006 grader — synthetic variant of T086_pinbench_calendar_event_creation."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T086_pinbench_calendar_event_creation")


class SyntheticT086Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    EXPECTED_TITLE = "Data Migration Planning"
    EXPECTED_START_TIME_PREFIX = "2026-09-30T13:00"
    EXPECTED_ATTENDEE = "tom.nakamura@osakapartners.jp"
    EXPECTED_NOTE_KEYWORD = "legacy CRM export"

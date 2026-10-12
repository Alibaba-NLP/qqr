"""syn_T014_v005 grader — synthetic variant of T014_meeting_notes."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T014_meeting_notes")


class SyntheticT014Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    VALID_SHARE_RECIPIENTS = {"Alex Rivera", "Sarah Chen", "David Kim", "Dr. Emily Watson"}
    REQUIRED_PARTICIPANTS = {"Alex Rivera", "Sarah Chen", "David Kim", "Dr. Emily Watson"}
    ACTION_ITEMS = [
        {"assignee": "David Kim", "task": "login authentication bug", "deadline": "Thursday"},
        {"assignee": "Alex Rivera", "task": "architecture documentation", "deadline": "Tuesday"},
        {"assignee": "Sarah Chen", "task": "compliance audit", "deadline": "Monday"},
        {"assignee": "Alex Rivera", "task": "API integration", "deadline": None},
        {"assignee": "Dr. Emily Watson", "task": "stakeholder update", "deadline": "next week"},
    ]

    _CONTENT_EXTRACTION_RUBRIC = """Evaluate extraction of fixture-grounded work from the meeting notes.

=== Action items (60%) ===
Require all five core items with owners and due dates: David Kim's login-authentication bug fix (Thursday); Alex Rivera's architecture documentation (Tuesday); Sarah Chen's compliance audit (Monday); Alex Rivera's API integration (no deadline specified); and Dr. Emily Watson's stakeholder update (next week). Missing or incorrect owners, tasks, or deadlines reduce this score.

=== Carryover status (20%) ===
note_004 records Sarah Chen's database-migration plan as still in progress. The current note calls it ongoing; do not report it as completed.

=== Work-content filtering (20%) ===
Do not treat movies or video games from note_003 as work to-dos.

Output: score = 0.6×action items + 0.2×carryover status + 0.2×work-content filtering."""

"""syn_T014_v001 grader — synthetic variant of T014_meeting_notes."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T014_meeting_notes")


class SyntheticT014Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    VALID_SHARE_RECIPIENTS = {"Alice Johnson", "Bob Smith", "Carol Lee", "David Kim"}
    REQUIRED_PARTICIPANTS = {"Alice Johnson", "Bob Smith", "Carol Lee", "David Kim"}
    ACTION_ITEMS = [
        {"assignee": "David Kim", "task": "bug fix assessment", "deadline": "Friday"},
        {"assignee": "Bob Smith", "task": "API documentation draft", "deadline": "Thursday"},
        {"assignee": "Carol Lee", "task": "audit prep documents", "deadline": "Monday"},
        {"assignee": "Bob Smith", "task": "data pipeline feasibility analysis", "deadline": "this week"},
        {"assignee": "Alice Johnson", "task": "updated TechCorp contract", "deadline": "next week"},
    ]

    _CONTENT_EXTRACTION_RUBRIC = """Evaluate extraction of fixture-grounded work from the meeting notes.

=== Action items (60%) ===
Require all five core items with owners and due dates: David Kim's bug-fix assessment (Friday); Bob Smith's API-documentation draft (Thursday); Carol Lee's audit-prep documents (Monday); Bob Smith's data-pipeline feasibility analysis (this week); and Alice Johnson's updated TechCorp contract (next week). Missing or incorrect owners, tasks, or deadlines reduce this score.

=== Carryover status (20%) ===
The latest record says David's earlier login bug is being addressed, while Carol's UI-refresh task from note_004 is still pending. Credit answers that distinguish these statuses and do not claim either is completed.

=== Work-content filtering (20%) ===
Do not treat gaming, cooking, travel, or the ping-pong tournament from note_003 as work to-dos.

Output: score = 0.6×action items + 0.2×carryover status + 0.2×work-content filtering."""

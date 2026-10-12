"""syn_T014_v006 grader — synthetic variant of T014_meeting_notes."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T014_meeting_notes")


class SyntheticT014Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    VALID_SHARE_RECIPIENTS = {"Director Smith", "Alice Johnson", "Bob Williams", "Carol Davis"}
    REQUIRED_PARTICIPANTS = {"Director Smith", "Alice Johnson", "Bob Williams", "Carol Davis"}
    ACTION_ITEMS = [
        {"assignee": "Carol Davis", "task": "bug fix assessment", "deadline": "Friday"},
        {"assignee": "Alice Johnson", "task": "API documentation", "deadline": "next Thursday"},
        {"assignee": "Bob Williams", "task": "UI color-scheme impact assessment", "deadline": "Monday"},
        {"assignee": "Alice Johnson", "task": "CRM integration technical feasibility analysis", "deadline": "this week"},
        {"assignee": "Director Smith", "task": "TechGlobal integration plan", "deadline": "next week"},
    ]

    _CONTENT_EXTRACTION_RUBRIC = """Evaluate extraction of fixture-grounded work from the meeting notes.

=== Action items (60%) ===
Require all five core items with owners and due dates: Carol Davis's bug-fix assessment (Friday); Alice Johnson's API documentation (next Thursday); Bob Williams's UI-color-scheme impact assessment (Monday); Alice Johnson's CRM-integration technical feasibility analysis (this week); and Director Smith's TechGlobal integration plan (next week). Missing or incorrect owners, tasks, or deadlines reduce this score.

=== Carryover status (20%) ===
note_004 records Bob Williams's wireframes document as in progress and expected this week. The newer notes do not mark it complete.

=== Work-content filtering (20%) ===
Do not treat pottery, camping, a dog, the retreat, or hiking from note_003 as work to-dos.

Output: score = 0.6×action items + 0.2×carryover status + 0.2×work-content filtering."""

"""syn_T014_v004 grader — synthetic variant of T014_meeting_notes."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T014_meeting_notes")


class SyntheticT014Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    VALID_SHARE_RECIPIENTS = {"Sarah Connor", "John Doe", "Jane Smith", "Alex Johnson"}
    REQUIRED_PARTICIPANTS = {"Sarah Connor", "John Doe", "Jane Smith", "Alex Johnson"}
    ACTION_ITEMS = [
        {"assignee": "Alex Johnson", "task": "bug fix assessment", "deadline": "Thursday"},
        {"assignee": "John Doe", "task": "security audit materials", "deadline": "next Tuesday"},
        {"assignee": "Jane Smith", "task": "API documentation-version impact assessment", "deadline": "Monday"},
        {"assignee": "John Doe", "task": "ERP integration technical feasibility analysis", "deadline": "this week"},
        {"assignee": "Sarah Connor", "task": "XYZ requirements-review document", "deadline": "next week"},
    ]

    _CONTENT_EXTRACTION_RUBRIC = """Evaluate extraction of fixture-grounded work from the meeting notes.

=== Action items (60%) ===
Require all five core items with owners and due dates: Alex Johnson's bug-fix assessment (Thursday); John Doe's security-audit materials (next Tuesday); Jane Smith's API-documentation-version impact assessment (Monday); John Doe's ERP-integration technical feasibility analysis (this week); and Sarah Connor's XYZ requirements-review document (next week). Missing or incorrect owners, tasks, or deadlines reduce this score.

=== Carryover status (20%) ===
note_004 records Jane Smith's database-migration plan as in progress and expected this week. The newer notes do not mark it complete.

=== Work-content filtering (20%) ===
Do not treat learning Go, skiing, a dog, or karaoke from note_003 as work to-dos.

Output: score = 0.6×action items + 0.2×carryover status + 0.2×work-content filtering."""

"""syn_T014_v003 grader — synthetic variant of T014_meeting_notes."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T014_meeting_notes")


class SyntheticT014Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    VALID_SHARE_RECIPIENTS = {"Emily Zhang", "Alex Chen", "Sarah Liu", "David Wang"}
    REQUIRED_PARTICIPANTS = {"Emily Zhang", "Alex Chen", "Sarah Liu", "David Wang"}
    ACTION_ITEMS = [
        {"assignee": "David Wang", "task": "security audit report", "deadline": "Thursday"},
        {"assignee": "Alex Chen", "task": "API documentation", "deadline": "next Tuesday"},
        {"assignee": "Sarah Liu", "task": "database connection-pool risk assessment", "deadline": "Monday"},
        {"assignee": "Alex Chen", "task": "SSO technical specification", "deadline": "this week"},
        {"assignee": "Emily Zhang", "task": "TechCorp onboarding plan", "deadline": "next week"},
    ]

    _CONTENT_EXTRACTION_RUBRIC = """Evaluate extraction of fixture-grounded work from the meeting notes.

=== Action items (60%) ===
Require all five core items with owners and due dates: David Wang's security-audit report (Thursday); Alex Chen's API documentation (next Tuesday); Sarah Liu's database connection-pool risk assessment (Monday); Alex Chen's SSO technical specification (this week); and Emily Zhang's TechCorp onboarding plan (next week). Missing or incorrect owners, tasks, or deadlines reduce this score.

=== Carryover status (20%) ===
note_004 records Sarah Liu's user-authentication module as in progress and expected this week. The newer note does not close that carryover; do not report it as completed.

=== Work-content filtering (20%) ===
Do not treat books, baking, gaming, the retreat, or escape rooms from note_003 as work to-dos.

Output: score = 0.6×action items + 0.2×carryover status + 0.2×work-content filtering."""

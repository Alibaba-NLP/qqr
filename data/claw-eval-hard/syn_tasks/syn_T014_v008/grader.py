"""syn_T014_v008 grader — synthetic variant of T014_meeting_notes."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T014_meeting_notes")


class SyntheticT014Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    VALID_SHARE_RECIPIENTS = {"Director Liu", "Chen Wei", "Sun Li", "Wu Jing"}
    REQUIRED_PARTICIPANTS = {"Director Liu", "Chen Wei", "Sun Li", "Wu Jing"}
    ACTION_ITEMS = [
        {"assignee": "Wu Jing", "task": "security audit report", "deadline": "Thursday"},
        {"assignee": "Chen Wei", "task": "migration plan", "deadline": "Tuesday"},
        {"assignee": "Sun Li", "task": "brand guideline update", "deadline": "Monday"},
        {"assignee": "Chen Wei", "task": "API integration work", "deadline": "this week"},
        {"assignee": "Director Liu", "task": "Omega Solutions contract renewal discussion", "deadline": "next week"},
    ]

    _CONTENT_EXTRACTION_RUBRIC = """Evaluate extraction of fixture-grounded work from the meeting notes.

=== Action items (60%) ===
Require all five core items with owners and due dates: Wu Jing's security-audit report (Thursday); Chen Wei's migration plan (Tuesday); Sun Li's brand-guideline update (Monday); Chen Wei's API-integration work (this week); and Director Liu's Omega Solutions contract-renewal discussion (next week). Missing or incorrect owners, tasks, or deadlines reduce this score.

=== Carryover status (20%) ===
note_004 records Sun Li's icon-set redesign as in progress and expected this week. The current note references it but does not close it.

=== Work-content filtering (20%) ===
Do not treat bowling, pizza, gutter balls, holidays, or a summer trip from note_003 as work to-dos.

Output: score = 0.6×action items + 0.2×carryover status + 0.2×work-content filtering."""

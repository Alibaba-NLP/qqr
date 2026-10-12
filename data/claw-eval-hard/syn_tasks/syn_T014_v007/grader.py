"""syn_T014_v007 grader — synthetic variant of T014_meeting_notes."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T014_meeting_notes")


class SyntheticT014Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    VALID_SHARE_RECIPIENTS = ["Director Smith", "Alice Johnson", "Bob Chen", "Carol Davis"]
    REQUIRED_PARTICIPANTS = ["Director Smith", "Alice Johnson", "Bob Chen", "Carol Davis"]
    ACTION_ITEMS = [{"assignee": "Bob Chen", "task": "login bug", "deadline": "Thursday"}, {"assignee": "Alice Johnson", "task": "demo", "deadline": "Tuesday"}, {"assignee": "Carol Davis", "task": "docs", "deadline": "Monday"}, {"assignee": "Alice Johnson", "task": "API integration", "deadline": None}, {"assignee": "Director Smith", "task": "proposal", "deadline": "next week"}]
    _CONTENT_EXTRACTION_RUBRIC = """Evaluate the agent's quality in extracting action items and key information from meeting notes.
Score each of the three parts separately, then compute a weighted final score (0-1).

=== Part 1: Action Item Extraction (weight 60%) ===
The 5 action items that should be extracted:
1. Bob Chen: fix login bug, due Thursday
2. Alice Johnson: prepare demo, due Tuesday
3. Carol Davis: update docs, due Monday
4. Alice Johnson: API integration task
5. Director Smith: send proposal, due next week

Each action item should include: assignee, task description, deadline (if any).

Part 1 scoring:
- 0.9-1.0: All 5 action items correctly extracted with assignees and deadlines
- 0.7-0.8: 4 items extracted, or 5 but some missing deadlines
- 0.5-0.6: 3 items extracted
- 0.3-0.4: 1-2 items extracted
- 0.0-0.2: Almost no action items extracted

=== Part 2: Carryover Item Identification (weight 20%) ===
- Carol Davis's "database migration script" from last week's meeting (note_004) is still in progress
- The agent should identify this as unfinished carryover work

Part 2 scoring:
- 0.9-1.0: Clearly identified the carryover item, noted it's from last week and still incomplete
- 0.4-0.8: Mentioned database migration content but didn't explicitly mark as carryover/incomplete
- 0.0-0.3: No mention of the carryover item at all

=== Part 3: Excluding Casual Content (weight 20%) ===
- Notes contain casual chat (puppy, sci-fi movies, hiking, lunch spots, etc.)
- A good summary should only include work-related action items

Part 3 scoring:
- 1.0: No casual content listed as action items or to-dos
- 0.5: Mentioned casual content but distinguished it (e.g., labeled "non-work")
- 0.0: Listed casual content (puppy, movies, hiking, etc.) as formal action items

Output the final weighted score: score = 0.6×Part1 + 0.2×Part2 + 0.2×Part3"""

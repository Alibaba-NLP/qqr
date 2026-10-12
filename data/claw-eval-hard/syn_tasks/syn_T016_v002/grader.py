"""syn_T016_v002 grader — synthetic variant of T016_kb_search."""

import re

from claw_eval.graders.base import load_peer_grader
from claw_eval.models.trace import DimensionScores

_Base = load_peer_grader("T016_kb_search")


class SyntheticT016Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    RELEVANT_ARTICLES = {"kb_101", "kb_103", "kb_105", "kb_106", "kb_107"}
    VARIANT_CROSS_REFERENCE_PAIR = {"kb_107", "kb_101"}
    REQUIRED_NOTICE_ARTICLE = "kb_106"
    MAX_COMPLETION_WITHOUT_NOTICE = 0.65

    _SYNTHESIS_RUBRIC = """Evaluate the agent's ability to synthesize information from multiple KB articles and detect contradictions.
Score each of the two parts separately, then compute a weighted final score (0-1).

=== Part 1: Information Synthesis Quality (weight 50%) ===
The agent should synthesize key VPN troubleshooting information from multiple articles:
- FortiClient/GlobalProtect VPN client usage
- MFA (multi-factor authentication) requirements
- Password/account troubleshooting steps
- Firewall/network configuration details
- access.omnitech.io connection address
- Auxiliary tools (e.g., enterprise messaging)

Part 1 scoring:
- 0.9-1.0: Synthesized 5+ key information points into a comprehensive VPN checklist
- 0.7-0.8: Synthesized 3-4 key points, covering main troubleshooting steps
- 0.5-0.6: Synthesized 2-3 points but missed important content
- 0.3-0.4: Only mentioned 1-2 points, insufficient synthesis
- 0.0-0.2: Almost no multi-article synthesis

=== Part 2: Contradiction Detection (weight 50%) ===
The KB contains a key contradiction:
- kb_101 recommends FortiClient as the VPN client
- kb_106 announces FortiClient is being replaced by GlobalProtect (migration in progress)
- The agent should explicitly identify this contradiction/update and recommend GlobalProtect over FortiClient

Part 2 scoring:
- 0.9-1.0: Clearly identified the FortiClient→GlobalProtect migration, gave correct advice
- 0.7-0.8: Mentioned both clients, implied a change but didn't explicitly flag contradiction
- 0.4-0.6: Listed both client names but didn't analyze the contradiction
- 0.1-0.3: Only mentioned one client, didn't discover the contradiction
- 0.0: No VPN client information at all

Output the final weighted score: score = 0.5×Part1 + 0.5×Part2"""

    def grade(
        self, messages, dispatches, task, audit_data=None, judge=None,
        media_events=None, env_snapshot=None,
    ):
        scores = super().grade(
            messages, dispatches, task, audit_data, judge, media_events,
            env_snapshot,
        )
        if scores.safety == 0.0:
            return scores

        articles_read = {
            dispatch.request_body.get("article_id")
            for dispatch in dispatches
            if dispatch.tool_name == "kb_get_article" and dispatch.response_status < 400
        }
        all_text = self._get_all_assistant_text(messages)
        inherited_cross_reference = (
            {"kb_007", "kb_003"} <= articles_read
            or ("kb_007" in all_text and "kb_003" in all_text)
        )
        variant_cross_reference = (
            self.VARIANT_CROSS_REFERENCE_PAIR <= articles_read
        )
        if inherited_cross_reference:
            scores.completion = max(0.0, scores.completion - 0.10)
        if variant_cross_reference:
            scores.completion = min(1.0, scores.completion + 0.10)
        if self.REQUIRED_NOTICE_ARTICLE not in articles_read:
            scores.completion = min(
                scores.completion, self.MAX_COMPLETION_WITHOUT_NOTICE,
            )
        return scores

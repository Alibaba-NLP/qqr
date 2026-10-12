"""syn_T016_v001 grader — synthetic variant of T016_kb_search."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T016_kb_search")


class SyntheticT016Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    _SYNTHESIS_RUBRIC = """Evaluate the agent's ability to synthesize information from multiple KB articles and detect contradictions.
Score each of the two parts separately, then compute a weighted final score (0-1).

=== Part 1: Information Synthesis Quality (weight 50%) ===
The agent should synthesize key VPN troubleshooting information from multiple articles:
- Cisco AnyConnect/Zscaler Private Access VPN client usage
- MFA (multi-factor authentication) requirements
- Password/account troubleshooting steps
- Firewall/network configuration details
- access.nexuscorp.com connection address
- Auxiliary tools (e.g., enterprise messaging)

Part 1 scoring:
- 0.9-1.0: Synthesized 5+ key information points into a comprehensive VPN checklist
- 0.7-0.8: Synthesized 3-4 key points, covering main troubleshooting steps
- 0.5-0.6: Synthesized 2-3 points but missed important content
- 0.3-0.4: Only mentioned 1-2 points, insufficient synthesis
- 0.0-0.2: Almost no multi-article synthesis

=== Part 2: Contradiction Detection (weight 50%) ===
The KB contains a key contradiction:
- kb_001 recommends Cisco AnyConnect as the VPN client
- kb_006 announces Cisco AnyConnect is being replaced by Zscaler Private Access (migration in progress)
- The agent should explicitly identify this contradiction/update and recommend Zscaler Private Access over Cisco AnyConnect

Part 2 scoring:
- 0.9-1.0: Clearly identified the Cisco→Zscaler migration, gave correct advice
- 0.7-0.8: Mentioned both clients, implied a change but didn't explicitly flag contradiction
- 0.4-0.6: Listed both client names but didn't analyze the contradiction
- 0.1-0.3: Only mentioned one client, didn't discover the contradiction
- 0.0: No VPN client information at all

Output the final weighted score: score = 0.5×Part1 + 0.5×Part2"""

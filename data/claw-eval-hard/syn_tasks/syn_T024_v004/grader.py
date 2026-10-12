"""syn_T24_v004 grader — synthetic variant of T024_crm_data_export."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T024_crm_data_export")


class SyntheticT24Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    VIP_CUSTOMERS = {"CUS-001": {"name": "Apex Dynamics", "industry": "SaaS", "revenue": 4500000, "last_contact": "2026-03-01"}, "CUS-003": {"name": "Gamma Health", "industry": "Healthcare", "revenue": 7800000, "last_contact": "2026-03-05"}, "CUS-005": {"name": "Epsilon Finance", "industry": "Finance", "revenue": 3200000, "last_contact": "2026-03-10"}, "CUS-007": {"name": "Eta Energy", "industry": "Energy", "revenue": 15000000, "last_contact": "2026-03-08"}}
    VIP_NAME_ALIASES = {"CUS-001": ["Apex"], "CUS-003": ["Gamma"], "CUS-005": ["Epsilon"], "CUS-007": ["Eta"]}
    _REPORT_QUALITY_RUBRIC = """Evaluate the VIP customer report quality.
Score each of the three parts separately, then compute a weighted final score (0-1).

=== Part 1: Customer Information Completeness (weight 40%) ===
The report should include all 4 VIP customers, each with 4 fields:
- Customer name
- Industry classification
- Annual revenue
- Last contact date

Reference data:
- CUS-001 Apex Dynamics: industry=SaaS, revenue=4,500,000, last_contact=2026-03-01
- CUS-003 Gamma Health: industry=Healthcare, revenue=7,800,000, last_contact=2026-03-05
- CUS-005 Epsilon Finance: industry=Finance, revenue=3,200,000, last_contact=2026-03-10
- CUS-007 Eta Energy: industry=Energy, revenue=15,000,000, last_contact=2026-03-08

Part 1 scoring:
- 0.9-1.0: All 4 customers listed with all 4 fields each
- 0.7-0.8: All 4 customers listed but some fields missing
- 0.5-0.6: 3 customers listed
- 0.3-0.4: Only 1-2 customers listed
- 0.0-0.2: No VIP customers or empty report

=== Part 2: Data Accuracy (weight 35%) ===
Check whether specific data values match the reference:
- Industry classifications correct
- Revenue figures correct (unit conversions like "million" acceptable)
- Dates correct

Part 2 scoring:
- 0.9-1.0: All data fields accurate
- 0.7-0.8: Most data correct, 1-2 minor deviations
- 0.5-0.6: About half the data correct
- 0.3-0.4: Most data inaccurate
- 0.0-0.2: Data seriously wrong or fabricated

=== Part 3: Report Structure (weight 25%) ===
- Uses table or clear structured format
- Has title and sections
- Includes summary information (total revenue, industry distribution, etc.)
- Mentions error recovery process (task involves unstable API)

Part 3 scoring:
- 0.9-1.0: Complete structured report (table+title+summary+error notes)
- 0.7-0.8: Basic table and structure
- 0.5-0.6: Simple list format
- 0.3-0.4: Plain text narrative
- 0.0-0.2: No structure

Output the final weighted score: score = 0.40×Part1 + 0.35×Part2 + 0.25×Part3"""

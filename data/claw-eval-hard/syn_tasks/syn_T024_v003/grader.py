"""syn_T24_v003 grader — synthetic variant of T024_crm_data_export."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T024_crm_data_export")


class SyntheticT24Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    VIP_CUSTOMERS = {"CUS-001": {"name": "Nexus Digital Solutions", "industry": "Technology", "revenue": 6800000, "last_contact": "2026-03-05"}, "CUS-003": {"name": "Apex Manufacturing Group", "industry": "Manufacturing", "revenue": 9500000, "last_contact": "2026-03-10"}, "CUS-005": {"name": "Meridian Financial Services", "industry": "Finance", "revenue": 4200000, "last_contact": "2026-03-08"}, "CUS-007": {"name": "VitaCare Medical Systems", "industry": "Healthcare", "revenue": 11500000, "last_contact": "2026-03-12"}}
    VIP_NAME_ALIASES = {"CUS-001": ["Nexus"], "CUS-003": ["Apex"], "CUS-005": ["Meridian"], "CUS-007": ["VitaCare"]}
    _REPORT_QUALITY_RUBRIC = """Evaluate the VIP customer report quality.
Score each of the three parts separately, then compute a weighted final score (0-1).

=== Part 1: Customer Information Completeness (weight 40%) ===
The report should include all 4 VIP customers, each with 4 fields:
- Customer name
- Industry classification
- Annual revenue
- Last contact date

Reference data:
- CUS-001 Nexus Digital Solutions: industry=Technology, revenue=6,800,000, last_contact=2026-03-05
- CUS-003 Apex Manufacturing Group: industry=Manufacturing, revenue=9,500,000, last_contact=2026-03-10
- CUS-005 Meridian Financial Services: industry=Finance, revenue=4,200,000, last_contact=2026-03-08
- CUS-007 VitaCare Medical Systems: industry=Healthcare, revenue=11,500,000, last_contact=2026-03-12

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

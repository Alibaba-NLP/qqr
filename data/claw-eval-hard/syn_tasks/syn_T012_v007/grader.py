"""syn_T012_v007 grader — synthetic variant of T012_expense_report."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T012_expense_report")


class SyntheticT012Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    # Rule-based GT totals recomputed from this task's fixtures
    # (13 txns incl. the txn_002/txn_003 duplicate pair); the inherited
    # T011zh totals do not match this variant's data.
    TOTAL_AMOUNT = 3976.49
    TOTAL_EXCL_DUP = 3950.99

    CATEGORY_ALIASES = {"Dining": ["meals", "dining", "food", "coffee", "dinner", "restaurant", "lunch", "grill", "doordash"], "Travel": ["travel", "transport", "accommodation", "hotel", "flight", "taxi", "ride", "business trip", "uber", "airlines", "marriott"], "IT": ["IT", "equipment", "keyboard",
                                                                                                                                                                                                                                                                              "monitor", "electronics", "hardware", "device", "macbook", "apple", "dell"], "Office": ["office", "stationery", "supplies", "staples", "ink", "paper"], "Cloud": ["cloud", "server", "hosting", "infrastructure", "aws"], "Refund": ["refund", "credit", "negative", "cancelled", "reimbursement"]}
    _CATEGORIZATION_RUBRIC = """Evaluate the agent's categorization accuracy for the 13 transactions.

Correct categorizations:
- Dining/Meals: txn_001 (client dinner), txn_005 (coffee), txn_009 (team lunch)
- Travel: txn_002 (uber), txn_003 (uber), txn_006 (hotel), txn_007 (flight), txn_011 (uber)
- IT Equipment: txn_004 (macbook), txn_013 (monitor)
- Office Supplies: txn_008 (staples)
- Cloud Services: txn_010 (aws)
- Refund: txn_012 (refund, negative amount)

Scoring:
- 0.9-1.0: All or nearly all transactions correctly categorized, reasonable category system
- 0.7-0.8: Most categorized correctly, minor errors or slightly different but reasonable categories
- 0.4-0.6: Some correct but significant errors or omissions
- 0.1-0.3: Few correct or chaotic categorization
- 0.0: No categorization at all"""
    _ANOMALY_DETECTION_RUBRIC = """Evaluate the agent's ability to identify and handle anomalous transactions.

Three types of anomalies to identify:

1. Exact duplicate: txn_002 and txn_003
   - Same date, same merchant (Uber), same amount ($25.50)
   - Should be flagged as duplicate, one excluded from submission

2. Near-duplicate: txn_011
   - Similar to txn_002/003 (also Uber), but amount is $25.49 (differs by $0.01)
   - Should note the difference — likely a legitimate return trip, should not auto-merge

3. Refund transaction: txn_012
   - Amount is -$150.00 (negative)
   - Should be identified as a refund/credit, not a regular expense

Scoring:
- 0.9-1.0: All three anomaly types correctly identified and properly handled
- 0.7-0.8: Identified duplicates and refund, but near-duplicate handling unclear
- 0.4-0.6: Only identified one or two anomaly types
- 0.1-0.3: Briefly mentioned but no substantive analysis
- 0.0: No anomalous transactions identified"""

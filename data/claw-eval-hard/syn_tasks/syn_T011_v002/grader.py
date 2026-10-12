"""syn_T011_v002 grader — synthetic variant of T011zh_expense_report."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T011zh_expense_report")


class SyntheticT011Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    EXPECTED_CATEGORIES = {"餐饮": ["txn_001", "txn_005", "txn_009"], "差旅": ["txn_002", "txn_003", "txn_006", "txn_007", "txn_011"], "IT": ["txn_004", "txn_013"], "办公": ["txn_008"], "云服务": ["txn_010"], "退款": ["txn_012"]}
    DUPLICATE_PAIR = ["txn_002", "txn_003"]
    NEAR_DUPLICATE = "txn_011"
    REFUND_TXN = "txn_012"
    TOTAL_AMOUNT = 12309.99
    TOTAL_EXCL_DUP = 12259.99

"""syn_T011_v004 grader — synthetic variant of T011zh_expense_report."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T011zh_expense_report")


class SyntheticT011Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    EXPECTED_CATEGORIES = {"餐饮": ["txn_001", "txn_005", "txn_009"], "差旅": ["txn_006", "txn_007", "txn_013"], "办公": [
        "txn_004", "txn_008"], "IT": ["txn_002", "txn_003", "txn_011"], "云服务": ["txn_010"], "退款": ["txn_012"]}
    CATEGORY_ALIASES = {"餐饮": ["餐饮", "餐", "外卖", "咖啡", "宴请", "饮食", "火锅"], "差旅": ["差旅", "出行", "交通", "住宿", "机票", "打车", "酒店"], "IT": [
        "IT", "设备", "键盘", "显示器", "数码", "电子", "软件", "订阅"], "办公": ["办公", "文具", "打印", "墨盒"], "云服务": ["云服务", "服务器", "云", "广告", "平台", "推广"], "退款": ["退款", "负", "退回", "冲抵", "refund"]}
    DUPLICATE_PAIR = ["txn_002", "txn_003"]
    NEAR_DUPLICATE = "txn_011"
    REFUND_TXN = "txn_012"
    # Rule-based GT totals recomputed from this task's fixtures
    # (13 txns incl. the txn_002/txn_003 duplicate pair; generator
    # arithmetic was off by 300).
    TOTAL_AMOUNT = 9774.99
    TOTAL_EXCL_DUP = 9475.99

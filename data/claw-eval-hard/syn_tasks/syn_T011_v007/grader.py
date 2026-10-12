"""syn_T011_v007 grader — synthetic variant of T011zh_expense_report."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T011zh_expense_report")


class SyntheticT011Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    EXPECTED_CATEGORIES = {"餐饮": ["txn_001", "txn_005", "txn_009"], "差旅": ["txn_002", "txn_003", "txn_006", "txn_007", "txn_011"], "IT": ["txn_004", "txn_013"], "办公": ["txn_008"], "云服务": ["txn_010"], "退款": ["txn_012"]}
    CATEGORY_ALIASES = {"餐饮": ["餐饮", "餐", "外卖", "咖啡", "宴请", "饮食", "海底捞", "瑞幸", "饿了么", "聚餐"], "差旅": ["差旅", "出行", "交通", "住宿", "机票", "打车", "高德", "希尔顿", "酒店", "航空"], "IT": ["IT", "设备", "键盘", "显示器", "数码", "电子", "苹果", "配件", "京东"], "办公": ["办公", "文具", "得力"], "云服务": ["云服务", "服务器", "云", "阿里"], "退款": ["退款", "负", "退回", "冲抵", "refund"]}
    DUPLICATE_PAIR = ["txn_002", "txn_003"]
    NEAR_DUPLICATE = "txn_011"
    REFUND_TXN = "txn_012"
    TOTAL_AMOUNT = 12704.99
    TOTAL_EXCL_DUP = 12644.99

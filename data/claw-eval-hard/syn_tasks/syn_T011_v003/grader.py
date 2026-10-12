"""syn_T011_v003 grader — synthetic variant of T011zh_expense_report."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T011zh_expense_report")


class SyntheticT011Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    EXPECTED_CATEGORIES = {"餐饮": ["txn_001", "txn_008", "txn_009"], "差旅": ["txn_004", "txn_005", "txn_006", "txn_012", "txn_013"], "IT": ["txn_002", "txn_010"], "办公": ["txn_007"], "云服务": ["txn_003"], "退款": ["txn_011"]}
    DUPLICATE_PAIR = ["txn_004", "txn_005"]
    NEAR_DUPLICATE = "txn_006"
    REFUND_TXN = "txn_011"
    TOTAL_AMOUNT = 7279.9
    TOTAL_EXCL_DUP = 7229.9

    _CATEGORIZATION_RUBRIC = """评估agent对费用交易分类和汇总的准确性。

应正确分类：餐饮 txn_001、txn_008、txn_009；IT txn_002、txn_010；云服务 txn_003；办公 txn_007；差旅 txn_004、txn_005、txn_006、txn_012、txn_013；退款 txn_011。

高分答案应给出结构化分类，并正确说明全部交易原始合计为 7279.90 元。错误归类、遗漏交易或把退款当作普通支出应降低分数。"""

    _ANOMALY_DETECTION_RUBRIC = """评估agent对费用异常和提交总额的识别。

- txn_004 与 txn_005 是 2026-03-05 的曹操出行 50.00 元精确重复项，必须只排除其中一笔。
- txn_006 是 49.90 元的返程差旅，金额相近但不是重复项，必须保留。
- txn_011 是 -200.00 元退款，必须作为贷项处理。
- 排除一笔精确重复项后的报告总额必须为 7229.90 元。

高分要求准确区分精确重复、近似但独立的交易和退款，并给出正确提交总额。"""

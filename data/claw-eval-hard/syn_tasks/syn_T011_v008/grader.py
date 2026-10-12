"""syn_T011_v008 grader — synthetic variant of T011zh_expense_report."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T011zh_expense_report")


class SyntheticT011Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    EXPECTED_CATEGORIES = {"餐饮": ["txn_001", "txn_005", "txn_009"], "差旅": ["txn_002", "txn_003", "txn_006", "txn_007", "txn_011"], "IT": [
        "txn_004", "txn_013"], "办公": ["txn_008"], "云服务": ["txn_010"], "退款": ["txn_012"]}
    CATEGORY_ALIASES = {"餐饮": ["餐饮", "餐", "外卖", "咖啡", "宴请", "饮食"], "差旅": ["差旅", "出行", "交通", "住宿", "机票", "打车"], "IT": [
        "IT", "设备", "键盘", "显示器", "数码", "电子"], "办公": ["办公", "文具"], "云服务": ["云服务", "服务器", "云"], "退款": ["退款", "负", "退回", "冲抵", "refund"]}
    DUPLICATE_PAIR = ["txn_002", "txn_003"]
    NEAR_DUPLICATE = "txn_011"
    REFUND_TXN = "txn_012"
    # Rule-based GT totals recomputed from this task's fixtures
    # (13 txns incl. the txn_002/txn_003 duplicate pair; generator
    # arithmetic was off by 100).
    TOTAL_AMOUNT = 13448.99
    TOTAL_EXCL_DUP = 13398.99
    _CATEGORIZATION_RUBRIC = """评估 agent 对 13 笔交易的分类准确度。

正确的分类：
- 餐饮：txn_001（海底捞）、txn_005（咖啡）、txn_009（外卖）
- 差旅：txn_002（打车）、txn_003（打车）、txn_006（酒店）、txn_007（机票）、txn_011（打车）
- IT 设备：txn_004（键盘）、txn_013（显示器）
- 办公：txn_008（文具）
- 云服务：txn_010（服务器）
- 退款：txn_012（退款，负金额）

评分标准：
- 0.9-1.0: 全部或绝大部分交易分类正确，分类体系合理
- 0.7-0.8: 大部分分类正确，个别有误或分类维度略有不同但合理
- 0.4-0.6: 部分分类正确但有较多错误或遗漏
- 0.1-0.3: 仅少量分类正确或分类混乱
- 0.0: 完全没有分类"""
    _ANOMALY_DETECTION_RUBRIC = """评估 agent 识别和处理异常交易的能力。

三类异常交易需要识别：

1. 完全重复：txn_002 和 txn_003
   - 同一日期、同一商家（高德打车）、同一金额（50 元）
   - 应标记为重复，提交时排除其一

2. 近似重复：txn_011
   - 与 txn_002/003 类似（也是高德打车、也是打车），但金额为 49.99（差 0.01 元）
   - 应指出差异，可能是返程的正常交易，不应自动合并

3. 退款交易：txn_012
   - 金额为 -350 元（负数）
   - 应识别为退款/冲抵，而非普通消费

评分标准：
- 0.9-1.0: 三类异常全部正确识别并恰当处理
- 0.7-0.8: 识别了重复和退款，但近似重复处理不够清晰
- 0.4-0.6: 只识别了一两类异常
- 0.1-0.3: 仅简单提及，没有实质性分析
- 0.0: 未识别任何异常交易"""

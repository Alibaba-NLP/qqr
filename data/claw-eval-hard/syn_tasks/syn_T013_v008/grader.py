"""syn_T013_v008 grader — synthetic variant of T013zh_meeting_notes."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T013zh_meeting_notes")


class SyntheticT013Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    ACTION_ITEMS = [{"assignee": "吴测试", "task": "bug", "deadline": "周五"}, {"assignee": "陈工", "task": "技术评审", "deadline": "下周二"}, {"assignee": "林设计师", "task": "影响范围评估", "deadline": "下周一"}, {"assignee": "陈工", "task": "ERP", "deadline": None}, {"assignee": "刘总监", "task": "需求评估", "deadline": "下周"}]
    VALID_SHARE_RECIPIENTS = ["刘总监", "陈工", "林设计师", "吴测试"]
    REQUIRED_PARTICIPANTS = ["刘总监", "陈工", "林设计师", "吴测试"]
    _CONTENT_EXTRACTION_RUBRIC = """评估 agent 从会议记录中提取行动项和关键信息的质量。
请分别评估以下三个部分，然后按权重计算最终加权分（0-1 之间）。

=== 第一部分：行动项提取（权重 60%）===
应该提取的 5 个行动项：
1. 吴测试：修复 bug，截止周五
2. 陈工：技术评审，截止下周二
3. 林设计师：影响范围评估，截止下周一
4. 陈工：ERP 相关任务
5. 刘总监：需求评估，截止下周

每个行动项应包含：负责人、任务内容、截止日期（如有）。

本部分评分标准：
- 0.9-1.0: 全部 5 个行动项均正确提取，负责人和截止日期完整
- 0.7-0.8: 提取了 4 个行动项，或 5 个但部分缺少截止日期
- 0.5-0.6: 提取了 3 个行动项
- 0.3-0.4: 提取了 1-2 个行动项
- 0.0-0.2: 几乎没有提取行动项

=== 第二部分：遗留项识别（权重 20%）===
- 林设计师的"更新品牌指南"是上周会议 (note_004) 的遗留任务，仍在进行中
- agent 应识别出这是未完成的遗留工作并在整理中提及

本部分评分标准：
- 0.9-1.0: 明确识别了遗留项，说明了它来自上周会议且仍未完成
- 0.4-0.8: 提到了品牌指南相关内容但未明确标注为遗留/未完成
- 0.0-0.3: 完全没有提及遗留项

=== 第三部分：排除闲聊内容（权重 20%）===
- 笔记中有闲聊内容（学 Go、画展、狗、沙发、cosplay 等）
- 好的整理应该只包含工作相关的行动项，不应把闲聊当作待办

本部分评分标准：
- 1.0: 完全没有将闲聊内容列为行动项或待办
- 0.5: 提到了闲聊内容但做了区分（如标注为"非工作相关"）
- 0.0: 将闲聊内容（学 Go、画展等）作为正式行动项列出

请输出最终的加权分数：score = 0.6×第一部分 + 0.2×第二部分 + 0.2×第三部分"""

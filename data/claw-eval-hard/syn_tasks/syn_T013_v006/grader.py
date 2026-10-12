"""syn_T013_v006 grader — synthetic variant of T013zh_meeting_notes."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T013zh_meeting_notes")


class SyntheticT013Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    ACTION_ITEMS = [
        {"assignee": "吴伟", "task": "设计稿终稿确认", "deadline": "周五之前"},
        {"assignee": "陈静", "task": "评审材料", "deadline": "下周三"},
        {"assignee": "周敏", "task": "竞品分析模板样本测试", "deadline": "下周一之前"},
        {"assignee": "陈静", "task": "CRM 集成技术可行性分析", "deadline": "本周内"},
        {"assignee": "刘总", "task": "XYZ 合作评估文档", "deadline": "下周"},
    ]
    VALID_SHARE_RECIPIENTS = {"刘总", "陈静", "吴伟", "周敏"}
    REQUIRED_PARTICIPANTS = {"刘总", "陈静", "吴伟", "周敏"}

    _CONTENT_EXTRACTION_RUBRIC = """评估agent从会议记录中提取行动项、按时间顺序核对状态和排除非工作内容的质量。

=== 第一部分：行动项提取（权重60%）===
应从 note_001 和 note_002 提取以下5项：
1. 吴伟：周五之前完成设计稿终稿确认并邮件通知团队。
2. 陈静：准备下周三评审材料。
3. 周敏：下周一之前完成竞品分析模板的样本测试。
4. 陈静：本周内完成 CRM 集成技术可行性分析。
5. 刘总：下周向 XYZ 发送正式合作评估文档。
每项均应包含负责人、任务和截止时间。全部正确为高分；遗漏、错配负责人或截止时间应扣分。

=== 第二部分：历史遗留项的时间顺序（权重20%）===
note_004 中周敏的渠道联络列表曾是进行中，但更新的 note_001 明确说她已经处理完成。高分答案应说明这是已完成的历史遗留项，不能将其错误报告为当前未完成待办。

=== 第三部分：排除闲聊（权重20%）===
note_003 的摄影、钓鱼、宠物和团建讨论不是工作待办。高分答案不应把这些闲聊列为行动项。

输出最终加权分数：score = 0.6×第一部分 + 0.2×第二部分 + 0.2×第三部分"""

"""syn_T003_v001 grader — synthetic variant."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T003zh_calendar_scheduling")


class SyntheticT003Grader(_Base):
    """Synthetic variant with correct busy slots for attendees."""

    # Busy slots extracted from fixture calendar events for the manager (张经理)
    USER_BUSY = [
        (9, 0, 9, 30),
        (11, 0, 11, 30),
        (13, 30, 14, 0),
        (15, 0, 16, 0),
    ]
    ATTENDEE1_BUSY = [(9, 0, 9, 30), (16, 30, 17, 30)]  # 刘洋
    ATTENDEE2_BUSY = [(9, 0, 9, 30), (13, 0, 13, 30), (17, 0, 18, 0)]  # 李明

    ATTENDEE_NAMES = ['刘洋', '李明']
    TITLE_KEYWORDS = ['项目同步', '同步会', '周会', '例会', '推进会', '协作会', '对齐会']
    TARGET_DATE = "2026-03-02"

    _SCHEDULING_ANALYSIS_RUBRIC = """\
评估agent的排期分析质量以及创建的事件是否使用了合适的标题。

一、排期分析（主要考察）：
1. 展示各参会人的日程冲突情况（提到刘洋、李明各自什么时间有会议/忙碌）
2. 说明哪些时间段是所有人都空闲的
3. 解释为什么选择了最终的时间槽（如"14:30-15:30所有人都空闲"）
4. 提及关键的日程事件作为避让依据

核心考察：agent是否展示了"看冲突→找空闲→选最优"的完整分析逻辑，\
而非直接给出一个时间而不解释。

二、事件标题（次要考察）：
- 创建的日历事件标题是否与"项目同步会"相关
- 可接受的标题如：项目同步会、项目同步、同步会议、团队同步等
- 标题应该体现"项目同步"这一主题

评分标准：
- 0.9-1.0: 完整排期分析（冲突、空闲、选择理由），且事件标题合适
- 0.7-0.8: 排期分析基本完整但有遗漏，标题合适
- 0.5-0.6: 排期分析不完整，或标题不太相关
- 0.2-0.4: 仅简单提及时间，缺乏分析过程
- 0.0-0.1: 完全没有排期分析"""

    def _is_valid_slot_all_attendees(self, start_str, end_str):
        """Check if the proposed time doesn't conflict with any attendee."""
        try:
            start = self._parse_dt(start_str)
            end = self._parse_dt(end_str)
        except (ValueError, TypeError):
            return False

        from datetime import datetime as _dt
        target = _dt.fromisoformat(self.TARGET_DATE)
        if start.month != target.month or start.day != target.day:
            return False
        if start.hour < 9 or end.hour > 18 or (end.hour == 18 and end.minute > 0):
            return False

        duration_min = (end - start).total_seconds() / 60
        if duration_min < 55:
            return False

        slot_start = start.hour * 60 + start.minute
        slot_end = end.hour * 60 + end.minute

        all_busy = [self.USER_BUSY, self.ATTENDEE1_BUSY, self.ATTENDEE2_BUSY]
        for busy_schedule in all_busy:
            if _Base._slot_conflicts(slot_start, slot_end, busy_schedule):
                return False
        return True

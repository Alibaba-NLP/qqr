"""syn_T015_v001 grader — synthetic variant of T015zh_kb_search."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T015zh_kb_search")


class SyntheticT015Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    # set required: peer grade() computes `articles_read & RELEVANT_ARTICLES`
    # (set intersection); a list would raise TypeError and zero the score.
    RELEVANT_ARTICLES = {"kb_101", "kb_103", "kb_105", "kb_106", "kb_107"}
    VARIANT_CROSS_REFERENCE_PAIR = {"kb_107", "kb_103"}
    _SYNTHESIS_RUBRIC = """评估 agent 从多篇知识库文章中综合信息和发现矛盾的能力。
请分别评估以下两个部分，然后按权重计算最终加权分（0-1 之间）。

=== 第一部分：信息综合质量（权重 50%）===
agent 应从多篇文章中综合内部系统问题排查的关键信息：
- SecureTunnel/NetGuard 客户端的使用方法
- MFA 多因素认证的要求
- 密码/账号相关的排查步骤
- 防火墙/网络配置相关信息
- access.futuretech.io 连接地址
- 钉钉等辅助工具

本部分评分标准：
- 0.9-1.0: 综合了 5 个以上关键信息点，形成完整的排查清单
- 0.7-0.8: 综合了 3-4 个关键信息点，基本覆盖主要排查步骤
- 0.5-0.6: 综合了 2-3 个信息点，但遗漏了重要内容
- 0.3-0.4: 仅提及 1-2 个信息点，综合不足
- 0.0-0.2: 几乎没有综合多篇文章的信息

=== 第二部分：矛盾信息识别（权重 50%）===
知识库中存在关键矛盾：
- kb_101 推荐使用 SecureTunnel 作为客户端
- kb_106 通知 SecureTunnel 将被 NetGuard 替代（迁移中）
- agent 应明确指出这一矛盾/更新，建议用户使用 NetGuard 而非 SecureTunnel

本部分评分标准：
- 0.9-1.0: 明确识别了 SecureTunnel→NetGuard 的迁移/替换，给出了正确建议
- 0.7-0.8: 提到了两种客户端，暗示有变更但未明确说明矛盾
- 0.4-0.6: 仅列出了两种客户端名称，未分析矛盾关系
- 0.1-0.3: 只提到一种客户端，未发现矛盾
- 0.0: 完全没有涉及客户端的信息

请输出最终的加权分数：score = 0.5×第一部分 + 0.5×第二部分"""

    def grade(
        self, messages, dispatches, task, audit_data=None, judge=None,
        media_events=None, env_snapshot=None,
    ):
        scores = super().grade(
            messages, dispatches, task, audit_data, judge, media_events,
            env_snapshot,
        )
        if scores.safety == 0.0:
            return scores

        articles_read = {
            dispatch.request_body.get("article_id")
            for dispatch in dispatches
            if dispatch.tool_name == "kb_get_article" and dispatch.response_status < 400
        }
        all_text = self._get_all_assistant_text(messages)
        inherited_cross_reference = (
            {"kb_007", "kb_003"} <= articles_read
            or ("kb_007" in all_text and "kb_003" in all_text)
        )
        variant_cross_reference = (
            self.VARIANT_CROSS_REFERENCE_PAIR <= articles_read
        )
        if inherited_cross_reference:
            scores.completion = max(0.0, scores.completion - 0.10)
        if variant_cross_reference:
            scores.completion = min(1.0, scores.completion + 0.10)
        return scores

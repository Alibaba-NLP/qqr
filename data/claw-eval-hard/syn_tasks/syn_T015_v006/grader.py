"""syn_T015_v006 grader — synthetic variant of T015zh_kb_search."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T015zh_kb_search")


class SyntheticT015Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    # set required: peer grade() computes `articles_read & RELEVANT_ARTICLES`
    # (set intersection); a list would raise TypeError and zero the score.
    RELEVANT_ARTICLES = {"kb_001", "kb_003", "kb_005", "kb_006", "kb_007"}
    _SYNTHESIS_RUBRIC = """评估 agent 从多篇知识库文章中综合信息和发现矛盾的能力。
请分别评估以下两个部分，然后按权重计算最终加权分（0-1 之间）。

=== 第一部分：信息综合质量（权重 50%）===
agent 应从多篇文章中综合 VPN 问题排查的关键信息：
- FortiClient/GlobalProtect VPN 客户端的使用方法
- MFA 多因素认证的要求
- 密码/账号相关的排查步骤
- 防火墙/网络配置相关信息
- vpn.aurora.tech 连接地址
- 企业微信等辅助工具

本部分评分标准：
- 0.9-1.0: 综合了 5 个以上关键信息点，形成完整的 VPN 排查清单
- 0.7-0.8: 综合了 3-4 个关键信息点，基本覆盖主要排查步骤
- 0.5-0.6: 综合了 2-3 个信息点，但遗漏了重要内容
- 0.3-0.4: 仅提及 1-2 个信息点，综合不足
- 0.0-0.2: 几乎没有综合多篇文章的信息

=== 第二部分：矛盾信息识别（权重 50%）===
知识库中存在关键矛盾：
- kb_001 推荐使用 FortiClient 作为 VPN 客户端
- kb_006 通知 FortiClient 将被 GlobalProtect 替代（迁移中）
- agent 应明确指出这一矛盾/更新，建议用户使用 GlobalProtect 而非 FortiClient

本部分评分标准：
- 0.9-1.0: 明确识别了 FortiClient→GlobalProtect 的迁移/替换，给出了正确建议
- 0.7-0.8: 提到了两种客户端，暗示有变更但未明确说明矛盾
- 0.4-0.6: 仅列出了两种客户端名称，未分析矛盾关系
- 0.1-0.3: 只提到一种客户端，未发现矛盾
- 0.0: 完全没有涉及 VPN 客户端的信息

请输出最终的加权分数：score = 0.5×第一部分 + 0.5×第二部分"""

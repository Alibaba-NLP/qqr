"""postcheck_judge：claw-eval 校准后“安全与忠实性复查器”。

与共享 RTI 流水线接口一致（agents/postcheck.py）：

    build(user_prompt, history, original, corrected, tools, label=None)

判断“校准后的这一步”是否应被接受。最重要的职责是抓住 corrector_draft / deleak 可能漏掉的**泄漏**——
尤其是 “根据反馈，我需要：1)… 2)…” / “Based on feedback, I need to: 1)…” 这种照抄监督清单、暴露看过
反馈的开头。校准思路借鉴 tau2_bench，但面向 claw 的中文 groundtruth 参考，并给出中英双语的泄漏词表。
"""

import json


def build(user_prompt, history, original, corrected, tools=None, label: dict = None) -> str:
    label = label or {}
    groundtruth = label.get("groundtruth") or ""
    tools_block = json.dumps(tools, ensure_ascii=False, indent=2) if tools else "[]"
    original_block = (
        json.dumps(original, ensure_ascii=False, indent=2)
        if not isinstance(original, str) else original
    )
    corrected_block = (
        json.dumps(corrected, ensure_ascii=False, indent=2)
        if not isinstance(corrected, str) else corrected
    )

    return f"""
你是 claw-eval 多服务沙箱智能体的“校准后安全与忠实性复查器”。请判断【校准后回复】是否应被接受。

可见信息 = 用户任务 + 工具列表 + 对话历史（含工具返回）。校准后回复中不允许出现任何其他来源的信息。

[用户任务]
{user_prompt}

[参考解法 —— 仅供你复查判断，校准后回复绝不能泄漏它]
{groundtruth}

[可用工具]
{tools_block}

[Agent 对话历史（可见）]
{history}

[原回复]
{original_block}

[校准后回复]
{corrected_block}

以下任一成立即判定 ok=false：
1) 泄漏（leak）。**只在以下两种“确凿”情况判泄漏，不要仅凭语气/自信去推断**：
   a) 明确提及先验来源：反馈 / 监督 / 指导 / 提示 / 参考 / 参考解法 / 答案 / 标准答案 / 评估或评分标准 /
      “先前 / 上一版草稿”等（中英皆算：“根据反馈”“参考解法”“based on feedback”“the feedback”
      “reference”“groundtruth”“the correct answer”…），或照抄监督清单式开头（“**问题识别：**”“**FIX：**”
      “我需要：1)… 2)…”这种逐条搬运监督修复清单）。
   b) 使用了“可见历史（含工具返回）里根本没出现”的具体事实——特定人名 / 邮箱 / 记录ID / 数值 / 约束——
      且恰好与参考解法一致。判断依据是“该事实是否出现在可见历史里”，而不是语气是否自信。
2) 伪造工具结果（fabricated_tool_result）：声称某个历史中并不存在的具体工具返回值 / 结果。
3) 参数错误（wrong_tool_arguments）：参数与可见历史矛盾，或类型错误（如需要布尔 true 却用字符串 "True"），
   或忽略了历史中已明确给出的必需取值。
4) 校准不合理（unreasonable_calibration）：不是合理的下一步，或相比原回复没有改进。

以下情况**不要**判为问题（属于校准的预期行为）：
- **自信但可由可见历史推导的推理**：例如“我已定位根因是 X”“为符合 Y 策略需错开 Z”“enabled 应为布尔 true”——
  只要 X / Y / Z（根因、策略、约束、取值）能在可见历史（工单 / 记录 / KB / 工具返回 / schema）中找到依据，
  就属于正常的胜任表现，**不算泄漏**，绝不要因为语气自信、结论明确而判 ok=false。
- 补上任务隐含的合理下一步（如在更新/删除等写操作前先向用户确认；先查询记录再行动）。
- 合规的谨慎行为（“我先核实一下……”）。
- 与参考不同但可由可见历史推导、且不含任何泄漏的有效路径。

只输出 JSON：
{{
  "ok": true/false,
  "reasons": ["leak", "fabricated_tool_result", "wrong_tool_arguments", "unreasonable_calibration"],
  "diagnosis": "若 ok=false：指出确切问题并引用泄漏/违规的具体片段；若 ok=true：一句话说明为何可接受。"
}}

**判泄漏要“看内容、不看语气”：只有当回复明确提到先验来源（反馈 / 参考 / 参考解法 / 答案 / 评估标准 /
上一版草稿），或使用了可见历史中不存在的具体事实（人名 / 邮箱 / ID / 数值）时，才判 ok=false；对可由可见
历史推导出来的自信推理（哪怕结论明确、语气笃定），一律不要误判为泄漏。**
""".strip()

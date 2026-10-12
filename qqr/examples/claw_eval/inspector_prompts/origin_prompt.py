"""origin_prompt：claw-eval 反事实基线（不含任何先验信息）。

共享 RTI 流水线在 qqr/off_policy/rti/inspector.py 中会导入 ``{prompt_base}.origin_prompt``，
因此必须存在。签名与 corrector_draft 一致：

    build(history, original_response, feedback_json, label, tools, output_format="json")

它精确复刻智能体在正常 rollout 中所见——只有可见对话历史 + 工具，绝无任何先验（无 groundtruth、
无反馈）。校准后的这一步会同时在“本基线 prompt”和“校准器 prompt”下打分，供 ref-KL gate 度量校准
引入的概率偏移。因此这里必须保持干净，不含任何参考 / 反馈内容。
"""

import json
from typing import List


def _output_instruction(output_format: str) -> str:
    if output_format == "tool_call_xml":
        return (
            "# 输出格式（Qwen3.5 工具调用 XML）\n"
            "<think>你对“当前这一步”的推理</think>\n"
            "<tool_call><function=工具名>\n<parameter=参数名>\n值\n</parameter>\n</function></tool_call>\n"
            "若直接回复用户，则只输出 <think>...</think> 与回复正文。"
        )
    if output_format == "tool_call_json":
        return (
            "# 输出格式\n"
            "<think>你对“当前这一步”的推理</think>\n"
            '<tool_call>{"name": "工具名", "arguments": {"参数": "值"}}</tool_call>'
        )
    return (
        "# 输出格式\n"
        "只输出 JSON：\n"
        "{\n"
        '  "reasoning": "你对当前状态与下一步最佳行动的推理",\n'
        '  "tool_calls": [{"name": "工具名", "arguments": {"参数": "值"}}],\n'
        '  "content": "给用户的回复正文"\n'
        "}"
    )


def build(
    history: str,
    original_response=None,
    feedback_json: str = "",
    label: dict = None,
    tools: List = None,
    output_format: str = "json",
) -> str:
    tools_block = json.dumps(tools, ensure_ascii=False, indent=2) if tools else "[]"
    return f"""
你是 claw-eval 多服务沙箱智能体（crm / helpdesk / calendar / contacts / gmail / kb / scheduler，
Qwen3.5 工具调用）。请基于目前的对话历史，给出“当前这一步”的行动。

[对话历史]
{history}

[可用工具]
{tools_block}

{_output_instruction(output_format)}
""".strip()

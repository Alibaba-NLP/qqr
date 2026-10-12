import json
from typing import Any, Dict, List

import json
from typing import Any, Dict, List

def build(history: str, original_response: dict, feedback_json: str, label: dict, tools: List, output_format: str) -> str:
    # 提取真实信息用于内部逻辑判断（但仍不在 prompt 中直接暴露）
    gt = label.get("groundtruth")
    ref = label.get("ref") or []
    order_str = label.get("order") or ""

    output_instruction = """
    <tool_call>{{"name": "tool_name1", "arguments": {{"arg1": "value1"}}}}</tool_call>
    <tool_call>{{"name": "tool_name2", "arguments": {{"arg1": "value1"}}}}</tool_call>
    """
    if output_format == "tool_call_xml":
        output_instruction = """
        <tool_call><function=name><parameter=k>v</parameter></function></tool_call>
        <tool_call><function=name><parameter=k>v</parameter></function></tool_call>
        """

    return f"""
        # 输入说明

        [历史对话]
        {history}

        [可用工具]
        {json.dumps(tools, ensure_ascii=False, indent=4)}

        # 按照以下格式输出：
        - 最后一轮对话之前你必须根据用户当前信息、用户请求、历史交互信息做出推理和行动，按照以下格式输出：
        <think>你的推理和下一步行动的计划
        </think>
        {output_instruction}
        - 要求最后一次工具调用必须调用但此direction工具，规划好origin，destination以及waypoints。
        - 最后一轮对话可直接回答。
        """
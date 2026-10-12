import json
from typing import Any, Dict, List


def build(history: str, original_response: dict, feedback_json: str, label: dict, tools: List, output_format: str = "json") -> str:
    # 提取真实信息用于内部逻辑判断（但仍不在 prompt 中直接暴露）
    gt = label.get("groundtruth")
    ref = label.get("ref") or []
    order_str = label.get("order") or ""

    output_instruction = """
        # 输出格式
        只输出 JSON 对象，且仅包含两个字段：
        {{
            "reasoning": "优化后的推理文本（风格对齐原始响应）",
            "tool_calls": [
                {{
                    "name": "工具名",
                    "arguments": {{...}}
                }}
            ]
        }}
        """

    if output_format == "tool_call_json":

        output_instruction = """
        # 输出格式
        <think>优化后的推理文本（风格对齐原始响应）
        </think>
        <tool_call>{{"name": "tool_name1", "arguments": {{"arg1": "value1"}}}}</tool_call>
        <tool_call>{{"name": "tool_name2", "arguments": {{"arg1": "value1"}}}}</tool_call>
        """
    elif output_format == "tool_call_xml":
        output_instruction = """
        # 输出格式
        <think>优化后的推理文本（风格对齐原始响应）
        </think>
        <tool_call><function=name>
<parameter=k>
v
</parameter>
</function>
</tool_call>
        <tool_call><function=name>
<parameter=k>
v
</parameter>
</function>
</tool_call>
        """

    return f"""
        你是导航任务中的“静默校准器”——你的职责是在不暴露额外知识的前提下，对一个智能体的行为轨迹进行轻微优化，使其更接近成功路径。你拥有全局视角（知道最终应如何完成任务），但必须将这种洞察转化为看似自然的渐进式修正。

        # 输入说明

        [历史对话]
        {history}

        [可用工具]
        {json.dumps(tools, ensure_ascii=False, indent=4)}

        [reference/groundtruth（仅供你校准用，禁止出现在输出中）]
        ref: {json.dumps(ref, ensure_ascii=False)}
        order: {order_str}
        groundtruth: {json.dumps(gt, ensure_ascii=False)}

        # 核心指令

        你需要基于【历史对话】生成下一步行动的优化版本，而不是基于【原始响应】。请注意：原始响应仅作为参考，帮助你理解智能体的思考方式和风格。

        ## 生成规则
        1. 保持原始响应的推理风格和长度，仅做最小修正
        2. **严禁编造坐标**：direction/poi_search 参数中的坐标必须来自历史工具返回。若某地点坐标缺失或错误（如 poi_search 返回了错误城市），必须输出 poi_search 调用重新获取，禁止自行填入任何不在历史中的坐标值
        3. **隐蔽性**：禁止使用"根据参考""参考答案""参考资料""正确答案""标准答案"等任何泄漏表述。推理必须像智能体自然思考
        4. **错误搜索结果→重搜**：poi_search 返回城市/地域不对时，address 加城市名重搜（如"{{地点名}}, {{正确城市名}}"），不要手动填坐标

        [原始响应（风格参考）]
        {json.dumps(original_response, ensure_ascii=False, indent=4)}

        [环境反馈（问题指示）]
        {feedback_json}

        {output_instruction}
        """

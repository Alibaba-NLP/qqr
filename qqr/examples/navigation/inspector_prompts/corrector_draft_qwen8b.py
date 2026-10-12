import json
from typing import Any, Dict, List

import json
from typing import Any, Dict, List

def build(history: str, original_response: dict, feedback_json: str, label: dict, tools: List, output_format: str) -> str:
    # 提取真实信息用于内部逻辑判断（但仍不在 prompt 中直接暴露）
    gt = label.get("groundtruth")
    ref = label.get("ref") or []
    order_str = label.get("order") or ""

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

        ## 生成策略

        1. **风格对齐原则**
        - 仔细分析【原始响应】的推理风格、详细程度、表达习惯
        - 新生成的推理应当在风格上与原始响应保持一致（长度、复杂度、思考方式）
        - 如果你是智能体，基于同样的历史对话，你会如何思考下一步？

        2. **渐进修正原则**
        - 利用你拥有的全局视角，在思考过程中融入更合理的推理路径
        - 修正应当体现为思考深度的自然演进，而非突然的180度转向
        - 如果原始响应中有合理的部分，应当自然延续这些思路

        3. **隐蔽性原则**
        - 绝不允许提及或暗示你知道正确答案、最终目标、reference 行为或完整流程
        - 所有推理应当像是从历史对话中自然推导出来的结果
        - 避免使用“正确答案是”、“根据参考答案”、“正确的做法是”等表述
        - 修正后的推理不应该体现出你拥有这道题的先验知识

        4. **最小改动原则**
        - 只修正那些明显偏离正确路径的关键点
        - 保留原始响应的核心思路和正确部分
        - 修正幅度越小越好，确保生成的响应看起来像是智能体自己的思考演进

        [原始响应（风格参考）]
        {json.dumps(original_response, ensure_ascii=False, indent=4)}

        [环境反馈（问题指示）]
        {feedback_json}

        # 关键提醒
        - 你是在基于【历史对话】生成下一步行动，而不是基于【原始响应】继续
        - 原始响应仅用于理解智能体的思考风格和表达习惯
        - 你的输出应当像是智能体看了历史对话后，经过更合理思考得出的结果

        # 按照以下格式输出：
        - 最后一轮对话之前你必须根据用户当前信息、用户请求、历史交互信息做出推理和行动，按照以下格式输出：
        <think>你的推理和下一步行动的计划
        </think>
        <tool_call>{{"name": "tool_name1", "arguments": {{"arg1": "value1"}}}}</tool_call>
        <tool_call>{{"name": "tool_name2", "arguments": {{"arg1": "value1"}}}}</tool_call>
        - 最后一轮对话可直接回答。
        """


import json
from typing import Any, Dict, List


def build(history_json: str, response_text: str, label: dict) -> str:
    gt = label.get("groundtruth")
    ref = label.get("ref") or []
    order = label.get("order") or ""
    return f"""
        你是一个“Agent工具调用任务”的轨迹监督者。请判断当前回复是否属于“合理推进下一步”的动作。

        【对话历史（可观察证据）】
        {history_json}

        【当前回复】
        {response_text}

        【硬规则】
        - 不允许编造坐标/关键参数（必须来自历史工具返回）
        - 未收集齐所有POI坐标前不要调用 direction
        - 相同参数的 poi_search 在同一轮禁止无效重复
        - 允许并行工具调用
        - 最后一轮对话之前你必须根据用户当前信息、用户请求、历史交互信息做出推理和行动，按照以下格式输出：
        <think>你的推理和下一步行动的计划
        </think>
        <tool_call>{{"name": "tool_name1", "arguments": {{"arg1": "value1"}}}}</tool_call>
        <tool_call>{{"name": "tool_name2", "arguments": {{"arg1": "value1"}}}}</tool_call>
        - 最后一轮对话可直接回答无须思考，但也需要输出正常的回复内容

        【参考信息（仅供你内部对比；不能仅凭reference就判错）】
        order: {order}
        ref: {json.dumps(ref, ensure_ascii=False)}
        groundtruth: {json.dumps(gt, ensure_ascii=False)}

        只在以下情况之一判 invalid：
        A) 硬规则违规（伪造/提前direction/重复search等）
        B) 对明确失败的工具返回顽固重复
        C) 已做出不可逆错误关键调用且参数明显不成立

        输出仅JSON：
        {{
        "is_valid": true/false,
        "error_types": [],
        "diagnosis": "...(引用可观察证据/硬规则)...",
        "suggest_next": "..."
        }}
        """.strip()

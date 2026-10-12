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
        - 允许且鼓励并行工具调用：一轮内可同时发起多个 poi_search

        【轮次预算（关键）】
        - 全程只有 6 轮（当前轮次 = 历史中 assistant 回复数 + 1），轮次极易耗尽
        - 还有多个地点待查时，本轮必须并行搜索；逐轮只搜一个会耗尽轮次，最终无法调用 direction
        - 某次 poi_search 返回地域明显不符或为空时，应改写 query/region 重搜该地点，而不是沿用错误参数搜下一个

        【参考信息（仅供你内部对比；不能仅凭reference就判错，但可用于 diagnosis/suggest_next 指出参数问题）】
        order: {order}
        ref: {json.dumps(ref, ensure_ascii=False)}
        groundtruth: {json.dumps(gt, ensure_ascii=False)}

        只在以下情况之一判 invalid：
        A) 硬规则违规（伪造/提前direction/重复search等）
        B) 对明确失败的工具返回顽固重复，或沿用明显错误的搜索参数（region/query 与历史返回矛盾）不纠正
        C) 已做出不可逆错误关键调用且参数明显不成立
        D) 轮次浪费：还有多个地点待查时，本轮只串行搜一个而不并行，将导致轮次耗尽

        输出仅JSON：
        {{
        "is_valid": true/false,
        "error_types": [],
        "diagnosis": "...(引用可观察证据/硬规则)...",
        "suggest_next": "..."
        }}
        """.strip()

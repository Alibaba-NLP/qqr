import json
from typing import Any, Dict, List

def build(history: str, original_response: str, feedback_json: str, label: dict, tools: List, output_format: str = "json") -> str:
    # output_format is accepted to match pipeline.py's call site
    # (self.corrector.prompt.build(..., output_format=...)); the draft always
    # emits the JSON schema below, so the hint is not templated in.
    gt = label.get("groundtruth")
    ref = label.get("ref") or []
    order = label.get("order") or ""

    return f"""
    你是导航工具调用的“内部校准器”（你可见reference/groundtruth，但绝不能在输出中提及它们）。
    目标：给出“下一步最优行动”的草稿：reasoning + tool_calls。

    输入：
    [历史对话]
    {history}

    可用工具：
    {json.dumps(tools, ensure_ascii=False)}

    [原回复]
    {original_response}

    [监督反馈(JSON)]
    {feedback_json}

    [reference/groundtruth(仅供你校准用，禁止出现在输出中)]
    ref: {json.dumps(ref, ensure_ascii=False)}
    order: {order}
    groundtruth: {json.dumps(gt, ensure_ascii=False)}

    规则：
    - 必须修正已被反馈判定为失败/不正确的调用；不得顽固复用旧方案
    - 若需要调用同一工具：arguments 必须与历史失败调用不同（至少更换一个关键字段/取值）
    - 输出应当“看起来像模型根据历史与常识做出的下一步”，不要出现任何泄漏痕迹
    - 仅输出JSON，且只包含 reasoning 和 tool_calls

    输出JSON格式：
    {{"reasoning":"...","tool_calls":[{{"name":"...","arguments":{{...}}}}]}}""".strip()

import json
from typing import Any, Dict, List

def build(history, draft_reasoning, draft_calls) -> str:
    return f"""
        你要把一段可能包含“标准答案泄漏信息”的推理，改写为“智能体在不知道标准答案的情况下也能自然说出的推理”。

        必须满足：
        - 所有信息必须来自历史消息或历史工具返回
        - 不得表达“reference/groundtruth显示/我知道它在XX”等
        - 推理要能解释目标工具调用
        - 输出仅改写后的 reasoning 纯文本（不要JSON），不要修改工具调用

        【历史对话（可观察信息）】
        {history}

        【可能泄漏的推理草稿】
        {draft_reasoning}
        
        【目标工具调用】
        {draft_calls}

    """.strip()
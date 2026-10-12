import json
from typing import Any, Dict, List

def build(history, draft_reasoning, draft_calls) -> str:
    return f"""

        【历史对话（可观察信息）】
        {history}

        【推理草稿】
        {draft_reasoning}
        
        【目标工具调用】
        {draft_calls}

    """.strip()

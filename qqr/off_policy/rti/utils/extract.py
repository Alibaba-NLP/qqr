import re
from typing import Any, Dict, List

from qqr.data.prompts.qwen import QwenPrompt

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_prompter = QwenPrompt()


def extract_tool_calls_from_text(txt: str) -> List[Dict[str, Any]]:
    """Extract tool calls from Qwen3/Qwen3.5 style response text."""
    txt = txt or ""
    calls: List[Dict[str, Any]] = []
    for m in _TOOL_CALL_RE.finditer(txt):
        result = _prompter._parse_tool_block(m.group(1).strip())
        if result is not None:
            calls.append({
                "name": result["function"]["name"],
                "arguments": result["function"]["arguments"],
            })
    return calls

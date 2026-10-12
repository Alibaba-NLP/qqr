import json
from typing import Any, Dict, List

def json_sanitize(x: Any) -> Any:
    try:
        json.dumps(x, ensure_ascii=False)
        return x
    except Exception:
        return str(x)

def strip_sensitive_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for c in tool_calls or []:
        out.append({"name": c.get("name"), "arguments": dict(c.get("arguments") or {})})
    return out

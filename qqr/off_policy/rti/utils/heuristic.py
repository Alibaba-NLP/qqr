# qqr/off_policy/rti/utils/heuristic.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from .extract import extract_tool_calls_from_text
from .text import COORD_RE


def _extract_poi_queries(tool_calls: List[Dict[str, Any]]) -> List[str]:
    qs: List[str] = []
    for c in tool_calls or []:
        if c.get("name") == "poi_search":
            args = c.get("arguments") or {}
            q = args.get("address") or args.get("query") or ""
            if isinstance(q, str) and q.strip():
                qs.append(q.strip())
    return qs


def _has_duplicate_poi_search(tool_calls: List[Dict[str, Any]]) -> bool:
    qs = _extract_poi_queries(tool_calls)
    return len(qs) != len(set(qs))


def _direction_args_match_gt(tool_call: Dict[str, Any], gt: Optional[Dict[str, Any]]) -> bool:
    if not gt:
        return True
    if tool_call.get("name") != gt.get("name"):
        return False
    return (tool_call.get("arguments") or {}) == (gt.get("arguments") or {})


def is_obviously_bad(
    *,
    messages: List[Dict[str, Any]],
    tool_calls: List[Dict[str, Any]],
    gt: Optional[Dict[str, Any]],
    strict_direction_match: bool,
    tools: Optional[List[Dict[str, Any]]],
) -> Tuple[bool, Dict[str, Any]]:
    history_has_tool_return = any(m.get("role") in (
        "tool", "function") for m in (messages or []))

    if not tool_calls:
        return False, {"reason": "no_tool_calls"}

    if tools and any(t['name'] not in tools for t in tool_calls):
        return True, {"reason": "tool_call_not_in_agent_tool_list"}

    if _has_duplicate_poi_search(tool_calls):
        return True, {"reason": "duplicate_poi_search_same_turn"}

    for c in tool_calls:
        if c.get("name") == "direction":
            args = c.get("arguments") or {}
            if not args.get("origin") or not args.get("destination"):
                return True, {"reason": "direction_missing_origin_or_destination"}
            if not history_has_tool_return:
                return True, {"reason": "direction_called_without_grounded_tool_returns"}
            if strict_direction_match and gt and not _direction_args_match_gt(c, gt):
                return True, {"reason": "direction_not_equal_groundtruth_strict"}

    return False, {"reason": "ok"}


def bootstrap_with_groundtruth(gt: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    if not gt or gt.get("name") != "direction":
        return None
    return [{"name": "direction", "arguments": gt.get("arguments") or {}}]


def heuristic_postcheck(
    *,
    corrected_response: str,
    correction_mode: str,
    postcheck_allow_gt_bootstrap: bool,
    messages: List[Dict[str, Any]],
) -> Tuple[bool, Dict[str, Any]]:
    meta: Dict[str, Any] = {"ok": True, "reasons": [], "mode": correction_mode}

    if correction_mode == "gt_bootstrap" and postcheck_allow_gt_bootstrap:
        return True, {"ok": True, "reasons": ["allow_gt_bootstrap_bypass"], "mode": correction_mode}

    txt = corrected_response or ""
    low = txt.lower()

    leak_keywords = [
        "groundtruth", "reference", "标准答案", "标答", "参考答案",
        "参考信息", "参考资料", "参考数据", "根据参考",
        "依据参考", "参照参考", "据参考", "参考给出", "参考提供",
        "我看到了", "根据ref", "根据reference", "根据groundtruth",
    ]
    if any(k.lower() in low for k in leak_keywords):
        meta["ok"] = False
        meta["reasons"].append("leak_keyword")

    return meta["ok"], meta

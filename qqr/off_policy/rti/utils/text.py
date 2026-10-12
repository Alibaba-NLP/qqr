# qqr/off_policy/rti/utils/text.py
import re
import time

COORD_RE = re.compile(r"(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)")

def now_ts() -> float:
    return time.time()

def truncate(s: str, max_chars: int = 40960) -> str:
    s = s or ""
    return s if len(s) <= max_chars else (s[:max_chars] + " ...[truncated]")

def strip_code_fence(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()

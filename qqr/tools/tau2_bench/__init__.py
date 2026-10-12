"""
tau2-bench tool executor for qqr.

Provides:
  - get_environment(domain)   – stateful tau2 Environment for one episode
  - execute_tool_call(env, tool_call_dict)  – async wrapper
  - get_tool_definitions(domain)  – OpenAI function-calling schema list (agent tools)
  - get_user_tool_definitions(domain)  – OpenAI function-calling schema list (user tools)
  - DOMAINS                   – list of supported domain names
  - _get_env_constructor(domain) – internal: returns env constructor callable
"""

import sys as _sys

# ── Fix dual-import: make bare 'tau2' resolve to the embedded copy ────────────
# tau2 may be pip-installed externally. The embedded evaluators do
# 'from tau2.xxx import Yyy'.  If those resolve to the external copy while
# _build_tau2_task creates objects from the embedded copy, pydantic will reject
# them as different types.  We redirect sys.modules so 'tau2' always points
# to our embedded copy.
from . import tau2 as _embedded_tau2

_stale = [k for k in _sys.modules if k == 'tau2' or k.startswith('tau2.')]
for _k in _stale:
    del _sys.modules[_k]
_sys.modules['tau2'] = _embedded_tau2

# Also register already-loaded submodules under bare 'tau2.*' names
_prefix = 'qqr.tools.tau2_bench.tau2.'
for _key in list(_sys.modules.keys()):
    if _key.startswith(_prefix):
        _bare = 'tau2.' + _key[len(_prefix):]
        _sys.modules[_bare] = _sys.modules[_key]
# ──────────────────────────────────────────────────────────────────────────────

from .executor import (
    DOMAINS,
    _get_env_constructor,
    execute_tool_call,
    get_environment,
    get_tool_definitions,
    get_user_tool_definitions,
)

__all__ = [
    "DOMAINS",
    "_get_env_constructor",
    "execute_tool_call",
    "get_environment",
    "get_tool_definitions",
    "get_user_tool_definitions",
]

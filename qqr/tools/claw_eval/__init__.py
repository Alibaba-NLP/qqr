"""claw-eval tool adapter for qqr.

Bridges the external claw-eval benchmark repo (``CLAW_EVAL_ROOT``) into the
qqr rollout pipeline. claw-eval ships its own agent tool dispatch, sandbox
runner, mock-service manager and simulated user agent under
``claw_eval.runner.*``; this package makes those importable (via a single,
idempotent ``sys.path`` injection) and exposes thin lazy accessors so the
rollout / reward code does not have to manage import paths directly.

Used by:
  * qqr.examples.claw_eval.config   – sandbox config loading + path injection
  * qqr.examples.claw_eval.rollout  – tool dispatch / sandbox / service / user-agent
"""

from .bridge import (
    ensure_claw_eval_on_path,
    get_claw_eval_root,
    get_sandbox_runner_cls,
    get_sandbox_tool_dispatcher_cls,
    get_sandbox_tool_names,
    get_sandbox_tools,
    get_service_manager_cls,
    get_tool_dispatcher_cls,
    get_user_agent_cls,
    load_sandbox_config,
)

__all__ = [
    "ensure_claw_eval_on_path",
    "get_claw_eval_root",
    "get_sandbox_runner_cls",
    "get_sandbox_tool_dispatcher_cls",
    "get_sandbox_tool_names",
    "get_sandbox_tools",
    "get_service_manager_cls",
    "get_tool_dispatcher_cls",
    "get_user_agent_cls",
    "load_sandbox_config",
]

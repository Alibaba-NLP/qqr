"""claw-eval tool bridge.

The claw-eval benchmark (an external repo pointed to by ``CLAW_EVAL_ROOT``)
ships its own agent tooling under ``claw_eval.runner.*``:

  * ``ToolDispatcher``            – HTTP/API tool dispatch
  * ``SandboxToolDispatcher``     – wraps HTTP dispatch + E2B sandbox tools
  * ``E2BSandboxRunner``          – sandbox lifecycle (start / inject files / stop)
  * ``ServiceManager``            – mock service (API) lifecycle
  * ``UserAgent``                 – simulated user for multi-turn (C-type) tasks
  * ``SANDBOX_TOOLS`` / ``SANDBOX_TOOL_NAMES``

Rather than let the rollout / reward code manage ``sys.path`` and import paths
directly, this bridge centralises the integration:

  * :func:`ensure_claw_eval_on_path` injects ``CLAW_EVAL_ROOT/src`` into
    ``sys.path`` exactly once (idempotent, safe to call from any Ray worker).
  * The ``get_*`` accessors lazily import the corresponding claw-eval class so
    the (heavy) claw-eval dependencies are only loaded inside rollout workers.

All accessors are lazy on purpose: importing this module must stay cheap and
must not require the claw-eval repo to be present at import time.
"""

import os
import sys
from pathlib import Path

__all__ = [
    "get_claw_eval_root",
    "ensure_claw_eval_on_path",
    "load_sandbox_config",
    "get_service_manager_cls",
    "get_sandbox_runner_cls",
    "get_user_agent_cls",
    "get_tool_dispatcher_cls",
    "get_sandbox_tool_dispatcher_cls",
    "get_sandbox_tools",
    "get_sandbox_tool_names",
]

# The claw-eval repo lives outside this project; point CLAW_EVAL_ROOT at a
# checkout (must contain src/, normally also tasks/ fixtures/ config_*.yaml).
_CLAW_EVAL_ROOT_ENV = "CLAW_EVAL_ROOT"


def get_claw_eval_root() -> Path:
    """Return the claw-eval repo root.

    ``CLAW_EVAL_ROOT`` wins when set; otherwise fall back to the checkout of an
    installed ``claw_eval`` package (editable installs point at
    ``<root>/src/claw_eval``). Raises when neither is available.
    """
    env_root = os.getenv(_CLAW_EVAL_ROOT_ENV)
    if env_root:
        return Path(env_root)
    try:
        import importlib.util

        spec = importlib.util.find_spec("claw_eval")
        if spec is not None and spec.origin:
            return Path(spec.origin).resolve().parents[2]
    except Exception:
        pass
    raise RuntimeError(
        "CLAW_EVAL_ROOT is not set and claw_eval is not importable. Export it "
        "to a claw-eval checkout (containing src/), e.g.: "
        "export CLAW_EVAL_ROOT=/path/to/claw-eval"
    )


def ensure_claw_eval_on_path() -> str:
    """Add ``CLAW_EVAL_ROOT/src`` to ``sys.path`` (idempotent).

    Returns the src path that was ensured. Safe to call repeatedly and from
    multiple Ray workers.
    """
    src = str(get_claw_eval_root() / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


def load_sandbox_config(config_name: str = "config_multimodal.yaml"):
    """Load claw-eval's sandbox config section from a repo-root yaml file."""
    ensure_claw_eval_on_path()
    from claw_eval.config import load_config

    cfg = load_config(get_claw_eval_root() / config_name)
    return cfg.sandbox


# ── Lazy class / value accessors ─────────────────────────────────

def get_service_manager_cls():
    ensure_claw_eval_on_path()
    from claw_eval.runner.services import ServiceManager

    return ServiceManager


def get_sandbox_runner_cls():
    ensure_claw_eval_on_path()
    from claw_eval.runner.sandbox_runner import E2BSandboxRunner

    return E2BSandboxRunner


def get_user_agent_cls():
    ensure_claw_eval_on_path()
    from claw_eval.runner.user_agent import UserAgent

    return UserAgent


def get_tool_dispatcher_cls():
    ensure_claw_eval_on_path()
    from claw_eval.runner.dispatcher import ToolDispatcher

    return ToolDispatcher


def get_sandbox_tool_dispatcher_cls():
    ensure_claw_eval_on_path()
    from claw_eval.runner.sandbox_dispatcher import SandboxToolDispatcher

    return SandboxToolDispatcher


def get_sandbox_tools():
    ensure_claw_eval_on_path()
    from claw_eval.runner.sandbox_tools import SANDBOX_TOOLS

    return SANDBOX_TOOLS


def get_sandbox_tool_names():
    ensure_claw_eval_on_path()
    from claw_eval.runner.sandbox_tools import SANDBOX_TOOL_NAMES

    return SANDBOX_TOOL_NAMES

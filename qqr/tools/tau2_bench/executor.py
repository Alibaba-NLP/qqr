"""
tau2_bench tool executor for qqr.

Uses embedded tau2 source code (qqr/tools/tau2_bench/tau2/) instead of
external installation, making the project self-contained.

Key responsibilities:
  1. Provide per-episode stateful tau2 Environment instances (one per rollout).
  2. Execute agent tool calls against the live environment (sync I/O wrapped async).
  3. Expose OpenAI function-calling schemas for all four domains.

Design notes:
  - Each rollout episode needs its OWN environment (fresh DB copy).
    Callers must call get_environment(domain) at the start of every episode
    and keep the returned object alive for that episode's lifetime.
  - execute_tool_call() is the main dispatch function; it wraps the sync
    tau2 Environment.get_response() in an executor so the asyncio loop
    stays unblocked.
  - get_tool_definitions(domain) returns the OpenAI-compatible list of tool
    schemas that should be placed in sample.train_metadata["tools"].
"""

import asyncio
import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Root directory that contains the per-sample RL DB files.
# Individual samples store db_path as "tau2_rl_database/<file>.json"
# (relative to this root), so resolved = _RL_DATABASE_ROOT / db_path is
# always correct.  The qqr repo ships these files at:
#   data/tau2_bench/tau2_rl_database/<file>.json
# Resolution order:
#   1. TAU2_RL_DATABASE_ROOT environment variable, if set (custom layouts).
#   2. <repo root>/data/tau2_bench, next to the qqr python package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_RL_DATABASE_ROOT = Path(
    os.environ.get("TAU2_RL_DATABASE_ROOT", _REPO_ROOT / "data" / "tau2_bench")
)

# ── Domain constants ──────────────────────────────────────────────────────────

DOMAINS = ["airline", "retail", "telecom", "mock"]


# ── Local tau2 imports (bare 'tau2.*' redirected to embedded copy by __init__.py)

def _get_env_constructor(domain: str):
    """Return the get_environment callable for the given domain."""
    from tau2.domains.airline.environment import get_environment as airline_env
    from tau2.domains.retail.environment import get_environment as retail_env
    from tau2.domains.telecom.environment import get_environment as telecom_env
    from tau2.domains.mock.environment import get_environment as mock_env

    if domain == "airline":
        return airline_env
    elif domain == "retail":
        return retail_env
    elif domain == "telecom":
        return telecom_env
    elif domain == "mock":
        return mock_env
    else:
        raise ValueError(f"Unknown tau2-bench domain: {domain!r}. "
                         f"Supported: {DOMAINS}")


def get_environment(domain: str, db_path: Optional[str] = None):
    """
    Create and return a fresh stateful tau2 Environment for the given domain.

    Each call returns a NEW environment with its own DB copy – safe to use
    for a single rollout episode.  Do NOT share across parallel rollouts.

    Args:
        domain: One of DOMAINS (airline, retail, telecom, mock).
        db_path: Optional relative path to a specific DB file
            (e.g. "tau2_rl_database/tau2_airline_new_db_1.json").
            The path is resolved relative to _RL_DATABASE_ROOT.
            When None, the domain's default embedded DB is used.
    """
    from tau2.domains.airline.data_model import FlightDB
    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.telecom.data_model import TelecomDB

    constructor = _get_env_constructor(domain)

    if db_path is None:
        return constructor()

    # Resolve the absolute path to the custom DB file
    resolved = _RL_DATABASE_ROOT / db_path
    if not resolved.exists():
        logger.warning(
            f"[tau2-bench] db_path '{db_path}' not found at '{resolved}'. "
            "Falling back to default DB."
        )
        return constructor()

    # Load domain-specific DB from the custom path and pass it in
    try:
        if domain == "airline":
            db = FlightDB.load(str(resolved))
            return constructor(db=db)
        elif domain == "retail":
            db = RetailDB.load(str(resolved))
            return constructor(db=db)
        elif domain == "telecom":
            db = TelecomDB.load(str(resolved))
            return constructor(db=db)
        else:
            # mock and others: no custom DB support, fall back
            return constructor()
    except Exception as e:
        logger.warning(
            f"[tau2-bench] Failed to load custom db_path '{db_path}': {e}. "
            "Falling back to default DB."
        )
        return constructor()


# ── Tool definition cache ─────────────────────────────────────────────────────

_TOOL_DEF_CACHE: dict[str, list[dict]] = {}
_USER_TOOL_DEF_CACHE: dict[str, list[dict]] = {}


def get_tool_definitions(domain: str) -> list[dict]:
    """
    Return the OpenAI function-calling schema list for the given domain (agent tools).

    The result is cached after the first call (schemas are static per domain).
    """
    if domain not in _TOOL_DEF_CACHE:
        env = get_environment(domain)
        tools = env.get_tools()
        _TOOL_DEF_CACHE[domain] = [t.openai_schema for t in tools]
    return _TOOL_DEF_CACHE[domain]


def get_user_tool_definitions(domain: str) -> list[dict]:
    """
    Return the OpenAI function-calling schema list for user tools in the given domain.
    
    Returns empty list if the domain has no user tools.
    The result is cached after the first call.
    """
    if domain not in _USER_TOOL_DEF_CACHE:
        try:
            env = get_environment(domain)
            user_tools = env.get_user_tools()
            _USER_TOOL_DEF_CACHE[domain] = [t.openai_schema for t in user_tools]
        except (ValueError, AttributeError):
            # Domain has no user tools
            _USER_TOOL_DEF_CACHE[domain] = []
    return _USER_TOOL_DEF_CACHE[domain]

def _get_tool_schema(env, tool_name: str, requestor: str) -> Optional[dict]:
    """Get the OpenAI schema for a tool by name."""
    try:
        if requestor == "user":
            tools = env.get_user_tools()
        else:
            tools = env.get_tools()
        
        for tool in tools:
            if tool.name == tool_name:
                return tool.openai_schema
    except Exception:
        pass
    return None


def _execute_tool_call_sync(env, tool_call_dict: dict) -> dict:
    """
    Execute a single tool call dict against a live tau2 Environment (sync).

    tool_call_dict has the OpenAI format:
      {
        "id": "call_xxx",
        "type": "function",
        "function": {"name": "...", "arguments": "{...}" | {...}},
        "requestor": "assistant" | "user"  # Optional, defaults to "assistant"
      }

    Returns a qqr-style tool-role message dict.
    """
    from tau2.data_model.message import ToolCall

    tool_name = tool_call_dict["function"]["name"]
    tool_call_id = tool_call_dict.get("id", "call_0")
    # Support both "requestor" field and infer from context
    requestor = tool_call_dict.get("requestor", "assistant")

    if tool_name == 'find_user_id_by_name_zip':
        print()

    # Parse arguments
    try:
        args_raw = tool_call_dict["function"].get("arguments", "{}")
        if isinstance(args_raw, dict):
            arguments = args_raw
        else:
            arguments = json.loads(args_raw) if args_raw else {}
    except (json.JSONDecodeError, TypeError) as e:
        return {
            "role": "tool",
            "content": f"[Error] Invalid JSON arguments: {e}",
            "tool_call_id": tool_call_id,
        }

    # Build a tau2 ToolCall and dispatch through env.get_response()
    try:
        tc = ToolCall(
            id=tool_call_id,
            name=tool_name,
            arguments=arguments,
            requestor=requestor,
        )
        tool_message = env.get_response(tc)
        content = tool_message.content or ""
    except Exception as e:
        logger.exception(f"[tau2-bench] Tool '{tool_name}' execution error: {e}")
        content = f"[Error] Tool '{tool_name}' failed: {e}"

    return {
        "role": "tool",
        "content": content,
        "tool_call_id": tool_call_id,
    }


async def execute_tool_call(env, tool_call_dict: dict) -> dict:
    """
    Async wrapper for executing a single tool call against the given environment.

    Runs the sync tau2 execution in a thread-pool executor so the asyncio
    event loop is not blocked.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _execute_tool_call_sync, env, tool_call_dict
    )

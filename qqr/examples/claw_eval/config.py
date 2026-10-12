import os
from pathlib import Path
from types import SimpleNamespace

__all__ = [
    "claw_eval_root",
    "tasks_dir",
    "tasks_dirs",
    "fixtures_dir",
    "config_path",
    "mock_services_dir",
    "service_cwd",
    "max_steps",
    "timeout_seconds",
    "sandbox_tools",
    "max_images_per_rollout",
    "screenshot_max_images",
    "rollout_dump_root",
]

# ── claw-eval paths ──────────────────────────────────────────────
# Code root: an external claw-eval checkout (CLAW_EVAL_ROOT, or the checkout
# of an installed claw_eval package). tasks/ fixtures/ config_*.yaml are read
# from it unless overridden below; mock_services/ is vendored (see below).
from qqr.tools.claw_eval.bridge import get_claw_eval_root

claw_eval_root = get_claw_eval_root()

# Support multiple task directories via colon-separated CLAW_EVAL_TASKS_DIR.
# If unset, defaults to <claw_eval_root>/tasks.
_tasks_dir_env = os.getenv("CLAW_EVAL_TASKS_DIR")
if _tasks_dir_env:
    tasks_dirs = [Path(p) for p in _tasks_dir_env.split(os.pathsep) if p]
else:
    tasks_dirs = [claw_eval_root / "tasks"]
tasks_dir = tasks_dirs[0]

fixtures_dir = Path(os.getenv("CLAW_EVAL_FIXTURES_DIR", str(claw_eval_root / "fixtures")))
config_path = Path(os.getenv("CLAW_EVAL_CONFIG_PATH", str(claw_eval_root / "config_general.yaml")))

# Task services (task.yaml `services:`) are spawned with commands like
# `python mock_services/<name>/server.py`, resolved against the service cwd.
# Use the vendored copy under qqr/tools/claw_eval/mock_services; fall back to
# the checkout's own mock_services/ only if the vendored copy is missing.
_vendored_mock_services_dir = (
    Path(__file__).resolve().parents[2] / "tools" / "claw_eval" / "mock_services"
)
if _vendored_mock_services_dir.is_dir():
    mock_services_dir = _vendored_mock_services_dir
else:
    mock_services_dir = claw_eval_root / "mock_services"
service_cwd = mock_services_dir.parent


def resolve_task_dir(task_id: str) -> Path | None:
    """Return the task directory containing ``task.yaml`` for ``task_id``.

    Searches all directories configured via ``CLAW_EVAL_TASKS_DIR``.  Matching
    is exact: if the task ID is not found, ``None`` is returned.
    """
    for root in tasks_dirs:
        d = root / task_id
        if (d / "task.yaml").exists():
            return d
    return None


def resolve_grader_path(task_id: str) -> Path | None:
    """Return the path to ``grader.py`` for ``task_id``."""
    task_dir = resolve_task_dir(task_id)
    if task_dir is None:
        return None
    return task_dir / "grader.py"

# ── Agent loop parameters ────────────────────────────────────────
max_steps = 40
timeout_seconds = 1800
sandbox_tools = True

# ── Multimodal image budget ────────────────────────────────────
max_images_per_rollout = 32

# Whether to feed screenshots/images from tool results into the model as
# multimodal inputs. Set CLAW_EVAL_MULTIMODAL_INPUT_ENABLED=0 to force a
# unimodal (text-only) rollout even when tools return images.
MULTIMODAL_INPUT_ENABLED = os.environ.get(
    "CLAW_EVAL_MULTIMODAL_INPUT_ENABLED", "true"
).strip().lower() in ("1", "true", "yes", "on")

# ── Output artifact collection ──────────────────────────────────
artifact_max_chars = 20000

# ── Tool result truncation ──────────────────────────────────────
# Cap the per-tool-call text that is inserted back into the conversation.
# Long OCR / document-extraction results (hundreds of thousands of chars)
# otherwise exhaust the rollout context budget and force TRUNCATED samples.
# Overridable via CLAW_TOOL_RESULT_MAX_CHARS.
tool_result_max_chars = int(
    os.environ.get("CLAW_TOOL_RESULT_MAX_CHARS", "10000")
)

qqr_path = Path(os.getenv("QQR_PATH", str(Path(__file__).resolve().parents[3])))

# ── Rollout data dump dir ───────────────────────────────────────
# Train/eval rollout samples are dumped here for offline inspection.
# Overridable via env; defaults under the current project's outputs/.
# When TASK_NAME indicates a claw-eval-hard experiment, default to the
# split output directory so dumps don't collide with regular claw-eval runs.
_task_name = os.getenv("TASK_NAME", "")
if "claw-eval-hard" in _task_name:
    _default_rollout_dump_root = str(qqr_path / "outputs" / "claw-eval-hard" / "rollout_dumps")
else:
    _default_rollout_dump_root = str(qqr_path / "outputs" / "claw-eval" / "rollout_dumps")
rollout_dump_root = Path(
    os.getenv("CLAW_ROLLOUT_DUMP_ROOT", _default_rollout_dump_root)
)

# ── Official grader judge (eval_grader_adapter._get_judge) ────────
# Let the yaml point the OFFICIAL claw-eval grader's LLM judge at a fast local
# vLLM endpoint instead of the default DashScope qwen3.7-max. Read at call time
# (Ray workers get env vars late). When base_url/model are unset, _get_judge
# falls back to claw-eval's config_general.yaml.
def _get_grader_judge_base_url():
    return os.getenv("GRADER_JUDGE_BASE_URL")

def _get_grader_judge_api_key():
    return os.getenv("GRADER_JUDGE_API_KEY")

def _get_grader_judge_model():
    return os.getenv("GRADER_JUDGE_MODEL")

def _get_grader_judge_concurrency():
    # Bounds the thread pool used to run the (blocking) official grader
    # concurrently across trajectories / eval trials.
    return int(os.getenv("GRADER_JUDGE_CONCURRENCY", "16"))

# ── Simulated user agent (UserAgent) ──────────────────────────────
# Let the yaml point the claw-eval simulated user at a configurable
# OpenAI-compatible endpoint instead of the previous hard-coded local URL.
# Read at call time (Ray workers get env vars late).
def _get_user_agent_base_url():
    return os.getenv("USER_AGENT_BASE_URL", "")

def _get_user_agent_api_key():
    return os.getenv("USER_AGENT_API_KEY", "EMPTY")

def _get_user_agent_model():
    return os.getenv("USER_AGENT_MODEL", "Qwen3.5-397B-A17B")

def _ensure_claw_path():
    # Delegate to the shared qqr.tools.claw_eval bridge so sys.path injection stays
    # centralised (see qqr/tools/claw_eval/bridge.py).
    from qqr.tools.claw_eval import ensure_claw_eval_on_path
    ensure_claw_eval_on_path()

_sandbox_config_cache = None

def _get_sandbox_config():
    global _sandbox_config_cache
    if _sandbox_config_cache is None:
        from qqr.tools.claw_eval import load_sandbox_config
        _sandbox_config_cache = load_sandbox_config("config_multimodal.yaml")
    return _sandbox_config_cache

_user_agent_config_cache = None

def _get_user_agent_config():
    global _user_agent_config_cache
    if _user_agent_config_cache is None:
        _user_agent_config_cache = SimpleNamespace(
            api_key=_get_user_agent_api_key(),
            base_url=_get_user_agent_base_url(),
            model_id=_get_user_agent_model(),
            context_window=128000,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
    return _user_agent_config_cache

# def _get_user_agent_config():
#     global _user_agent_config_cache
#     if _user_agent_config_cache is None:
#         _ensure_claw_path()
#         from claw_eval.config import load_config
#         cfg = load_config(claw_eval_root / "config_user_agent.yaml")
#         _user_agent_config_cache = cfg.user_agent_model
#     return _user_agent_config_cache

screenshot_max_images = 4


# ─── RTI Configuration ───
RTI_ENABLED = os.environ.get("RTI_ENABLED", "false").lower() == "true"
RTI_REPLACE_NUM = int(os.environ.get("RTI_REPLACE_NUM", "2"))
RTI_SUCCESS_THRESHOLD = float(os.environ.get("RTI_SUCCESS_THRESHOLD", "0.5"))


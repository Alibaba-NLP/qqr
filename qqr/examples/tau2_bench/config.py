"""config for tau2_bench rollout."""

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "max_steps",
    "DOMAINS",
]

# ── Episode control ───────────────────────────────────────────────────────────

max_steps = 15  # max agent turns per episode (each turn = one assistant response)

# ── Domain list ───────────────────────────────────────────────────────────────

DOMAINS = ["airline", "retail", "telecom", "mock"]

# ── Policy paths ──────────────────────────────────────────────────────────────
# Absolute paths to policy files for each domain.

import pathlib as _pl

_DATA_ROOT = _pl.Path(__file__).resolve().parent.parent.parent / "tools" / "tau2_bench" / "data" / "tau2" / "domains"

# Values can be a single Path or a list of Paths.
# For telecom, two policy files are concatenated (mirrors TelecomEnvironment.get_environment()):
#   <main_policy>...</main_policy><tech_support_policy>...</tech_support_policy>
POLICY_PATHS = {
    "airline": _DATA_ROOT / "airline" / "policy.md",
    "retail":  _DATA_ROOT / "retail"  / "policy.md",
    "telecom": [
        _DATA_ROOT / "telecom" / "main_policy.md",
        _DATA_ROOT / "telecom" / "tech_support_manual.md",
    ],
    "mock":    _DATA_ROOT / "mock"    / "policy.md",
}

# ── User Simulator LLM config ─────────────────────────────────────────────────
# Separate LLM for user simulation (generates customer requests and replies).
# Supply the endpoint via --user-sim-openai-base-url (see run scripts).

user_sim_openai_base_url = ""
user_sim_openai_api_key = "EMPTY"
user_sim_model = "Qwen3.5-397B-A17B"
user_sim_temperature = 0.7
user_sim_max_new_tokens = 4096
user_sim_timeout = 60

# User simulator retry config
user_sim_max_retries = 3
user_sim_retry_base_backoff = 2
user_sim_retry_max_backoff = 30

# ── Judge LLM config (for NL assertion evaluation) ─────────────────────────────
# Optional: for evaluating NL assertions in reward computation

judge_model = "gpt-4o-mini"
judge_openai_base_url = ""
judge_openai_api_key = ""
judge_temperature = 0.0
judge_max_new_tokens = 1024
judge_timeout = 60

# ── Inspector config (mirrors aime) ──────────────────────────────────────────

@dataclass
class InspectorConfig:
    enabled: bool = False
    use_judge_llm: bool = False
    openai_base_url: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_new_tokens: int = 800
    timeout: int = 60
    log_dir: Optional[str] = None
    allow_gt_bootstrap: bool = True
    strict_direction_match: bool = False
    enable_postcheck: bool = True
    postcheck_allow_gt_bootstrap: bool = True
    postcheck_use_judge_llm: bool = True
    max_retries: int = 10
    retry_base_backoff: int = 10
    retry_max_backoff: int = 300

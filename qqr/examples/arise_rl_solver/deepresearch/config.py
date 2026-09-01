"""
DeepResearch - Reward-Gated Reverse KL (RG-KL) solver training configuration

Core algorithm:
- Three-phase rollout: student (k_s=8, no memory) -> the LLM coach produces memory -> teacher (k_m=4, with memory)
- Teachers do not enter the PPO loss; they only contribute the scalar Δ_r = mean(r_m) - mean(r_s)
- Student samples carry guided_tokens = [p_m_prompt, τ_s_response], which triggers slime's
  _compute_guided_teacher_log_probs forward pass yields π_θ(τ_s | p_m) as the KL teacher
- λ(Δ_r) = λ_0 · gate(Δ_r) · warmup · cosine_decay
- L = L_GRPO_student + λ(Δ_r) · D_KL(π_θ(·|p_s) ‖ π_θ(·|p_m)) + β · KL_ref_penalty

At deployment the student needs no memory: self-distillation has internalised the memory-guided behaviour into its weights.
"""

import os

from qqr.mcp import MCPServerManager, MCPServerStdioCacheable, MCPServerStdioParams
from qqr.utils.envs import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    SEARCH_API_KEY,
    SEARCH_API_URL,
    PYTHONPATH,
    SERPER_API_KEY,
)

__all__ = [
    "max_steps",
    "tool_response_max_chars",
    "mcp_manager",
    "llm_judge_api_key",
    "llm_judge_base_url",
    "llm_judge_model",
    "llm_judge_concurrency_limit",
    "enable_rg_kl",
    "k_student",
    "k_teacher",
    "rg_kl_lambda_0",
    "rg_kl_delta_threshold",
    "rg_kl_gate_temperature",
    "rg_kl_warmup_iters",
    "rg_kl_decay_iters",
    "rg_kl_min_lambda_ratio",
    "enable_guided_tokens_for_student",
    "kl_final_only",
]


# ============ MCP server settings (Google Search only) ============

google_search_server_params = MCPServerStdioParams(
    command="python",
    args=["-m", "qqr.tools.google_search"],
    env={
        "PYTHONPATH": PYTHONPATH,
        "SERPER_API_KEY": SERPER_API_KEY or "",
        "SEARCH_API_KEY": SEARCH_API_KEY or "",
        "SEARCH_API_URL": SEARCH_API_URL or "",
        "SEARCH_BACKEND": os.getenv("SEARCH_BACKEND", "google"),
    },
)
google_search_server = MCPServerStdioCacheable(
    name="GoogleSearch",
    params=google_search_server_params,
    cache_tools_list=True,
    client_session_timeout_seconds=300,
    max_retry_attempts=3,
    blocklist=[],
    cache_ttl=600,
    cache_maxsize=8192,
    concurrency_limit=4,
)

mcp_manager = MCPServerManager([google_search_server], connect_in_parallel=True)


# ============ Solver agent settings ============

max_steps = 40  # the solver gets at most 40 rounds of tool interaction, as in the paper

# ============ Tool response truncation settings ============

tool_response_max_chars = int(os.environ.get("TOOL_RESPONSE_MAX_CHARS", "7500"))


# ============ LLM judge settings (rubric scoring + coach) ============

llm_judge_api_key = DASHSCOPE_API_KEY
llm_judge_base_url = DASHSCOPE_BASE_URL
llm_judge_model = os.environ.get("LLM_JUDGE_MODEL", "gpt-5.2-2025-12-11")
llm_judge_concurrency_limit = int(os.environ.get("LLM_JUDGE_CONCURRENCY_LIMIT", "10"))


# ============ Reward-Gated Reverse KL (RG-KL) settings ============

enable_rg_kl = os.environ.get("ENABLE_RG_KL", "true").lower() == "true"

# Three-phase rollout split: n_samples_per_prompt = k_student + k_teacher
k_student = int(os.environ.get("K_STUDENT", "8"))
k_teacher = int(os.environ.get("K_TEACHER", "8"))

# RG-KL gating base coefficient: λ(Δ_r) = λ_0 · gate(Δ_r) · warmup · cosine_decay
rg_kl_lambda_0 = float(os.environ.get("RG_KL_LAMBDA_0", "0.5"))

# Δ_r threshold: apply the KL term only when the memory genuinely helps (Δ_r > δ_thresh)
rg_kl_delta_threshold = float(os.environ.get("RG_KL_DELTA_THRESHOLD", "0.05"))
rg_kl_gate_temperature = float(os.environ.get("RG_KL_GATE_TEMPERATURE", "0.0125"))

# Warmup: for the first N rollouts let GRPO stabilise the student alone, ramping the KL term linearly from 0 to 1
rg_kl_warmup_iters = int(os.environ.get("RG_KL_WARMUP_ITERS", "20"))

# Cosine decay: λ_0 decays to min_lambda_ratio by the end
rg_kl_min_lambda_ratio = float(os.environ.get("RG_KL_MIN_LAMBDA_RATIO", "0.1"))
rg_kl_decay_iters = int(os.environ.get("RG_KL_DECAY_ITERS", "200"))

# safety switch: whether to build [p_m, τ_s_response] guided_tokens for student samples
# when False, degenerates to standard GRPO (no KL distillation)
enable_guided_tokens_for_student = (
    os.environ.get("ENABLE_GUIDED_TOKENS_FOR_STUDENT", "true").lower() == "true"
)

# KL distillation scope: when true, only the final response turn gets the p_m swap; intermediate tool-call turns use identity
# rationale: intermediate turns are tool-call JSON, and applying KL there tends to force the memory's tool usage onto the student,
#   which then mismatches at deployment time where there is no memory; the final turn is a natural-language report, so distilling how it is written after reading the memory
#   can be internalised directly into the deployment path
# when False, fall back to the all-turn scope
kl_final_only = os.environ.get("KL_FINAL_ONLY", "true").lower() == "true"




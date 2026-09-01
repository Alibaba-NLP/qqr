"""
Travel rubrics - Reward-Gated Reverse KL (RG-KL) solver training configuration

Core algorithm:
- Three-phase rollout: student (k_s=8, no memory) -> the LLM coach produces memory -> teacher (k_m=4, with memory)
- Teachers do not enter the PPO loss; they only contribute the scalar Δ_r = mean(r_m) - mean(r_s)
- Student samples carry guided_tokens = [p_m_prompt, τ_s_response], which triggers slime's
  _compute_guided_teacher_log_probs forward pass yields π_θ(τ_s | p_m) as the KL teacher
- λ(Δ_r) = λ_0 · max(Δ_r - δ_thresh, 0) · warmup · cosine_decay
- L = L_GRPO_student + λ(Δ_r) · D_KL(π_θ(·|p_s) ‖ π_θ(·|p_m)) + β · KL_ref_penalty

At deployment the student needs no memory: self-distillation has internalised the memory-guided behaviour into its weights.
"""

import os

from qqr.mcp import MCPServerManager, MCPServerStdioCacheable, MCPServerStdioParams
from qqr.utils.envs import (
    AMAP_MAPS_API_KEY,
    BAILIAN_WEB_SEARCH_API_KEY,
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    PYTHONPATH,
)

__all__ = [
    "max_steps",
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
    "rg_kl_min_lambda",
    "rg_kl_warmup_iters",
    "rg_kl_decay_iters",
]


# ============ MCP server settings ============

amap_server_params = MCPServerStdioParams(
    command="python",
    args=["-m", "qqr.tools.amap"],
    env={"AMAP_MAPS_API_KEY": AMAP_MAPS_API_KEY, "PYTHONPATH": PYTHONPATH},
)
amap_server = MCPServerStdioCacheable(
    name="AMap", params=amap_server_params, cache_tools_list=True,
    client_session_timeout_seconds=60, max_retry_attempts=3, blocklist=[],
    cache_ttl=600, cache_maxsize=8192, concurrency_limit=16,
)

transport_server_params = MCPServerStdioParams(
    command="python",
    args=["-m", "qqr.tools.mock_transport"],
    env={"DASHSCOPE_API_KEY": DASHSCOPE_API_KEY, "DASHSCOPE_BASE_URL": DASHSCOPE_BASE_URL, "PYTHONPATH": PYTHONPATH},
)
transport_server = MCPServerStdioCacheable(
    name="Transport", params=transport_server_params, cache_tools_list=True,
    client_session_timeout_seconds=60, max_retry_attempts=3, blocklist=[],
    cache_ttl=600, cache_maxsize=8192, concurrency_limit=4,
)

web_search_server_params = MCPServerStdioParams(
    command="python",
    args=["-m", "qqr.tools.web_search"],
    env={"BAILIAN_WEB_SEARCH_API_KEY": BAILIAN_WEB_SEARCH_API_KEY, "PYTHONPATH": PYTHONPATH},
)
web_search_server = MCPServerStdioCacheable(
    name="WebSearch", params=web_search_server_params, cache_tools_list=True,
    client_session_timeout_seconds=60, max_retry_attempts=3, blocklist=[],
    cache_ttl=600, cache_maxsize=8192, concurrency_limit=1,
)

mcp_manager = MCPServerManager(
    [amap_server, transport_server, web_search_server], connect_in_parallel=True
)


# ============ Solver agent settings ============

max_steps = 40  # the solver gets at most 40 rounds of tool interaction, as in the paper


# ============ LLM judge settings ============

llm_judge_api_key = DASHSCOPE_API_KEY
llm_judge_base_url = DASHSCOPE_BASE_URL
llm_judge_model = os.environ.get("LLM_JUDGE_MODEL", "gpt-5.2-2025-12-11")
llm_judge_concurrency_limit = int(os.environ.get("LLM_JUDGE_CONCURRENCY_LIMIT", "10"))


# ============ Reward-Gated Reverse KL (RG-KL) settings ============

enable_rg_kl = os.environ.get("ENABLE_RG_KL", "true").lower() == "true"

# Three-phase rollout split: n_samples_per_prompt = k_student + k_teacher
k_student = int(os.environ.get("K_STUDENT", "8"))
k_teacher = int(os.environ.get("K_TEACHER", "8"))

# RG-KL gating base coefficient: λ(Δ_r) = λ_0 · max(Δ_r - δ_thresh, 0) · warmup · cosine_decay
rg_kl_lambda_0 = float(os.environ.get("RG_KL_LAMBDA_0", "0.5"))

# Δ_r threshold: apply the KL term only when the memory genuinely helps (Δ_r > δ_thresh)
rg_kl_delta_threshold = float(os.environ.get("RG_KL_DELTA_THRESHOLD", "0.05"))
rg_kl_gate_temperature = float(os.environ.get("RG_KL_GATE_TEMPERATURE", "0.0125"))

# Warmup: for the first N rollouts let GRPO stabilise the student alone, ramping the KL term linearly from 0 to 1
rg_kl_warmup_iters = int(os.environ.get("RG_KL_WARMUP_ITERS", "20"))

# Cosine decay: λ_0 decays to min_lambda by the end
rg_kl_min_lambda_ratio = float(os.environ.get("RG_KL_MIN_LAMBDA_RATIO", "0.1"))  # = 0.01 / 0.1
rg_kl_decay_iters = int(os.environ.get("RG_KL_DECAY_ITERS", "120"))

# safety switch: whether to build [p_m, τ_s_response] guided_tokens for student samples
# when False, degenerates to standard GRPO (no KL distillation)
enable_guided_tokens_for_student = (
    os.environ.get("ENABLE_GUIDED_TOKENS_FOR_STUDENT", "true").lower() == "true"
)

# KL scope: "all_response" (every turn contributes to the KL) vs "final_only" (only the final reply, skipping tool calls)
# empirically all_response is too strong for multi-turn agents: forcing tool-call tokens to match the memory's arguments degrades performance
kl_scope = os.environ.get("KL_SCOPE", "final_only").lower()
assert kl_scope in ("all_response", "final_only"), f"KL_SCOPE must be all_response or final_only, got: {kl_scope}"

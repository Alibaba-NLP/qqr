"""
VitaBench Reward-Gated Reverse KL (RG-KL) solver training configuration

Core algorithm (ported from arise_rl_solver/travel):
- Three-phase rollout: student (k_s, no memory) -> the LLM coach produces memory -> teacher (k_m, with memory)
- Teachers do not enter the PPO loss; they only contribute Δ_r = mean(r_m) - mean(r_s)
- Student samples carry guided_tokens = [p_m_prompt, τ_s_response], which triggers slime's
  _compute_guided_teacher_log_probs forward pass yields π_θ(τ_s | p_m) as the KL teacher
- λ(Δ_r) = λ_0 · gate(Δ_r) · warmup · cosine_decay
- L = L_GRPO_student + λ(Δ_r) · D_KL(π_θ(·|p_s) ‖ π_θ(·|p_m)) + β · KL_ref_penalty

At deployment the student needs no memory: self-distillation has internalised the memory-guided behaviour into its weights.

VitaBench specifics:
- Interactive mode: the UserSimulator (local Qwen3.5-397B-A17B) simulates the multi-turn user dialogue
- A sliding-window judge (the same model, deployed locally, thinking disabled)
- The coach uses external gpt-5.2 to produce query-specific memory
- max_steps=40 (interactive mode: user dialogue plus tool calls)
- VitaBench OTA-domain task data (environment, instructions and user_profile)
"""

import os

from qqr.utils.envs import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

__all__ = [
    "max_steps",
    "llm_judge_api_key",
    "llm_judge_base_url",
    "llm_judge_model",
    "llm_judge_concurrency_limit",
    "vitabench_data_dir",
    "default_domain",
    "user_simulator_api_key",
    "user_simulator_base_url",
    "user_simulator_model",
    "user_simulator_temperature",
    "coach_api_key",
    "coach_base_url",
    "coach_model",
    "coach_concurrency_limit",
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
]

# -- VitaBench data paths --
vitabench_data_dir = os.environ.get(
    "VITABENCH_DATA_DIR", "/path/to/vitabench/data/vita"  # points at the data/vita directory of the official VitaBench repo
)

# -- Default domain --
default_domain = os.environ.get("VITABENCH_DOMAIN", "ota")

# -- Agent settings --
max_steps = 40  # interactive mode needs more rounds (user dialogue plus tool calls)

# -- UserSimulator settings (multi-turn user simulation; the paper uses Qwen3.5-397B) --
user_simulator_api_key = os.environ.get("USER_SIMULATOR_API_KEY", "EMPTY")
user_simulator_base_url = os.environ.get(
    "USER_SIMULATOR_BASE_URL", DASHSCOPE_BASE_URL
)
user_simulator_model = os.environ.get("USER_SIMULATOR_MODEL", "Qwen3.5-397B-A17B")
user_simulator_temperature = float(os.environ.get("USER_SIMULATOR_TEMPERATURE", "0.0"))

# -- LLM judge (sliding-window rubric scoring; the paper uses gpt-5.2) --
llm_judge_api_key = os.environ.get("LLM_JUDGE_API_KEY", DASHSCOPE_API_KEY or "EMPTY")
llm_judge_base_url = os.environ.get("LLM_JUDGE_BASE_URL", DASHSCOPE_BASE_URL)
llm_judge_model = os.environ.get("LLM_JUDGE_MODEL", "gpt-5.2-2025-12-11")  # the paper uses gpt-5.2 for every judge
llm_judge_concurrency_limit = int(os.environ.get("LLM_JUDGE_CONCURRENCY_LIMIT", "16"))

# -- Coach (memory-guided coaching prompt, external gpt-5.2) --
coach_api_key = os.environ.get("COACH_API_KEY", DASHSCOPE_API_KEY)
coach_base_url = os.environ.get("COACH_BASE_URL", DASHSCOPE_BASE_URL)
coach_model = os.environ.get("COACH_MODEL", "gpt-5.2-2025-12-11")
coach_concurrency_limit = int(os.environ.get("COACH_CONCURRENCY_LIMIT", "10"))


# ============ Reward-Gated Reverse KL (RG-KL) settings ============

enable_rg_kl = os.environ.get("ENABLE_RG_KL", "true").lower() == "true"

# Three-phase rollout split: n_samples_per_prompt = k_student + k_teacher
# defaults to 16 (8 student + 8 teacher), matching G=16 in the paper:
# multi-turn sequences are long and rollouts are expensive, so start with a smaller group
k_student = int(os.environ.get("K_STUDENT", "8"))
k_teacher = int(os.environ.get("K_TEACHER", "8"))

# RG-KL gating base coefficient: λ(Δ_r) = λ_0 · gate(Δ_r) · warmup · cosine_decay
rg_kl_lambda_0 = float(os.environ.get("RG_KL_LAMBDA_0", "0.5"))

# Δ_r threshold: apply the KL term only when the memory genuinely helps (Δ_r > δ_thresh)
rg_kl_delta_threshold = float(os.environ.get("RG_KL_DELTA_THRESHOLD", "0.05"))
rg_kl_gate_temperature = float(os.environ.get("RG_KL_GATE_TEMPERATURE", "0.0125"))

# Warmup: for the first N rollouts let GRPO stabilise the student alone, ramping the KL term linearly from 0 to 1
rg_kl_warmup_iters = int(os.environ.get("RG_KL_WARMUP_ITERS", "20"))

# Cosine decay: λ decays to λ_0 · min_ratio by the end
rg_kl_decay_iters = int(os.environ.get("RG_KL_DECAY_ITERS", "120"))
rg_kl_min_lambda_ratio = float(os.environ.get("RG_KL_MIN_LAMBDA_RATIO", "0.1"))

# safety switch: whether to build [p_m, τ_s_response] guided_tokens for student samples
# when False, degenerates to standard GRPO (no KL distillation)
enable_guided_tokens_for_student = (
    os.environ.get("ENABLE_GUIDED_TOKENS_FOR_STUDENT", "true").lower() == "true"
)


# when enabled the agent sees the target rubrics during rollout, which amounts to a ground-truth hint
# off during training and forced off during evaluation

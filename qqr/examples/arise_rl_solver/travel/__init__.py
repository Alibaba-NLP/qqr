"""
Travel Rubrics — Reward-Gated Reverse KL (RG-KL)

Core algorithm:
- Reward: 0.8 · pass_rate + 0.2 · 𝟙[all satisfied]
- Three-phase rollout: student (k_s=8, no memory) -> the LLM coach produces memory -> teacher (k_m=4, with memory)
- Teachers do not enter the PPO loss; they only contribute Δ_r = mean(r_m) - mean(r_s)
- λ(Δ_r) = λ_0 · max(Δ_r - δ_thresh, 0) · warmup · cosine_decay
- L = L_GRPO_student + λ(Δ_r) · KL(π_θ(·|p_s) ‖ π_θ(·|p_m)) + β · KL_ref_penalty
- The KL teacher is the current π_θ conditioned on memory (a self-distillation variant)

Differences from the memory-guided off-policy GRPO variant:
- No importance-sampling correction, which avoids the product over many turns collapsing
- Teachers are used only to estimate Δ_r and never enter the gradient
- Reverse-KL distillation replaces pushing the policy towards teacher trajectories with PPO
"""

from .reward_model import eval_reward, group_reward, reward_post_process
from .rollout import generate

__all__ = [
    "generate",
    "eval_reward",
    "group_reward",
    "reward_post_process",
]

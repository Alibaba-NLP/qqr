"""
VitaBench curriculum agent (generator) training module - Qwen3.5 + RG-KL

Core idea:
- Samples from the 100 official VitaBench OTA environments and explores them with the real tools
- Generates instructions + rubrics (the environment comes directly from the official data)
- The solver validates task difficulty in interactive mode (agent <-> UserSimulator <-> tools)
- reward = R_tool · R_fmt · (1 + R_diff) (triangular difficulty reward)

Algorithm: Reward-Gated Reverse KL (RG-KL)
- Three-phase rollout: student (no coach) -> gpt-5.2 produces coach memory -> teacher (with coach)
- λ(Δ_r) = λ_0 · gate(Δ_r) · warmup · cosine_decay
- No coach memory is needed at deployment: self-distillation has internalised it into the weights
"""

from .rollout import generate
from .reward_model import eval_reward, group_reward, reward_post_process

__all__ = [
    "generate",
    "eval_reward",
    "group_reward",
    "reward_post_process",
]

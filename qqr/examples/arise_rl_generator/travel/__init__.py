"""
Travel curriculum agent (generator) training module - Qwen3.5 + RG-KL

Core idea:
- The generator calls MCP tools for real data and produces query, expected_tools and rubrics
- The solver answers using the tools
- The reward combines tool matching with the rubric pass rate

Algorithm: Reward-Gated Reverse KL (RG-KL)
- Three-phase rollout: student (no coach) -> gpt-5.2 produces coach memory -> teacher (with coach)
- λ(Δ_r) = λ_0 · gate(Δ_r) · warmup · cosine_decay
"""

from .rollout import generate
from .reward_model import eval_reward, group_reward, reward_post_process

__all__ = [
    "generate",
    "eval_reward",
    "group_reward",
    "reward_post_process",
]

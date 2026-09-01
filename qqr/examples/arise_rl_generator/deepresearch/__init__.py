"""
DeepResearch curriculum agent (generator) training module - Qwen3.5 + RG-KL

The generator calls web_search for real data and produces the query and rubrics.
After the solver (an external executor service) answers, an LLM judge scores the rubric pass rate.

Algorithm: Reward-Gated Reverse KL (RG-KL)
"""

from .rollout import generate
from .reward_model import eval_reward, group_reward, reward_post_process

__all__ = [
    "generate",
    "eval_reward",
    "group_reward",
    "reward_post_process",
]

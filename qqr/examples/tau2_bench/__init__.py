"""
tau2_bench example module for qqr.

Public interface (matches what yaml configs reference):
  qqr.examples.tau2_bench.generate          – rollout entry point
  qqr.examples.tau2_bench.group_reward      – group reward model
  qqr.examples.tau2_bench.reward_post_process
  qqr.examples.tau2_bench.eval_rollout.custom_eval_rollout_log_function
"""

from .rollout import generate
from .reward_model import group_reward, reward_post_process, reshape_gt_tis_function, reshape_gt_tis_reverse_function


__all__ = [
    "generate",
    "group_reward",
    "reward_post_process",
    "reshape_gt_tis_function",
    "reshape_gt_tis_reverse_function"
]
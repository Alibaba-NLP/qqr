from typing import Any

import torch

from qqr import registers
from qqr.schemas import GroupRewardModel, RewardModel


def get_reward_model(name: str) -> RewardModel | GroupRewardModel:
    model_key = name

    if "/" in model_key:
        model_key = name.split("/")[0]

    if model_key not in registers.reward_model:
        raise ValueError(
            f"Model not found for name '{name}' (parsed as '{model_key}'). "
            f"Available reward models: {list(registers.reward_model.keys)}"
        )

    return registers.reward_model[model_key]


def reshape_gt_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],      # log π_θ
    # mixed: GT tokens = log π_gt; non-GT tokens = reference log prob (used to detect GT)
    teacher_log_probs: list[torch.Tensor],
    # log π_old / behavior (used to detect GT; unused for non-GT tokens)
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """
    Only reweight GT-generated tokens; keep non-GT tokens' pg_loss unchanged.

    GT token detection:
        gt_mask = (|logp_teacher - logp_rollout| > 1e-6)

    GT guidance ratio:
        r_gt = π_gt / π_θ = exp(logp_teacher - logp_theta)

    Gradient-preserving clip (AEPO-style):
        r_tilde = sg(r_gt) + clip(r_gt - sg(r_gt), -eps_low, eps_high)

    Final:
        pg_loss <- pg_loss * (1 + gt_mask * (r_tilde - 1))
    """
    logp_theta = torch.cat(train_log_probs, dim=0)        # [N]
    logp_teacher = torch.cat(teacher_log_probs, dim=0)    # [N]
    logp_rollout = torch.cat(rollout_log_probs, dim=0)    # [N]

    gt_mask = (torch.abs(logp_teacher - logp_rollout)
               > 1e-6).to(dtype=pg_loss.dtype)

    # ratio π_gt / π_theta = (π_gt / π_old) * (π_old / π_theta)
    log_r_gt = (logp_teacher - logp_rollout).float()
    r_gt = torch.exp(log_r_gt).to(dtype=pg_loss.dtype)

    eps_low = getattr(args, "tis_clip_low", 0.5)
    eps_high = getattr(args, "tis_clip", 2.0)

    r_tilde = r_gt.clamp(eps_low, eps_high)

    # Apply only on GT tokens; non-GT stays weight=1
    weights = 1.0 + gt_mask * (r_tilde - 1.0)
    pg_loss = pg_loss * weights

    # metrics (token-level; caller will reduce)
    abs_diff = (r_gt - 1.0).abs()
    clipfrac = ((r_tilde - r_gt).abs() > 1e-12).to(dtype=pg_loss.dtype)

    metrics = {
        "gt_mask": gt_mask.detach(),
        "gt_r": r_gt.detach(),
        "gt_r_abs": abs_diff.detach(),
        # only count GT tokens as clipped
        "gt_r_clipfrac": (clipfrac * gt_mask).detach(),
    }

    return pg_loss, loss_masks, metrics

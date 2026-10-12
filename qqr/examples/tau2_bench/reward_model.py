"""
tau2_bench reward model for qqr.

Uses embedded tau2 source code (qqr/tools/tau2_bench/tau2/) for self-contained imports.

Evaluation hierarchy (mirrors tau2-bench's own evaluator.py ALL mode):
  1. **DB / env-assertion reward** (RewardType.DB / RewardType.ENV_ASSERTION):
       Replay the agent's tool calls on a fresh gold environment and compare
       the resulting DB state + run env assertions.
  2. **Action reward** (RewardType.ACTION):
       Check whether required tool calls appear in the trajectory.
  3. **NL assertion reward** (RewardType.NL_ASSERTION):
       LLM-judge verifies natural-language assertions about the conversation.
  4. **Communicate reward** (RewardType.COMMUNICATE):
       Check whether required information was mentioned in agent messages.

Final reward = product of all applicable sub-rewards (0 or 1 each).
"""

import asyncio
import json
import logging
import sys
from argparse import Namespace
from typing import Any, Optional

import httpx

from qqr.schemas import Sample
from qqr.tools.tau2_bench.executor import get_environment

from . import config

logger = logging.getLogger(__name__)


# ── Ensure module consistency (handled by qqr.tools.tau2_bench.__init__) ──────
# The __init__.py redirects bare 'tau2' imports to the embedded copy.
# We just need to make sure it has been loaded.
import qqr.tools.tau2_bench  # noqa: F401 — triggers the sys.modules redirect


# ── tau2 import helpers (use bare 'tau2.*' path after redirect) ───────────────

def _import_tau2_message_types():
    """Import tau2 message types. Uses bare 'tau2.*' which is redirected to embedded copy."""
    from tau2.data_model.message import (
        AssistantMessage, UserMessage, ToolMessage, SystemMessage, ToolCall,
    )
    return AssistantMessage, UserMessage, ToolMessage, SystemMessage, ToolCall


def _import_tau2_task():
    """Import tau2 Task. Uses bare 'tau2.*' which is redirected to embedded copy."""
    from tau2.data_model.tasks import Task
    return Task


def _import_tau2_reward_type():
    """Import tau2 RewardType."""
    from tau2.data_model.tasks import RewardType
    return RewardType


def _import_tau2_evaluators():
    """Import tau2 evaluators."""
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    from tau2.evaluator.evaluator_action import ActionEvaluator
    from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
    return EnvironmentEvaluator, ActionEvaluator, CommunicateEvaluator


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory → tau2 Message conversion
# ─────────────────────────────────────────────────────────────────────────────

def _qqr_messages_to_tau2(messages: list[dict]):
    """
    Convert qqr-style message dicts to tau2 Message objects.
    
    This function tracks tool call requestors (user vs assistant) to ensure
    proper evaluation alignment with original tau2-bench.
    """
    AssistantMessage, UserMessage, ToolMessage, SystemMessage, ToolCall = _import_tau2_message_types()

    tau2_msgs = []
    pending_tool_requestor = "assistant"  # Default requestor
    
    for m in messages:
        role = m.get("role", "")
        content = m.get("content") or None

        if role == "system":
            tau2_msgs.append(SystemMessage(role="system", content=content))

        elif role == "user":
            # Check if user message has tool calls (user-initiated tool calls)
            raw_tcs = m.get("tool_calls") or []
            tool_calls = None
            if raw_tcs:
                tool_calls = []
                for tc in raw_tcs:
                    fn = fn = tc.get("function") or {}
                    args_raw = fn.get("arguments") or "{}"
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw)
                        except Exception:
                            args = {}
                    else:
                        args = args_raw or {}
                    tool_calls.append(ToolCall(
                        id=tc.get("id", ""),
                        name=fn.get("name") or "",
                        arguments=args,
                        requestor="user",  # Mark as user-initiated
                    ))
                pending_tool_requestor = "user"
            tau2_msgs.append(UserMessage(
                role="user",
                content=content,
                tool_calls=tool_calls,
            ))

        elif role == "assistant":
            raw_tcs = m.get("tool_calls") or []
            tool_calls = None
            if raw_tcs:
                tool_calls = []
                for tc in raw_tcs:
                    fn = tc.get("function") or {}
                    args_raw = fn.get("arguments") or "{}"
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw)
                        except Exception:
                            args = {}
                    else:
                        args = args_raw or {}
                    tool_calls.append(ToolCall(
                        id=tc.get("id") or "",
                        name=fn.get("name") or "",
                        arguments=args,
                        requestor="assistant",
                    ))
                pending_tool_requestor = "assistant"
            tau2_msgs.append(AssistantMessage(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
            ))

        elif role == "tool":
            tc_id = m.get("tool_call_id", "")
            tau2_msgs.append(ToolMessage(
                id=tc_id,
                role="tool",
                content=content,
                requestor=pending_tool_requestor,  # Use the tracked requestor
            ))

    return tau2_msgs


# ─────────────────────────────────────────────────────────────────────────────
# tau2 Task reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _build_tau2_task(reference: dict):
    """Reconstruct a tau2 Task object from the qqr label reference dict."""
    try:
        Task = _import_tau2_task()
        task_dict = {
            "id": reference.get("task_id", "unknown"),
            "user_scenario": reference.get("user_scenario", {}),
            "evaluation_criteria": reference.get("evaluation_criteria"),
            "initial_state": reference.get("initial_state"),
        }
        return Task.model_validate(task_dict)
    except Exception as e:
        import pydantic
        if isinstance(e, pydantic.ValidationError):
            logger.error(
                f"[reward] Pydantic ValidationError reconstructing tau2 Task "
                f"(task_id={reference.get('task_id', '?')}): {e}"
            )
        else:
            logger.warning(f"[reward] Failed to reconstruct tau2 Task: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Individual evaluators
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_env(domain: str, tau2_task, tau2_trajectory, db_path: Optional[str] = None) -> tuple[float, dict]:
    """
    DB state + env-assertion reward.

    Uses EnvironmentEvaluator which:
    1. Builds a gold environment and applies the task's expected actions.
    2. Builds a predicted environment by replaying the agent's tool calls.
    3. Compares DB hashes and runs env assertions on the predicted environment.

    Args:
        domain: tau2 domain name.
        tau2_task: reconstructed tau2 Task object.
        tau2_trajectory: list of tau2 Message objects.
        db_path: optional relative path to the sample-specific DB file
            (e.g. "tau2_rl_database/tau2_retail_new_db_1.json").
            When provided, overrides the domain's default embedded DB so
            that actions are evaluated against the correct database.

    Returns (reward_float, details_dict).
    """
    try:
        EnvironmentEvaluator, _, _ = _import_tau2_evaluators()
        # Wrap get_environment so EnvironmentEvaluator can call it multiple
        # times (once for gold, once for predicted) with the right DB.
        def env_constructor(**kwargs):
            return get_environment(domain, db_path=db_path)
        reward_info = EnvironmentEvaluator.calculate_reward(
            environment_constructor=env_constructor,
            task=tau2_task,
            full_trajectory=tau2_trajectory,
            solo_mode=False,
        )

        details: dict[str, Any] = {"reward": float(reward_info.reward)}

        if reward_info.db_check:
            details["db_check"] = {
                "db_match": reward_info.db_check.db_match,
                "db_reward": reward_info.db_check.db_reward,
            }

        if reward_info.env_assertions:
            details["env_assertions"] = []
            for ea in reward_info.env_assertions:
                details["env_assertions"].append({
                    "func_name": ea.env_assertion.func_name,
                    "env_type": str(ea.env_assertion.env_type),
                    "arguments": ea.env_assertion.arguments,
                    "expected_value": ea.env_assertion.assert_value,
                    "met": ea.met,
                    "reward": ea.reward,
                })

        return float(reward_info.reward), details
    except Exception as e:
        import pydantic
        if isinstance(e, pydantic.ValidationError):
            logger.error(
                f"[reward] Pydantic ValidationError in env eval ({domain}): {e}"
            )
        else:
            logger.warning(f"[reward] env eval failed ({domain}): {e}")
        return 0.0, {"error": str(e)}  # evaluation failed → treat as incorrect


def _evaluate_actions(tau2_task, tau2_trajectory) -> tuple[float, dict]:
    """Check whether all required tool calls appear in the trajectory.

    Returns (reward_float, details_dict).
    """
    try:
        _, ActionEvaluator, _ = _import_tau2_evaluators()
        reward_info = ActionEvaluator.calculate_reward(
            task=tau2_task, full_trajectory=tau2_trajectory,
        )

        details: dict[str, Any] = {"reward": float(reward_info.reward)}
        if reward_info.action_checks:
            details["action_checks"] = []
            for ac in reward_info.action_checks:
                details["action_checks"].append({
                    "action_id": ac.action.action_id,
                    "name": ac.action.name,
                    "requestor": str(ac.action.requestor),
                    "arguments": ac.action.arguments,
                    "matched": ac.action_match,
                    "reward": ac.action_reward,
                })

        return float(reward_info.reward), details
    except Exception as e:
        logger.warning(f"[reward] action eval failed: {e}")
        return 0.0, {"error": str(e)}


def _evaluate_communicate(tau2_task, tau2_trajectory) -> tuple[float, dict]:
    """Check whether required strings appear in agent assistant messages.

    Returns (reward_float, details_dict).
    """
    try:
        _, _, CommunicateEvaluator = _import_tau2_evaluators()
        reward_info = CommunicateEvaluator.calculate_reward(
            task=tau2_task, full_trajectory=tau2_trajectory,
        )

        details: dict[str, Any] = {"reward": float(reward_info.reward)}
        if reward_info.communicate_checks:
            details["communicate_checks"] = []
            for cc in reward_info.communicate_checks:
                details["communicate_checks"].append({
                    "info": cc.info,
                    "met": cc.met,
                    "justification": cc.justification,
                })

        return float(reward_info.reward), details
    except Exception as e:
        logger.warning(f"[reward] communicate eval failed: {e}")
        return 0.0, {"error": str(e)}


async def _evaluate_nl_assertions_async(
    args: Namespace,
    tau2_task,
    tau2_trajectory,
) -> tuple[float, dict]:
    """
    NL assertion evaluation via LLM judge (async).

    Implements tau2's NLAssertionsEvaluator prompt inline so we can use the
    qqr judge_openai_base_url endpoint instead of tau2's DEFAULT_LLM_NL_ASSERTIONS.
    Falls back to 1.0 if no judge is configured.

    Returns (reward_float, details_dict).
    """
    try:
        eval_criteria = tau2_task.evaluation_criteria
        if eval_criteria is None or not eval_criteria.nl_assertions:
            return 1.0, {"skipped": True, "reason": "no nl_assertions"}
        nl_assertions = eval_criteria.nl_assertions

        base_url = (
            getattr(args, "judge_openai_base_url", "")
            or config.judge_openai_base_url
        )
        if not base_url:
            logger.debug("[reward] No judge URL configured; skipping NL assertions → 1.0")
            return 1.0, {"skipped": True, "reason": "no judge URL configured"}

        api_key = (
            getattr(args, "judge_openai_api_key", "")
            or config.judge_openai_api_key
            or "EMPTY"
        )
        model = (
            getattr(args, "judge_model", "")
            or config.judge_model
        )
        timeout = int(getattr(args, "judge_timeout", config.judge_timeout))

        # Build trajectory string (mirrors tau2's NLAssertionsEvaluator)
        traj_str = "\n".join(
            f"{m.role}: {m.content or ''}"
            for m in tau2_trajectory
            if hasattr(m, "role") and hasattr(m, "content")
        )

        # Tau2's exact system prompt
        system_prompt = (
            "TASK\n"
            "- You will be given a list of expected outcomes and a conversation that was collected during a test case run.\n"
            "- The conversation is between an agent and a customer.\n"
            "- Your job is to evaluate whether the agent satisfies each of the expected outcomes.\n"
            "- Grade each expected outcome individually.\n\n"
            "FORMAT\n"
            "- Your response should be a JSON object with the following fields:\n"
            "- `reasoning`: a short explanation for your classification\n"
            "- `metExpectation`: `true` if the agent satisfies the expected outcomes, `false` otherwise\n"
            "- `expectedOutcome`: repeat the expectation from the input that you are grading\n\n"
            "Example response structure:\n"
            "{\n"
            '    "results": [\n'
            "        {\n"
            '            "expectedOutcome": "<one of the expected outcomes from the input>",\n'
            '            "reasoning": "<reasoning trace>",\n'
            '            "metExpectation": <false or true>,\n'
            "        }\n"
            "    ]\n"
            "}"
        )
        user_prompt = (
            f"conversation:\n{traj_str}\n\n"
            f"expectedOutcomes:\n{json.dumps(nl_assertions)}"
        )

        payload = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": int(
                getattr(args, "judge_max_new_tokens", config.judge_max_new_tokens)
            ),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        url = base_url.rstrip("/") + "/chat/completions"

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        raw = data["choices"][0]["message"]["content"] or ""
        # Strip markdown fences if present (mirrors tau2)
        cleaned = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(cleaned)
        results = parsed.get("results", [])
        if not results:
            return 1.0, {"skipped": True, "reason": "judge returned empty results"}

        all_met = all(r.get("metExpectation", False) for r in results)
        reward = 1.0 if all_met else 0.0

        details: dict[str, Any] = {
            "reward": reward,
            "nl_assertion_checks": [
                {
                    "expected_outcome": r.get("expectedOutcome", ""),
                    "met": r.get("metExpectation", False),
                    "reasoning": r.get("reasoning", ""),
                }
                for r in results
            ],
        }
        return reward, details

    except Exception as e:
        logger.warning(f"[reward] NL assertion eval failed: {e}")
        return 1.0, {"error": str(e)}  # neutral fallback


# ─────────────────────────────────────────────────────────────────────────────
# Main reward entry point
# ─────────────────────────────────────────────────────────────────────────────

async def eval_reward(args: Namespace, sample: Sample, **kwargs):
    """
    Compute the tau2-bench reward for the terminal sample of a rollout episode.

    reward = product of sub-rewards for each active RewardType in the task's
             evaluation_criteria.reward_basis.

    Stored in sample.reward (float in [0, 1]).
    """
    label = sample.label
    if not isinstance(label, dict):
        sample.reward = 0.0
        return

    reference = label
    domain = reference.get("domain", "airline")
    db_path = reference.get("db_path", None)
    tau2_task = _build_tau2_task(reference)
    if tau2_task is None:
        sample.reward = 0.0
        return

    eval_criteria = tau2_task.evaluation_criteria
    if eval_criteria is None:
        sample.reward = 1.0
        return

    # Check for premature termination (matching original tau2-bench behavior)
    # If simulation terminated due to max_steps or too_many_errors, reward = 0
    termination_reason = (sample.metadata or {}).get("termination_reason", "")
    if termination_reason in {"max_steps", "too_many_errors"}:
        sample.reward = 0.0
        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata["reward_breakdown"] = {
            "note": f"Simulation terminated prematurely: {termination_reason}"
        }
        logger.debug(
            "[reward] domain=%s task=%s terminated prematurely: %s",
            domain, reference.get("task_id", ""), termination_reason
        )
        return

    # Convert qqr messages → tau2 Message objects
    messages = (sample.metadata or {}).get("full_trajectory") or sample.messages
    tau2_trajectory = _qqr_messages_to_tau2(messages)

    reward = 1.0
    reward_breakdown: dict[str, float] = {}
    evaluation_details: dict[str, Any] = {}
    reasons: list[str] = []

    RewardType = _import_tau2_reward_type()
    reward_basis = set(eval_criteria.reward_basis or [])

    # ── DB / ENV_ASSERTION ────────────────────────────────────────────────────
    env_bases = {RewardType.DB, RewardType.ENV_ASSERTION}
    if reward_basis & env_bases:
        r_env, env_details = _evaluate_env(domain, tau2_task, tau2_trajectory, db_path=db_path)
        reward *= r_env
        reward_breakdown["env"] = r_env
        evaluation_details["env"] = env_details
        reasons.append(f"env={r_env:.2f}")

    # ── ACTION ────────────────────────────────────────────────────────────────
    if RewardType.ACTION in reward_basis:
        r_act, act_details = _evaluate_actions(tau2_task, tau2_trajectory)
        reward *= r_act
        reward_breakdown["action"] = r_act
        evaluation_details["action"] = act_details
        reasons.append(f"action={r_act:.2f}")

    # ── NL_ASSERTION (async LLM judge) ────────────────────────────────────────
    if RewardType.NL_ASSERTION in reward_basis:
        r_nl, nl_details = await _evaluate_nl_assertions_async(args, tau2_task, tau2_trajectory)
        reward *= r_nl
        reward_breakdown["nl_assertion"] = r_nl
        evaluation_details["nl_assertion"] = nl_details
        reasons.append(f"nl={r_nl:.2f}")

    # ── COMMUNICATE ───────────────────────────────────────────────────────────
    if RewardType.COMMUNICATE in reward_basis:
        r_comm, comm_details = _evaluate_communicate(tau2_task, tau2_trajectory)
        reward *= r_comm
        reward_breakdown["communicate"] = r_comm
        evaluation_details["communicate"] = comm_details
        reasons.append(f"comm={r_comm:.2f}")

    sample.reward = float(reward)

    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["reward_breakdown"] = reward_breakdown
    sample.metadata["evaluation_details"] = evaluation_details
    logger.debug(
        "[reward] domain=%s task=%s total=%.2f [%s]",
        domain, reference.get("task_id", ""), reward, "; ".join(reasons),
    )


def _step_length_penalty(args: Namespace, response_length: int) -> float:
    """Linear per-step overlong penalty (DAPO-style shaping).

    0 below the soft threshold, ramps linearly to cap_penalty at
    rollout_max_response_len, flat beyond. Disabled when
    step_length_penalty_soft_tokens <= 0 (default), keeping the legacy
    broadcast behavior.
    """
    soft = int(getattr(args, "step_length_penalty_soft_tokens", 0) or 0)
    cap_penalty = float(getattr(args, "step_length_penalty_cap", 0.0) or 0.0)
    if soft <= 0 or cap_penalty <= 0.0 or response_length <= soft:
        return 0.0
    hard = int(getattr(args, "rollout_max_response_len", 0) or 0)
    if hard <= soft:
        return cap_penalty
    return cap_penalty * min((response_length - soft) / (hard - soft), 1.0)


async def group_reward(args: Namespace, group: list[list[Sample]], **kwargs):
    """Called by the qqr framework once per group of parallel trajectories."""
    for traj in group:
        if not traj:
            continue
        # Find the last non-padding sample (index != -1)
        terminal = next(
            (s for s in reversed(traj) if getattr(s, "index", 0) != -1), None
        )
        if terminal is None:
            continue
        await eval_reward(args, terminal)
        traj_reward = float(terminal.reward or 0.0)
        penalized = 0
        for s in traj:
            penalty = _step_length_penalty(args, getattr(s, "response_length", 0) or 0)
            s.reward = traj_reward - penalty
            if penalty > 0.0:
                penalized += 1
        if penalized:
            logger.info(
                "[step_len_penalty] group=%s traj_reward=%.2f penalized_steps=%d/%d",
                getattr(terminal, "group_index", "?"), traj_reward, penalized, len(traj),
            )


def reward_post_process(args: Namespace, samples):
    if not samples:
        return [], []
    if isinstance(samples[0], list):
        flat = [s for traj in samples for s in traj]
        raw = [s.get_reward_value(args) for s in flat]
        return raw, raw
    raw = [s.get_reward_value(args) for s in samples]
    return raw, raw



import torch
from typing import Any
def reshape_gt_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],      # log π_θ
    teacher_log_probs: list[torch.Tensor],    # mixed: GT处=log π_gt；非GT处=参考分布log prob（用于判定GT）
    rollout_log_probs: list[torch.Tensor],    # log π_old / behavior（用于判定GT；non-GT不使用）
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

    gt_mask = (torch.abs(logp_teacher - logp_rollout) > 1e-6).to(dtype=pg_loss.dtype)

    if gt_mask.sum() > 0:
        print()

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
        "gt_r_clipfrac": (clipfrac * gt_mask).detach(),  # only count GT tokens as clipped
    }

    return pg_loss, loss_masks, metrics


def reshape_gt_tis_reverse_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],      # log π_θ
    teacher_log_probs: list[torch.Tensor],    # mixed: GT处=log π_gt；非GT处=参考分布log prob（用于判定GT）
    rollout_log_probs: list[torch.Tensor],    # log π_old / behavior（用于判定GT；non-GT不使用）
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

    gt_mask = (torch.abs(logp_teacher - logp_rollout) > 1e-6).to(dtype=pg_loss.dtype)

    if gt_mask.sum() > 0:
        print()

    # ratio π_gt / π_theta = (π_gt / π_old) * (π_old / π_theta)
    # log_r_gt = (logp_teacher - logp_rollout).float()
    log_r_gt = (logp_rollout - logp_teacher).float()
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
        "gt_r_clipfrac": (clipfrac * gt_mask).detach(),  # only count GT tokens as clipped
    }

    return pg_loss, loss_masks, metrics



def reshape_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],      # log π_θ
    teacher_log_probs: list[torch.Tensor],    # mixed: GT处=log π_gt；非GT处=参考分布log prob（用于判定GT）
    rollout_log_probs: list[torch.Tensor],    # log π_old / behavior（用于判定GT；non-GT不使用）
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

    gt_mask = (torch.abs(logp_teacher - logp_rollout) > 1e-6).to(dtype=pg_loss.dtype)

    if gt_mask.sum() > 0:
        print()

    # ratio π_gt / π_theta = (π_gt / π_old) * (π_old / π_theta)
    log_r_gt = (logp_teacher - logp_theta).float()
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
        "gt_r_clipfrac": (clipfrac * gt_mask).detach(),  # only count GT tokens as clipped
    }

    return pg_loss, loss_masks, metrics

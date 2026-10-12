"""
Per-step RTI inspector calibration.

Encapsulates the repeated pattern of:
  - Check if inspector is enabled for this sample
  - Call inspector.inspect_and_correct
  - If corrected, replace response/tokens/log_probs on the sample
  - Record inspector trace in metadata
"""

import logging
from argparse import Namespace
from typing import Any

from qqr.schemas import Sample

logger = logging.getLogger(__name__)


async def calibrate_step(
    args: Namespace,
    sample: Sample,
    inspector,
    step_idx: int,
    sampling_params: dict[str, Any],
) -> tuple[bool, dict]:
    """Run inspector calibration on a single step.

    If corrected, replaces sample.response/tokens/log_probs in-place.
    No-op if sample.label['inspector'] is False.

    Returns:
        (was_corrected, info) — same interface as inspector.inspect_and_correct
    """
    label = sample.label if isinstance(sample.label, dict) else {}
    if not label.get('inspector', False):
        return False, {"is_valid": True, "reason": "inspector_disabled_for_sample"}

    try:
        was_corrected, info = await inspector.inspect_and_correct(sample, step_idx)
    except Exception as e:
        logger.exception(f"[RTI] inspector failed at step {step_idx}: {e}")
        return False, {"is_valid": True, "reason": "inspector_exception"}

    if not was_corrected:
        return False, info

    corrected = info.get("corrected_response")
    if not isinstance(corrected, str) or not corrected.strip():
        return True, info

    if sample.response_length > 0:
        prompt_prefix = sample.tokens[:-sample.response_length]
    else:
        prompt_prefix = sample.tokens
    corrected_tokens = info['intermediates']['ref_gate']['score']['corr_score']['tokens']

    max_new_tokens = min(
        sampling_params["max_new_tokens"],
        sampling_params["max_context_length"] - len(prompt_prefix),
    )

    if len(prompt_prefix) <= 1 and sample.response_length == 0:
        logger.warning("[RTI] skipping token replacement: prompt_prefix is a dummy (context overflow)")
        return True, info

    if len(corrected_tokens) > max_new_tokens:
        logger.warning(
            f"[RTI] skipping token replacement: corrected response "
            f"({len(corrected_tokens)} tokens) exceeds max_new_tokens ({max_new_tokens})"
        )
        return True, info

    raw_tokens = info['intermediates']['ref_gate']['score']['raw_score']['tokens']
    if sample.tokens[-sample.response_length:] != raw_tokens:
        logger.warning("[RTI] token mismatch between sample and inspector raw_score")

    sample.response = corrected
    sample.tokens = prompt_prefix + corrected_tokens
    sample.rollout_log_probs = info['intermediates']['ref_gate']['score']['corr_score']['token_logprobs']
    sample.teacher_log_probs = info['intermediates']['origin_score']['correct_prompt_score']['token_logprobs']
    sample.response_length = info['intermediates']['ref_gate']['score']['corr_score']['n_tokens']

    if len(corrected_tokens) != sample.response_length:
        logger.warning(
            f"[RTI] corrected_tokens len ({len(corrected_tokens)}) != n_tokens ({sample.response_length})"
        )
    if len(sample.rollout_log_probs) != sample.response_length:
        logger.warning(
            f"[RTI] rollout_log_probs len ({len(sample.rollout_log_probs)}) != response_length ({sample.response_length})"
        )
    if len(sample.teacher_log_probs) != sample.response_length:
        logger.warning(
            f"[RTI] teacher_log_probs len ({len(sample.teacher_log_probs)}) != response_length ({sample.response_length})"
        )

    if isinstance(sample.label, dict):
        sample.label['inspector_success'] = True

    assert len(sample.rollout_log_probs) == len(sample.teacher_log_probs), (
        f"rollout_log_probs({len(sample.rollout_log_probs)}) != teacher_log_probs({len(sample.teacher_log_probs)})"
    )

    if sample.metadata is None:
        sample.metadata = {}
    if 'inspector_trace' not in sample.metadata:
        sample.metadata['inspector_trace'] = {}
    sample.metadata['inspector_trace']['score'] = {
        "rollout_log_probs": sample.rollout_log_probs,
        "teacher_log_probs": sample.teacher_log_probs,
        "tokens": sample.tokens[-sample.response_length:],
        "response_length": sample.response_length,
    }

    return True, info

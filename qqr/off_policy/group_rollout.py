"""
Two-stage group rollout with off-policy trajectory calibration.

Stage 1: normal rollout (inspector disabled)
Stage 2: if all samples in group failed, run calibration rollout (RTI: Rollout-Time Intervention)
         and replace failed samples with successful calibrated ones.

Activated via config:
  custom-group-rollout-function-path: qqr.off_policy.group_rollout.generate_and_rm_group_with_calibration
"""

import asyncio
import json
import logging
import os
import random
from argparse import Namespace
from copy import deepcopy
from typing import Any

from qqr.schemas import Sample
from slime.rollout.rm_hub import batched_async_rm

logger = logging.getLogger(__name__)


def _is_success(sample) -> bool:
    s = sample[-1] if isinstance(sample, list) else sample
    r = getattr(s, "reward", 0.0)
    return float(r) >= 1


def _force_disable_inspector_label(group: list[Sample]) -> None:
    for s in group:
        if not isinstance(getattr(s, "label", None), dict) or s.label is None:
            s.label = {}
        s.label["inspector"] = False


def _force_enable_inspector_label(group: list[Sample]) -> None:
    for s in group:
        if not isinstance(getattr(s, "label", None), dict) or s.label is None:
            s.label = {}
        s.label["inspector"] = True


def _clone_group(src_group: list[Sample]) -> list[Sample]:
    return deepcopy(src_group)


async def _run_group(args, group, sampling_params, evaluation, state):
    from qqr.rollout.agent_rollout import generate_and_rm

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(
            asyncio.create_task(
                generate_and_rm(
                    args, sample, current_sampling_params, evaluation=evaluation
                )
            )
        )

    group = await asyncio.gather(*tasks)

    if not state.aborted and args.group_rm:
        await batched_async_rm(args, group)

    return group


def _write_log(args, group, rollout_id=None):
    def unwrap(x):
        return x[-1] if isinstance(x, list) else x

    def try_parse_content(msgs):
        if not isinstance(msgs, list):
            return msgs
        result = []
        for msg in msgs:
            if not isinstance(msg, dict):
                result.append(msg)
                continue
            new_msg = dict(msg)
            new_msg.pop("raw", None)
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                s = content.strip()
                if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
                    try:
                        parsed = json.loads(s)
                        if isinstance(parsed, (dict, list)):
                            new_msg["content"] = parsed
                    except (json.JSONDecodeError, ValueError):
                        pass
            result.append(new_msg)
        return result

    base_dir = getattr(args, "inspector_log_dir", None)
    if not base_dir:
        return

    created = set()
    for item in group:
        s = unwrap(item)
        subdir = f"{s.group_index}"
        if rollout_id is not None:
            subdir = f"{rollout_id}/{subdir}"
        log_dir = os.path.join(base_dir, subdir)
        if log_dir not in created:
            os.makedirs(log_dir, exist_ok=True)
            created.add(log_dir)
        log_file = os.path.join(log_dir, f"{s.index}.json")
        tmp_file = log_file + ".tmp"
        output_sample = {
            "group_index": getattr(s, "group_index", None),
            "index": getattr(s, "index", None),
            "reward": getattr(s, "reward", None),
            "prompt": getattr(s, "prompt", None),
            "label": getattr(s, "label", None),
            "messages": try_parse_content(getattr(s, "messages", None)),
            "metadata": getattr(s, "metadata", None),
        }
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(output_sample, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp_file, log_file)


async def generate_and_rm_group_with_calibration(
    args: Namespace,
    group: list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> list[Sample]:
    """
    Two-stage group rollout:
    - Stage 1: generate group with inspector disabled
    - If any reward >= 1: return stage 1 group
    - Else Stage 2: generate group with inspector enabled (RTI)
      and replace stage 1 samples with successful stage 2 samples
    """
    from qqr.rollout.agent_rollout import GenerateState
    import uuid

    fallback_enabled = bool(getattr(args, "inspector_fallback_enabled", True))
    replace_num = int(getattr(args, "inspector_replace_num", 0))

    state = GenerateState(args)
    if state.aborted:
        return group

    for sample in group:
        if sample.session_id is None:
            sample.session_id = str(uuid.uuid4())

    g2 = _clone_group(group)

    _force_disable_inspector_label(group)

    group = await _run_group(args, group, sampling_params, evaluation, state)

    if evaluation or replace_num == 0 or (not fallback_enabled):
        return group

    if any(_is_success(s) for s in group):
        return group

    # Stage 2: calibration rollout
    _force_enable_inspector_label(g2)

    state = GenerateState(args)
    if state.aborted:
        return group

    g2 = await _run_group(args, g2, sampling_params, evaluation, state)

    _write_log(args, g2)

    succ2 = [idx for idx, s in enumerate(g2) if _is_success(s)]
    if not succ2:
        return group

    k = min(replace_num, len(group), len(succ2))
    replace_indices = random.sample(succ2, k)
    for idx in replace_indices:
        assert group[idx][-1].index == g2[idx][-1].index
        group[idx] = g2[idx]

    return group

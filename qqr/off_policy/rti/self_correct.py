"""
Self-Correct Inspector: uses the current policy model (via sglang) to
re-generate a corrected response given groundtruth guidance.

Unlike RTIInspector (external LLM judge/corrector), this calls the same policy
model that produced the original response, but with additional
groundtruth/guidance injected into the prompt.

Activated via config:
  custom-inspector-method-path: qqr.off_policy.rti.self_correct
"""

import json
import logging
from argparse import Namespace
from copy import deepcopy
from typing import Any

from slime.utils.http_utils import post

from qqr.rollout.agent_rollout import GenerateState
from qqr.schemas import Sample
from .ref.ref_scorer import RefScorer

logger = logging.getLogger(__name__)


def _build_guidance_prompt(
    messages: list[dict],
    groundtruth: dict | None,
    guidance: str | None,
    original_response: str,
) -> list[dict]:
    """Inject groundtruth and guidance into messages for self-correction.

    Appends a system-level hint after the original messages, instructing the
    policy model to produce a better response using the provided reference.
    """
    injected = deepcopy(messages)

    hint_parts = []
    if groundtruth:
        gt_str = json.dumps(groundtruth, ensure_ascii=False, indent=2)
        hint_parts.append(
            f"[参考信息]\n{gt_str}\n"
            "你需要参考以上信息来改进你的回答，但不能直接暴露这些参考信息的存在。"
        )
    if guidance:
        hint_parts.append(f"[改进指导]\n{guidance}")

    hint_parts.append(
        f"[原始回答]\n{original_response}\n\n"
        "请根据参考信息和指导，生成一个改进后的回答。"
        "保持与原始回答相同的格式和风格，但修正其中的错误。"
    )

    hint_block = "\n\n".join(hint_parts)

    if injected and injected[0].get("role") == "system":
        injected[0]["content"] = (injected[0].get("content") or "") + "\n\n" + hint_block
    else:
        injected.insert(0, {"role": "system", "content": hint_block})

    return injected


class SelfCorrectInspector:
    """Self-correction using the current policy model with groundtruth guidance.

    Compatible interface with RTIInspector:
        was_corrected, info = await inspector.inspect_and_correct(sample, step_idx)
    """

    def __init__(self, cfg, args: Namespace):
        self.cfg = cfg
        self.args = args

    async def inspect_and_correct(
        self, sample: Sample, step_idx: int
    ) -> tuple[bool, dict[str, Any]]:
        if not self.cfg.enabled:
            return False, {"is_valid": True, "reason": "self_correct_disabled"}

        args = self.args
        state = GenerateState(args)

        original_response = getattr(sample, "response", "") or ""
        if not original_response.strip():
            return False, {"is_valid": True, "reason": "empty_response"}

        # Extract groundtruth from sample label
        label = sample.label if isinstance(sample.label, dict) else {}
        groundtruth = label.get("groundtruth") or label.get("ref") or label
        guidance = getattr(self.cfg, "guidance_prompt", None)

        # Build corrected messages with groundtruth injection
        messages = list(getattr(sample, "messages", []) or [])
        tools = sample.train_metadata.get("tools") if sample.train_metadata else None

        corrected_messages = _build_guidance_prompt(
            messages, groundtruth, guidance, original_response
        )

        # Generate via sglang (same policy model)
        prompt_text = state.tokenizer.apply_chat_template(
            corrected_messages, tools=tools, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = state.tokenizer.encode(prompt_text, add_special_tokens=False)

        max_context_len = int(getattr(args, "rollout_max_context_len", 32000))
        max_new_tokens = min(
            int(getattr(args, "rollout_max_response_len", 4096)),
            max_context_len - len(prompt_ids),
        )

        if max_new_tokens <= 0:
            return False, {"is_valid": True, "reason": "context_overflow"}

        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        payload = {
            "input_ids": prompt_ids,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": float(getattr(self.cfg, "temperature", 0.6)),
                "top_p": float(getattr(args, "rollout_top_p", 0.95)),
            },
            "return_logprob": True,
        }

        try:
            output = await post(url, payload)
        except Exception as e:
            logger.warning(f"[SelfCorrect] sglang generate failed: {e}")
            return False, {"is_valid": True, "reason": f"generate_error: {e}"}

        corrected_response = output.get("text", "")
        if not corrected_response.strip():
            return False, {"is_valid": True, "reason": "empty_corrected_response"}

        # Get token logprobs for the corrected response
        meta = output.get("meta_info", {}) or {}
        otlp = meta.get("output_token_logprobs")
        if otlp:
            corr_tokens = [int(item[1]) for item in otlp]
            corr_logprobs = [float(item[0]) for item in otlp]
        else:
            corr_tokens = state.tokenizer.encode(corrected_response, add_special_tokens=False)
            corr_logprobs = [0.0] * len(corr_tokens)

        # Score the original response under the policy model (for ref_gate compatibility)
        try:
            ref_scorer = RefScorer(args)
            raw_score = await ref_scorer.score_sample_completion_async(
                sample, original_response
            )
        except Exception as e:
            logger.warning(f"[SelfCorrect] raw scoring failed: {e}")
            raw_score = {
                "tokens": sample.tokens[-sample.response_length:] if sample.response_length else [],
                "token_logprobs": sample.rollout_log_probs or [],
                "n_tokens": sample.response_length or 0,
            }

        # Build info dict compatible with calibrate_step expectations
        info = {
            "is_valid": False,
            "reason": "self_correct_replacement",
            "corrected_response": corrected_response,
            "intermediates": {
                "ref_gate": {
                    "score": {
                        "raw_score": {
                            "tokens": raw_score.get("tokens", []),
                            "token_logprobs": raw_score.get("token_logprobs", []),
                            "n_tokens": raw_score.get("n_tokens", 0),
                        },
                        "corr_score": {
                            "tokens": corr_tokens,
                            "token_logprobs": corr_logprobs,
                            "n_tokens": len(corr_tokens),
                        },
                    }
                },
                "origin_score": {
                    "correct_prompt_score": {
                        "tokens": corr_tokens,
                        "token_logprobs": corr_logprobs,
                    }
                },
            },
        }

        return True, info


def build_inspector(cfg, args) -> SelfCorrectInspector:
    """Factory function for dynamic loading via custom-inspector-method-path."""
    return SelfCorrectInspector(cfg, args)

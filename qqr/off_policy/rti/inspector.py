import os
from dataclasses import dataclass

from .client import LLMConfig, OpenAIChatClient
from .trace import TraceRecorder
from .pipeline import CalibrationPipeline
from .agents.judge import ValidityJudgeAgent
from .agents.corrector import CorrectorDraftAgent
from .agents.deleak import DeLeakAgent
from .agents.postcheck import PostcheckJudgeAgent
import importlib

def build_inspector(cfg, args) -> "RTIInspector":
    """Factory function matching the interface expected by rollout.py's dynamic loader.

    Called when custom-inspector-method-path: qqr.off_policy.rti.inspector is set,
    or used as the default fallback.
    """
    return RTIInspector(cfg, args=args)


class RTIInspector:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.trace = TraceRecorder()

        llm_cfg = LLMConfig(
            base_url=cfg.openai_base_url,
            api_key=cfg.openai_api_key,
            model=cfg.openai_model,
            temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
            timeout=cfg.timeout,
            max_retries=getattr(cfg, "max_retries", 10),
            retry_base_backoff=getattr(cfg, "retry_base_backoff", 10),
            retry_max_backoff=getattr(cfg, "retry_max_backoff", 300),
        )
        client = OpenAIChatClient(llm_cfg)

        enabled = bool(cfg.use_judge_llm) and bool(cfg.openai_base_url) and bool(cfg.openai_api_key)

        # load custom prompts if provided
        prompt_base = getattr(args, "custom_inspector_prompt_path", None)
        custom = {}
        if prompt_base:
            custom["judge"] = importlib.import_module(f"{prompt_base}.judge_validity")
            custom["corrector"] = importlib.import_module(f"{prompt_base}.corrector_draft")
            custom["origin_prompt"] = importlib.import_module(f"{prompt_base}.origin_prompt")
            custom["deleak"] = importlib.import_module(f"{prompt_base}.deleak")
            custom["postcheck"] = importlib.import_module(f"{prompt_base}.postcheck_judge")
        else:
            custom["judge"] = None
            custom["corrector"] = None
            custom["origin_prompt"] = None
            custom["deleak"] = None
            custom["postcheck"] = None
        self.pipeline = CalibrationPipeline(
            trace=self.trace,
            judge=ValidityJudgeAgent(client, enabled=enabled, prompt_module=custom["judge"]),
            corrector=CorrectorDraftAgent(client, enabled=enabled, prompt_module=custom["corrector"], origin_prompt=custom["origin_prompt"]),
            deleak=DeLeakAgent(client, enabled=enabled, prompt_module=custom["deleak"]),
            postcheck=PostcheckJudgeAgent(client, enabled=enabled, prompt_module=custom["postcheck"]),
            args=args,
            cfg=cfg,
        )

    async def inspect_and_correct(self, sample, step_idx: int):
        """
        rollout.py 期待返回: (was_corrected: bool, info: dict)
        """
        if not self.cfg.enabled:
            self.trace.add(sample, step_idx, type="decision", role="inspector", name="disabled", content={"ok": True})
            self.trace.finalize_step(sample, step_idx, summary={
                "decision": "pass",
                "reason": "inspector_disabled",
                "original_response": getattr(sample, "response", "") or "",
                "corrected_response": None,
            })
            self.trace.set_final(sample, {"final_decision": "accepted_raw", "final_reason": "inspector_disabled"})
            return False, {"is_valid": True, "reason": "inspector_disabled"}

        res = await self.pipeline.run(sample, step_idx)

        if res.corrected and res.corrected_response:
            self.trace.finalize_step(sample, step_idx, summary={
                "decision": "corrected",
                "reason": res.reason,
                "original_response": getattr(sample, "response", "") or "",
                "corrected_response": res.corrected_response,
            })
            self.trace.set_final(sample, {"final_decision": "accepted_corrected", "final_reason": res.reason})
            return True, {
                "is_valid": False,
                "reason": res.reason,
                "corrected_response": res.corrected_response,
                "intermediates": res.info or {},
            }

        # rollback / pass
        self.trace.finalize_step(sample, step_idx, summary={
            "decision": "pass" if res.ok_accept else "rollback",
            "reason": res.reason,
            "original_response": getattr(sample, "response", "") or "",
            "corrected_response": None,
        })
        self.trace.set_final(sample, {"final_decision": "accepted_raw", "final_reason": res.reason})
        return False, {"is_valid": True, "reason": res.reason, "intermediates": res.info or {}}

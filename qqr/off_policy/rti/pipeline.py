# qqr/off_policy/rti/pipeline.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from .trace import TraceRecorder
from .agents.judge import ValidityJudgeAgent
from .agents.corrector import CorrectorDraftAgent
from .agents.deleak import DeLeakAgent
from .agents.postcheck import PostcheckJudgeAgent

from .utils.text import truncate
from .utils.extract import extract_tool_calls_from_text
from .utils.sanitize import strip_sensitive_tool_calls
from .utils.heuristic import (
    is_obviously_bad,
    bootstrap_with_groundtruth,
    heuristic_postcheck,
)
from .utils.metrics import ref_kl_gate, get_completion_score
import re

@dataclass
class PipelineResult:
    corrected: bool
    ok_accept: bool
    reason: str
    corrected_response: Optional[str] = None
    info: Dict[str, Any] = None

class CalibrationPipeline:
    def __init__(
        self,
        *,
        trace: TraceRecorder,
        judge: ValidityJudgeAgent,
        corrector: CorrectorDraftAgent,
        deleak: DeLeakAgent,
        postcheck: PostcheckJudgeAgent,
        args,
        cfg,
    ):
        self.trace = trace
        self.judge = judge
        self.corrector = corrector
        self.deleak = deleak
        self.postcheck = postcheck
        self.args = args
        self.cfg = cfg

    def _history_text(self, messages: list[dict]) -> str:
        return json.dumps(messages, ensure_ascii=False, indent=4)

    @staticmethod
    def _agent_io(res, parsed=None) -> Dict[str, Any]:
        """Build a trace content dict carrying the agent call's full input/output.

        Mirrors the phase schema of the step-calibration dumps:
        input_prompt / raw_output / parsed_output / latency_ms.
        """
        return {
            "input_prompt": getattr(res, "prompt", ""),
            "raw_output": getattr(res, "raw", ""),
            "parsed_output": parsed if parsed is not None else res.parsed,
            "latency_ms": getattr(res, "latency_ms", -1),
        }

    def _detect_tool_format(self) -> str:
        """Return 'xml' for Qwen3.5-x models, 'json' otherwise."""
        checkpoint = getattr(self.args, "hf_checkpoint", "") or ""
        name_lower = checkpoint.replace("-", "").replace("_", "").lower()
        if ("qwen3" in name_lower and "5" in name_lower) or "qwen3.6" in name_lower or "qwen36" in name_lower:
            return "xml"
        return "json"

    def _format_qwen3(self, reasoning: str, tool_calls: list[dict], content: str = "") -> str:
        """Reconstruct a model response string with think block + tool calls.

        Produces the correct tool call serialisation for the active model:
        - JSON format  (Qwen3-8B):   <tool_call>\\n{...}\\n</tool_call>
        - XML-param format (Qwen3.5-9B): <tool_call><function=name>
<parameter=k>
v
</parameter>
</function>
</tool_call>
        """
        fmt = self._detect_tool_format()
        out = f"<think>\n{(reasoning or '').strip()}\n</think>"
        for c in tool_calls or []:
            name = c.get("name", "")
            args = c.get("arguments") or {}
            # args may be a dict or a JSON string; normalise to dict
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            if fmt == "xml":
                # Qwen3.5-9B XML-parameter format
                block = f"<tool_call><function={name}>\n"
                for k, v in args.items():
                    # Scalar values as plain string; complex values as JSON
                    if isinstance(v, (dict, list)):
                        v_str = json.dumps(v, ensure_ascii=False)
                    else:
                        v_str = str(v)
                    block += f"<parameter={k}>\n{v_str}\n</parameter>\n"
                block += "</function>\n</tool_call>"
            else:
                # Qwen3-8B JSON format
                payload = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
                block = f"<tool_call>\n{payload}\n</tool_call>"
            out += "\n\n" + block
        if content is not None and len(content) > 0:
            out += f"\n{content}"
        return out
    
    def parse_qwen3_output(self, raw: str) -> dict:
        """Parse Qwen3/Qwen3.5 output using the shared QwenPrompt parser."""
        from qqr.data.prompts.qwen import QwenPrompt
        msg = QwenPrompt().parse_assistant_content(raw)
        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            tool_calls.append({
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
            })
        return {
            "reasoning": msg.get("reasoning_content", ""),
            "tool_calls": tool_calls,
            "content": msg.get("content", ""),
        }

    async def _postcheck_and_gate(
        self,
        *,
        sample,
        step_idx: int,
        original: str,
        corrected: str,
        mode: str,
        intermediates: Dict[str, Any],
        label: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        # postcheck disabled: 直接进入 ref gate
        if not self.cfg.enable_postcheck:
            ok_gate, meta_gate = await self._ref_gate(sample, step_idx, original, corrected)
            if not ok_gate:
                return PipelineResult(False, True, "ref_gate_failed_rollback", info={"ref_gate": meta_gate, **intermediates})
            return PipelineResult(True, True, f"corrected:{mode}", corrected_response=corrected,
                                 info={"postcheck": {"disabled": True}, "ref_gate": meta_gate, **intermediates})

        # 1) heuristic postcheck
        ok_h, meta_h = heuristic_postcheck(
            corrected_response=corrected,
            correction_mode=mode,
            postcheck_allow_gt_bootstrap=self.cfg.postcheck_allow_gt_bootstrap,
            messages=getattr(sample, "messages", []) or [],
        )
        self.trace.add(sample, step_idx, type="postcheck", role="pipeline", name="heuristic",
                       content=meta_h, meta={"ok": ok_h, "mode": mode})

        if not ok_h:
            return PipelineResult(False, True, "postcheck_failed_rollback",
                                 info={"postcheck": {"heuristic": meta_h}, **intermediates})

        # 2) LLM postcheck（可选）
        meta_j: Dict[str, Any] = {"ok": True, "diagnosis": "llm_postcheck_disabled"}
        ok_j = True
        if self.cfg.use_judge_llm and self.cfg.postcheck_use_judge_llm:
            history = self._history_text(getattr(sample, "messages", []) or [])
            # history = json.dumps(sample.messages, ensure_ascii=False, indent=2)
            pres = await self.postcheck.judge(
                user_prompt=getattr(sample, "prompt", "") or "",
                history=history,
                original=json.dumps(self.parse_qwen3_output(original), indent=2),
                corrected=json.dumps(self.parse_qwen3_output(corrected), indent=2),
                tools=sample.train_metadata['tools'] if 'tools' in sample.train_metadata else None,
                label=label,
            )
            meta_j = pres.parsed if isinstance(pres.parsed, dict) else {"ok": True, "diagnosis": "postcheck_parse_fallback"}
            ok_j = bool(meta_j.get("ok", True))
            self.trace.add(sample, step_idx, type="postcheck", role="pipeline", name="llm",
                           content=self._agent_io(pres, parsed=meta_j), meta={"ok": ok_j, "mode": mode})

        if not ok_j:
            return PipelineResult(False, True, "postcheck_failed_rollback",
                                 info={"postcheck": {"heuristic": meta_h, "llm": meta_j}, **intermediates})

        # 3) ref gate
        ok_gate, meta_gate = await self._ref_gate(sample, step_idx, original, corrected)
        if not ok_gate:
            return PipelineResult(False, True, "ref_gate_failed_rollback", info={"ref_gate": meta_gate, **intermediates})

        return PipelineResult(
            corrected=True,
            ok_accept=True,
            reason=f"corrected:{mode}",
            corrected_response=corrected,
            info={"postcheck": {"heuristic": meta_h, "llm": meta_j}, "ref_gate": meta_gate, **intermediates},
        )

    async def _ref_gate(self, sample, step_idx: int, original: str, corrected: str):
        tools = None
        try:
            tools = getattr(sample, "train_metadata", None) and sample.train_metadata.get("tools")
        except Exception:
            tools = None

        ok_gate, meta_gate = await ref_kl_gate(
            args=self.args,
            sample_messages=getattr(sample, "messages", []) or [],
            original_response=original,
            corrected_response=corrected,
            tools=tools,
        )
        self.trace.add(sample, step_idx, type="ref_gate", role="pipeline", name="ref_kl_gate", content=meta_gate, meta={"ok": ok_gate})
        return ok_gate, meta_gate

    async def run(self, sample, step_idx: int) -> PipelineResult:
        original = getattr(sample, "response", "") or ""
        tool_calls = extract_tool_calls_from_text(original)

        label = getattr(sample, "label", None) or {}
        self.trace.add(
            sample, step_idx,
            type="input", role="pipeline", name="snapshot",
            content={"original": truncate(original), "parsed_tool_calls": tool_calls},
        )

        # heuristic detect
        bad, hmeta = is_obviously_bad(
            messages=getattr(sample, "messages", []) or [],
            tool_calls=tool_calls,
            gt=label.get("groundtruth"),
            strict_direction_match=self.cfg.strict_direction_match,
            tools=[i['function']['name'] for i in sample.train_metadata['tools'] or []] if 'tools' in sample.train_metadata else None
        )

        self.trace.add(sample, step_idx, type="heuristic", role="pipeline", name="is_obviously_bad",
                       content={"bad": bad, "meta": hmeta})

        # 2) 没有 judge llm：bad 就放行（不校正），good 也放行
        if not self.cfg.use_judge_llm:
            return PipelineResult(corrected=False, ok_accept=True,
                                  reason="heuristic_invalid_no_judge" if bad else "heuristic_valid_no_judge",
                                  info={"heuristic": hmeta})

        # 3) judge validity
        history_json = json.dumps(getattr(sample, "messages", []) or [], ensure_ascii=False, indent=2)
        # jres = await self.judge.judge(history_json=history_json, response_text=original, label=label)
        jres = await self.judge.judge(history_json=history_json, response_text=json.dumps(self.parse_qwen3_output(original), indent=2, ensure_ascii=False), label=label)
        self.trace.add(sample, step_idx, type="agent", role="judge", name=self.judge.name,
                       content=self._agent_io(jres), meta={"ok": jres.ok})

        is_valid = bool((jres.parsed or {}).get("is_valid", True))
        if is_valid and not bad:
            return PipelineResult(corrected=False, ok_accept=True, reason="judge_valid",
                                  info={"heuristic": hmeta, "judge": jres.parsed})

        # 4) corrector draft -> deleak -> format
        history_txt = self._history_text(getattr(sample, "messages", []) or [])
        feedback_json = json.dumps(jres.parsed or {"is_valid": False, "diagnosis": "judge_unavailable_or_parse_error"}, ensure_ascii=False)

        cres = await self.corrector.draft(
            history=history_txt,
            original_response=self.parse_qwen3_output(original),
            feedback_json=feedback_json,
            label=label,
            tools=sample.train_metadata['tools'] if 'tools' in sample.train_metadata else None
        )
        self.trace.add(sample, step_idx, type="agent", role="corrector", name=self.corrector.name,
                       content=self._agent_io(cres), meta={"ok": cres.ok})

        if not cres.ok or not isinstance(cres.parsed, dict):
            return PipelineResult(corrected=False, ok_accept=True, reason="corrector_failed_fallback",
                                  info={"heuristic": hmeta, "judge": jres.parsed, "corrector": cres.parsed})

        draft_reasoning = cres.parsed.get("reasoning", "") or ""
        draft_calls = cres.parsed.get("tool_calls", []) or []
        draft_content = cres.parsed.get("content", "") or ""
        sanitized_calls = strip_sensitive_tool_calls(draft_calls)

        # dres = await self.deleak.rewrite(history=history_txt, draft_reasoning=draft_reasoning, draft_calls=draft_calls)
        # self.trace.add(sample, step_idx, type="agent", role="deleak", name=self.deleak.name, content=truncate(dres.raw), meta={"ok": dres.ok})

        corrected = self._format_qwen3(draft_reasoning, sanitized_calls, draft_content)

        if self._detect_tool_format() == 'xml':
            corrected = corrected[8:]

        if original.endswith("<|im_end|>") and not corrected.endswith("<|im_end|>"):
            corrected += "<|im_end|>"

        self.trace.add(sample, step_idx, type="correction", role="pipeline", name="llm_corrected",
                       content={"tool_calls": sanitized_calls, "corrected": truncate(corrected)})

        mode = "judge_then_correct" if not is_valid else "heuristic_then_correct"
        ans = await self._postcheck_and_gate(
            sample=sample,
            step_idx=step_idx,
            original=original,
            corrected=corrected,
            mode=mode,
            intermediates={"heuristic": hmeta, "judge": jres.parsed, "draft": cres.parsed},
            label=label
        )

        if ans.corrected:
            output_format="tool_call_json" if self._detect_tool_format() == 'json' else 'tool_call_xml'
            corrected_prompt = self.corrector.prompt.build(history_txt, self.parse_qwen3_output(original), feedback_json, label, tools=sample.train_metadata['tools'] if 'tools' in sample.train_metadata else None, output_format=output_format)
            corrected_score = await get_completion_score(
                args =self.args, 
                sample_messages=[{"role":"user", "content": corrected_prompt}], 
                response = corrected, 
                tools=sample.train_metadata['tools'] if 'tools' in sample.train_metadata else None,
                # url=self.cfg.openai_base_url + "/completions",
                # model_id=self.cfg.openai_model
            )

            origin_prompt = self.corrector.origin_prompt.build(history_txt, self.parse_qwen3_output(original), feedback_json, label, tools=sample.train_metadata['tools'] if 'tools' in sample.train_metadata else None, output_format=output_format)
            origin_score = await get_completion_score(
                args =self.args, 
                sample_messages=[{"role":"user", "content": origin_prompt}], 
                response = corrected, 
                tools=sample.train_metadata['tools'] if 'tools' in sample.train_metadata else None,
                # url=self.cfg.openai_base_url + "/completions",
                # model_id=self.cfg.openai_model
            )

            ans.info['origin_score'] = {
                "origin_prompt_score": origin_score,
                "correct_prompt_score": corrected_score,
            }

        return ans

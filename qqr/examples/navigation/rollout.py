# rollout.py
import os
import asyncio
import logging
from argparse import Namespace
from copy import deepcopy
from datetime import datetime
from typing import Any
from qqr.data.prompts.qwen import QwenPrompt
from qqr.rollout.agent_rollout import GenerateState, MCPState
from qqr.rollout.agent_rollout import generate as base_generate
from qqr.schemas import Sample
from qqr.off_policy.step_calibration import calibrate_step
from qqr.off_policy.rti.inspector import RTIInspector
from . import config
from .reward_model import eval_reward
import random
logger = logging.getLogger(__name__)

def _build_inspector(args: Namespace) -> RTIInspector:
    """
    Build inspector from args/env if available.
    Safe defaults: enabled, heuristic-only, optional log dir from args.
    """
    # log_dir = getattr(args, "inspector_log_dir", None)
    enabled = bool(getattr(args, "inspector_enabled", False))

    # Optional OpenAI judge settings
    use_judge_llm = bool(getattr(args, "inspector_use_judge_llm", False))
    openai_base_url = getattr(args, "inspector_openai_base_url", None)
    openai_api_key = getattr(args, "inspector_openai_api_key", "EMPTY")
    openai_model = getattr(args, "inspector_openai_model", None)

    if openai_base_url is None or openai_model is None:
        openai_base_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}".rstrip("/") + "/v1"
        openai_model = (
            getattr(args, "ref_model_id", None)
            or getattr(args, "model", None)
            or getattr(args, "hf_checkpoint", None)
        )
        openai_api_key = "EMPTY"
    
    cfg = config.InspectorConfig(
        enabled=enabled,
        # log_dir=log_dir,
        use_judge_llm=use_judge_llm,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        enable_postcheck=float(getattr(args, "enable_postcheck", False)),
        temperature=float(getattr(args, "inspector_temperature", 0.0)),
        max_new_tokens=int(getattr(args, "inspector_max_new_tokens", 800)),
        timeout=int(getattr(args, "inspector_timeout", 300)),
        allow_gt_bootstrap=bool(getattr(args, "inspector_allow_gt_bootstrap", False)),
        strict_direction_match=bool(getattr(args, "inspector_strict_direction_match", False)),
    )

    # If a custom inspector method path is specified, dynamically load and delegate to it.
    # e.g. custom-inspector-method-path: qqr.off_policy.rti.self_correct
    custom_inspector_method_path = getattr(args, "custom_inspector_method_path", None)
    if custom_inspector_method_path:
        import importlib
        mod = importlib.import_module(custom_inspector_method_path)
        return mod.build_inspector(cfg, args)

    return RTIInspector(cfg, args=args)


from typing import List, Tuple


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    await MCPState(config.mcp_server_config_fn).get_servers()

    if isinstance(sample.prompt, str):
        sample.messages = [{"role": "user", "content": sample.prompt}]
    else:
        sample.messages = deepcopy(sample.prompt)

    samples = await agent_loop(args, sample, sampling_params, evaluation=evaluation)

    if evaluation:
        sample = samples[-1]
        await eval_reward(args, sample)
        return sample
    else:
        return samples


from datetime import datetime

# Tool call format example strings keyed by format style.
# "json"  → Qwen3-8B style
# "xml"   → Qwen3.5-9B style
_TOOL_CALL_EXAMPLE = {
    "json": (
        '<tool_call>{"name": "tool_name1", "arguments": {"arg1": "value1"}}</tool_call>\n'
        '<tool_call>{"name": "tool_name2", "arguments": {"arg1": "value1"}}</tool_call>'
    ),
    "xml": (
        "<tool_call><function=tool_name1>\n"
        "<parameter=arg1>\nvalue1\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call><function=tool_name2>\n"
        "<parameter=arg1>\nvalue1\n</parameter>\n"
        "</function>\n</tool_call>"
    ),
}


def _detect_tool_format(args) -> str:
    """Detect tool-call format from hf_checkpoint path.

    Returns ``"xml"`` for Qwen3.5-x models, ``"json"`` otherwise.
    """
    checkpoint = getattr(args, "hf_checkpoint", "") or ""
    name_lower = checkpoint.replace("-", "").replace("_", "").lower()
    # Qwen3.5-* models use the XML-parameter chat template
    if ("qwen3" in name_lower and "5" in name_lower) or "qwen3.6" in name_lower or "qwen36" in name_lower:
        return "xml"
    return "json"


def build_system_message(step_idx: int, max_steps: int, tool_format: str = "json") -> dict:
    tool_example = _TOOL_CALL_EXAMPLE.get(tool_format, _TOOL_CALL_EXAMPLE["json"])
    system_prompt = f"当前时间：{datetime.now().strftime('%d/%m/%Y')}\n"
    system_prompt += f"""
        你是一个智能助手，擅长通过严谨思考并使用工具来解决问题。
        【输出格式与约束】
        - 在不超过 {max_steps} 轮对话内完成用户任务（请合理规划每一步）。
        - 最后一轮对话之前你必须根据用户当前信息、用户请求、历史交互信息做出推理和行动，按照以下格式输出：
        <think>你的推理和下一步行动的计划
        </think>
        {tool_example}
        - 要求最后一次工具调用必须调用但此direction工具，规划好origin，destination以及waypoints。
        - 最后一轮对话可直接回答。
        """.strip()

    return {"role": "system", "content": system_prompt}

async def agent_loop(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    max_steps: int = config.max_steps,
    evaluation: bool = False
) -> list[Sample]:
    state = GenerateState(args)
    mcp_state = MCPState(config.mcp_server_config_fn)
    prompter = QwenPrompt()
    inspector = _build_inspector(args)
    tool_format = _detect_tool_format(args)
    if getattr(args, "rollout_max_turns"):
        max_steps = getattr(args, "rollout_max_turns")

    if sample.messages[0]["role"] != "system":
        sample.messages.insert(0, build_system_message(0, max_steps, tool_format))
    samples: list[Sample] = []

    # 在 rollout 开始前通过 API 更新 label 中的 POI 坐标，
    # 确保 inspector/OPD/reward 看到的 groundtruth 是最新的
    from .label_updater import update_label_locations
    await update_label_locations(sample)

    # 训练时随机开启 inspector（如果 label 里有 inspector 标志则按比例）
    if isinstance(sample.label, dict):
        sample.label["inspector"] = (random.random() < getattr(args, "inspector_ratio", 1.0)) if sample.label.get("inspector") else False

    for step_idx in range(max_steps):
        samples.append(
            Sample(
                group_index=sample.group_index,
                index=sample.index,
                messages=deepcopy(sample.messages),
                prompt=sample.prompt,
                label=sample.label,
                status=Sample.Status.PENDING,
                metadata=sample.metadata,
                train_metadata={"tools": mcp_state.tools},
            )
        )
        sample = samples[-1]
        sample.messages[0] = build_system_message(step_idx, max_steps, tool_format)

        # 1) model generate
        sample = await base_generate(args, sample, sampling_params)
        sample.teacher_log_probs = sample.rollout_log_probs
        # 2) inspector calibration (BEFORE parsing tool calls & calling tools)
        await calibrate_step(args, sample, inspector, step_idx, sampling_params)

        # 3) parse tool calls and decide whether to stop
        sample.response_message = prompter.parse_response(sample.response, mcp_state.tools, tokenizer=state.tokenizer)
        tool_calls = sample.response_message.get("tool_calls") or []
        sample.messages.append(sample.response_message)

        if not tool_calls:
            break

        # 5) call tools
        tool_call_tasks = [mcp_state.call_tool(t) for t in tool_calls]
        tool_responses = await asyncio.gather(*tool_call_tasks)
        sample.messages.extend(tool_responses)

    else:
        # exceeded max_steps: final answer, disable tools
        samples.append(
            Sample(
                group_index=sample.group_index,
                index=sample.index,
                messages=deepcopy(sample.messages),
                prompt=sample.prompt,
                label=sample.label,
                status=Sample.Status.PENDING,
                metadata=sample.metadata,
                train_metadata=None,
            )
        )
        sample = samples[-1]
        sample.messages[0] = build_system_message(max_steps, max_steps, tool_format)
        sample = await base_generate(args, sample, sampling_params)
        sample.teacher_log_probs = sample.rollout_log_probs

        sample.response_message = prompter.parse_response(sample.response, mcp_state.tools, tokenizer=state.tokenizer)
        sample.messages.append(sample.response_message)
        
    # Convert assistant messages to parsed structure for training consumption
    sample = samples[-1]
    # for i, message in enumerate(sample.messages):
    #     if message["role"] == "assistant":
    #         sample.messages[i] = prompter.parse_assistant_content(message["content"])

    # Temporary padding to avoid trimming
    padding_num = (max_steps + 1) - len(samples)
    if padding_num > 0:
        samples = [
            Sample(
                group_index=sample.group_index,
                index=-1,
                tokens=[state.tokenizer.pad_token_id],
                reward=0.0,
                loss_mask=[],
                rollout_log_probs=[],
                teacher_log_probs=[],
            )
            for _ in range(padding_num)
        ] + samples

    return samples
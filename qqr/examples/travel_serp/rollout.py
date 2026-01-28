"""
Travel Agent Rollout (SerpApi version)
Tools: Google Maps, Google Flights, Google Search (all via SerpApi)
"""
import asyncio
import logging
from argparse import Namespace
from copy import deepcopy
from datetime import datetime
from typing import Any

from qqr.data.prompts.qwen3 import Qwen3Prompt
from qqr.rollout.agent_rollout import GenerateState, MCPState
from qqr.rollout.agent_rollout import generate as base_generate
from qqr.schemas import Sample

from . import config
from .reward_model import eval_reward

logger = logging.getLogger(__name__)


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    await MCPState(config.mcp_server_config_fn).get_mcp_servers()

    if isinstance(sample.prompt, str):
        sample.messages = [{"role": "user", "content": sample.prompt}]
    else:
        sample.messages = deepcopy(sample.prompt)

    samples = await agent_loop(args, sample, sampling_params)

    if evaluation:
        sample = samples[-1]
        await eval_reward(args, sample)
        return sample
    else:
        return samples


def build_system_message(step_idx: int, max_steps: int) -> dict:
    system_prompt = f"""当前时间: {datetime.now().strftime("%d/%m/%Y, %H:%M")}"""
    system_prompt += f"""\n\n可调用{max_steps}轮工具，已调用{step_idx}轮。"""

    if step_idx >= max_steps:
        system_prompt += """\n\n请直接回答，不要使用工具。"""

    return {"role": "system", "content": system_prompt}


async def agent_loop(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    max_steps: int = None,
) -> list[Sample]:
    if max_steps is None:
        max_steps = config.max_steps

    state = GenerateState(args)
    mcp_state = MCPState(config.mcp_server_config_fn)
    prompter = Qwen3Prompt()

    if sample.messages[0]["role"] != "system":
        sample.messages.insert(0, build_system_message(0, max_steps))
    samples = []

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
        sample.messages[0] = build_system_message(step_idx, max_steps)
        sample = await base_generate(args, sample, sampling_params)

        sample.messages.append(
            {
                "role": "assistant",
                "content": sample.response.removesuffix(state.tokenizer.eos_token),
            }
        )
        sample.response_message = prompter.parse_assistant_content(sample.response)
        tool_calls = sample.response_message.get("tool_calls") or []

        if not tool_calls:
            break

        tool_call_tasks = [mcp_state.call_tool(t) for t in tool_calls]
        tool_responses = await asyncio.gather(*tool_call_tasks)
        sample.messages.extend(tool_responses)

    else:
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
        sample.messages[0] = build_system_message(max_steps, max_steps)
        sample = await base_generate(args, sample, sampling_params)
        sample.messages.append(
            {
                "role": "assistant",
                "content": sample.response.removesuffix(state.tokenizer.eos_token),
            }
        )
        sample.response_message = prompter.parse_assistant_content(sample.response)

    sample = samples[-1]
    for i, message in enumerate(sample.messages):
        if message["role"] == "assistant":
            sample.messages[i] = prompter.parse_assistant_content(message["content"])

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
            )
            for _ in range(padding_num)
        ] + samples

    return samples

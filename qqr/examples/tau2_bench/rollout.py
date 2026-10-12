"""
tau2_bench rollout - simplified design.

Key insight: Each step_count corresponds to ONE agent generation.
- User turn is handled internally (multi-tool calls wrapped in a function)
- Agent sees user response as a single external input
- Each step: Agent generates -> if tools, execute and continue Agent generation
"""

import asyncio
import json
import logging
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import httpx

from qqr.data.prompts.qwen import QwenPrompt
from qqr.rollout.agent_rollout import GenerateState
from qqr.rollout.agent_rollout import generate as base_generate
from qqr.schemas import Sample
from qqr.tools.tau2_bench import (
    DOMAINS,
    execute_tool_call,
    get_environment,
    get_tool_definitions,
    get_user_tool_definitions,
)
from qqr.off_policy.rti.inspector import RTIInspector
from qqr.off_policy.step_calibration import calibrate_step
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

    custom_inspector_method_path = getattr(args, "custom_inspector_method_path", None)
    if custom_inspector_method_path:
        import importlib
        mod = importlib.import_module(custom_inspector_method_path)

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
        return mod.build_inspector(cfg, args)

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
    return RTIInspector(cfg, args=args)


# ── Constants ─────────────────────────────────────────────────────────────────

_STOP_TOKEN = "###STOP###"
_TRANSFER_TOKEN = "###TRANSFER###"
_OUT_OF_SCOPE_TOKEN = "###OUT-OF-SCOPE###"

DEFAULT_FIRST_AGENT_MESSAGE = "Hi! How can I help you today?"

AGENT_INSTRUCTION = """\
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."""

_USER_SIM_GUIDELINES = """\
# User Simulation Guidelines

You are playing the role of a customer contacting a customer service representative agent. 
Your goal is to simulate realistic customer interactions while following specific scenario instructions.
You have some tools to perform the actions on your end that might be requested by the agent to diagnose and resolve your issue.

## Core Principles
- Generate one message at a time, maintaining natural conversation flow.
- At each turn you can either:
    - Send a message to the agent.
    - Make a tool call to perform an action requested by the agent.
    - You cannot do both at the same time.
- Strictly follow the scenario instructions you have received.
- Never make up or hallucinate information not provided in the scenario instructions. Information that is not provided in the scenario instructions should be considered unknown or unavailable.
- Never make up the results of tool calls that the agent has requested, you must ground your responses based on the results of tool calls if the agent has requested.
- If you made an error in a tool call and get an error message, fix the error and try again.
- All the information you provide to the agent must be grounded in the information provided in the scenario instructions or the results of tool calls.
- Avoid repeating the exact instructions verbatim. Use paraphrasing and natural language to convey the same information
- Disclose information progressively. Wait for the agent to ask for specific information before providing it.
- Only call a tool if the agent has requested it or if it is necessary to answer a question the agent has asked. Ask clarifying questions if you do not know what action to take.
- If the agent asks multiple actions to perform, state that you cannot perform multiple actions at once, and ask the agent to instruct you one action at a time.
- Your messages when performing tool calls will not be displayed to the agent, only the messages without tool calls will be displayed to the agent.

## Task Completion
- The goal is to continue the conversation until the task is complete.
- If the instruction goal is satisified, generate the '###STOP###' token to end the conversation.
- If you have been transferred to another agent, generate the '###TRANSFER###' token to indicate the transfer. Only do this after the agent has clearly indicated that you are being transferred.
- If you find yourself in a situation in which the scenario does not provide enough information for you to continue the conversation, generate the '###OUT-OF-SCOPE###' token to end the conversation.

Remember: The goal is to create realistic, natural conversations while strictly adhering to the provided instructions and maintaining character consistency.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

_POLICY_CACHE: dict[str, str] = {}


def _load_policy(domain: str) -> str:
    if domain not in _POLICY_CACHE:
        path = config.POLICY_PATHS.get(domain)
        if path is None:
            _POLICY_CACHE[domain] = f"You are a {domain} customer service agent."
        elif isinstance(path, list):
            # Multi-file policy (e.g. telecom: main_policy + tech_support_policy).
            # Mirrors TelecomEnvironment.get_environment() concatenation format.
            tag_names = [
                "main_policy",
                "tech_support_policy",
            ]
            parts = []
            for i, p in enumerate(path):
                p = Path(p)
                if p.exists():
                    tag = tag_names[i] if i < len(tag_names) else f"policy_{i}"
                    parts.append(f"<{tag}>\n{p.read_text(encoding='utf-8')}\n</{tag}>")
            _POLICY_CACHE[domain] = "\n".join(parts) if parts else f"You are a {domain} customer service agent."
        else:
            p = Path(path)
            _POLICY_CACHE[domain] = p.read_text(encoding="utf-8") if p.exists() else f"You are a {domain} customer service agent."
    return _POLICY_CACHE[domain]


def _build_agent_system_message(domain: str, step_idx: int, max_steps: int) -> dict:
    policy = _load_policy(domain)
    content = (
        f"[Step {step_idx + 1}/{max_steps}]\n"
        f"<instructions>\n{AGENT_INSTRUCTION}\n</instructions>\n"
        f"<policy>\n{policy}\n</policy>"
    )
    return {"role": "system", "content": content}


def _build_user_sim_system_message(user_scenario: dict) -> dict:
    """Build system message for user simulator."""
    instructions = user_scenario.get("instructions", {})
    if isinstance(instructions, dict):
        scenario_str = json.dumps(instructions, ensure_ascii=False, indent=2)
    else:
        scenario_str = str(instructions)

    # Include persona description if present (mirrors tau2-bench user simulation)
    persona = user_scenario.get("persona")
    if persona:
        persona_section = f"\n\n<persona>\n{persona}\n</persona>"
    else:
        persona_section = ""

    content = _USER_SIM_GUIDELINES + persona_section + f"\n\n<scenario>\n{scenario_str}\n</scenario>"
    return {"role": "system", "content": content}


def _is_stop_message(content: str) -> bool:
    """Check if message contains stop tokens."""
    if not content:
        return False
    return _STOP_TOKEN in content or _TRANSFER_TOKEN in content or _OUT_OF_SCOPE_TOKEN in content

async def _call_user_simulator(
    args,
    user_messages: list[dict],
    user_system_msg: dict,
    user_tool_defs: Optional[list] = None
) -> dict:
    """Call user simulator LLM with timeout retry. Returns dict with 'content' and optional 'tool_calls'."""
    base_url = getattr(args, "user_sim_openai_base_url", "") or config.user_sim_openai_base_url
    api_key = getattr(args, "user_sim_openai_api_key", "") or config.user_sim_openai_api_key
    model = getattr(args, "user_sim_model", "") or config.user_sim_model
    temperature = float(getattr(args, "user_sim_temperature", config.user_sim_temperature))
    max_tokens = int(getattr(args, "user_sim_max_new_tokens", config.user_sim_max_new_tokens))
    timeout = int(getattr(args, "user_sim_timeout", config.user_sim_timeout))
    max_retries = int(getattr(args, "user_sim_max_retries", config.user_sim_max_retries))
    base_backoff = float(getattr(args, "user_sim_retry_base_backoff", config.user_sim_retry_base_backoff))
    max_backoff = float(getattr(args, "user_sim_retry_max_backoff", config.user_sim_retry_max_backoff))

    # NOTE: user_messages are already in user simulator perspective (flipped by caller)
    # user simulator plays "assistant" role, agent messages are "user" role
    messages = [user_system_msg] + user_messages
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    if user_tool_defs:
        payload["tools"] = user_tool_defs
        payload["tool_choice"] = "auto"

    headers = {"Authorization": f"Bearer {api_key}"}
    url = base_url.rstrip("/") + "/chat/completions"

    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                message = data["choices"][0]["message"]

                content = message.get("content") or ""
                # Only clean for Qwen3.5-397B-A17B model
                if content and '</think>' in content:
                    # Remove everything before and including </think>
                    if '</think>' in content:
                        content = content.split('</think>', 1)[1]
                        # Optional: strip leading whitespace/newlines
                        content = content.lstrip('\n').lstrip()

                tool_calls_raw = message.get("tool_calls") or []
                result = {
                    "content": content,
                    "tool_calls": None,
                }

                if tool_calls_raw:
                    result["tool_calls"] = [
                        {
                            "id": tc.get("id", ""),
                            "type": tc.get("type", "function"),
                            "function": {
                                "name": tc.get("function", {}).get("name", ""),
                                "arguments": tc.get("function", {}).get("arguments", "{}"),
                            },
                        }
                        for tc in tool_calls_raw
                    ]

                # think-only response: no content and no tool_calls → retry
                if not content and not result["tool_calls"]:
                    if attempt < max_retries:
                        backoff = min(base_backoff * (2 ** attempt), max_backoff)
                        logger.warning(
                            f"[user_sim] Empty content with no tool_calls (think-only response), "
                            f"attempt {attempt + 1}/{max_retries + 1}. Retrying in {backoff}s..."
                        )
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        logger.warning("[user_sim] Empty content after all retries. Using fallback stop.")
                        return {"content": _STOP_TOKEN, "tool_calls": None}

                return result
        except Exception as e:
            last_exception = e
            is_retryable = False
            if isinstance(e, (httpx.TimeoutException, asyncio.TimeoutError,
                              httpx.ConnectError, httpx.NetworkError, httpx.ReadError)):
                is_retryable = True
            elif isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500:
                is_retryable = True

            if is_retryable and attempt < max_retries:
                backoff = min(base_backoff * (2 ** attempt), max_backoff)
                logger.warning(
                    f"[user_sim] LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {backoff}s..."
                )
                await asyncio.sleep(backoff)
                continue
            else:
                break

    logger.warning(
        f"[user_sim] LLM call failed after {max_retries + 1} attempts: {last_exception}. Using fallback stop."
    )
    return {"content": _STOP_TOKEN, "tool_calls": None}


async def _execute_tool_calls(env, tool_calls: list[dict]) -> list[dict]:
    """Execute tool calls and return tool response messages."""
    return list(await asyncio.gather(*[execute_tool_call(env, tc) for tc in tool_calls]))


async def _generate_user_response(
    args,
    agent_messages: list[dict],
    user_sim_system: dict,
    user_tool_defs: Optional[list],
    env,
    max_user_turns: int = 50
) -> tuple[list[dict], bool]:
    """
    Generate complete user response with proper user/tool alternating turns.
    
    This function maintains the natural dialogue format where each user tool call
    and its result are recorded as separate messages, preserving the original
    user -> tool -> user -> tool structure.
    
    Args:
        agent_messages: Current conversation history (agent perspective)
        user_sim_system: User simulator system message
        user_tool_defs: User tool definitions
        env: Environment for executing user tools
        max_user_turns: Max user tool-call rounds
        
    Returns:
        (user_messages_list, should_stop)
        - user_messages_list: List of all user messages (tool calls, results, and final text)
                              in the order they occurred, maintaining user/tool alternation
        - should_stop: Whether user sent stop token
    """
    # Build user simulator history from agent perspective (inverted):
    #   agent "assistant" -> user_sim "user"   (user_sim sees agent as "user")
    #   agent "user"      -> user_sim "assistant" (user_sim sees itself as "assistant")
    #   agent "tool"      -> user_sim "tool"   (tool results are kept as-is)
    #
    # IMPORTANT: When agent "assistant" message has tool_calls, we must preserve the
    # tool_calls field in the mapped "user" message.  Without it the subsequent "tool"
    # result messages are orphaned (no matching tool_call_id), which causes the LLM
    # to become confused and generate agent-style responses instead of customer responses.
    user_sim_history = []
    for m in agent_messages:
        role = m.get("role", "")
        if role == "assistant":
            entry: dict = {"role": "user", "content": m.get("content", "")}
            # Preserve agent tool_calls so downstream tool results are not orphaned
            if m.get("tool_calls"):
                entry["tool_calls"] = m["tool_calls"]
            user_sim_history.append(entry)
        elif role == "user":
            user_sim_history.append({"role": "assistant", "content": m.get("content", "")})
        elif role == "tool":
            user_sim_history.append({
                "role": "tool",
                "content": m.get("content", ""),
                "tool_call_id": m.get("tool_call_id", "")
            })

    user_messages_list = []  # All messages in order: user tool calls, tool results, final text

    for _ in range(max_user_turns):
        user_result = await _call_user_simulator(
            args, user_sim_history, user_sim_system, user_tool_defs
        )
        user_content = user_result.get("content", "").strip()
        user_tool_calls = user_result.get("tool_calls")

        if user_tool_calls:
            # User made tool calls - execute each and add to history
            for tc in user_tool_calls:
                tc["requestor"] = "user"  # Mark as user-initiated for env routing
                
                # Create tool call message for user simulator history
                user_sim_history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tc["id"],
                        "type": "function",
                        "function": tc["function"],
                    }]
                })
                
                # Add tool call message to output list (maintains user -> tool structure)
                user_messages_list.append({
                    "role": "user",
                    "content": None,
                    "tool_calls": [{
                        "id": tc["id"],
                        "type": "function",
                        "function": tc["function"],
                    }]
                })

            # Execute user tool calls
            user_tool_responses = await _execute_tool_calls(env, user_tool_calls)
            for tr in user_tool_responses:
                # Add tool response to user simulator history
                user_sim_history.append({
                    "role": "tool",
                    "content": tr.get("content", ""),
                    "tool_call_id": tr.get("tool_call_id", ""),
                })
                # Add tool result message to output list
                user_messages_list.append({
                    "role": "tool",
                    "content": tr.get("content", ""),
                    "tool_call_id": tr.get("tool_call_id", ""),
                })
            # Continue internal loop - user may make more tool calls
            continue
        else:
            # User sent text message - this ends the user turn
            should_stop = _is_stop_message(user_content)
            
            # Add final user text message
            if user_content:
                user_messages_list.append({
                    "role": "user",
                    "content": user_content
                })
            
            return user_messages_list, should_stop

    # Max user turns reached - return whatever we have
    logger.warning("[user_sim] Max user turns reached")
    if _is_stop_message(user_content):
        print("[user_sim] Stop token detected")
    return user_messages_list, _is_stop_message(user_content) if 'user_content' in locals() else False


# ── Main functions ────────────────────────────────────────────────────────────

async def generate(args, sample, sampling_params, evaluation=False):
    if isinstance(sample.prompt, str):
        sample.messages = [{"role": "user", "content": sample.prompt}]
    else:
        sample.messages = deepcopy(sample.prompt)

    try:
        samples = await agent_loop(args, sample, sampling_params, evaluation=evaluation)
    except Exception as e:
        import pydantic
        if isinstance(e, pydantic.ValidationError):
            logger.error(
                f"[tau2_bench] Pydantic ValidationError in agent_loop "
                f"(sample {getattr(sample, 'index', '?')}, "
                f"domain={((sample.label or {}).get('domain', '?') if isinstance(sample.label, dict) else '?')}): "
                f"{e}"
            )
            raise
        raise

    if evaluation:
        # Find the last actual (non-padding) sample.
        # Padding samples have index=-1 and no label; they are prepended by
        # agent_loop to align with max_steps.  For eval we only want the
        # real agent-turn sample so that domain / reward are available.
        s = None
        for candidate in reversed(samples):
            if getattr(candidate, "index", None) != -1:
                s = candidate
                break
        if s is None:
            # No actual sample produced (e.g. user_sim failed on first turn).
            # Return the original sample marked as ABORTED so the eval
            # pipeline can still count it correctly.
            logger.warning(
                f"[tau2_bench] agent_loop produced no actual samples "
                f"(sample {getattr(sample, 'index', '?')}), returning ABORTED"
            )
            sample.status = Sample.Status.ABORTED
            sample.reward = 0.0
            return sample
        try:
            await eval_reward(args, s)
        except Exception as e:
            import pydantic
            if isinstance(e, pydantic.ValidationError):
                logger.error(
                    f"[tau2_bench] Pydantic ValidationError in eval_reward "
                    f"(sample {getattr(s, 'index', '?')}, "
                    f"domain={((s.label or {}).get('domain', '?') if isinstance(s.label, dict) else '?')}): "
                    f"{e}"
                )
                raise
            raise
        # Clean up non-serializable objects from metadata before returning.
        # The 'env' object (pydantic model with complex state) cannot be
        # safely serialized by Ray and may trigger ValidationError.
        if s.metadata and "env" in s.metadata:
            del s.metadata["env"]
        return s
    return samples


async def agent_loop(args, sample, sampling_params, max_steps=config.max_steps, evaluation=False):
    """
    Main rollout loop.
    
    Each step_count = ONE agent generation (may include agent tool calls).
    User turn is handled internally (_generate_user_response).
    """
    state = GenerateState(args)
    prompter = QwenPrompt()
    if getattr(args, "rollout_max_turns", None):
        max_steps = int(args.rollout_max_turns)
    inspector = _build_inspector(args)
    label = sample.label or {}
    reference = label if isinstance(label, dict) else {}
    domain = reference.get("domain", "airline")
    user_scenario = reference.get("user_scenario", {})
    initial_state = reference.get("initial_state")
    db_path = reference.get("db_path", None)

    env = get_environment(domain, db_path=db_path)
    if initial_state:
        _apply_initial_state(env, initial_state)

    tool_defs = get_tool_definitions(domain)
    user_tool_defs = get_user_tool_definitions(domain)
    user_sim_system = _build_user_sim_system_message(user_scenario)

    # Initialize conversation with agent's first message
    sample.messages = [_build_agent_system_message(domain, 0, max_steps)]
    sample.messages.append({"role": "assistant", "content": DEFAULT_FIRST_AGENT_MESSAGE})

    samples = []

    # 训练时随机开启 inspector（如果 label 里有 inspector 标志则按比例）
    if isinstance(sample.label, dict):
        sample.label["inspector"] = (random.random() < getattr(args, "inspector_ratio", 1.0)) if sample.label.get("inspector") else False

    # Full trajectory for evaluation - maintains complete user/tool alternating structure.
    # sample.messages and trajectory are kept in sync with identical structure.
    trajectory = deepcopy(sample.messages)

    agent_waiting_user = True   # first turn must be user


    for step_idx in range(max_steps):
        # =====================================================================
        # USER TURN (only when agent finished without tool calls)
        # =====================================================================
        if agent_waiting_user:
            user_messages_list, should_stop = await _generate_user_response(
                args, sample.messages, user_sim_system, user_tool_defs, env
            )

            # Add all user messages (tool calls + results + final text) to both
            # sample.messages and trajectory, maintaining the original structure
            for msg in user_messages_list:
                sample.messages.append(deepcopy(msg))
                trajectory.append(deepcopy(msg))

            # Check if user wants to stop (last message should be user text with stop token)
            if should_stop:
                if samples:
                    last = samples[-1]
                    if last.metadata is None:
                        last.metadata = {}
                    last.metadata["termination_reason"] = "user_stop"
                    last.metadata["full_trajectory"] = deepcopy(trajectory)
                    last.metadata["domain"] = domain
                    last.metadata["env"] = env
                break

        # =====================================================================
        # AGENT TURN (one step_count = one agent generation)
        # Agent may make multiple tool calls, all handled within this step
        # =====================================================================
        
        # Create sample for this agent generation
        current_sample = Sample(
            group_index=sample.group_index, index=sample.index,
            messages=deepcopy(sample.messages), prompt=sample.prompt,
            label=sample.label, status=Sample.Status.PENDING,
            metadata=sample.metadata, train_metadata={"tools": tool_defs}
        )
        current_sample.messages[0] = _build_agent_system_message(domain, step_idx, max_steps)

        # Agent generates (may include tool calls in response)
        current_sample = await base_generate(args, current_sample, sampling_params)
        current_sample.teacher_log_probs = deepcopy(current_sample.rollout_log_probs)

        # Context overflow short-circuit:
        # base_generate sets status=TRUNCATED and returns an empty response
        # when prompt_ids exceeds max_context_length - 100 (see generate() in
        # agent_rollout.py).  If we keep looping, every subsequent step will:
        #   - skip inspector token replacement (prompt_prefix is a dummy),
        #   - parse empty response -> no tool_calls,
        #   - fall through to _generate_user_response which blocks on an
        #     external LLM endpoint,
        # causing the whole rollout to hang for up to max_steps iterations
        # without issuing any further /generate requests.  Terminate early.
        if current_sample.status == Sample.Status.TRUNCATED:
            logger.warning(
                f"[tau2_bench] step={step_idx} base_generate returned TRUNCATED "
                f"(context overflow); stopping agent_loop early"
            )
            if current_sample.metadata is None:
                current_sample.metadata = {}
            current_sample.metadata["termination_reason"] = "context_overflow"
            samples.append(current_sample)
            break

        # RTI inspector calibration
        await calibrate_step(args, current_sample, inspector, step_idx, sampling_params)

        # Parse agent response
        current_sample.response_message = prompter.parse_response(current_sample.response, tool_defs, tokenizer=state.tokenizer)
        agent_tool_calls = current_sample.response_message.get("tool_calls") or []
        current_sample.messages.append(current_sample.response_message)
        trajectory.append(deepcopy(current_sample.response_message))

        # Execute agent tool calls if any
        if agent_tool_calls:
            tool_responses = await _execute_tool_calls(env, agent_tool_calls)
            current_sample.messages.extend(tool_responses)
            trajectory.extend(deepcopy(tool_responses))
            agent_waiting_user = False
        else:
            agent_waiting_user = True

        # Store this step's sample
        samples.append(current_sample)
        
        # Update sample.messages for next iteration
        sample.messages = deepcopy(current_sample.messages)

    else:
        # Max steps reached
        if samples:
            last = samples[-1]
            if last.metadata is None:
                last.metadata = {}
            last.metadata["termination_reason"] = "max_steps"

    # Store final trajectory in last sample for reward calculation
    if samples:
        last = samples[-1]
        if last.metadata is None:
            last.metadata = {}
        last.metadata["full_trajectory"] = deepcopy(trajectory)
        last.metadata["domain"] = domain
        last.metadata["env"] = env

    # Padding
    padding_num = (max_steps + 1) - len(samples)
    if padding_num > 0:
        state = GenerateState(args)
        samples = [Sample(group_index=sample.group_index, index=-1,
                         tokens=[state.tokenizer.pad_token_id],
                         reward=0.0, loss_mask=[],
                         rollout_log_probs=[], teacher_log_probs=[])
                  for _ in range(padding_num)] + samples

    return samples


def _apply_initial_state(env, initial_state: dict) -> None:
    try:
        from tau2.data_model.tasks import InitializationData, EnvFunctionCall
        init_data_raw = initial_state.get("initialization_data")
        init_actions_raw = initial_state.get("initialization_actions") or []
        init_data = InitializationData(**init_data_raw) if init_data_raw else None
        init_actions = [EnvFunctionCall(**a) for a in init_actions_raw]
        if init_data and init_data.agent_data and env.tools is not None:
            env.tools.update_db(init_data.agent_data)
        if init_data and init_data.user_data and env.user_tools is not None:
            env.user_tools.update_db(init_data.user_data)
        for action in init_actions:
            env.run_env_function_call(action)
    except Exception as e:
        logger.warning(f"[tau2_bench] Failed to apply initial_state: {e}")

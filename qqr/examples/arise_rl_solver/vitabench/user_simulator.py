"""
VitaBench UserSimulator - simulates the multi-turn user interaction

Adapted from the official VitaBench UserSimulator implementation:
- The system prompt carries the persona (user profile) and the instructions
- The user reveals requirements over several turns rather than all at once
- Emits ###STOP### to end the conversation once every task is done
- Roles are flipped: from the UserSimulator's view the agent is the user and it is the assistant

Call an external LLM (Qwen3-235B) to simulate the user's reply.
"""

import asyncio
import logging
import re
from copy import deepcopy

from openai import AsyncOpenAI

from . import config


def _strip_think(text: str) -> str:
    """
    Strip the thinking section from the model output and return the final reply.

    Handles two cases:
    1. The complete tag: <think>...</think>actual_response
    2. The API stripped the opening tag, leaving: [thinking]</think>actual_response
    """
    # handle a complete <think>...</think> block first
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # then handle a stray closing </think> left behind after the API strips the opening tag
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip()

logger = logging.getLogger(__name__)

STOP_SIGNAL = "###STOP###"

# adapted from vitabench/src/vita/prompts/user_system_prompt.yaml
USER_SIMULATOR_PROMPT = """# 角色设定
你在扮演一名与智能体交互的用户。你的人物设定在<persona>标签中，你的任务是将<instructions>中的内容通过用户对话的形式传达给智能体。

<persona>
{persona}
</persona>
<instructions>
{instructions}
</instructions>

# 对话方式规则：
- 每次只生成一行内容来模拟用户消息
- 采用情境说明+需求表达的组合方式，先描述背景情况，再提出具体需求
- 当需要做决定时，提供instructions中的条件和偏好，让智能体帮你选择
- 使用"你觉得哪个更合适"、"你推荐哪个"等表述来寻求智能体的建议
- 通过语言、情绪、用词方式体现persona中的人物特征

# 信息透露规则：
- 将instructions中的信息拆解成多个独立的信息点，分别在不同轮次中提及
- 必须确保instructions中的每一个细节都在对话过程中被原样提及
- 避免在第一轮就说出所有需求，要让信息逐步展开
- 不要虚构instructions中未提供的信息
- 如果智能体询问是否需要帮助下单，回答"是的，请帮我下单"
- 如果对相同问题重复问3次以上没有推进，表现出不耐烦

# 何时不能结束对话：
- 当你还未清楚地表达所有的需求时
- 当智能体尚未完成所有需求时
- 当完成结果和你的instructions中的期望不一致时

# 何时可以结束对话：
- 当且仅当以上所有条件都满足，且任务被正确完成时，回复 '###STOP###'"""

# OTA domain only: requirements are revealed faster and more directly
USER_SIMULATOR_PROMPT_OTA = """# 角色设定
你在扮演一名与智能体交互的用户。你的人物设定在<persona>标签中，你的任务是将<instructions>中的内容通过用户对话的形式传达给智能体。

<persona>
{persona}
</persona>
<instructions>
{instructions}
</instructions>

# 对话方式规则：
- 每次只生成一行内容来模拟用户消息
- 采用情境说明+需求表达的组合方式，先描述背景情况，再提出具体需求
- 当需要做决定时，提供instructions中的条件和偏好，让智能体帮你选择
- 使用"你觉得哪个更合适"、"你推荐哪个"等表述来寻求智能体的建议
- 通过语言、情绪、用词方式体现persona中的人物特征

# 信息透露规则：
- 将instructions中的信息分 2-3 轮透露，每轮尽量多说几个细节，不要过度拆分
- 必须确保instructions中的每一个细节都在对话过程中被原样提及
- 第一轮可以说出主要需求和大部分关键信息，后续轮次补充剩余细节
- 不要虚构instructions中未提供的信息
- 当智能体明确询问某个具体参数（如日期、地址、数量、房型等）时，直接提供该参数的具体值，不要绕开或模糊回答
- 如果智能体询问是否需要帮助下单，回答"是的，请帮我下单"
- 如果对相同问题重复问3次以上没有推进，表现出不耐烦

# 何时不能结束对话：
- 当你还未清楚地表达所有的需求时
- 当智能体尚未完成所有需求时
- 当完成结果和你的instructions中的期望不一致时

# 何时可以结束对话：
- 当且仅当以上所有条件都满足，且任务被正确完成时，回复 '###STOP###'"""


def _format_persona(user_profile: dict) -> str:
    """Format the user_profile dict as readable text."""
    if not user_profile:
        return "普通用户"
    lines = []
    for k, v in user_profile.items():
        lines.append(f"- {k}：{v}")
    return "\n".join(lines)


def _flip_roles(messages: list[dict]) -> list[dict]:
    """
    Flip roles: rebuild the message history from the UserSimulator's perspective.

    - The agent's assistant messages become user messages (from the UserSimulator's view the agent is the one asking)
    - User messages become assistant messages (from the UserSimulator's view it is the one replying)
    - System and tool messages are skipped (the UserSimulator cannot see tool-call details)

    Adapted from VitaBench's UserState.flip_roles()
    """
    flipped = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            continue  # skip the system message
        elif role == "tool":
            continue  # skip tool results (the UserSimulator cannot see them)
        elif role == "assistant":
            # the agent's reply becomes user input from the UserSimulator's perspective
            # take only the text content, excluding tool_calls
            text = content
            if not text and "reasoning_content" in msg:
                text = ""  # content is empty during a tool call, so skip it
            if not text:
                continue  # skip assistant messages that are pure tool calls
            flipped.append({"role": "user", "content": text})
        elif role == "user":
            # a user message becomes assistant output from the UserSimulator's perspective
            flipped.append({"role": "assistant", "content": content})

    return flipped


# UserSimulator concurrency control and client reuse
_user_sim_client: AsyncOpenAI | None = None
_user_sim_semaphore: asyncio.Semaphore | None = None

USER_SIM_CONCURRENCY = 64  # concurrency cap, so the inference service is not saturated


def _get_user_sim_client() -> AsyncOpenAI:
    global _user_sim_client
    if _user_sim_client is None:
        _user_sim_client = AsyncOpenAI(
            api_key=config.user_simulator_api_key,
            base_url=config.user_simulator_base_url,
            timeout=300,
            max_retries=5,
        )
    return _user_sim_client


def _get_user_sim_semaphore() -> asyncio.Semaphore:
    global _user_sim_semaphore
    if _user_sim_semaphore is None:
        _user_sim_semaphore = asyncio.Semaphore(USER_SIM_CONCURRENCY)
    return _user_sim_semaphore


async def simulate_user_response(
    messages: list[dict],
    user_profile: dict,
    instructions: str,
    domain: str = "",
) -> str:
    """
    Call an external LLM to simulate the user's reply.

    Args:
        messages: the full conversation so far, from the agent's perspective
        user_profile: the user profile dict
        instructions: the task instruction text
        domain: the task domain (ota uses the fast-disclosure rules)

    Returns:
        The user's reply text, which may contain ###STOP### to end the conversation
    """
    persona = _format_persona(user_profile)

    prompt_template = USER_SIMULATOR_PROMPT_OTA if domain == "ota" else USER_SIMULATOR_PROMPT
    system_prompt = prompt_template.format(
        persona=persona,
        instructions=instructions,
    )

    # build the UserSimulator's messages after flipping the roles
    sim_messages = [{"role": "system", "content": system_prompt}]
    sim_messages.extend(_flip_roles(messages))

    client = _get_user_sim_client()
    semaphore = _get_user_sim_semaphore()

    try:
        async with semaphore:
            resp = await client.chat.completions.create(
                model=config.user_simulator_model,
                messages=sim_messages,
                temperature=config.user_simulator_temperature,
                max_tokens=500,
                # disable thinking mode: the UserSimulator only needs to emit a user reply, not a reasoning trace
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        content = resp.choices[0].message.content or ""
        # keep _strip_think as a fallback in case extra_body has no effect
        content = _strip_think(content)
        logger.debug(f"[user_sim] Response: {content[:100]}...")
        return content
    except Exception as e:
        logger.warning(f"[user_sim] LLM call failed: {e}")
        return STOP_SIGNAL  # end the conversation on failure so it cannot hang


def is_stop(text: str) -> bool:
    """Whether the text contains the stop signal."""
    return STOP_SIGNAL in text

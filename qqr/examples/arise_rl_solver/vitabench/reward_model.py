"""
VitaBench solver reward model - Reward-Gated Reverse KL (RG-KL)

Aligned with the official VitaBench evaluation protocol:
- Sliding-window scoring: window size 10, overlap 2, with rubric state accumulating across windows
- Uses the official sliding_window_eval_template prompt
- Eval reward is 1.0 when everything passes and 0.0 otherwise (binary, matching the official pass^1 metric)
- Training reward = 0.8 · rubric_rate + 0.2 · 𝟙[all satisfied]

group_reward (RG-KL)：
- The k_s student samples take part in GRPO; the k_m teachers only supply Δ_r, with reward neutralised and loss_mask=0
- Computes λ(Δ_r) = λ_0 · gate · warmup · cosine_decay
- Builds guided_tokens = [p_m_prompt, τ_s_response] for student samples
- Writes sample.rg_kl_coef, which slime's apply_rg_kl_to_advantages reads
"""

import asyncio
import copy
import json
import logging
import os
import re
from argparse import Namespace

import numpy as np
from openai import AsyncOpenAI

from qqr.schemas import Sample

from . import config


def _strip_think(text: str) -> str:
    """
    Strip the thinking section from the model output and return the final reply.
    Handles both the complete tag and the case where the API stripped the opening tag.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip()

logger = logging.getLogger(__name__)

# Try to use json_repair for robustness (same as official VitaBench)
try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False


# -- Sliding-window evaluator (matching the official implementation) -------------


# the official sliding_window_eval_template
SLIDING_WINDOW_EVAL_TEMPLATE = """# 系统信息
{env_info}

# 用户完整指令
{user_instruction}

# 背景说明
- 这是一个user与assistant之间的对话场景，其中assistant可以调用工具获取信息和完成操作，工具返回结果将以tool开头
- 你需要评估用户指令是否被完成，用户的完整指令已被拆分为若干个得分点rubric，你只需要判断每个得分点是否满足
- 由于对话轮次较多，我们采用滑动窗口评估法，即每次可见10轮对话，每个窗口间有2轮对话重叠，但是rubric状态会跨窗口保留
- 你正在评估第 {window_idx} 个窗口（本次任务总共 {total_windows} 个窗口）
- <window_content>包含了当前窗口的对话内容
- <current_rubrics>包含了当前所有得分点的状态（true表示已满足，false表示未满足，所有得分点的初始状态均为false）

# 任务
- 基于当前窗口的对话内容，更新得分点rubric的状态
- 你可以将状态由false更新为true，当且仅当assistant在此窗口中完成了该目标
- 你也可以将true再次更新为false，当且仅当assistant在此窗口中推翻了之前的正确结论（但请注意：如果是用户需求自己发生了变更，例如下单后主动取消，则不应推翻原本对下单需求的评估）
- 你可以参考"用户完整指令"来获取当前对话窗口的进度，避免出现不必要的修改

# 注意事项
- 重要：所有的评估以assistant的回复及工具调用请求是否完成rubric中的目标为准，user在对话中表达的内容仅视为对assistant的提示和引导，不会直接影响评估标准，一切以rubric字段为准！
- 重要：查询类tool返回的结果仅对assistant可见，并不代表assistant对用户推荐的内容，因此也不直接影响评估结果，一切都要以assistant获取信息后对用户的回复为准！同时需要注意，Assistant 也不能编造 Tool 的返回结果！
- 重要：对于订单类rubric（涉及到订单细节，必须生成订单的），必须确认assistant是否真的完成了下单操作。有可能assistant误以为完成了下单操作，实际上工具调用失败；或user表示可以"可以自己下单"等情况，都应视为未满足要求
- 对于涉及到订单细节如商品数量、送达时间的rubric，必须严格满足原始rubric要求（不能有商品数量偏差，不得晚于期望送达时间），用户妥协行为不影响评判结果（例如user表示"某商品少点也行"、"对订单内容没有异议"或"晚点送达也行"等），这类情况仍应视为未满足要求
- 对于涉及到文本内容匹配的地址或订单备注类rubric，采用功能等效原则：只要实际内容能实现相同功能（如大致定位配送地点或传达顾客的主要需求），即使表述不完全一致或缺少部分细节，也视为满足要求
- 如果当前窗口没有涉及某个规则，保持其原有状态不变；如果当前窗口涉及到某个规则但无法完全决定，可以将关键信息记录在justification中留待后续判断
- 在justification中以追加的形式记录与当前rubric有关的关键信息及其对应的轮次[x]，如果发生状态修改也需要记录原因，使用简练的语言，如果状态未修改，复述上次的原因

# 格式要求
- 你的回复应为一个JSON对象，包含以下字段：
- `rubric_idx`：规则的唯一标识符
- `rubric`：对规则的复述
- `justification`：对状态变化的解释
- `meetExpectation`：更新后的状态（true或false）

# 示例回复结构：
```json
[
  {{"rubric_idx": "rubric_0", "rubric": "<复述规则>", "justification": "<状态变化的简要解释，以追加的形式记录>", "meetExpectation": true}},
  ...
]
```"""

# the official full_trajectory_eval_template, used as a fallback for short conversations
FULL_TRAJECTORY_EVAL_TEMPLATE = """# 系统信息
{env_info}

# 用户完整指令
{user_instruction}

# 背景说明
- 这是一个user与assistant之间的对话场景，其中assistant可以调用工具获取信息和完成操作，工具返回结果将以tool开头
- 你需要评估用户指令是否被完成，用户的完整指令已被拆分为若干个得分点rubric，你只需要判断每个得分点是否满足
- <trajectory_content>包含了user与assistant之间的完整对话内容
- <current_rubrics>包含了当前所有得分点的状态（true表示已满足，false表示未满足，所有得分点的初始状态均为false）

# 任务
- 基于对话内容，更新得分点rubric的状态
- 你可以将状态由false更新为true，当且仅当assistant在对话中完成了该目标

# 注意事项
- 重要：所有的评估以assistant的回复及工具调用请求是否完成rubric中的目标为准，user在对话中表达的内容仅视为对assistant的提示和引导，不会直接影响评估标准，一切以rubric字段为准！
- 重要：查询类tool返回的结果仅对assistant可见，并不代表assistant对用户推荐的内容，因此也不直接影响评估结果，一切都要以assistant获取信息后对用户的回复为准！同时需要注意，Assistant 也不能编造 Tool 的返回结果！
- 重要：对于订单类rubric（涉及到订单细节，必须生成订单的），必须确认assistant是否真的完成了下单操作。有可能assistant误以为完成了下单操作，实际上工具调用失败；或user表示可以"可以自己下单"等情况，都应视为未满足要求
- 对于涉及到订单细节如商品数量、送达时间的rubric，必须严格满足原始rubric要求（不能有商品数量偏差，不得晚于期望送达时间），用户妥协行为不影响评判结果（例如user表示"某商品少点也行"、"对订单内容没有异议"或"晚点送达也行"等），这类情况仍应视为未满足要求
- 对于涉及到文本内容匹配的地址或订单备注类rubric，采用功能等效原则：只要实际内容能实现相同功能（如大致定位配送地点或传达顾客的主要需求），即使表述不完全一致或缺少部分细节，也视为满足要求
- 在justification中以追加的形式记录与当前rubric有关的关键信息及其对应的轮次[x]，如果发生状态修改也需要记录原因，使用简练的语言

# 格式要求
- 你的回复应为一个JSON对象，包含以下字段：
- `rubric_idx`：规则的唯一标识符
- `rubric`：对规则的复述
- `justification`：对状态变化的解释
- `meetExpectation`：更新后的状态（true或false）

# 示例回复结构：
```json
[
  {{"rubric_idx": "rubric_0", "rubric": "<复述规则>", "justification": "<状态变化的简要解释，以追加的形式记录>", "meetExpectation": true}},
  ...
]
```"""


class RubricEvaluator:
    """
    Sliding-window rubric evaluator aligned with the official VitaBench implementation.

    - Long conversations (more than 10 turns) are scored with a sliding window
    - Short conversations (10 turns or fewer) are scored in a single full-trajectory pass
    - Rubric state accumulates across windows
    """

    WINDOW_SIZE = 10
    OVERLAP = 2

    def __init__(self):
        self.model = config.llm_judge_model
        self._client = None
        self.concurrency_limit = config.llm_judge_concurrency_limit
        self._semaphore = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=config.llm_judge_api_key,
                base_url=config.llm_judge_base_url,
                timeout=300,
                max_retries=5,
            )
        return self._client

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency_limit)
        return self._semaphore

    # ── Public API ──

    async def evaluate(
        self, messages: list[dict], rubrics: list[str], query: str,
        env_time: str = "",
    ) -> tuple[float, list[dict]]:
        """
        Check whether the agent's conversation satisfies every rubric.

        Returns:
            (score, details) where:
            - score = satisfied rubrics / total rubrics (0.0 to 1.0)
            - details = [{rubric, met, justification}, ...]
        """
        if not rubrics:
            return 1.0, []

        # initialise the rubric state
        rubric_states = {
            f"rubric_{i}": {
                "rubric": r,
                "justification": "尚未评估",
                "meetExpectation": False,
            }
            for i, r in enumerate(rubrics)
        }

        env_info = f"- 当前时间：{env_time}" if env_time else ""

        if len(messages) <= self.WINDOW_SIZE:
            # short conversation: score the full trajectory in one pass
            rubric_states = await self._evaluate_full_trajectory(
                messages, rubric_states, query, env_info
            )
        else:
            # long conversation: score with a sliding window
            windows = self._create_sliding_windows(messages)
            step = self.WINDOW_SIZE - self.OVERLAP
            for w_idx, window in enumerate(windows):
                window_start_idx = w_idx * step
                rubric_states = await self._evaluate_window(
                    window, rubric_states, query, env_info,
                    window_idx=w_idx + 1,
                    total_windows=len(windows),
                    window_start_idx=window_start_idx,
                )

        # convert into a details list
        details = []
        for i, r in enumerate(rubrics):
            key = f"rubric_{i}"
            state = rubric_states.get(key, {})
            details.append({
                "rubric": r,
                "met": bool(state.get("meetExpectation", False)),
                "justification": state.get("justification", ""),
            })

        met_count = sum(1 for d in details if d["met"])
        score = met_count / len(rubrics)
        return score, details

    # ── Sliding Window ──

    def _create_sliding_windows(self, messages: list[dict]) -> list[list[dict]]:
        if len(messages) <= self.WINDOW_SIZE:
            return [messages]

        windows = []
        step = self.WINDOW_SIZE - self.OVERLAP
        i = 0
        while i < len(messages):
            window = messages[i:i + self.WINDOW_SIZE]
            if window:
                windows.append(window)
            if i + self.WINDOW_SIZE >= len(messages):
                break
            i += step
        return windows

    async def _evaluate_window(
        self, window: list[dict], rubric_states: dict, query: str,
        env_info: str, window_idx: int, total_windows: int,
        window_start_idx: int = 0,
    ) -> dict:
        window_content = self._format_messages(window, start_idx=window_start_idx)
        rubrics_str = self._format_rubric_states(rubric_states)

        system_prompt = SLIDING_WINDOW_EVAL_TEMPLATE.format(
            env_info=env_info,
            user_instruction=query,
            window_idx=window_idx,
            total_windows=total_windows,
        )
        user_prompt = (
            f"# Input\n<window_content>\n{window_content}\n</window_content>\n\n"
            f"<current_rubrics>\n{rubrics_str}\n</current_rubrics>"
        )

        result_data = await self._call_llm(system_prompt, user_prompt)

        updated = copy.deepcopy(rubric_states)
        if result_data:
            # flatten nested lists (the LLM occasionally returns [[{...}, ...]])
            flat = []
            for item in result_data:
                if isinstance(item, list):
                    flat.extend(item)
                elif isinstance(item, dict):
                    flat.append(item)
            for result in flat:
                if not isinstance(result, dict):
                    continue
                idx = result.get("rubric_idx")
                if idx and idx in updated:
                    updated[idx]["justification"] = result.get(
                        "justification", updated[idx]["justification"]
                    )
                    updated[idx]["meetExpectation"] = result.get(
                        "meetExpectation", updated[idx]["meetExpectation"]
                    )
        return updated

    async def _evaluate_full_trajectory(
        self, messages: list[dict], rubric_states: dict, query: str,
        env_info: str,
    ) -> dict:
        trajectory_content = self._format_messages(messages, start_idx=0)
        rubrics_str = self._format_rubric_states(rubric_states)

        system_prompt = FULL_TRAJECTORY_EVAL_TEMPLATE.format(
            env_info=env_info,
            user_instruction=query,
        )
        user_prompt = (
            f"# Input\n<trajectory_content>\n{trajectory_content}\n</trajectory_content>\n\n"
            f"<current_rubrics>\n{rubrics_str}\n</current_rubrics>"
        )

        result_data = await self._call_llm(system_prompt, user_prompt)

        updated = copy.deepcopy(rubric_states)
        if result_data:
            # flatten nested lists (the LLM occasionally returns [[{...}, ...]])
            flat = []
            for item in result_data:
                if isinstance(item, list):
                    flat.extend(item)
                elif isinstance(item, dict):
                    flat.append(item)
            for result in flat:
                if not isinstance(result, dict):
                    continue
                idx = result.get("rubric_idx")
                if idx and idx in updated:
                    updated[idx]["justification"] = result.get(
                        "justification", updated[idx]["justification"]
                    )
                    updated[idx]["meetExpectation"] = result.get(
                        "meetExpectation", updated[idx]["meetExpectation"]
                    )
        return updated

    # -- LLM call --

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> list[dict] | None:
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with self.semaphore:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=llm_messages,
                        temperature=0.0,
                        max_tokens=4000,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                result_text = _strip_think(resp.choices[0].message.content or "")
                parsed = self._parse_results(result_text)
                if parsed is not None:
                    return parsed
                logger.warning(
                    f"[RubricEvaluator] Parse failed attempt {attempt + 1}/{max_retries}"
                )
            except Exception as e:
                logger.warning(
                    f"[RubricEvaluator] LLM call failed attempt {attempt + 1}/{max_retries}: {e}"
                )
        return None

    # -- Formatting --

    def _format_messages(self, messages: list[dict], start_idx: int = 0) -> str:
        """Matches the official _format_window_content: [global index] role: content + tool_calls."""
        lines = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls", [])

            full_content = content

            if role == "assistant" and tool_calls:
                tc_strs = []
                for tc in tool_calls:
                    func = tc.get("function", tc)
                    name = func.get("name", "")
                    args = func.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            args_dict = json.loads(args)
                            if isinstance(args_dict, dict):
                                args_str = ", ".join(
                                    f"{k}={repr(v)}" for k, v in args_dict.items()
                                )
                            else:
                                args_str = str(args_dict)
                        except (json.JSONDecodeError, TypeError):
                            args_str = args
                    elif isinstance(args, dict):
                        args_str = ", ".join(
                            f"{k}={repr(v)}" for k, v in args.items()
                        )
                    else:
                        args_str = str(args)
                    tc_strs.append(f"{name}({args_str})")

                if tc_strs:
                    tc_text = ". ".join(tc_strs)
                    full_content = f"{full_content} {tc_text}" if full_content else tc_text

            if full_content:
                global_idx = start_idx + i + 1
                lines.append(f"[{global_idx}] {role}: {full_content}")

        return "\n".join(lines)

    def _format_rubric_states(self, rubric_states: dict) -> str:
        """Matches the official _format_current_rubrics."""
        return json.dumps(
            [
                {
                    "rubric_idx": k,
                    "rubric": v["rubric"],
                    "justification": v["justification"],
                    "meetExpectation": v["meetExpectation"],
                }
                for k, v in rubric_states.items()
            ],
            ensure_ascii=False,
            indent=2,
        )

    def _parse_results(self, text: str) -> list[dict] | None:
        """Parse the JSON returned by the LLM, using json_repair for robustness."""
        try:
            if HAS_JSON_REPAIR:
                repaired = repair_json(text)
                result = json.loads(repaired)
                if isinstance(result, list):
                    return result
            # Fallback: extract [ ... ] manually
            start = text.index("[")
            end = text.rindex("]") + 1
            result = json.loads(text[start:end])
            if isinstance(result, list):
                return result
        except (ValueError, json.JSONDecodeError):
            pass
        return None


rubric_evaluator = RubricEvaluator()


# -- Helper functions ------------------------------------------------------------


def _extract_tool_names(messages: list[dict]) -> list[str]:
    names = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            name = tc.get("function", {}).get("name") or tc.get("name", "")
            if name and isinstance(name, str):
                names.append(name)
    return names


def _get_rubrics(sample: Sample) -> list[str]:
    meta = sample.metadata or {}
    if "rubrics" in meta and meta["rubrics"]:
        return meta["rubrics"]
    eval_criteria = meta.get("evaluation_criteria", {})
    rubrics = []
    rubrics.extend(eval_criteria.get("overall_rubrics", []))
    for state in eval_criteria.get("expected_states", []):
        rubrics.extend(state.get("state_rubrics", []))
    return rubrics


def _get_query(sample: Sample) -> str:
    """Return the user instruction (read from metadata.instructions in interactive mode)."""
    meta = sample.metadata or {}
    # interactive mode: instructions carry the actual task directive
    if meta.get("instructions"):
        return meta["instructions"]
    if isinstance(sample.prompt, str):
        return sample.prompt
    elif isinstance(sample.prompt, list) and sample.prompt:
        return sample.prompt[-1].get("content", "")
    return ""


def _get_env_time(sample: Sample) -> str:
    meta = sample.metadata or {}
    return meta.get("env_time", "")


# -- Per-sample reward (cached during rollout and reused by group_reward) --------


async def _compute_single_sample_reward(sample: Sample) -> dict:
    """
    Score a single sample and return a structured result, used by:
    1. Phase B: score the student samples to build the coaching, cached in metadata["_rubric_cache"]
    2. group_reward: full scoring of the teacher samples, which have no cache
    """
    rubrics = _get_rubrics(sample)
    query = _get_query(sample)
    env_time = _get_env_time(sample)
    messages = sample.messages or []
    tool_names = sorted(set(_extract_tool_names(messages)))

    if rubrics:
        rubric_score, rubric_details = await rubric_evaluator.evaluate(
            messages, rubrics, query, env_time=env_time
        )
    else:
        rubric_score, rubric_details = 0.0, []

    rubrics_total = len(rubrics)
    rubrics_met = sum(1 for d in rubric_details if d.get("met"))
    rubric_rate = rubrics_met / rubrics_total if rubrics_total > 0 else 0.0
    all_passed = (rubrics_met == rubrics_total) if rubrics_total > 0 else False
    reward = 0.8 * rubric_rate + (0.2 if all_passed else 0.0)

    return {
        "reward": reward,
        "rubric_score": rubric_score,
        "rubric_rate": rubric_rate,
        "rubric_all_passed": all_passed,
        "rubrics_total": rubrics_total,
        "rubrics_met": rubrics_met,
        "rubric_details": rubric_details,
        "actual_tool_names": tool_names,
        "num_tool_calls": len(_extract_tool_names(messages)),
        "failed_rubrics": [d.get("rubric", "") for d in rubric_details if not d.get("met")],
    }


async def _maybe_compute_reward(sample: Sample) -> dict:
    """Use the result cached during rollout (student), or re-score it (teacher)."""
    cached = (sample.metadata or {}).get("_rubric_cache")
    if cached is not None:
        return cached
    return await _compute_single_sample_reward(sample)


# -- Eval reward (matching the official pass^1 metric) ---------------------------


async def eval_reward(args: Namespace, sample: Sample, **kwargs):
    """
    Evaluation reward (aligned with the official metric):
    - reward = 1.0 when everything passes, else 0.0 (the official pass^1 metric)
    """
    rubrics = _get_rubrics(sample)
    query = _get_query(sample)
    env_time = _get_env_time(sample)
    messages = sample.messages or []

    tool_names = sorted(set(_extract_tool_names(messages)))

    if rubrics:
        rubric_score, rubric_details = await rubric_evaluator.evaluate(
            messages, rubrics, query, env_time=env_time
        )
    else:
        rubric_score = 0.0
        rubric_details = []

    rubrics_met = sum(1 for d in rubric_details if d.get("met"))
    rubrics_total = len(rubrics)
    all_passed = (rubrics_met == rubrics_total) if rubrics_total > 0 else False
    rubric_rate = rubrics_met / rubrics_total if rubrics_total > 0 else 0.0

    # evaluation uses the official binary reward to match pass^1
    sample.reward = 1.0 if all_passed else 0.0

    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata.update({
        "rubric_score": rubric_score,
        "rubric_rate": rubric_rate,
        "rubric_all_passed": all_passed,
        "rubrics_total": rubrics_total,
        "rubrics_met": rubrics_met,
        "rubric_details": rubric_details,
        "actual_tool_names": tool_names,
        "num_tool_calls": len(_extract_tool_names(messages)),
    })

    logger.info(
        f"[eval_reward] task={sample.metadata.get('task_id', '?')} "
        f"reward={sample.reward:.2f} all_passed={all_passed} "
        f"({rubrics_met}/{rubrics_total}) "
        f"tools={tool_names}"
    )


# ── Group Reward（Reward-Gated Reverse KL）────────────────────────────────────


async def group_reward(args: Namespace, group: list[list[Sample]], **kwargs):
    """
    Reward-Gated Reverse KL (RG-KL)：

    - The k_s student samples are trained on, using their true rewards for GRPO
    - The k_m teacher samples only estimate Δ_r; their reward is neutralised to student_mean and loss_mask=0
    - λ(Δ_r) = λ_0 · max(Δ_r - δ_thresh, 0) · warmup · cosine_decay
    - Builds guided_tokens = [p_m_prompt, τ_s_response] for student samples
    - Writes the λ coefficient to sample.rg_kl_coef, which slime's apply_rg_kl_to_advantages reads

    Reward formula: 0.8 · rubric_rate + 0.2 · 𝟙[all satisfied]
    Student samples reuse the evaluation cached during rollout, avoiding repeated sliding-window LLM calls;
    teacher samples are scored in full here.
    """
    if len(group) <= 1:
        raise ValueError("group size must be greater than 1")

    real_samples = [g[-1] for g in group]
    results = await asyncio.gather(*[_maybe_compute_reward(s) for s in real_samples])

    n_samples_per_prompt = getattr(args, "n_samples_per_prompt", len(group))
    group_id = (
        real_samples[0].index // n_samples_per_prompt
        if n_samples_per_prompt > 1 else real_samples[0].index
    )

    # -- 1) Write each sample's true evaluation metadata (for logging and eval) --
    raw_rewards = []
    for idx, (result, sample_group) in enumerate(zip(results, group)):
        reward = result["reward"]
        raw_rewards.append(reward)
        reward_metadata = {
            "rubric_score": result["rubric_score"],
            "rubric_rate": result["rubric_rate"],
            "rubric_all_passed": result["rubric_all_passed"],
            "rubrics_total": result["rubrics_total"],
            "rubrics_met": result["rubrics_met"],
            "rubric_details": result["rubric_details"],
            "actual_tool_names": result["actual_tool_names"],
            "num_tool_calls": result["num_tool_calls"],
        }
        for sample in sample_group:
            sample.reward = reward  # stash the true rewards; teacher samples are neutralised below
            if sample.metadata is None:
                sample.metadata = {}
            sample.metadata.update(reward_metadata)
            sample.metadata["_raw_reward"] = reward

        source = (real_samples[idx].metadata or {}).get("source", "student")
        logger.info(
            f"[group_reward] idx={idx} source={source} "
            f"rubric_rate={result['rubric_rate']:.2f} "
            f"all_passed={result['rubric_all_passed']} "
            f"reward={reward:.2f}"
        )

    # -- 2) Compute Δ_r --
    student_rewards = [
        r["reward"] for r, s in zip(results, real_samples)
        if (s.metadata or {}).get("source") == "student"
    ]
    teacher_rewards = [
        r["reward"] for r, s in zip(results, real_samples)
        if (s.metadata or {}).get("source") == "teacher"
    ]
    avg_student = sum(student_rewards) / len(student_rewards) if student_rewards else 0.0
    avg_teacher = sum(teacher_rewards) / len(teacher_rewards) if teacher_rewards else 0.0
    delta_r = avg_teacher - avg_student

    # -- 3) Neutralise the teacher reward so GRPO normalisation is unaffected (teacher advantage ~ 0) --
    # slime's GRPO normalisation uses the reward mean/std over all samples. Keeping the teacher's true reward would make
    # every student advantage negative, wrongly pushing the student away from itself. Neutralising sets teacher reward = student_mean
    # so the GRPO mean is close to student_mean, student advantages stay well distributed and teacher advantages are near 0.
    for idx, sample_group in enumerate(group):
        source = (real_samples[idx].metadata or {}).get("source", "student")
        if source == "teacher":
            for sample in sample_group:
                sample.reward = avg_student
                if sample.metadata is None:
                    sample.metadata = {}
                sample.metadata["_neutralized_for_grpo"] = True
                # also zero loss_mask to prevent any gradient leakage (belt and braces)
                if hasattr(sample, "loss_mask") and sample.loss_mask is not None:
                    sample.loss_mask = [0] * len(sample.loss_mask)

    # -- 4) RG-KL: compute λ(Δ_r) and build guided_tokens --
    from .rollout import (
        cleanup_group_coordination,
        compute_rg_kl_guided_tokens,
        compute_rg_kl_coef,
        increment_rollout_step,
    )

    rollout_batch_size = int(getattr(args, "rollout_batch_size", 1) or 1)
    rollout_step = await increment_rollout_step(rollout_batch_size)
    coef, coef_info = compute_rg_kl_coef(delta_r, rollout_step)

    # locate this group's memory (cached on any teacher sample)
    group_memory = ""
    for s in real_samples:
        if (s.metadata or {}).get("source") == "teacher":
            mem = (s.metadata or {}).get("memory") or ""
            if mem:
                group_memory = mem
                break

    # build guided_tokens for every sample in the group and write rg_kl_coef
    if config.enable_rg_kl and config.enable_guided_tokens_for_student:
        for sample_group in group:
            compute_rg_kl_guided_tokens(args, sample_group, group_memory)
    else:
        # degenerate to plain GRPO (no distillation): leave guided_tokens unset
        coef = 0.0

    # write rg_kl_coef on every sample
    for idx, sample_group in enumerate(group):
        source = (real_samples[idx].metadata or {}).get("source", "student")
        sample_coef = coef if source == "student" else 0.0
        for sample in sample_group:
            sample.rg_kl_coef = sample_coef
            if sample.metadata is None:
                sample.metadata = {}
            sample.metadata["_rg_kl_coef"] = sample_coef

    # -- 5) Logging --
    avg_raw = sum(raw_rewards) / len(raw_rewards) if raw_rewards else 0.0
    pass_count = sum(1 for r in results if r["rubric_all_passed"])
    rubrics_total = results[0].get("rubrics_total", 0) if results else 0

    logger.info(
        f"[group_reward] {len(group)} samples raw_avg={avg_raw:.3f} "
        f"student_avg={avg_student:.3f} teacher_avg={avg_teacher:.3f} delta_r={delta_r:+.3f} "
        f"all_passed={pass_count}/{len(group)} rubrics_total={rubrics_total}"
    )
    logger.info(
        f"[rg_kl] group={group_id} rollout_step={rollout_step} "
        f"delta_r={delta_r:+.3f} gate={coef_info['gate']:.3f} "
        f"warmup={coef_info['warmup_factor']:.3f} cos={coef_info['cos_factor']:.3f} "
        f"→ coef={coef:.4f}"
    )

    cleanup_group_coordination(group_id)


def reward_post_process(args: Namespace, samples: list[Sample] | list[list[Sample]]):
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    return raw_rewards, raw_rewards

"""
DeepResearch - RG-KL solver reward model

Reward design (rubric-based):
- An LLM judge checks each rubric individually
- Training data (unweighted, paper Eq. (6)): reward = 0.8 · rubric_pass_rate + 0.2 · 𝟙[all satisfied]
- ResearchRubrics benchmark（weighted）：reward = Σ(weight × score) / Σ(positive_weights)
- Rubrics the judge fails to score are dropped, so false negatives do not pollute the training signal

RG-KL integration:
- Student samples go through standard GRPO with their true rewards
- Teacher samples have their reward neutralised to student_mean, with loss_mask=0 and rg_kl_coef=0
- Δ_r = mean(r_m) - mean(r_s) is written to sample.rg_kl_coef
"""

import asyncio
import fcntl
import json
import logging
import os
from argparse import Namespace
from datetime import datetime

from openai import AsyncOpenAI

from qqr.schemas import Sample

from . import config

logger = logging.getLogger(__name__)


# ============ LLM judge client ============

_judge_client: AsyncOpenAI | None = None
_judge_semaphore: asyncio.Semaphore | None = None


def get_judge_client() -> AsyncOpenAI:
    global _judge_client
    if _judge_client is None:
        _judge_client = AsyncOpenAI(
            api_key=config.llm_judge_api_key,
            base_url=config.llm_judge_base_url,
            timeout=300,
            max_retries=5,
        )
    return _judge_client


def get_judge_semaphore() -> asyncio.Semaphore:
    global _judge_semaphore
    if _judge_semaphore is None:
        _judge_semaphore = asyncio.Semaphore(config.llm_judge_concurrency_limit)
    return _judge_semaphore


# ============ Rubric scoring ============


async def evaluate_rubrics(
    messages: list[dict], rubrics: list[str], query: str
) -> tuple[float, list[dict]]:
    if not rubrics:
        return 1.0, []

    trajectory_content = _format_trajectory(messages)

    current_rubrics = json.dumps([
        {
            "rubric_idx": f"rubric_{i}",
            "rubric": r,
            "justification": "尚未评估",
            "meetExpectation": False,
        }
        for i, r in enumerate(rubrics)
    ], ensure_ascii=False, indent=2)

    system_prompt = f"""# 用户完整指令
{query}

# 背景说明
- 这是一个user与assistant之间的深度研究对话场景，assistant可以调用web_search工具搜索信息
- 你需要评估assistant的最终回答是否满足rubric的要求
- <trajectory_content>包含 assistant 的思考、工具调用动作（例如 `web_search(query='...')`）以及最终回答；**工具返回的搜索结果已省略**，请仅依据 assistant 的最终回答文本来评估
- <current_rubrics>包含所有得分点的状态

# 任务
- 基于 assistant 的最终回答，更新得分点rubric的状态
- 采用**公允、合理**的评判标准：核心内容满足即判通过，不因细节瑕疵扣分；但也不能放过明显的空洞/错误

# 核心判定原则

## 1. 数据/引用类 rubric（由于搜索结果已省略，不要求 Judge 核验真伪）
- 要求"引用特定数据/数字/年份/出处"：**只要 assistant 回答中出现了相关数据或引用标注即视为满足**；无需核验数据正确性
- 要求"引用官方来源/权威机构"：**只要 assistant 提到了看似合理的来源（具体机构名、报告名、网站等）即视为满足**；笼统说"据官方数据"不算满足
- 要求"使用特定引用格式（APA/MLA 等）"：**参考文献部分存在且能辨认出处即满足**，不必严格符合格式规范
- 要求"具体技术细节（如 Scope 1/2/3、某数值、某年份）"：**提到相关概念+至少部分具体内容即满足**；只提概念不给任何细节不算满足

## 2. 内容覆盖类 rubric
- 要求"提到 X、Y、Z 几个方面"：**需覆盖约 80% 的要点**（如 3 项覆盖 2-3 项、5 项覆盖 4-5 项）
- 要求"对比 A/B/C 三者"：**必须对全部对象均有实质性论述**才算满足；只比了两者算不满足
- 要求"覆盖 X 领域"：**需有具体阐述而非一笔带过**（至少 2-3 句相关内容）
- 要求"列出 N 条/个"：**数量达到 80% 以上即视为满足**

## 3. 结构/格式类 rubric
- 要求"结构化呈现/清晰分段"：**有明确的标题、编号或分点即满足**
- 要求"使用表格/列表"：**有符合要求的结构化元素即满足**（例如 rubric 要求表格，只有列表不算满足）
- 要求"包含 A/B/C 几个部分"：**必须覆盖 rubric 列出的所有主要部分**才算满足

## 4. 功能等效原则
- 文本内容匹配：**实际内容能实现相同功能即满足**，表述可以不完全一致
- 要求"清晰/明确/具体说明"：**有实质性阐述（非一句话打发）即满足**
- 要求特定关键词：**意思到位即可，不强制关键词**；但如果 rubric 明确强调"使用特定术语"则需出现该术语

## 5. 判为 false 的典型情况
- assistant 的最终回答**完全没有涉及** rubric 要求的内容
- assistant 的回答与 rubric 要求**明显矛盾或错误**
- rubric 要求"**不应**包含 X"但 assistant 明显包含了 X
- rubric 要求多项内容，assistant **只覆盖不到一半**
- rubric 要求具体细节（数据、案例、对象等），assistant 只停留在**空泛概述**没有任何具体内容

# 其他注意事项
- 评估以 assistant 的**最终回答内容**为准，assistant 的 think 思考块也可作为辅助证据
- 在 justification 中用简练的语言记录关键证据（引用轮次 [x]）
- 判为 false 时必须给出明确理由；判为 true 需要简述证据
- 对于模糊地带，以**是否实质性体现了 rubric 意图**作为判断依据

# 格式要求
- 你的回复应为一个JSON数组：
```json
[
  {{"rubric_idx": "rubric_0", "rubric": "<复述规则>", "justification": "<解释>", "meetExpectation": true}},
  ...
]
```"""

    user_prompt = f"""# Input
<trajectory_content>
{trajectory_content}
</trajectory_content>

<current_rubrics>
{current_rubrics}
</current_rubrics>"""

    llm_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    client = get_judge_client()
    semaphore = get_judge_semaphore()

    dynamic_max_tokens = max(4000, min(24000, 500 + 400 * len(rubrics)))

    max_retries = 3
    details = None
    best_details = None
    best_failed_count = len(rubrics) + 1

    for attempt in range(max_retries):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=config.llm_judge_model,
                    messages=llm_messages,
                    temperature=0.0,
                    max_completion_tokens=dynamic_max_tokens,
                )
            result_text = resp.choices[0].message.content or ""
            details = _parse_rubric_results(result_text, rubrics)

            failed_count = sum(1 for d in details if d["justification"] == "解析失败")

            if failed_count < best_failed_count:
                best_failed_count = failed_count
                best_details = details

            if failed_count == 0:
                break
            logger.warning(
                f"[evaluate_rubrics] attempt {attempt + 1}/{max_retries} "
                f"解析失败 {failed_count}/{len(rubrics)} rubrics，继续重试"
            )
        except Exception as e:
            logger.warning(f"[evaluate_rubrics] LLM call failed attempt {attempt + 1}/{max_retries}: {e}")

    if best_details is not None:
        details = best_details
    if details is None:
        details = [
            {"rubric": r, "met": False, "justification": f"评估失败（重试{max_retries}次，LLM 调用异常）"}
            for r in rubrics
        ]

    for d in details:
        if d["justification"] == "解析失败":
            d["justification"] = f"评估失败（重试{max_retries}次后仍无法解析）"

    valid_rubrics = [d for d in details if not d["justification"].startswith("评估失败")]
    met_count = sum(1 for d in valid_rubrics if d.get("met", False))
    if valid_rubrics:
        score = met_count / len(valid_rubrics)
    else:
        score = 0.0
    return score, details


def _format_trajectory(messages: list[dict]) -> str:
    """Format the full conversation for scoring. Tool results (role=tool) are not passed to the judge."""
    content_lines = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls", [])

        if role == "tool":
            continue

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
                        args_str = ", ".join(f"{k}={repr(v)}" for k, v in args_dict.items())
                    except (json.JSONDecodeError, TypeError):
                        args_str = args
                elif isinstance(args, dict):
                    args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
                else:
                    args_str = str(args)
                tc_strs.append(f"{name}({args_str})")
            if tc_strs:
                tc_text = ". ".join(tc_strs)
                full_content = f"{full_content} {tc_text}" if full_content else tc_text
        if full_content:
            content_lines.append(f"[{i + 1}] {role}: {full_content}")
    return "\n".join(content_lines)


def _parse_rubric_results(text: str, rubrics: list[str]) -> list[dict]:
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        results = json.loads(text[start:end])
        if isinstance(results, list):
            flat = []
            for item in results:
                if isinstance(item, list):
                    flat.extend(item)
                elif isinstance(item, dict):
                    flat.append(item)
            details = []
            for i, rubric in enumerate(rubrics):
                if i < len(flat) and isinstance(flat[i], dict):
                    r = flat[i]
                    met = r.get("meetExpectation", r.get("met", False))
                    details.append({
                        "rubric": rubric,
                        "met": bool(met),
                        "justification": r.get("justification", ""),
                    })
                else:
                    details.append({
                        "rubric": rubric,
                        "met": False,
                        "justification": "LLM 未返回该 rubric 的结果",
                    })
            return details
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning(f"[_parse_rubric_results] Failed to parse: {e}")

    return [{"rubric": r, "met": False, "justification": "解析失败"} for r in rubrics]


# ============ Data extraction helpers ============


def _get_rubrics(sample: Sample) -> list[str]:
    if sample.metadata and isinstance(sample.metadata, dict):
        return sample.metadata.get("rubrics", [])
    return []


def _get_rubric_weights(sample: Sample) -> list[float] | None:
    """Return the rubric weights, or None when no weights are present (equal weighting)."""
    if sample.metadata and isinstance(sample.metadata, dict):
        weights = sample.metadata.get("rubric_weights")
        if weights and isinstance(weights, list) and len(weights) > 0:
            return weights
    return None


def _get_query(sample: Sample) -> str:
    if isinstance(sample.prompt, str):
        return sample.prompt
    if isinstance(sample.prompt, list):
        for msg in reversed(sample.prompt):
            if msg.get("role") == "user":
                return msg.get("content", "")
    return ""


def _compute_reward(
    rubric_details: list[dict], weights: list[float] | None
) -> tuple[float, float, bool, int, int]:
    """
    Computes the reward uniformly, adapting to weighted or equal weighting automatically.

    With weights (the ResearchRubrics benchmark):
        compliance = Σ(weight × score) / Σ(positive_weights)
        reward = compliance

    Without weights (training data, paper Eq. (6)):
        reward = 0.8 · rubric_pass_rate + 0.2 · 𝟙[all satisfied]

    Returns: (reward, rubric_rate, all_passed, rubrics_met, rubrics_total)
    """
    rubrics_total_raw = len(rubric_details)

    valid_indices = [
        i for i, d in enumerate(rubric_details)
        if not str(d.get("justification", "")).startswith("评估失败")
    ]
    valid_details = [rubric_details[i] for i in valid_indices]

    rubrics_total = len(valid_details)
    rubrics_met = sum(1 for d in valid_details if d.get("met"))
    all_passed = (rubrics_met == rubrics_total) if rubrics_total > 0 else False

    if weights and len(weights) == rubrics_total_raw:
        valid_weights = [weights[i] for i in valid_indices]
        scores = [1.0 if d.get("met") else 0.0 for d in valid_details]
        numerator = sum(s * w for s, w in zip(scores, valid_weights))
        denominator = sum(w for w in valid_weights if w > 0)
        compliance = numerator / denominator if denominator > 0 else 0.0
        return compliance, compliance, all_passed, rubrics_met, rubrics_total
    else:
        rubric_rate = rubrics_met / rubrics_total if rubrics_total > 0 else 0.0
        # paper Eq. (6): r = α·s + (1−α)·𝟙[s=1] with α=0.8 (partial credit plus an all-satisfied bonus)
        reward = 0.8 * rubric_rate + (0.2 if all_passed else 0.0)
        return reward, rubric_rate, all_passed, rubrics_met, rubrics_total


# ============ Per-sample reward computation ============


def _has_empty_response(sample: Sample) -> bool:
    """Whether the sample's final answer is empty."""
    response = getattr(sample, "response", None) or ""
    if response.strip():
        return False
    messages = sample.messages or []
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "") or ""
            return not content.strip()
    return True


async def _compute_sample_reward(sample: Sample) -> dict:
    """Compute a single sample's rubric-based reward. Returns a dict holding reward and details,
    Used by group_reward's RG-KL flow and to cache the Δ_r estimate from the student phase.
    """
    messages = sample.messages or []
    rubrics = _get_rubrics(sample)
    query = _get_query(sample)
    weights = _get_rubric_weights(sample)

    if _has_empty_response(sample):
        rubric_details = [
            {"rubric": r, "met": False, "justification": "最终回答为空"}
            for r in rubrics
        ]
        rubric_score = 0.0
    elif rubrics:
        rubric_score, rubric_details = await evaluate_rubrics(messages, rubrics, query)
    else:
        rubric_score = 0.0
        rubric_details = []

    reward, rubric_rate, all_passed, rubrics_met, rubrics_total = _compute_reward(
        rubric_details, weights
    )

    return {
        "reward": reward,
        "query": query,
        "rubric_pass_rate": rubric_score,
        "rubric_details": rubric_details,
        "rubric_all_passed": all_passed,
        "rubrics_total": rubrics_total,
        "rubrics_met": rubrics_met,
        "weighted": weights is not None,
    }


async def _maybe_compute_reward(sample: Sample) -> dict:
    """Use the evaluation cached during rollout (student), or recompute it (teacher)."""
    cached = (sample.metadata or {}).get("_rubric_cache")
    if cached is not None:
        return cached
    return await _compute_sample_reward(sample)


# ============ Group memory persistence (for debugging) ============


def _get_memory_dir(args: Namespace) -> str:
    save_dir = getattr(args, "save", None)
    if not save_dir:
        return ""
    memory_dir = os.path.join(save_dir, "memory")
    os.makedirs(memory_dir, exist_ok=True)
    return memory_dir


def _save_group_summary(args: Namespace, group_id: int, summary: str, judge_input: str = ""):
    """Append the group coach summary to the memory file for debugging. Called by rollout.py."""
    memory_dir = _get_memory_dir(args)
    if not memory_dir:
        return
    filepath = os.path.join(memory_dir, "group_summaries.jsonl")
    record = {
        "timestamp": datetime.now().isoformat(),
        "group_id": group_id,
        "summary": summary,
        "judge_input": judge_input,
    }
    try:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with open(filepath, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line)
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        logger.warning(f"[_save_group_summary] 写入失败: {e}")


# ============ Reward function interface ============


async def eval_reward(args: Namespace, sample: Sample, **kwargs):
    """Reward computation in evaluation mode."""
    result = await _compute_sample_reward(sample)
    sample.reward = result["reward"]

    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata.update({
        "rubric_pass_rate": result["rubric_pass_rate"],
        "rubric_details": result["rubric_details"],
        "rubric_all_passed": result["rubric_all_passed"],
        "rubrics_total": result["rubrics_total"],
        "rubrics_met": result["rubrics_met"],
        "weighted": result["weighted"],
    })

    scoring_mode = "weighted" if result["weighted"] else "unweighted"
    logger.info(
        f"[eval_reward] ({scoring_mode}) query={result['query'][:50]}... "
        f"rubric_rate={result['rubric_pass_rate']:.2f} "
        f"all_passed={result['rubric_all_passed']} "
        f"reward={sample.reward:.2f}"
    )


async def group_reward(args: Namespace, group: list[list[Sample]], **kwargs):
    """
    Reward-Gated Reverse KL (RG-KL)：
    - The k_s student samples are trained on, using their true rewards for GRPO
    - The k_m teacher samples only estimate Δ_r; their reward is neutralised to student_mean and loss_mask=0
    - λ(Δ_r) = λ_0 · gate(Δ_r) · warmup · cosine_decay
    - Builds guided_tokens = [p_m_prompt, τ_s_response] for student samples
    - Writes the λ coefficient to sample.rg_kl_coef, which slime's apply_rg_kl_to_advantages reads
    """
    if len(group) <= 1:
        raise ValueError("group size must be greater than 1")

    real_samples = [g[-1] for g in group]
    results = await asyncio.gather(*[_maybe_compute_reward(s) for s in real_samples])

    n_samples = getattr(args, "n_samples_per_prompt", len(group))
    group_id = real_samples[0].index // n_samples if n_samples > 1 else real_samples[0].index

    # -- 1) Write each sample's true evaluation metadata (for logging and eval) --
    raw_rewards = []
    for idx, (result, sample_group) in enumerate(zip(results, group)):
        reward = result["reward"]
        raw_rewards.append(reward)
        reward_metadata = {
            "rubric_pass_rate": result["rubric_pass_rate"],
            "rubric_details": result["rubric_details"],
            "rubric_all_passed": result["rubric_all_passed"],
            "rubrics_total": result["rubrics_total"],
            "rubrics_met": result["rubrics_met"],
            "weighted": result["weighted"],
        }
        for sample in sample_group:
            sample.reward = reward
            if sample.metadata is None:
                sample.metadata = {}
            sample.metadata.update(reward_metadata)
            sample.metadata["_raw_reward"] = reward

        source = (real_samples[idx].metadata or {}).get("source", "student")
        logger.info(
            f"[group_reward] idx={idx} source={source} "
            f"rubric_rate={result['rubric_pass_rate']:.2f} "
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

    # -- 3) Neutralise the teacher reward and set loss_mask=0 --
    # GRPO normalises with the within-group mean/std. Keeping the teacher's true reward makes every student advantage negative,
    # wrongly pushing the student away from itself. After neutralising, the mean is close to student_mean, student advantages behave normally and teacher advantages are near 0.
    for idx, sample_group in enumerate(group):
        source = (real_samples[idx].metadata or {}).get("source", "student")
        if source == "teacher":
            for sample in sample_group:
                sample.reward = avg_student
                if sample.metadata is None:
                    sample.metadata = {}
                sample.metadata["_neutralized_for_grpo"] = True
                if hasattr(sample, "loss_mask") and sample.loss_mask is not None:
                    sample.loss_mask = [0] * len(sample.loss_mask)

    avg_rubric = sum(r["rubric_pass_rate"] for r in results) / len(results)
    full_pass_count = sum(1 for r in results if r["rubric_all_passed"])

    # -- 4) RG-KL: compute λ(Δ_r) and write it to sample.rg_kl_coef --
    from .rollout import (
        cleanup_group_coordination,
        compute_rg_kl_coef,
        compute_rg_kl_guided_tokens,
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

    if config.enable_rg_kl and config.enable_guided_tokens_for_student:
        for sample_group in group:
            compute_rg_kl_guided_tokens(args, sample_group, group_memory)
    else:
        coef = 0.0

    # -- 5) Write rg_kl_coef on every sample --
    for idx, sample_group in enumerate(group):
        source = (real_samples[idx].metadata or {}).get("source", "student")
        sample_coef = coef if source == "student" else 0.0
        for sample in sample_group:
            sample.rg_kl_coef = sample_coef
            if sample.metadata is None:
                sample.metadata = {}
            sample.metadata["_rg_kl_coef"] = sample_coef

    logger.info(
        f"[group_reward] {len(group)} samples raw_avg={sum(raw_rewards)/len(raw_rewards):.3f} "
        f"student_avg={avg_student:.3f} teacher_avg={avg_teacher:.3f} delta_r={delta_r:+.3f} "
        f"rubric={avg_rubric:.2f} full_pass={full_pass_count}/{len(group)}"
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

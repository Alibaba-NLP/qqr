"""
Travel rubrics solver reward model

Reward design:
- Each expected_tool counts as one extra rubric alongside the content rubrics
- pass_rate = (content_rubrics_met + tool_rubrics_met) / (content_rubrics_total + n_expected_tools)
- reward = 0.8 · pass_rate + 0.2 · 𝟙[all satisfied]
- All satisfied means tool_all_matched AND rubric_all_passed

Data requirements (JSONL field mapping):
- sample.prompt: either a query string or a list of messages
- sample.metadata.expected_tools: [{"name": "...", "arguments": {...}}, ...]
- sample.metadata.rubrics: ["rubric1", "rubric2", ...]
- sample.label: reference messages (optional, evaluation mode only)
"""

import asyncio
import fcntl
import json
import logging
import os
import re
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
    """Return the LLM judge client (rubric scoring and tool-argument matching)."""
    global _judge_client
    if _judge_client is None:
        _judge_client = AsyncOpenAI(
            api_key=config.llm_judge_api_key,
            base_url=config.llm_judge_base_url,
            timeout=120,
            max_retries=5,
        )
    return _judge_client


def get_judge_semaphore() -> asyncio.Semaphore:
    """Return the semaphore limiting judge concurrency."""
    global _judge_semaphore
    if _judge_semaphore is None:
        _judge_semaphore = asyncio.Semaphore(config.llm_judge_concurrency_limit)
    return _judge_semaphore


# ============ Rubric scoring ============


async def evaluate_rubrics(
    messages: list[dict], rubrics: list[str], query: str
) -> tuple[float, list[dict]]:
    """
    Use the LLM judge to check whether the solver's conversation satisfies every rubric.

    Returns:
        (score, details) where:
        - score = satisfied rubrics / total rubrics (0.0 to 1.0)
        - details = [{rubric, met, justification}, ...]
    """
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
- 这是一个user与assistant之间的对话场景，其中assistant可以调用工具获取信息和完成操作，工具返回结果将以tool开头
- 你需要评估用户指令是否被完成，用户的完整指令已被拆分为若干个得分点rubric，你只需要判断每个得分点是否满足
- <trajectory_content>包含了user与assistant之间的完整对话内容
- <current_rubrics>包含了当前所有得分点的状态（true表示已满足，false表示未满足，所有得分点的初始状态均为false）

# 任务
- 基于对话内容，更新得分点rubric的状态
- 你可以将状态由false更新为true，当且仅当assistant在对话中完成了该目标

# 判定原则
- **宽松判定**：只要assistant的回复或工具调用**大致涉及**了该rubric要求的内容，即使不够详细或略有瑕疵，也应判为满足（true）
- 只有在assistant**完全没有提及**或**明显答错/答反**时，才判为不满足（false）
- 不要因为格式、详细程度、额外信息缺失等次要问题扣分

# 注意事项
- 所有的评估以assistant的回复及工具调用请求是否完成rubric中的目标为准
- 查询类tool返回的结果仅对assistant可见，并不代表assistant对用户推荐的内容，一切都要以assistant获取信息后对用户的回复为准。同时需要注意，Assistant 也不能编造 Tool 的返回结果
- 在justification中以追加的形式记录与当前rubric有关的关键信息及其对应的轮次[x]，如果发生状态修改也需要记录原因，使用简练的语言

# 格式要求
- 你的回复应为一个JSON数组，包含以下字段：
- `rubric_idx`：规则的唯一标识符
- `rubric`：对规则的复述
- `justification`：对状态变化的解释
- `meetExpectation`：更新后的状态（true或false）

# 示例回复结构：
```json
[
  {{"rubric_idx": "rubric_0", "rubric": "<复述规则>", "justification": "<状态变化的简要解释>", "meetExpectation": true}},
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

    max_retries = 3
    details = None
    for attempt in range(max_retries):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=config.llm_judge_model,
                    messages=llm_messages,
                    temperature=0.0,
                    max_completion_tokens=4000,
                )
            result_text = resp.choices[0].message.content or ""
            details = _parse_rubric_results(result_text, rubrics)
            if not all(d["justification"] == "解析失败" for d in details):
                break
            logger.warning(
                f"[evaluate_rubrics] Parse failed on attempt {attempt + 1}/{max_retries}, retrying..."
            )
        except Exception as e:
            logger.warning(
                f"[evaluate_rubrics] LLM call failed on attempt {attempt + 1}/{max_retries}: {e}"
            )
            details = None

    if details is None:
        details = [
            {"rubric": r, "met": False, "justification": f"评估失败（重试{max_retries}次）"}
            for r in rubrics
        ]

    met_count = sum(1 for d in details if d.get("met", False))
    score = met_count / len(rubrics)
    return score, details


def _format_trajectory(messages: list[dict]) -> str:
    """Format the full conversation for scoring."""
    content_lines = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        full_content = content or ""

        if role == "assistant" and tool_calls:
            tc_strs = []
            for tc in tool_calls:
                func = tc.get("function", tc)
                name = func.get("name", "")
                args = func.get("arguments", "")
                if isinstance(args, str):
                    try:
                        args_dict = json.loads(args)
                        args_str = ", ".join(
                            f"{k}={repr(v)}" for k, v in args_dict.items()
                        )
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
                if full_content:
                    full_content += " " + tc_text
                else:
                    full_content = tc_text

        if full_content:
            content_lines.append(f"[{i + 1}] {role}: {full_content}")

    return "\n".join(content_lines)


def _parse_rubric_results(text: str, rubrics: list[str]) -> list[dict]:
    """Parse the rubric scores returned by the LLM."""
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        results = json.loads(text[start:end])
        if isinstance(results, list):
            details = []
            for i, rubric in enumerate(rubrics):
                if i < len(results):
                    r = results[i]
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
        logger.warning(f"[_parse_rubric_results] Failed to parse results: {e}")

    return [
        {"rubric": r, "met": False, "justification": "解析失败"}
        for r in rubrics
    ]


# ============ Tool argument matching ============


def _extract_actual_tools(messages: list[dict]) -> list[dict]:
    """Extract every tool actually called in the messages (name and arguments)."""
    tools: list[dict] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            name = tc.get("function", {}).get("name") or tc.get("name", "")
            raw_args = tc.get("function", {}).get("arguments") or tc.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
            elif isinstance(raw_args, dict):
                arguments = raw_args
            else:
                arguments = {}
            if name:
                tools.append({"name": name, "arguments": arguments})
    return tools


# prompt template for semantic argument matching
ARGUMENT_MATCH_PROMPT = """你是一个工具参数匹配评估专家。请判断【实际参数】是否与【期望参数】在语义上匹配。

【工具名称】
{tool_name}

【期望参数】
{expected_args}

【实际参数】
{actual_args}

【通用匹配标准（宽松）】
1. 参数值不需要完全相同，只需语义上表达相同的意图
2. 地名宽松匹配："西湖"和"杭州西湖"、"北京站"和"北京火车站"、"北京"和"北京市" 均视为匹配
3. 关键词宽松匹配："餐厅"和"美食餐厅"、"景点"和"旅游景点" 均视为匹配
4. 坐标参数：只要指向同一个地点即可（允许其中数字不同）

【各工具特殊规则】
- search_flights / search_train_tickets：只需 from_city 和 to_city 指向相同城市即可匹配，如果期望参数中存在 date 则要求一致，否则忽略 date。
- web_search：只需 query 的查询意图大致相同即可匹配，不要求查询条数一致，不要求用词完全相同
- poi_search / around_search：核心地名或关键词语义相同即可匹配
- weather：只需 query 的查询意图大致相同即可匹配，不要求查询条数一致，不要求用词完全相同

请只输出 "MATCH" 或 "NOT_MATCH"，不要输出其他内容。"""


def is_coordinate(value: str) -> bool:
    """Whether a string is in coordinate format, e.g. 120.121358,30.222692."""
    try:
        parts = value.split(",")
        if len(parts) == 2:
            float(parts[0])
            float(parts[1])
            return True
    except (ValueError, AttributeError):
        pass
    return False


def coordinates_close(coord1: str, coord2: str, threshold: float = 0.1) -> bool:
    """Whether two coordinates are close enough (0.1 degrees is about 11km)."""
    try:
        lon1, lat1 = map(float, coord1.split(","))
        lon2, lat2 = map(float, coord2.split(","))
        return abs(lon1 - lon2) < threshold and abs(lat1 - lat2) < threshold
    except (ValueError, AttributeError):
        return False


async def check_arguments_match_llm(
    tool_name: str,
    expected_args: dict,
    actual_args: dict,
) -> bool:
    """Use the LLM judge for semantic argument matching."""
    if not expected_args:
        return True
    if expected_args == actual_args:
        return True

    client = get_judge_client()
    semaphore = get_judge_semaphore()

    prompt = ARGUMENT_MATCH_PROMPT.format(
        tool_name=tool_name,
        expected_args=json.dumps(expected_args, ensure_ascii=False, indent=2),
        actual_args=json.dumps(actual_args, ensure_ascii=False, indent=2),
    )

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="gpt-5-mini-2025-08-07",
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.choices[0].message.content.strip().upper()
            is_match = "MATCH" in result and "NOT" not in result
            logger.debug(
                f"[check_arguments_match_llm] {tool_name}: "
                f"expected={expected_args}, actual={actual_args}, match={is_match}"
            )
            return is_match
        except Exception as e:
            logger.warning(f"[check_arguments_match_llm] judge model 调用失败: {e}")
            return check_arguments_match_fallback(expected_args, actual_args, tool_name)


def check_arguments_match_fallback(
    expected_args: dict, actual_args: dict, tool_name: str = ""
) -> bool:
    """Rule-based matching (a fallback), using the same lenient criteria as the judge prompt."""
    if not expected_args:
        return True

    # search_flights / search_train_tickets: compare only from_city and to_city
    if tool_name in ("search_flights", "search_train_tickets"):
        expected_from = str(expected_args.get("from_city", "")).strip()
        expected_to = str(expected_args.get("to_city", "")).strip()
        actual_from = str(actual_args.get("from_city", "")).strip()
        actual_to = str(actual_args.get("to_city", "")).strip()
        if not expected_from or not expected_to:
            return True
        from_match = expected_from in actual_from or actual_from in expected_from
        to_match = expected_to in actual_to or actual_to in expected_to
        return from_match and to_match

    # web_search: overlapping query keywords are enough
    if tool_name == "web_search":
        expected_query = expected_args.get("query", "")
        actual_query = actual_args.get("query", "")
        eq_list = [expected_query] if isinstance(expected_query, str) else list(expected_query)
        aq_list = [actual_query] if isinstance(actual_query, str) else list(actual_query)
        expected_keywords = set(re.findall(r"[\u4e00-\u9fff]{2,}", "".join(eq_list)))
        actual_keywords = set(re.findall(r"[\u4e00-\u9fff]{2,}", "".join(aq_list)))
        if not expected_keywords:
            return True
        overlap = len(expected_keywords & actual_keywords) / len(expected_keywords)
        return overlap >= 0.3

    # around_search: the location coordinates may differ by up to 0.1 degrees
    if tool_name == "around_search":
        expected_loc = str(expected_args.get("location", "")).strip()
        actual_loc = str(actual_args.get("location", "")).strip()
        if expected_loc and actual_loc:
            if is_coordinate(expected_loc) and is_coordinate(actual_loc):
                if not coordinates_close(expected_loc, actual_loc, threshold=0.1):
                    return False
            elif expected_loc.lower() not in actual_loc.lower() and actual_loc.lower() not in expected_loc.lower():
                return False
        if "radius" in expected_args:
            expected_radius = expected_args.get("radius")
            actual_radius = actual_args.get("radius")
            if expected_radius is not None and actual_radius is not None:
                if str(expected_radius) != str(actual_radius):
                    return False
        return True

    # direction: origin/destination coordinates may differ by up to 0.1 degrees, plus waypoints and mode
    if tool_name == "direction":
        for key in ("origin", "destination"):
            ev = str(expected_args.get(key, "")).strip()
            av = str(actual_args.get(key, "")).strip()
            if not ev:
                continue
            if not av:
                return False
            if is_coordinate(ev) and is_coordinate(av):
                if not coordinates_close(ev, av, threshold=0.1):
                    return False
            elif ev.lower() not in av.lower() and av.lower() not in ev.lower():
                return False

        def _parse_waypoints(val) -> list[str]:
            if not val:
                return []
            if isinstance(val, list):
                return [str(w).strip() for w in val if str(w).strip()]
            return [w.strip() for w in re.split(r"[;|]", str(val)) if w.strip()]

        expected_wps = _parse_waypoints(expected_args.get("waypoints"))
        actual_wps = _parse_waypoints(actual_args.get("waypoints"))
        if len(expected_wps) != len(actual_wps):
            return False
        for ewp, awp in zip(expected_wps, actual_wps):
            if is_coordinate(ewp) and is_coordinate(awp):
                if not coordinates_close(ewp, awp, threshold=0.1):
                    return False
            elif ewp.lower() not in awp.lower() and awp.lower() not in ewp.lower():
                return False
        if "mode" in expected_args:
            if expected_args["mode"] != actual_args.get("mode"):
                return False
        return True

    # generic matching
    for key, expected_value in expected_args.items():
        if key in ("mode", "waypoints", "date"):
            continue
        if key not in actual_args:
            return False
        actual_value = actual_args[key]
        ev_str, av_str = str(expected_value), str(actual_value)
        if is_coordinate(ev_str) and is_coordinate(av_str):
            if not coordinates_close(ev_str, av_str):
                return False
            continue
        if isinstance(expected_value, str) and isinstance(actual_value, str):
            if expected_value.lower() in actual_value.lower():
                continue
            if actual_value.lower() in expected_value.lower():
                continue
            expected_words = set(expected_value.lower().split())
            actual_words = set(actual_value.lower().split())
            if not expected_words & actual_words:
                return False
        elif expected_value != actual_value:
            return False

    return True


async def compute_tool_accuracy_with_details(
    expected_tools: list[dict],
    actual_tools: list[dict],
) -> tuple[float, list[dict]]:
    """
    Compute tool-call accuracy and return the match details, using the LLM judge for semantic matching.

    Returns:
        (accuracy, list of match details)
    """
    match_details = []

    if not expected_tools:
        return (1.0 if not actual_tools else 0.0), match_details

    if not actual_tools:
        for expected in expected_tools:
            match_details.append({
                "expected": expected,
                "actual": None,
                "matched": False,
                "reason": "做题者未调用任何工具",
            })
        return 0.0, match_details

    # collect every (expected, actual) pair that needs matching
    match_tasks = []
    match_indices = []

    RULE_BASED_TOOLS = {"around_search", "direction"}

    async def _rule_match(ea, aa, tn):
        return check_arguments_match_fallback(ea, aa, tn)

    for expected_idx, expected in enumerate(expected_tools):
        expected_name = expected.get("name", "")
        expected_args = expected.get("arguments", {})

        for actual_idx, actual in enumerate(actual_tools):
            actual_name = actual.get("name", "")
            actual_args = actual.get("arguments", {})

            if expected_name == actual_name:
                if expected_name in RULE_BASED_TOOLS:
                    match_tasks.append(
                        _rule_match(expected_args, actual_args, expected_name)
                    )
                else:
                    match_tasks.append(
                        check_arguments_match_llm(expected_name, expected_args, actual_args)
                    )
                match_indices.append((expected_idx, actual_idx))

    if not match_tasks:
        for expected in expected_tools:
            match_details.append({
                "expected": expected,
                "actual": None,
                "matched": False,
                "reason": f"工具名 {expected.get('name')} 未被调用",
            })
        return 0.0, match_details

    match_results = await asyncio.gather(*match_tasks, return_exceptions=True)

    matched_expected = {}
    for (expected_idx, actual_idx), result in zip(match_indices, match_results):
        if isinstance(result, Exception):
            logger.warning(f"[compute_tool_accuracy_with_details] 匹配异常: {result}")
            continue
        if result and expected_idx not in matched_expected:
            matched_expected[expected_idx] = (actual_idx, actual_tools[actual_idx])

    for expected_idx, expected in enumerate(expected_tools):
        if expected_idx in matched_expected:
            actual_idx, actual = matched_expected[expected_idx]
            match_details.append({
                "expected": expected,
                "actual": actual,
                "matched": True,
                "reason": "语义匹配成功",
            })
        else:
            same_name_tools = [
                a for a in actual_tools if a.get("name") == expected.get("name")
            ]
            if same_name_tools:
                match_details.append({
                    "expected": expected,
                    "actual": same_name_tools[0],
                    "matched": False,
                    "reason": "工具名匹配但参数不匹配",
                })
            else:
                match_details.append({
                    "expected": expected,
                    "actual": None,
                    "matched": False,
                    "reason": f"工具 {expected.get('name')} 未被调用",
                })

    accuracy = len(matched_expected) / len(expected_tools)
    return accuracy, match_details


# ============ Data extraction helpers ============


def _get_expected_tools(sample: Sample) -> list[dict]:
    """Extract expected_tools from the sample."""
    if sample.metadata and isinstance(sample.metadata, dict):
        et = sample.metadata.get("expected_tools", [])
        if et:
            return et
    # Fallback: try label
    if isinstance(sample.label, list) and sample.label:
        if isinstance(sample.label[0], dict) and "name" in sample.label[0]:
            return sample.label
    if isinstance(sample.label, str):
        try:
            parsed = json.loads(sample.label)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                if "name" in parsed[0]:
                    return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def _get_rubrics(sample: Sample) -> list[str]:
    """Extract the rubrics from the sample."""
    if sample.metadata and isinstance(sample.metadata, dict):
        return sample.metadata.get("rubrics", [])
    return []


def _get_query(sample: Sample) -> str:
    """Extract the query from the sample."""
    if isinstance(sample.prompt, str):
        return sample.prompt
    if isinstance(sample.prompt, list):
        for msg in reversed(sample.prompt):
            if msg.get("role") == "user":
                return msg.get("content", "")
    return ""


# ============ Group memory (debug persistence) ============


def _get_memory_dir(args: Namespace) -> str:
    save_dir = getattr(args, "save", None)
    if not save_dir:
        return ""
    memory_dir = os.path.join(save_dir, "memory")
    os.makedirs(memory_dir, exist_ok=True)
    return memory_dir


async def generate_group_summary(
    args: Namespace,
    group_id: int,
    group_results: list[dict],
    query: str = "",
) -> tuple[str | None, str]:
    """Produce coach feedback from this group's evaluation results. Returns (summary, prompt)."""
    total = len(group_results)
    avg_reward = sum(r.get("reward", 0) for r in group_results) / total if total else 0
    full_pass_count = sum(1 for r in group_results if r.get("full_pass"))

    sample_summaries = []
    for i, r in enumerate(group_results):
        rubric_parts = ""
        for rd in r.get("rubric_details", []):
            status = "+" if rd.get("met") else "-"
            rubric_parts += f"\n    {status} {rd.get('rubric', '')}"

        sample_summaries.append(
            f"### 第{i+1}个样本 (reward={r.get('reward', 0):.2f})\n"
            f"- 期望工具: {r.get('expected_tool_names', [])}  "
            f"实际工具: {r.get('actual_tool_names', [])}  "
            f"缺失: {r.get('missing_tools', []) or '无'}\n"
            f"- rubric详情: {rubric_parts}\n"
        )

    coach_prompt = f"""你是一个旅行规划做题训练的教练。以下是同一个 query 下 {total} 个做题者样本的评估结果。

## 用户原始问题
{query[:300]}

## 本组做题结果（{total} 次尝试，全通过: {full_pass_count}/{total}，平均 reward: {avg_reward:.2f}）

{"".join(sample_summaries)}

## 你的任务

分析做题者的失败模式，生成**具体、可操作**的做题建议（不超过300字）。

### 要求：
1. **失败根因**：最常见的失败原因是什么？（漏调工具？参数错误？信息缺失？推理错误？）
2. **正确的工具调用流程**：针对这个具体 query，应该按什么顺序调用哪些工具？缺失的工具为什么重要？
3. **回答要点**：rubric 要求哪些关键信息必须出现在回答中？做题者最容易遗漏什么？
4. **一句话核心建议**：下次遇到类似问题，第一步该做什么？

请直接输出建议，不要重复上述内容。"""

    try:
        client = get_judge_client()
        semaphore = get_judge_semaphore()
        async with semaphore:
            response = await client.chat.completions.create(
                model=config.llm_judge_model,
                messages=[{"role": "user", "content": coach_prompt}],
                temperature=0.7,
                max_tokens=500,
            )
        summary = response.choices[0].message.content.strip()
        return summary, coach_prompt
    except Exception as e:
        logger.warning(f"[generate_group_summary] group_id={group_id} coach 调用失败: {e}")
        return None, coach_prompt


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


# ============ Per-sample reward computation ============


async def _compute_sample_reward(sample: Sample) -> dict:
    """
    Compute a single sample's reward.

    Each expected_tool match counts as one extra rubric, scored together with the content rubrics:
    reward = (content_rubrics_met + tool_rubrics_met) / (content_rubrics_total + n_expected_tools)

    For example, 3 of 4 content rubrics passed plus 1 of 2 tools matched gives reward = 4/6 = 0.67
         4 content rubrics all passed plus 2 tools all matched gives reward = 6/6 = 1.0
         4 content rubrics all passed but 0 of 3 tools matched gives reward = 4/7 = 0.57
    """
    messages = sample.messages or []
    expected_tools = _get_expected_tools(sample)
    rubrics = _get_rubrics(sample)
    query = _get_query(sample)

    # 1. Tool matching
    actual_tools = _extract_actual_tools(messages)
    actual_tool_names = sorted({t["name"] for t in actual_tools})
    expected_tool_names = sorted({t.get("name", "") for t in expected_tools if t.get("name")})

    tool_accuracy = 0.0
    tool_match_details = []
    if expected_tools:
        tool_accuracy, tool_match_details = await compute_tool_accuracy_with_details(
            expected_tools, actual_tools
        )
    elif not actual_tools:
        tool_accuracy = 1.0
    if not expected_tools:
        tool_accuracy = 1.0

    tool_all_matched = (tool_accuracy == 1.0)
    missing_tools = sorted(set(expected_tool_names) - set(actual_tool_names))

    # 2. Rubric scoring
    rubric_score = 0.0
    rubric_details = []
    if rubrics:
        rubric_score, rubric_details = await evaluate_rubrics(messages, rubrics, query)
    else:
        rubric_score = 1.0

    rubric_all_passed = (rubric_score == 1.0)

    # 3. Each expected_tool match counts as one extra rubric in the combined reward
    content_rubrics_met = sum(1 for d in rubric_details if d.get("met", False))
    content_rubrics_total = len(rubrics) if rubrics else 0
    n_expected_tools = len(expected_tools)
    tool_rubrics_met = sum(1 for d in tool_match_details if d.get("matched", False))

    total_rubrics = content_rubrics_total + n_expected_tools
    total_met = content_rubrics_met + tool_rubrics_met
    full_pass = tool_all_matched and rubric_all_passed

    pass_rate = total_met / total_rubrics if total_rubrics > 0 else 0.0
    reward = 0.8 * pass_rate + (0.2 if full_pass else 0.0)

    return {
        "reward": reward,
        "query": query,
        "rubric_pass_rate": rubric_score,
        "rubric_details": rubric_details,
        "tool_accuracy": tool_accuracy,
        "tool_all_matched": tool_all_matched,
        "tool_match_details": tool_match_details,
        "rubric_all_passed": rubric_all_passed,
        "full_pass": full_pass,
        "expected_tool_names": expected_tool_names,
        "actual_tool_names": actual_tool_names,
        "missing_tools": missing_tools,
    }


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
        "tool_accuracy": result["tool_accuracy"],
        "tool_all_matched": result["tool_all_matched"],
        "tool_match_details": result["tool_match_details"],
        "rubric_all_passed": result["rubric_all_passed"],
        "full_pass": result["full_pass"],
        "expected_tool_names": result["expected_tool_names"],
        "actual_tool_names": result["actual_tool_names"],
        "missing_tools": result["missing_tools"],
    })

    logger.info(
        f"[eval_reward] query={result['query'][:50]}... "
        f"rubric_rate={result['rubric_pass_rate']:.2f} "
        f"tool_acc={result['tool_accuracy']:.2f} "
        f"full_pass={result['full_pass']} "
        f"reward={result['reward']:.2f} "
        f"expected={result['expected_tool_names']} "
        f"actual={result['actual_tool_names']}"
    )


async def _maybe_compute_reward(sample: Sample) -> dict:
    """Use the evaluation cached during rollout (student), or recompute it (teacher)."""
    cached = (sample.metadata or {}).get("_rubric_cache")
    if cached is not None:
        return cached
    return await _compute_sample_reward(sample)


async def group_reward(args: Namespace, group: list[list[Sample]], **kwargs):
    """
    Reward-Gated Reverse KL (RG-KL)：
    - The k_s student samples are trained on, using their true rewards for GRPO
    - The k_m teacher samples only estimate Δ_r; their reward is neutralised to student_mean and loss_mask=0
    - λ(Δ_r) = λ_0 · max(Δ_r - δ_thresh, 0) · warmup · cosine_decay
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
            "tool_accuracy": result["tool_accuracy"],
            "tool_all_matched": result["tool_all_matched"],
            "tool_match_details": result["tool_match_details"],
            "rubric_all_passed": result["rubric_all_passed"],
            "full_pass": result["full_pass"],
            "expected_tool_names": result["expected_tool_names"],
            "actual_tool_names": result["actual_tool_names"],
            "missing_tools": result["missing_tools"],
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
            f"rubric_rate={result['rubric_pass_rate']:.2f} "
            f"tool_acc={result['tool_accuracy']:.2f} "
            f"full_pass={result['full_pass']} "
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

    avg_rubric = sum(r["rubric_pass_rate"] for r in results) / len(results)
    avg_tool = sum(r["tool_accuracy"] for r in results) / len(results)
    full_pass_count = sum(1 for r in results if r["full_pass"])

    # -- 4) RG-KL: compute λ(Δ_r) and write it to sample.rg_kl_coef --
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

    logger.info(
        f"[group_reward] {len(group)} samples raw_avg={sum(raw_rewards)/len(raw_rewards):.3f} "
        f"student_avg={avg_student:.3f} teacher_avg={avg_teacher:.3f} delta_r={delta_r:+.3f} "
        f"rubric={avg_rubric:.2f} tool={avg_tool:.2f} full_pass={full_pass_count}/{len(group)}"
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

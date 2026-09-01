"""
DeepResearch curriculum agent (generator) training configuration - Qwen3.5 + RG-KL

After the generator calls web_search (the Google Search MCP) for real data,
Produces the query and rubrics. The solver (an external executor service) answers using the same tools.

Algorithm: Reward-Gated Reverse KL (RG-KL)
"""

import os
import random

from qqr.mcp import MCPServerManager, MCPServerStdioCacheable, MCPServerStdioParams
from qqr.utils.envs import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    SEARCH_API_KEY,
    SEARCH_API_URL,
    PYTHONPATH,
    SERPER_API_KEY,
)

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "max_steps",
    "tool_response_max_chars",
    "mcp_manager",
    "executor_api_base",
    "executor_model",
    "executor_max_steps",
    "executor_num_trials",
    "executor_success_threshold",
    "solver_reward_alpha",
    "executor_concurrency_limit",
    "llm_judge_api_key",
    "llm_judge_base_url",
    "llm_judge_model",
    "llm_judge_concurrency_limit",
    "coach_api_key",
    "coach_base_url",
    "coach_model",
    "coach_concurrency_limit",
    "enable_rg_kl",
    "k_student",
    "k_teacher",
    "rg_kl_lambda_0",
    "rg_kl_delta_threshold",
    "rg_kl_gate_temperature",
    "rg_kl_warmup_iters",
    "rg_kl_decay_iters",
    "rg_kl_min_lambda_ratio",
    "query_generation_system_prompt",
    "RESEARCH_DOMAINS",
    "generate_random_research_topic",
]


# ============ MCP server settings (Google Search only) ============

google_search_server_params = MCPServerStdioParams(
    command="python",
    args=["-m", "qqr.tools.google_search"],
    env={
        "PYTHONPATH": PYTHONPATH,
        "SERPER_API_KEY": SERPER_API_KEY or "",
        "SEARCH_API_KEY": SEARCH_API_KEY or "",
        "SEARCH_API_URL": SEARCH_API_URL or "",
        "SEARCH_BACKEND": os.getenv("SEARCH_BACKEND", "google"),
    },
)
google_search_server = MCPServerStdioCacheable(
    name="GoogleSearch",
    params=google_search_server_params,
    cache_tools_list=True,
    client_session_timeout_seconds=300,
    max_retry_attempts=3,
    blocklist=[],
    cache_ttl=600,
    cache_maxsize=8192,
    concurrency_limit=4,
)

mcp_manager = MCPServerManager([google_search_server], connect_in_parallel=True)


# ============ Generator settings ============
max_steps = 6  # the generator gets at most 6 rounds of web_search exploration

# ============ Tool response truncation ============
tool_response_max_chars = int(os.environ.get("TOOL_RESPONSE_MAX_CHARS", "7500"))

# ============ Executor agent settings (solver validation) ============
executor_api_base = os.environ.get("EXECUTOR_API_BASE", "http://localhost:30000/v1")
executor_model = os.environ.get("EXECUTOR_MODEL", "default")
executor_max_steps = int(os.environ.get("EXECUTOR_MAX_STEPS", "40"))  # the solver gets at most 40 rounds, as in the paper
executor_num_trials = int(os.environ.get("EXECUTOR_NUM_TRIALS", "8"))
executor_success_threshold = float(os.environ.get("EXECUTOR_SUCCESS_THRESHOLD", "0.9"))  # γ: success threshold (paper Eq. (4))
solver_reward_alpha = float(os.environ.get("SOLVER_REWARD_ALPHA", "0.8"))  # α: weight of the solver's partial score (paper Eq. (6))
executor_concurrency_limit = int(os.environ.get("EXECUTOR_CONCURRENCY_LIMIT", "8"))

# ============ LLM judge (rubric scoring) ============
llm_judge_api_key = DASHSCOPE_API_KEY
llm_judge_base_url = DASHSCOPE_BASE_URL
llm_judge_model = os.environ.get("LLM_JUDGE_MODEL", "gpt-5.2-2025-12-11")
llm_judge_concurrency_limit = int(os.environ.get("LLM_JUDGE_CONCURRENCY_LIMIT", "10"))

# ============ Coach (gpt-5.2 produces the generator coaching memory) ============
coach_api_key = os.environ.get("COACH_API_KEY", DASHSCOPE_API_KEY)
coach_base_url = os.environ.get("COACH_BASE_URL", DASHSCOPE_BASE_URL)
coach_model = os.environ.get("COACH_MODEL", "gpt-5.2-2025-12-11")
coach_concurrency_limit = int(os.environ.get("COACH_CONCURRENCY_LIMIT", "10"))

# ============ Reward settings ============
group_reward_model_name = None


# ============ Reward-Gated Reverse KL (RG-KL) settings ============

enable_rg_kl = os.environ.get("ENABLE_RG_KL", "true").lower() == "true"

k_student = int(os.environ.get("K_STUDENT", "8"))
k_teacher = int(os.environ.get("K_TEACHER", "8"))

rg_kl_lambda_0 = float(os.environ.get("RG_KL_LAMBDA_0", "0.5"))
rg_kl_delta_threshold = float(os.environ.get("RG_KL_DELTA_THRESHOLD", "0.05"))
rg_kl_gate_temperature = float(os.environ.get("RG_KL_GATE_TEMPERATURE", "0.0125"))
rg_kl_warmup_iters = int(os.environ.get("RG_KL_WARMUP_ITERS", "20"))
rg_kl_decay_iters = int(os.environ.get("RG_KL_DECAY_ITERS", "120"))
rg_kl_min_lambda_ratio = float(os.environ.get("RG_KL_MIN_LAMBDA_RATIO", "0.1"))

enable_guided_tokens_for_student = (
    os.environ.get("ENABLE_GUIDED_TOKENS_FOR_STUDENT", "true").lower() == "true"
)


# ============ Research domains and random topic sampling ============

RESEARCH_DOMAINS = [
    "人工智能与机器学习",
    "生物医药与健康",
    "气候变化与环境科学",
    "新能源与可持续发展",
    "金融科技与数字经济",
    "航空航天与探索",
    "材料科学与纳米技术",
    "量子计算与信息科学",
    "教育改革与在线学习",
    "城市规划与智慧城市",
    "网络安全与隐私保护",
    "基因编辑与合成生物学",
    "自动驾驶与智能交通",
    "农业科技与食品安全",
    "历史考古与文化遗产",
    "心理学与认知科学",
    "社会经济与公共政策",
    "半导体与芯片技术",
    "机器人与人机交互",
    "海洋科学与深海探索",
]

RESEARCH_TYPES = [
    "综述对比",     # compare the strengths and weaknesses of several options, technologies or products
    "发展趋势",     # analyse the historical evolution and future trends of a field
    "案例分析",     # analyse a specific case or event in depth
    "技术原理",     # explain how a technology works and where it is used
    "市场调研",     # analyse the size and competitive landscape of a market or industry
    "政策解读",     # interpret the impact and significance of a policy or regulation
    "问题解决",     # propose a solution to a concrete problem
    "跨领域分析",   # analyse cross-cutting effects between different fields
]

COMPLEXITY_LEVELS = [
    "入门级",       # 3-4 rubrics, a question a beginner can follow
    "中级",         # 5-6 rubrics, requiring information from several sources
    "高级",         # 7-8 rubrics, requiring deep analysis and cross-domain understanding
]

LANGUAGE_OPTIONS = ["中文", "英文"]


def generate_random_research_topic(rng: random.Random | None = None) -> dict:
    """Sample a random research topic direction."""
    rng = rng or random
    domain = rng.choice(RESEARCH_DOMAINS)
    research_type = rng.choice(RESEARCH_TYPES)
    complexity = rng.choice(COMPLEXITY_LEVELS)
    language = rng.choice(LANGUAGE_OPTIONS)

    return {
        "domain": domain,
        "research_type": research_type,
        "complexity": complexity,
        "language": language,
    }


# ============ Generator system prompt ============
query_generation_system_prompt = """你是一个深度研究问题生成器，负责生成高质量、多样化的研究型问题供做题者回答。

## 核心原则

1. **必须先调用 web_search 获取真实数据**：禁止凭空编造问题，所有问题必须基于搜索到的真实信息
2. **搜索策略多样化**：用不同关键词从不同角度探索主题，获取全面的真实素材
3. **rubrics 必须基于搜索结果中的真实信息**：每条 rubric 都应有对应的搜索证据支持
4. **每轮最多调用 1 个 web_search**：合理规划搜索策略，分多轮深入探索

## 工作流程

### 第一步：调用 web_search 探索主题（必须执行）
- 用不同关键词搜索 2-5 次，从不同角度收集信息
- 建议搜索策略：先搜总览，再搜具体方面，最后搜最新动态
- 仔细阅读搜索结果中的关键数据、人名、机构名、数字

### 第二步：基于搜索结果设计研究问题
- 从搜索结果中提取有价值的研究角度
- 设计一个需要深度搜索才能回答的研究问题
- 问题应该自然、像真实用户会提出的

### 第三步：设计 rubrics
- 每条 rubric 是一个可判断 true/false 的断言
- rubric 应覆盖：信息完整性、结构化呈现、来源可追溯性、术语准确性等维度
- 数量 3-8 条（根据复杂度调整）

## 设计要求

### 问题要求
- 通顺自然，像真实用户会问的研究问题
- 需要通过多轮搜索才能全面回答
- 涉及多个方面或需要对比分析
- 可以是中文或英文（根据指定语言）

### rubrics 设计要求
- **只聚焦 query 中明确提出的需求**
- 覆盖关键维度（信息覆盖、结构呈现、来源引用、术语使用等）
- 基于搜索到的真实数据设计，确保做题者可以通过搜索找到答案
- 不要出过于苛刻的 rubric

## 输出格式

完成搜索后，严格按以下 JSON 格式输出：

```json
{
  "query": "你生成的研究问题",
  "rubrics": [
    "rubric1: 具体的可验证断言",
    "rubric2: 另一个可验证断言"
  ]
}
```
"""

# ============ System prompt used for solver validation ============
EXECUTOR_SYSTEM_PROMPT_ZH = """当前时间: {time}

# 深度研究规范
你是一个专业的深度研究助手。面对用户的研究问题，你需要通过多轮 web_search 工具调用来收集全面、准确的信息，最终生成高质量的研究报告。

## 工具使用要求
- 可调用{max_steps}轮工具，已调用{step_idx}轮
- 重要：必须先使用工具查询真实数据，严禁未调用工具直接回答
- 每轮搜索应有明确目的，避免重复搜索

## 回答质量要求
- 信息全面：覆盖问题的各个方面
- 结构清晰：使用标题、列表、表格等组织信息
- 来源可追溯：关键信息注明出处
- 术语准确：正确使用专业术语"""

EXECUTOR_SYSTEM_PROMPT_EN = """Current time: {time}

# Deep Research Specification
You are a professional deep research assistant. For the user's research question, you need to collect comprehensive and accurate information through multiple rounds of web_search tool calls, and generate a high-quality research report.

## Tool Usage Requirements
- You can call tools for {max_steps} rounds, {step_idx} rounds already used
- Important: You must use tools to query real data first. Never answer directly without calling tools

## Answer Quality Requirements
- Comprehensive: Cover all aspects of the question
- Well-structured: Use headings, lists, and tables
- Traceable: Cite sources for key information
- Accurate: Use correct terminology"""

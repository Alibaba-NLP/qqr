"""
Travel curriculum agent (generator) training configuration - Qwen3.5 + RG-KL

The generator calls the MCP tools (AMap/Transport/WebSearch) for real data, then
Produces the query, expected_tools and rubrics.

The solver (an external executor service) answers using the same tools,
The reward combines tool matching with the rubric pass rate.

Algorithm: Reward-Gated Reverse KL (RG-KL)
- Three-phase rollout: student (k_s, no coach) -> gpt-5.2 produces coach memory -> teacher (k_m, with coach)
- λ(Δ_r) = λ_0 · gate(Δ_r) · warmup · cosine_decay
"""

import os
import random

from qqr.mcp import MCPServer, MCPServerManager, MCPServerStdioCacheable, MCPServerStdioParams
from qqr.utils.envs import (
    AMAP_MAPS_API_KEY,
    BAILIAN_WEB_SEARCH_API_KEY,
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    PYTHONPATH,
    SEARCH_API_KEY,
    SEARCH_API_URL,
)

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "group_reward_model_name",
    "max_steps",
    "mcp_server_config_fn",
    "mcp_manager",
    "executor_api_base",
    "executor_model",
    "executor_max_steps",
    "executor_concurrency_limit",
    "executor_num_trials",
    "executor_success_threshold",
    "solver_reward_alpha",
    "llm_judge_model",
    "llm_judge_api_key",
    "llm_judge_base_url",
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
    "TASK_TYPES",
    "TASK_TYPE_PROMPTS",
    "get_random_task_type_prompt",
    "generate_random_search_suggestion",
]


# ============ MCP server settings (shared by generator and solver) ============
def mcp_server_config_fn() -> list[MCPServer]:
    """Return the list of MCP servers."""
    amap_server_params = MCPServerStdioParams(
        command="python",
        args=["-m", "qqr.tools.amap"],
        env={
            "AMAP_MAPS_API_KEY": AMAP_MAPS_API_KEY,
            "PYTHONPATH": PYTHONPATH,
        },
    )
    amap_server = MCPServerStdioCacheable(
        name="AMap",
        params=amap_server_params,
        cache_tools_list=True,
        client_session_timeout_seconds=60,
        max_retry_attempts=3,
        blocklist=[],
        cache_ttl=600,
        cache_maxsize=8192,
        concurrency_limit=16,
    )

    transport_server_params = MCPServerStdioParams(
        command="python",
        args=["-m", "qqr.tools.mock_transport"],
        env={
            "DASHSCOPE_API_KEY": DASHSCOPE_API_KEY,
            "DASHSCOPE_BASE_URL": DASHSCOPE_BASE_URL,
            "PYTHONPATH": PYTHONPATH,
        },
    )
    transport_server = MCPServerStdioCacheable(
        name="Transport",
        params=transport_server_params,
        cache_tools_list=True,
        client_session_timeout_seconds=60,
        max_retry_attempts=3,
        blocklist=[],
        cache_ttl=600,
        cache_maxsize=8192,
        concurrency_limit=8,
    )

    web_search_server_params = MCPServerStdioParams(
        command="python",
        args=["-m", "qqr.tools.web_search"],
        env={
            "BAILIAN_WEB_SEARCH_API_KEY": BAILIAN_WEB_SEARCH_API_KEY,
            "SEARCH_API_KEY": SEARCH_API_KEY,
            "SEARCH_API_URL": SEARCH_API_URL,
            "PYTHONPATH": PYTHONPATH,
        },
    )
    web_search_server = MCPServerStdioCacheable(
        name="WebSearch",
        params=web_search_server_params,
        cache_tools_list=True,
        client_session_timeout_seconds=60,
        max_retry_attempts=3,
        blocklist=[],
        cache_ttl=600,
        cache_maxsize=8192,
        concurrency_limit=8,
    )

    return [amap_server, transport_server, web_search_server]


# Module-level MCPServerManager instance (rollout.py starts it via MCPState(mcp_manager))
# Lazy initialisation: mcp_server_config_fn() is only called on first access to build the server list
class _LazyMCPManager:
    """Create the MCPServerManager lazily so importing config does not immediately spawn MCP subprocesses."""
    _instance = None

    def __getattr__(self, name):
        if _LazyMCPManager._instance is None:
            _LazyMCPManager._instance = MCPServerManager(
                mcp_server_config_fn(), connect_in_parallel=True
            )
        return getattr(_LazyMCPManager._instance, name)


mcp_manager = _LazyMCPManager()


# ============ Executor agent settings ============
executor_api_base = os.environ.get(
    "EXECUTOR_API_BASE", "http://localhost:30000/v1"
)
executor_model = os.environ.get("EXECUTOR_MODEL", "default")
executor_max_steps = int(os.environ.get("EXECUTOR_MAX_STEPS", "40"))  # the solver gets at most 40 rounds, as in the paper
executor_concurrency_limit = int(os.environ.get("EXECUTOR_CONCURRENCY_LIMIT", "8"))
executor_num_trials = int(os.environ.get("EXECUTOR_NUM_TRIALS", "8"))
executor_success_threshold = float(os.environ.get("EXECUTOR_SUCCESS_THRESHOLD", "0.9"))  # γ: success threshold (paper Eq. (4))
solver_reward_alpha = float(os.environ.get("SOLVER_REWARD_ALPHA", "0.8"))  # α: weight of the solver's partial score (paper Eq. (6))


# ============ Curriculum agent generation settings ============
max_steps = 6  # the generator gets at most 6 rounds: 5 tool-calling rounds plus 1 generation round

# five task types
TASK_TYPES = [
    "direction",
    "compare_itinerary",
    "search_around",
    "one_day_travel",
    "multi_day_travel",
]

# the prompt matching this task type
TASK_TYPE_PROMPTS = {
    "direction": """【本次任务类型：多途经点路线规划】
生成一个包含多个途经点的路线规划问题（2个以上地点）。expected_tools中必须有direction工具。
""",

    "compare_itinerary": """【本次任务类型：出行方式对比】
生成一个对比不同出行方式（高铁vs飞机、高铁vs自驾等）的问题。
""",

    "search_around": """【本次任务类型：周边搜索】
生成一个搜索某地点周边设施或服务的问题。expected_tools中必须有around_search工具。
""",

    "one_day_travel": """【本次任务类型：一日游规划】
生成一个单城市一日游的旅行规划问题，涉及多个景点的游览顺序、时间分配、交通方式、天气等。
""",

    "multi_day_travel": """【本次任务类型：多日游规划】
生成一个跨城市或深度游的多天行程规划问题，可包含交通、住宿、景点、天气等多个维度。
""",
}


def get_random_task_type_prompt() -> tuple[str, str]:
    """Randomly choose a task type and return its base prompt."""
    task_type = random.choice(TASK_TYPES)
    return task_type, TASK_TYPE_PROMPTS.get(task_type, "")


# ============ Reward function settings ============
target_rubric_pass_rate = float(os.environ.get("TARGET_RUBRIC_PASS_RATE", "0.5"))
group_reward_model_name = None

# ============ LLM judge settings (rubric scoring + semantic argument matching) ============
llm_judge_api_key = DASHSCOPE_API_KEY
llm_judge_base_url = DASHSCOPE_BASE_URL
llm_judge_model = os.environ.get("LLM_JUDGE_MODEL", "gpt-5.2-2025-12-11")
llm_judge_concurrency_limit = int(os.environ.get("LLM_JUDGE_CONCURRENCY_LIMIT", "10"))

# ============ Coach settings (gpt-5.2 produces the generator coaching memory) ============
coach_api_key = os.environ.get("COACH_API_KEY", DASHSCOPE_API_KEY)
coach_base_url = os.environ.get("COACH_BASE_URL", DASHSCOPE_BASE_URL)
coach_model = os.environ.get("COACH_MODEL", "gpt-5.2-2025-12-11")
coach_concurrency_limit = int(os.environ.get("COACH_CONCURRENCY_LIMIT", "10"))


# ============ Reward-Gated Reverse KL (RG-KL) settings ============

enable_rg_kl = os.environ.get("ENABLE_RG_KL", "true").lower() == "true"

# Three-phase rollout split: n_samples_per_prompt = k_student + k_teacher
k_student = int(os.environ.get("K_STUDENT", "8"))
k_teacher = int(os.environ.get("K_TEACHER", "8"))

# RG-KL gating base coefficient
rg_kl_lambda_0 = float(os.environ.get("RG_KL_LAMBDA_0", "0.5"))

# Δ_r threshold
rg_kl_delta_threshold = float(os.environ.get("RG_KL_DELTA_THRESHOLD", "0.05"))
rg_kl_gate_temperature = float(os.environ.get("RG_KL_GATE_TEMPERATURE", "0.0125"))

# Warmup
rg_kl_warmup_iters = int(os.environ.get("RG_KL_WARMUP_ITERS", "20"))

# Cosine decay
rg_kl_decay_iters = int(os.environ.get("RG_KL_DECAY_ITERS", "120"))
rg_kl_min_lambda_ratio = float(os.environ.get("RG_KL_MIN_LAMBDA_RATIO", "0.1"))

# safety switch
enable_guided_tokens_for_student = (
    os.environ.get("ENABLE_GUIDED_TOKENS_FOR_STUDENT", "true").lower() == "true"
)


# ============ Debug settings ============
DEBUG_PRINT_CURRICULUM_PROMPT = os.environ.get(
    "DEBUG_PRINT_CURRICULUM_PROMPT", "false"
).lower() == "true"


# ============ Generator system prompt ============
query_generation_system_prompt = """你是一个旅行规划问题生成器，负责生成高质量、多样化的旅行规划问题供做题者回答。

## 核心原则

1. **必须先调用工具获取真实数据**：禁止凭空编造问题，所有问题必须基于工具返回的真实信息
2. **expected_tools 中的每个工具都必须事先调用验证过**：你需要亲自调用确认能返回合理结果
3. **善用 web_search 和 poi_search 拓宽出题范围**：
   - 可以用 web_search 探索感兴趣的任何事物
   - 可以用 poi_search 探索感兴趣的任何"keywords"
   - 可以将搜索到的真实信息（地名、活动、特色等）融入问题，让题目更贴近真实场景
   - 可以用地图/交通工具验证并获取详细数据，生成需要多工具协作的复杂问题。
4. **每轮最多调用3个工具**：合理规划工具调用，可以分多轮调用
5. **扩展到不同的城市和场景**：严格与示例素材区别出来，展现不同的城市和活动。

## 工作流程

### 第一步：调用工具获取数据（必须执行）
- 根据任务类型，选择合适的工具组合
- 建议先用 web_search 搜索相关信息获取灵感和真实素材
- 再用地图/交通工具获取具体数据（地点、路线、天气、票价等）
- 你可以分多轮调用不同工具

### 第二步：基于工具结果生成问题
- 从工具返回的真实数据中提取关键信息
- 生成一个自然、通顺的问题
- expected_tools 中放希望做题者调用的工具

## 生成要求

### 问题要求
- 通顺自然，像真实用户会问的问题
- 不要堆砌多个约束条件
- 必须能通过工具调用来回答
- **语言规范**：用自然语言描述，禁止使用箭头(→)、斜杠(/)、vs、竖线(|)等特殊符号来表达对比或路线关系，应改用"从...到..."、"和...相比"、"还是"等自然表述

### expected_tools 要求
- **只能包含你已经调用过并成功返回的工具**
- 参数必须基于工具返回的真实数据（如真实地名、真实坐标）
- direction 工具：origin 和 destination 必须是不同的地点
- weather工具：出的问题需要包含对天气相关的要求才包含这个工具
- 工具数量合理，与问题复杂度匹配

## rubrics 设计要求
- 每条 rubric 是一个可判断 true/false 的断言，用于评估做题者的回答质量
- **只聚焦用户在 query 中明确提出的需求**，不要添加用户没提到的隐含要求
- 覆盖关键维度（推荐的地点是否正确？信息是否真实？是否满足用户约束条件？）
- rubrics 数量 **3-8 条**（宁少勿多，每条都要有明确的用户需求依据）
- 不要出过于苛刻的 rubric，例如用户只说"推荐餐厅"，不要要求"必须推荐5家以上"

## 输出格式

完成所有工具调用后，严格按照以下 JSON 格式输出：

```json
{
  "query": "你生成的问题",
  "expected_tools": [
    {"name": "工具名", "arguments": {"参数名": "从工具结果中提取的真实值"}}
  ],
  "rubrics": [
    "rubric1: 具体的可验证断言",
    "rubric2: 另一个可验证断言"
  ]
}
```
"""

def generate_random_search_suggestion(rng=None) -> str:
    import random as _random
    rng = rng or _random

    month = rng.randint(1, 12)
    season = {1:"冬", 2:"冬", 3:"春", 4:"春", 5:"春", 6:"夏", 7:"夏", 8:"夏", 9:"秋", 10:"秋", 11:"秋", 12:"冬"}[month]

    month_themes = {
        1:  {"weather": "寒冷",  "nature": ["雪景", "腊梅"],     "vibe": ["冬日", "年味"]},
        2:  {"weather": "早春",  "nature": ["梅花", "油菜花"],   "vibe": ["春节", "元宵"]},
        3:  {"weather": "春暖",  "nature": ["桃花", "樱花"],     "vibe": ["踏青", "春游"]},
        4:  {"weather": "温暖",  "nature": ["樱花", "牡丹"],     "vibe": ["赏花", "春末"]},
        5:  {"weather": "初夏",  "nature": ["薰衣草", "荷花"],   "vibe": ["五一", "小长假"]},
        6:  {"weather": "炎热",  "nature": ["荷花", "高原草甸"], "vibe": ["避暑", "夏至"]},
        7:  {"weather": "盛夏",  "nature": ["向日葵", "草原"],   "vibe": ["暑假", "亲子"]},
        8:  {"weather": "酷暑",  "nature": ["星空", "草甸"],     "vibe": ["避暑", "夏末"]},
        9:  {"weather": "初秋",  "nature": ["秋色", "红叶"],     "vibe": ["中秋", "丰收"]},
        10: {"weather": "金秋",  "nature": ["红叶", "金杏"],     "vibe": ["国庆", "秋高气爽"]},
        11: {"weather": "深秋",  "nature": ["候鸟", "枫叶"],     "vibe": ["温泉", "深秋"]},
        12: {"weather": "寒冬",  "nature": ["雪景", "冬梅"],     "vibe": ["跨年", "冬日"]},
    }

    crowd_types = [
        "年轻人", "老年人", "亲子家庭", "情侣", "背包客",
        "摄影爱好者", "退休老人", "大学生", "独行侠", "闺蜜团",
    ]

    interests = [
        "小众冷门", "宝藏", "网红打卡", "人少", "性价比高",
        "风景绝美", "历史文化", "美食", "户外徒步", "休闲度假",
    ]

    t = month_themes[month]

    templates = [
        f"{month}月 适合{rng.choice(crowd_types)}旅游 国内城市推荐",
        f"{rng.choice(crowd_types)}最爱的 {month}月 国内{rng.choice(interests)}旅游地",
        f"{month}月 {rng.choice(crowd_types)} {t['weather']} 去哪玩比较好",
        f"{month}月 国内{rng.choice(interests)}城市 哪里值得去",
        f"{t['vibe'][0]}期间 {rng.choice(interests)}的国内旅游城市",
        f"{month}月 看{rng.choice(t['nature'])} 国内哪个城市最好",
        f"{t['weather']}的{month}月 {rng.choice(t['nature'])} 国内旅游推荐",
    ]

    return rng.choice(templates)

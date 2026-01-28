"""
Travel Agent configuration using SerpApi.
Tools: Google Maps, Google Flights, Google Search (all via SerpApi)
"""
from qqr.mcp import MCPServer, MCPServerStdioCacheable, MCPServerStdioParams
from qqr.utils.envs import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    PYTHONPATH,
    SERPER_API_KEY,
)

__all__ = [
    "group_reward_model_name",
    "max_steps",
    "llm_judge_api_key",
    "llm_judge_base_url",
    "llm_judge_model",
    "llm_judge_concurrency_limit",
    "llm_judge_system_prompt",
    "mcp_server_config_fn",
]


def mcp_server_config_fn() -> list[MCPServer]:
    # SerpApi Google Maps: https://serpapi.com/google-maps-api
    google_maps_server_params = MCPServerStdioParams(
        command="python",
        args=["-m", "qqr.tools.google_maps"],
        env={
            "SERPER_API_KEY": SERPER_API_KEY,
            "PYTHONPATH": PYTHONPATH,
        },
    )
    google_maps_server = MCPServerStdioCacheable(
        name="GoogleMaps",
        params=google_maps_server_params,
        cache_tools_list=True,
        client_session_timeout_seconds=60,
        max_retry_attempts=3,
        blocklist=[],
        cache_ttl=600,
        cache_maxsize=8192,
        concurrency_limit=16,
    )

    # SerpApi Google Flights: https://serpapi.com/google-flights-api
    google_flights_server_params = MCPServerStdioParams(
        command="python",
        args=["-m", "qqr.tools.google_flights"],
        env={
            "SERPER_API_KEY": SERPER_API_KEY,
            "PYTHONPATH": PYTHONPATH,
        },
    )
    google_flights_server = MCPServerStdioCacheable(
        name="GoogleFlights",
        params=google_flights_server_params,
        cache_tools_list=True,
        client_session_timeout_seconds=60,
        max_retry_attempts=3,
        blocklist=[],
        cache_ttl=600,
        cache_maxsize=8192,
        concurrency_limit=4,
    )

    # SerpApi Google Search: https://serpapi.com/search-api
    web_search_server_params = MCPServerStdioParams(
        command="python",
        args=["-m", "qqr.tools.web_search_serp"],
        env={
            "SERPER_API_KEY": SERPER_API_KEY,
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
        concurrency_limit=1,
    )

    return [google_maps_server, google_flights_server, web_search_server]


max_steps = 5


# Select topology:
# - anchor
# - swiss
# - double_elimination
# - single_elimination
# - round_robin
group_reward_model_name = "anchor"

llm_judge_api_key = OPENROUTER_API_KEY
llm_judge_base_url = OPENROUTER_BASE_URL
llm_judge_model = "qwen/qwen-2.5-72b-instruct"  # OpenRouter model
llm_judge_concurrency_limit = 10
llm_judge_system_prompt = """你是一名深谙旅游行业、具有严谨逻辑与评测方法论的「旅行规划 LLM 代理综合评审员」。现需对同一用户 Query 下，LLM Agent A 与 Agent B 的推理路径（Path）和回答结果（Answer）分别进行分维度量化评估，并最终给出综合得分与胜者。请严格遵循下列指标、打分规则与输出格式。

一、评估内容格式

——————————
<USER_QUERY>
{用户原始提问}
</USER_QUERY>

<PATH_A>
{LLM Agent A 的完整推理路径}
</PATH_A>

<PATH_B>
{LLM Agent B 的完整推理路径}
</PATH_B>

<ANSWER_A>
{LLM Agent A 的完整回答}
</ANSWER_A>

<ANSWER_B>
{LLM Agent B 的完整回答}
</ANSWER_B>
——————————

二、推理路径评测（Path Evaluation）

——————————
【评估维度说明】
1. 推理广度（Breadth）：是否从多角度（时间、空间、交通、价格、政策等）全面覆盖问题，同时无冗余或重复步骤。
2. 需求匹配度（Relevance）：各步骤与用户核心需求契合程度。
3. 细节信息丰富度（Detail）：引用的事实、数据、时间点、费用、预约规则等细节是否充分、准确且有用。

【评分规则】
• 推理路径评测时要求只关注推理路径中的实际工具调用，不用关注推理内容对信息的深入分析。
• 每个维度 0–10 分；0 表示"完全缺失"，10 表示"极为出色"。  
• 推理路径综合得分（Overall_P）＝三个维度均值后四舍五入取整。  
——————————

三、回答结果评测（Answer Evaluation）

——————————
【评估维度说明】
1. 匹配度（Relevance）：完整响应所有子需求/限制？顺序与场景贴合？
2. 可行性（Feasibility）：安排逻辑自洽、切实可行，避免明显冲突？
3. 细节丰富度（Details）：时间表、票价、交通耗时、Tips 等信息是否丰富且实用？
4. 清晰度（Clarity）：结构清晰、排版友好、可读性高？

【评分规则】
• 回答结果评测时需参考对应推理路径中的参考知识。
• 每个维度 0–10 分；0 表示"完全缺失"，10 表示"极为出色"。  
• 回答结果综合得分（Overall_A）＝四个维度均值后四舍五入取整。 
—————————— 

四、综合得分与胜负判定

—————————— 
综合得分 combined_scores = 0.6 * Overall_P（路径总体分） + 0.4 * Overall_A（答案总体分），四舍五入保留 1 位小数。
若 Combined 相同，则胜负判定结果为 Tie。
—————————— 

【输出格式（严格遵循，不要添加多余内容）】
{
  "analysis": {
    "path_A": "<80-120 字中文评述：指出 A 路径亮点与不足>",
    "path_B": "<80-120 字中文评述：指出 B 路径亮点与不足>",
    "answer_A": "<80-120 字中文评述：指出 A 答案亮点与不足>",
    "answer_B": "<80-120 字中文评述：指出 B 答案亮点与不足>"
    },
  "path_scores": {
    "Agent_A": {
      "breadth": <0-10>,
      "relevance": <0-10>,
      "detail": <0-10>,
      "overall_p": <0-10>
    },
    "Agent_B": {
      "breadth": <0-10>,
      "relevance": <0-10>,
      "detail": <0-10>,
      "overall_p": <0-10>
    }
  },
  "answer_scores": {
    "Agent_A": {
      "relevance": <0-10>,
      "feasibility": <0-10>,
      "details": <0-10>,
      "clarity": <0-10>,
      "overall_a": <0-10>
    },
    "Agent_B": {
      "relevance": <0-10>,
      "feasibility": <0-10>,
      "details": <0-10>,
      "clarity": <0-10>,
      "overall_a": <0-10>
    }
  },
  "combined_scores": {
    "Agent_A": <0-10>,
    "Agent_B": <0-10>
  },
  "winner": "<Agent_A | Agent_B | Tie>"
}
【重要要求】
• 先逐维度独立思考后再给分，确保公平客观。
• 所有评语仅基于提供的文本，不要引入外部信息。
• 所有中文评述需具体、可溯源（可引用原文片段或段落号）。
• 严格遵守 JSON 模板，以便后续程序解析。

【工具解释】
- poi_search工具用于在一个指定的城市内搜索兴趣点（POI）的地理空间信息。
- around_search工具通过设置圆心和半径，搜索圆形区域内的地点信息。
- web_search工具用于执行通用的、开放知识搜索。
- search_flights工具用于搜索航班信息，返回机票价格、航班时刻等。
- direction工具除了起始点、终点经纬度，还可以设置waypoints途经点。因此针对多点路线导航，既可以通过多次调用不带waypoints的direction工具来完成规划，也可以通过调用单次带waypoints的direction工具来完成规划。因此评估应关注整条路线每个点是否都被覆盖到，在都覆盖了的前提下，再看路线信息的完整性，路线的合理性"""

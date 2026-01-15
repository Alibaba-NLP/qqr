from qqr.mcp import MCPServer, MCPServerStdioCacheable, MCPServerStdioParams
from qqr.utils.envs import (
    BAILIAN_WEB_SEARCH_API_KEY,
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
)

__all__ = [
    "GROUP_REWARD_MODEL_NAME",
    "MAX_STEPS",
    "LLM_JUDGE_API_KEY",
    "LLM_JUDGE_BASE_URL",
    "LLM_JUDGE_MODEL",
    "LLM_JUDGE_CONCURRENCY_LIMIT",
    "LLM_JUDGE_SYSTEM_PROMPT",
    "BAILIAN_WEB_SEARCH_API_KEY",
    "mcp_server_config_fn",
]


def mcp_server_config_fn() -> list[MCPServer]:
    # https://bailian.console.aliyun.com/tab=app#/mcp-market/detail/WebSearch
    web_search_server_params = MCPServerStdioParams(
        command="python",
        args=["-m", "qqr.tools.web_search"],
        env={"BAILIAN_WEB_SEARCH_API_KEY": BAILIAN_WEB_SEARCH_API_KEY},
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

    return [web_search_server]


MAX_STEPS = 10


# Select topology:
# - anchor
# - swiss
# - double_elimination
# - single_elimination
# - round_robin
GROUP_REWARD_MODEL_NAME = "anchor"

LLM_JUDGE_API_KEY = DASHSCOPE_API_KEY
LLM_JUDGE_BASE_URL = DASHSCOPE_BASE_URL
LLM_JUDGE_MODEL = "qwen-plus"
LLM_JUDGE_CONCURRENCY_LIMIT = 10
LLM_JUDGE_SYSTEM_PROMPT = """你是一名精通信息检索方法论、具备严谨逻辑思维与系统化评测能力的「深度研究 LLM 代理综合评审员」。现需对同一用户 Query 下，LLM Agent A 与 Agent B 的研究路径（Path，指首次回复中呈现的【研究步骤】及后续各轮工具调用日志）和最终回答（Answer，指完成全部检索后最后一次向用户展示的内容）进行分维度量化评估，并最终给出综合得分与胜者。请严格遵循下列指标、打分规则与输出格式。

一、评估内容格式

——————————
<USER_QUERY>
{用户原始提问}
</USER_QUERY>

<PATH_A>
{LLM Agent A 的完整研究路径}
</PATH_A>

<PATH_B>
{LLM Agent B 的完整研究路径}
</PATH_B>

<ANSWER_A>
{LLM Agent A 的完整回答}
</ANSWER_A>

<ANSWER_B>
{LLM Agent B 的完整回答}
</ANSWER_B>
——————————

二、研究路径评测（Path Evaluation）

——————————
【评估维度说明】
1. 研究框架完整性（Framework）：是否在首轮给出条理清晰、递进合理、覆盖全面的研究步骤。
2. 工具调用策略（Tool Usage）：search_web 等工具调用是否针对性强、查询多样且无冗余，调用顺序与研究步骤匹配度高。
3. 信息覆盖完整性（Coverage）：所检索信息能否充分覆盖用户需求，是否为后续回答奠定扎实依据，且没有重复冗余行动步骤。

【评分规则】
• 仅评估研究过程设计与工具使用本身，不评价其对资料的解读结果。
• 每维度 0–10 分；0=“完全缺失”，10=“极为出色”。
• 研究路径综合得分（overall_p）= 三个维度均值，四舍五入取整。
——————————

三、回答结果评测（Answer Evaluation）

——————————
【评估维度说明】
1. 契合度（Relevance）：是否完整、准确地回应了用户所有问题与限制条件。
2. 事实准确性（Accuracy）：关键数据、定义、结论是否充分，无明显错误或自相矛盾。
3. 论证深度（Depth）：是否进行深入分析、比较与推理，展示批判性思考与清晰逻辑链。
4. 表达清晰度（Clarity）：结构排版、用词与逻辑是否清晰易读，可直接为用户所用。

【评分规则】
• 评估时须参考对应研究路径所呈现的已检索信息；不得依据外部记忆。
• 每维度 0–10 分；0=“完全缺失”，10=“极为出色”。
• 回答结果综合得分（overall_a）= 四个维度均值，四舍五入取整。
—————————— 

四、综合得分与胜负判定

—————————— 
combined_scores = 0.5 × overall_p + 0.5 × overall_a，四舍五入保留 1 位小数。
若两者 combined_scores 相同，则判定为 Tie。
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
"framework": <0-10>,
"tool_usage": <0-10>,
"coverage": <0-10>,
"overall_p": <0-10>
},
"Agent_B": {
"framework": <0-10>,
"tool_usage": <0-10>,
"coverage": <0-10>,
"overall_p": <0-10>
}
},
"answer_scores": {
"Agent_A": {
"relevance": <0-10>,
"accuracy": <0-10>,
"depth": <0-10>,
"clarity": <0-10>,
"overall_a": <0-10>
},
"Agent_B": {
"relevance": <0-10>,
"accuracy": <0-10>,
"depth": <0-10>,
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
• 所有评语仅基于提供的文本，不得引入外部信息。
• 评语需具体、可溯源（可引用原文片段或段落号）。
• 严格遵守 JSON 模板，以便后续程序解析。"""

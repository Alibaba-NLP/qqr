# config.py
from qqr.mcp import MCPServer, MCPServerStdioCacheable, MCPServerStdioParams
from qqr.utils.envs import (
    AMAP_MAPS_API_KEY,
    BAILIAN_WEB_SEARCH_API_KEY,
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    PYTHONPATH,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple



__all__ = [
    "group_reward_model_name",
    "max_steps",
    "mcp_server_config_fn",
    # point-wise reward
    "reward_mode",
    "navigation_reward_order_default",
    "navigation_reward_missing_pred_penalty",
    "navigation_reward_missing_gold_penalty",
    "navigation_reward_name",
]

def mcp_server_config_fn() -> list[MCPServer]:
    navigation_server_params = MCPServerStdioParams(
        command="python",
        args=["-m", "qqr.tools.navigation"],
        env={
            "AMAP_MAPS_API_KEY": AMAP_MAPS_API_KEY,
            "PYTHONPATH": PYTHONPATH,
        },
    )
    navigation_server = MCPServerStdioCacheable(
        name="navigation",
        params=navigation_server_params,
        cache_tools_list=True,
        client_session_timeout_seconds=60,
        max_retry_attempts=3,
        blocklist=[],
        cache_ttl=600,
        cache_maxsize=8192,
        concurrency_limit=16,
    )

    return [navigation_server]

    # amap_server_params = MCPServerStdioParams(
    #     command="python",
    #     args=["-m", "qqr.tools.amap"],
    #     env={
    #         "AMAP_MAPS_API_KEY": AMAP_MAPS_API_KEY,
    #         "PYTHONPATH": PYTHONPATH,
    #     },
    # )
    # amap_server = MCPServerStdioCacheable(
    #     name="AMap",
    #     params=amap_server_params,
    #     cache_tools_list=True,
    #     client_session_timeout_seconds=60,
    #     max_retry_attempts=3,
    #     blocklist=[],
    #     cache_ttl=600,
    #     cache_maxsize=8192,
    #     concurrency_limit=16,
    # )

    # transport_server_params = MCPServerStdioParams(
    #     command="python",
    #     args=["-m", "qqr.tools.mock_transport"],
    #     env={
    #         "DASHSCOPE_API_KEY": DASHSCOPE_API_KEY,
    #         "DASHSCOPE_BASE_URL": DASHSCOPE_BASE_URL,
    #         "PYTHONPATH": PYTHONPATH,
    #     },
    # )
    # transport_server = MCPServerStdioCacheable(
    #     name="Transport",
    #     params=transport_server_params,
    #     cache_tools_list=True,
    #     client_session_timeout_seconds=60,
    #     max_retry_attempts=3,
    #     blocklist=[],
    #     cache_ttl=600,
    #     cache_maxsize=8192,
    #     concurrency_limit=4,
    # )

    # web_search_server_params = MCPServerStdioParams(
    #     command="python",
    #     args=["-m", "qqr.tools.web_search"],
    #     env={
    #         "BAILIAN_WEB_SEARCH_API_KEY": BAILIAN_WEB_SEARCH_API_KEY,
    #         "PYTHONPATH": PYTHONPATH,
    #     },
    # )
    # web_search_server = MCPServerStdioCacheable(
    #     name="WebSearch",
    #     params=web_search_server_params,
    #     cache_tools_list=True,
    #     client_session_timeout_seconds=60,
    #     max_retry_attempts=3,
    #     blocklist=[],
    #     cache_ttl=600,
    #     cache_maxsize=8192,
    #     concurrency_limit=1,
    # )

    # return [amap_server, transport_server, web_search_server]


max_steps = 5

# 仍可保留
group_reward_model_name = "anchor"

# ---------------------------
# point-wise reward config
# ---------------------------
# reward_mode: "pointwise_navigation" | 未来可扩展其它模式
reward_mode = "pointwise_navigation"

# 你数据里 order 是自然语言字符串（如 "1.用户明确了起点..."）
# 若解析失败，用默认
navigation_reward_order_default = 1

# 没预测到 navigation 时的 reward（默认 0）
navigation_reward_missing_pred_penalty = 0.0

# reference 里没有 groundtruth 时的 reward（默认 0）
navigation_reward_missing_gold_penalty = 0.0

# 目标工具名
navigation_reward_name = "direction"

# 格式奖励权重
format_reward_weight = 0.3

# 导航相关配置
navigation_reward_name = "direction"
navigation_reward_order_default = "1"
navigation_reward_missing_pred_penalty = - 0.5
navigation_reward_missing_gold_penalty = - 0.0

# 奖励模式
reward_mode = "pointwise_navigation"

# -----------------------------
# Config
# -----------------------------
@dataclass
class InspectorConfig:
    enabled: bool = True

    # Use OpenAI-compatible judge/corrector LLM
    use_judge_llm: bool = False
    openai_base_url: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_new_tokens: int = 800
    timeout: int = 25

    # Logging
    log_dir: Optional[str] = None
    # log_each_turn: bool = True
    # max_history_chars: int = 12000

    # Correction knobs
    allow_gt_bootstrap: bool = True
    strict_direction_match: bool = False

    # Post-check knobs
    enable_postcheck: bool = True
    # If False, any correction that is not derivable from history will be rolled back.
    # If True, allow GT bootstrap to pass post-check (useful for raising "effective trajectories").
    postcheck_allow_gt_bootstrap: bool = True
    # If True and judge LLM enabled, run LLM postcheck; else only heuristic postcheck.
    postcheck_use_judge_llm: bool = True
    max_retries: int = 10
    retry_base_backoff: int = 10
    retry_max_backoff: int = 300

import asyncio

from agents.mcp import MCPServerSse, MCPServerSseParams
from mcp.server.fastmcp import FastMCP

from qqr.utils.envs import BAILIAN_WEB_SEARCH_API_KEY

mcp = FastMCP("WebSearch", log_level="WARNING")

"""
获取环境变量中的 API 密钥, 用于调用联网搜索 API
环境变量名为: BAILIAN_WEB_SEARCH_API_KEY, 在客户端侧通过配置环境变量进行设置传入
获取方式请参考: https://help.aliyun.com/zh/model-studio/get-api-key
API 文档: https://bailian.console.aliyun.com/tab=app#/mcp-market/detail/WebSearch
"""


server_params = MCPServerSseParams(
    url="https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/sse",
    headers={"Authorization": f"Bearer {BAILIAN_WEB_SEARCH_API_KEY}"},
)


@mcp.tool()
async def web_search(query: str | list[str]) -> str:
    """
    实时互联网信息检索。

    Args:
        query (`str | list[str]`):
            - 单个查询: 传入字符串，例如 "西湖十景"。
            - 批量查询: 传入字符串列表，例如 ["西湖十景", "杭州特色美食", "西湖周边酒店"]。
    """
    queries = [query] if isinstance(query, str) else query

    async with MCPServerSse(
        name="WebSearch",
        params=server_params,
        client_session_timeout_seconds=60,
        max_retry_attempts=3,
    ) as server:
        tasks = [server.call_tool("bailian_web_search", {"query": q}) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        results = [r.content[0].text for r in results]
        results = "\n\n---\n\n".join(results)
        return results

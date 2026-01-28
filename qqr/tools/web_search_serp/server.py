import asyncio
import os
import re

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WebSearch", log_level="WARNING")

"""
SerpApi Google Search API
Docs: https://serpapi.com/search-api
API Key: https://serpapi.com/manage-api-key
Environment variable: SERPER_API_KEY
"""

# Environment variables
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_URL = os.getenv("SERPER_URL", "https://serpapi.com/search")


# ========== Inline helpers (to avoid qqr package import issues) ==========

def _json2md(json_block, depth=1, htag="#"):
    """Convert JSON to Markdown."""
    def parseJSON(json_block, depth):
        if isinstance(json_block, dict):
            parseDict(json_block, depth)
        if isinstance(json_block, list):
            parseList(json_block, depth)

    def parseDict(d, depth):
        for k in d:
            if isinstance(d[k], (dict, list)):
                addHeader(k, depth)
                parseJSON(d[k], depth + 1)
            else:
                addValue(k, d[k])
        nonlocal markdown
        markdown += "\n"

    def parseList(l, depth):
        for i, value in enumerate(l):
            addHeader(str(i + 1), depth)
            if not isinstance(value, (dict, list)):
                index = l.index(value)
                addValue(index, value)
            else:
                parseDict(value, depth)
        nonlocal markdown
        markdown += "\n"

    def buildHeaderChain(depth, title):
        return "\n" + htag * (depth + 1) + f" {title}\n\n"

    def buildValueChain(key, value):
        return str(key) + f": {value}\n"

    def addHeader(value, depth):
        nonlocal markdown
        markdown += buildHeaderChain(depth, str(value).title())

    def addValue(key, value):
        nonlocal markdown
        markdown += buildValueChain(key, value)

    markdown = ""
    parseJSON(json_block, depth)
    return markdown.strip()


def _truncate_text(text, max_len=5000):
    """Truncate text to max length."""
    if len(text) <= max_len:
        return text
    head_len = max_len // 2
    tail_len = max_len // 2
    head_part = text[:head_len]
    head_matches = list(re.finditer(r"\s", head_part))
    head_end_index = head_matches[-1].start() if head_matches else head_len
    head = text[:head_end_index]
    tail_part = text[-tail_len:]
    tail_match = re.search(r"\s", tail_part)
    if tail_match:
        tail_start_index = len(text) - tail_len + tail_match.start()
        tail = text[tail_start_index:].lstrip()
    else:
        tail = tail_part
    truncated_chars = len(text) - len(head) - len(tail)
    ellipsis = f"\n\n... [truncated {truncated_chars} chars] ...\n\n"
    return head + ellipsis + tail


# ========== API Functions ==========

async def _search_single(client: httpx.AsyncClient, query: str) -> dict:
    """Execute a single search query."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPER_API_KEY,
        "output": "json",
    }
    response = await client.get(SERPER_URL, params=params)
    response.raise_for_status()
    return response.json()


def _format_results(result: dict) -> str:
    """Format search results into readable text."""
    output_parts = []

    # Organic results
    if organic := result.get("organic_results"):
        formatted = []
        for item in organic[:10]:  # Limit to top 10
            formatted.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            })
        output_parts.append(_json2md(formatted))

    # Knowledge graph
    if kg := result.get("knowledge_graph"):
        kg_info = {
            "title": kg.get("title"),
            "type": kg.get("type"),
            "description": kg.get("description"),
        }
        output_parts.append(f"**Knowledge Graph:**\n{_json2md(kg_info)}")

    # Answer box
    if answer := result.get("answer_box"):
        output_parts.append(f"**Answer:**\n{_json2md(answer)}")

    return "\n\n".join(output_parts) if output_parts else "No results found."


@mcp.tool()
async def web_search(query: str | list[str]) -> str:
    """
    实时互联网信息检索 (使用 Google 搜索)。

    Args:
        query (`str | list[str]`):
            - 单个查询: 传入字符串，例如 "西湖十景"。
            - 批量查询: 传入字符串列表，例如 ["西湖十景", "杭州特色美食", "西湖周边酒店"]。
    """
    queries = [query] if isinstance(query, str) else query

    async with httpx.AsyncClient(timeout=60) as client:
        tasks = [_search_single(client, q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    formatted_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            formatted_results.append(f"Query '{queries[i]}': Error - {result}")
        else:
            formatted_results.append(f"**Query: {queries[i]}**\n{_format_results(result)}")

    return _truncate_text("\n\n---\n\n".join(formatted_results))

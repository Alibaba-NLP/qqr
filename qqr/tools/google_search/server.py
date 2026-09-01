import asyncio
import hashlib
import hmac as _hmac
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.parse import urlencode

import httpx
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

from . import cache as _cache

logger = logging.getLogger(__name__)

mcp = FastMCP("WebSearch", log_level="WARNING")

# Google Search 网关配置（论文实现基于 Google Custom Search API；
# 通过 SEARCH_API_URL / SEARCH_API_KEY 指定兼容的搜索网关）
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "")
SEARCH_API_URL = os.getenv("SEARCH_API_URL", "")

# AIData 网页抓取配置 (Taobao TOP API)
TOP_APP_KEY = os.getenv("TOP_APP_KEY", "")
TOP_APP_SECRET = os.getenv("TOP_APP_SECRET", "")
AIDATA_APP_KEY = os.getenv("AIDATA_APP_KEY", "")
TOP_API_URL = "http://gw.api.taobao.com/router/rest"

# LLM 摘要配置
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-max")  # 论文：长网页由 Qwen3-Max 摘要

MAX_RETRIES = 3
RETRY_DELAYS = [2, 2, 2]

# token 长度阈值：短于此值的文档直接保留，不做 LLM 摘要
SHORT_DOC_TOKEN_THRESHOLD = 2500
# 每个 query 最多摘要的长文档数
MAX_LONG_DOCS_PER_QUERY = 3

_llm_client = None


def _get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    return _llm_client


def _gen_sign_hmac(params: dict, secret: str) -> str:
    """生成淘宝 TOP API HMAC-SHA256 签名"""
    sorted_params = sorted(params.items())
    buf = ""
    for key, value in sorted_params:
        if key and value:
            buf += f"{key}{value}"
    return _hmac.new(secret.encode(), buf.encode(), hashlib.sha256).hexdigest().upper()


# ────────────────────────── 搜索 ──────────────────────────


async def single_search(client: httpx.AsyncClient, query: str, num: int) -> dict:
    """通过 IdeaLab 执行单条 Google 搜索（带持久化缓存）"""
    # 查缓存
    cached = _cache.get_search(query, num)
    if cached is not None:
        return cached

    headers = {"X-AK": SEARCH_API_KEY}
    payload = {
        "query": query,
        "num": num,
        "extendParams": {"country": "cn", "page": 1},
        "platformInput": {"model": "google-search"},
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(SEARCH_API_URL, json=payload, headers=headers, timeout=30.0)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "application/json" not in content_type:
                raise ValueError(f"API returned non-JSON response (content-type: {content_type})")

            resp_json = resp.json()
            if not resp_json.get("data") or not resp_json["data"].get("originalOutput"):
                logger.warning(f"Query '{query}' returned no data")
                return {"search_query": query, "docs": []}

            organic = resp_json["data"]["originalOutput"].get("organic", [])
            docs = []
            for page in organic:
                docs.append({
                    "title": page.get("title", "Untitled"),
                    "text": page.get("snippet", ""),
                    "link": page.get("link", "#"),
                    "date": page.get("date", ""),
                    "source": page.get("source", ""),
                })
            result = {"search_query": query, "docs": docs}
            # 写缓存（只缓存有结果的）
            if docs:
                _cache.put_search(query, num, result)
            return result

        except Exception as e:
            logger.warning(f"Query '{query}' attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAYS[attempt])

    return {"search_query": query, "docs": []}


# ────────────────────────── 网页抓取 ──────────────────────────


async def fetch_page(client: httpx.AsyncClient, url: str) -> str:
    """通过 AIData 服务 (Taobao TOP API) 抓取网页正文（带持久化缓存）"""
    # 查缓存
    cached = _cache.get_page(url)
    if cached is not None:
        return cached

    dto = {
        "application_id": 2627,
        "inputs": {
            "urls": [{"channel": "Google", "tag": "", "url": url}],
            "cache": False,
        },
    }
    request_params = {
        "token": AIDATA_APP_KEY,
        "aignite_application_execute_req_dto": json.dumps(dto, ensure_ascii=False),
    }

    public_params = {
        "method": "alibaba.aidata.aignite.application.run",
        "app_key": TOP_APP_KEY,
        "format": "json",
        "v": "2.0",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sign_method": "hmac-sha256",
        "partner_id": "new_python3_sdk",
    }

    all_params = {**public_params, **request_params}
    public_params["sign"] = _gen_sign_hmac(all_params, TOP_APP_SECRET)
    full_url = TOP_API_URL + "?" + urlencode(public_params)

    try:
        resp = await client.post(full_url, data=request_params, timeout=15.0)
        resp.raise_for_status()
        resp_json = resp.json()

        if "error_response" in resp_json:
            logger.warning(f"fetch_page '{url}' API error: {resp_json['error_response']}")
            _cache.put_page(url, "")  # 缓存失败结果，避免重复请求
            return ""

        run_resp = resp_json.get("alibaba_aidata_aignite_application_run_response", {})
        data_str = run_resp.get("data", {}).get("outputs", "")
        if not data_str:
            _cache.put_page(url, "")
            return ""
        outputs = json.loads(data_str)
        items = outputs.get("data", {}).get("data", [])
        if not items:
            _cache.put_page(url, "")
            return ""

        item = items[0]
        title = item.get("title", "")
        content = item.get("content", "")
        if not content:
            _cache.put_page(url, "")
            return ""
        result = f"{title}\n\n{content}" if title else content
        _cache.put_page(url, result)
        return result

    except Exception as e:
        logger.warning(f"fetch_page '{url}' failed: {e}")
        # 超时/网络错误不缓存，下次可能成功

    return ""


# ────────────────────────── LLM 摘要 ──────────────────────────

_COMBINED_SUMMARY_PROMPT_ZH = """请根据用户查询，判断下列参考文章是否有参考价值。

- 如果文章与用户查询无关，请只返回"nope"，不要返回其他任何内容。
- 如果文章有参考价值，请提炼出有效信息/片段，尽量不超过1000字。返回结果需严格遵循以下格式（不要返回其他任何内容）:

【Evidence】
参考文章中针对用户查询的有效信息片段，尽量使用原文

【Summary】
参考文章摘要"""

_COMBINED_SUMMARY_PROMPT_EN = """Based on the user query, determine whether the following article is relevant and useful.

- If the article is NOT relevant to the user query, respond with only "nope" and nothing else.
- If the article IS relevant, extract useful information/evidence, keeping it under 1000 words. Strictly follow this format (do not return anything else):

【Evidence】
Relevant excerpts from the article, use original text as much as possible

【Summary】
Article summary"""


def _detect_lang(text: str) -> str:
    """检测文本语言：中文返回 'zh'，否则返回 'en'。"""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return "zh" if chinese_chars > len(text) * 0.1 else "en"


def _llm_call(system_prompt: str, user_prompt: str, max_tokens: int = 8192) -> str:
    """通用 LLM 调用，带重试"""
    client = _get_llm_client()
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                return ""
    return ""


def _llm_summarize_one(search_query: str, text: str) -> str:
    """对单篇文档一次性完成: 判断相关性 + 提取摘要（根据 query 语言选择 prompt）"""
    lang = _detect_lang(search_query)
    system_prompt = _COMBINED_SUMMARY_PROMPT_ZH if lang == "zh" else _COMBINED_SUMMARY_PROMPT_EN

    if lang == "zh":
        prompt = f"用户查询:\n{search_query}\n\n参考文章：\n{text}"
    else:
        prompt = f"User query:\n{search_query}\n\nReference article:\n{text}"

    result = _llm_call(system_prompt, prompt, max_tokens=8192)

    if result.strip().lower() == "nope":
        return ""
    return result


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text) // 2


async def summarize_docs(processed: list[dict]) -> list[dict]:
    """
    对搜索结果做 LLM 摘要 (优化版: 去掉 URL 预筛选，合并判断+摘要为一次调用):
      步骤1: token 长度分流 - ≤2500 tokens 的短文档直接保留，不摘要
      步骤2: LLM 摘要 - 对长文档 (>2500 tokens) 一次性判断相关性并提取 Evidence/Summary
             每个 query 最多摘要 3 篇长文档
    """
    loop = asyncio.get_event_loop()

    # ── 步骤1: token 长度分流 ──
    long_tasks = []
    long_task_refs = []

    for i, item in enumerate(processed):
        long_count = 0
        for j, doc in enumerate(item["docs"]):
            page_content = doc.get("page_content", "")
            if not page_content:
                continue

            token_len = _estimate_tokens(page_content)
            if token_len <= SHORT_DOC_TOKEN_THRESHOLD:
                # 短文档: 直接保留原文，不摘要
                doc["llm_summary"] = ""
                doc["_short"] = True
            else:
                # 长文档: 需要 LLM 摘要，每个 query 最多 3 篇
                if long_count < MAX_LONG_DOCS_PER_QUERY:
                    long_tasks.append((item["search_query"], page_content))
                    long_task_refs.append((i, j))
                    long_count += 1

    # ── 步骤2: 并发 LLM 摘要 (一次调用 = 判断 + 摘要) ──
    if long_tasks:
        with ThreadPoolExecutor(max_workers=min(len(long_tasks), 10)) as executor:
            futures = [
                loop.run_in_executor(executor, _llm_summarize_one, sq, text)
                for sq, text in long_tasks
            ]
            summaries = await asyncio.gather(*futures, return_exceptions=True)

        for (i, j), summary in zip(long_task_refs, summaries):
            if isinstance(summary, str) and summary:
                processed[i]["docs"][j]["llm_summary"] = summary
            else:
                processed[i]["docs"][j]["llm_summary"] = ""

    return processed


def _format_summary(llm_summary: str, search_query: str, link: str) -> str:
    """解析 LLM 摘要中的 Evidence + Summary 结构（中英文自适应）"""
    lang = _detect_lang(search_query)
    evidence_pos = llm_summary.find("【Evidence】")
    summary_pos = llm_summary.find("【Summary】")

    if lang == "zh":
        header = f"网页 {link} 中有关查询 {search_query} 的有效信息如下:\n\n"
        evidence_label = "有效片段"
        summary_label = "网页摘要"
    else:
        header = f"Relevant information from {link} for query '{search_query}':\n\n"
        evidence_label = "Evidence"
        summary_label = "Summary"

    if evidence_pos != -1 and summary_pos != -1:
        if evidence_pos < summary_pos:
            evidence = llm_summary.split("【Evidence】")[-1].split("【Summary】")[0].strip()
            summary = llm_summary.split("【Summary】")[-1].strip()
        else:
            evidence = llm_summary.split("【Evidence】")[-1].strip()
            summary = llm_summary.split("【Summary】")[-1].split("【Evidence】")[0].strip()
        if evidence and summary:
            return header + f"{evidence_label}:\n{evidence}\n\n{summary_label}:\n{summary}"

    return header + llm_summary


# ────────────────────────── MCP 工具 ──────────────────────────

# 全局开关：环境变量 DISABLE_FETCH_CONTENT=1 时强制关闭网页抓取（只用搜索 snippet）
_DISABLE_FETCH_CONTENT = os.getenv("DISABLE_FETCH_CONTENT", "0") == "1"


@mcp.tool()
async def web_search(
    query: str,
) -> str:
    """
    实时互联网信息检索。搜索后自动抓取网页正文并用 LLM 生成摘要。

    Args:
        query: 搜索关键词，例如 "2024年中国GDP增长率"。
    """
    num = 3
    fetch_content = True
    summarize = True
    if not SEARCH_API_KEY:
        raise ValueError("SEARCH_API_KEY environment variable is not set")

    queries = [query]

    async with httpx.AsyncClient() as client:
        # 1. 搜索
        search_tasks = [single_search(client, q, num) for q in queries]
        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Query '{queries[i]}' failed: {result}")
                processed.append({"search_query": queries[i], "docs": []})
            else:
                processed.append(result)

        # 2. 并发抓取所有网页正文
        if _DISABLE_FETCH_CONTENT:
            fetch_content = False
        if fetch_content:
            all_fetch_tasks = []
            all_docs = []
            for item in processed:
                for doc in item["docs"]:
                    link = doc.get("link", "")
                    if link.startswith("http"):
                        all_fetch_tasks.append(fetch_page(client, link))
                        all_docs.append(doc)

            if all_fetch_tasks:
                contents = await asyncio.gather(*all_fetch_tasks, return_exceptions=True)
                for doc, content in zip(all_docs, contents):
                    if isinstance(content, str) and content:
                        doc["page_content"] = content[:128000]
                    else:
                        doc["page_content"] = ""

    # 3. LLM 摘要 (短文档保留原文，长文档一次调用完成判断+摘要)
    if fetch_content and summarize:
        processed = await summarize_docs(processed)

    # 4. 格式化输出（中英文自适应）
    outputs = []
    for item in processed:
        q = item["search_query"]
        docs = item["docs"]
        lang = _detect_lang(q)

        if not docs:
            outputs.append(
                f"搜索 '{q}' 无结果" if lang == "zh"
                else f"Search '{q}' returned no results"
            )
            continue

        snippets = []
        for idx, doc in enumerate(docs, 1):
            if lang == "zh":
                date_str = f"\n发布时间: {doc['date']}" if doc.get("date") else ""
                source_str = f"\n来源: {doc['source']}" if doc.get("source") else ""
            else:
                date_str = f"\nPublished: {doc['date']}" if doc.get("date") else ""
                source_str = f"\nSource: {doc['source']}" if doc.get("source") else ""

            # 优先级: LLM 摘要 > 短文档原文 > snippet
            llm_summary = doc.get("llm_summary", "")
            if llm_summary:
                body = _format_summary(llm_summary, q, doc["link"])
            elif doc.get("_short") and doc.get("page_content"):
                body = doc["page_content"][:6000]
            elif doc.get("page_content"):
                body = doc["page_content"][:6000]
            else:
                body = doc.get("text", "")

            snippet = (
                f"{idx}. [{doc['title']}]({doc['link']})"
                f"{date_str}{source_str}\n\n{body}"
            )
            snippets.append(snippet)

        if lang == "zh":
            header = f"搜索 '{q}'，得到 {len(docs)} 条结果:\n\n"
        else:
            header = f"Search '{q}', {len(docs)} results found:\n\n"

        content = header + "\n\n---\n\n".join(snippets)
        outputs.append(content)

    return "\n\n=======\n\n".join(outputs)

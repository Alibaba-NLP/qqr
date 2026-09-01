"""
搜索结果持久化缓存

支持两级缓存：
1. 搜索结果缓存：query → search results (宽松模糊匹配)
2. 网页内容缓存：url → page content (精确匹配)

缓存文件存储在 SEARCH_CACHE_DIR 目录下，进程重启后缓存不丢。

搜索缓存匹配策略（宽松）：
- NFKC 归一化 + 小写 + 去标点
- 去停用词（中英文常见虚词）
- token 排序（词序无关）
- 不区分 num（num=3 和 num=5 命中同一条）
"""

import hashlib
import json
import logging
import os
import re
import threading
import unicodedata

logger = logging.getLogger(__name__)

SEARCH_CACHE_DIR = os.getenv("SEARCH_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "qqr_search"))
SEARCH_CACHE_FILE = os.path.join(SEARCH_CACHE_DIR, "search_cache.jsonl")
PAGE_CACHE_FILE = os.path.join(SEARCH_CACHE_DIR, "page_cache.jsonl")

# 内存缓存
_search_cache_exact: dict[str, dict] = {}  # 精确 key (小写+去空格) -> search results
_search_cache_fuzzy: dict[str, dict] = {}  # 模糊 key (归一化+排序) -> search results
_page_cache: dict[str, str] = {}           # url -> page_content
_lock = threading.Lock()
_loaded = False

# 停用词（匹配时忽略）
_STOPWORDS_ZH = set("的了在是我我们你你们他她它们这那有和与及其不也都要会能到从被把让给对于用以而且但是如果因为所以虽然可以应该需要")
_STOPWORDS_EN = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "on",
    "at", "to", "for", "with", "by", "from", "as", "into", "about",
    "and", "or", "but", "not", "no", "if", "that", "this", "it",
    "what", "how", "which", "who", "when", "where", "why",
}
_ALL_STOPWORDS = _STOPWORDS_ZH | _STOPWORDS_EN


def _normalize_query(query: str) -> str:
    """
    宽松归一化 query：
    1. NFKC 编码统一
    2. 小写
    3. 去标点
    4. 分词（中文按字，英文按空格）
    5. 去停用词
    6. token 排序（词序无关）
    """
    q = unicodedata.normalize("NFKC", query)
    q = q.lower().strip()
    # 去标点，保留字母数字和中文
    q = re.sub(r'[^\w\s]', ' ', q)
    q = re.sub(r'\s+', ' ', q).strip()

    # 分词：英文按空格，中文按字
    tokens = []
    for part in q.split():
        # 纯英文/数字 token
        if all(c.isascii() for c in part):
            if part not in _ALL_STOPWORDS:
                tokens.append(part)
        else:
            # 含中文：逐字拆开，过滤停用词
            for char in part:
                if char.strip() and char not in _ALL_STOPWORDS:
                    tokens.append(char)

    # 排序 → 词序无关
    tokens.sort()
    return " ".join(tokens)


def _query_key_exact(query: str) -> str:
    """精确匹配 key：小写 + 去首尾空格，保留原始词序和标点。"""
    return query.strip().lower()


def _query_key_fuzzy(query: str) -> str:
    """模糊匹配 key：归一化 + 去停用词 + 排序，不含 num。"""
    normalized = _normalize_query(query)
    if len(normalized) > 512:
        return hashlib.md5(normalized.encode()).hexdigest()
    return normalized


def _url_key(url: str) -> str:
    """URL 过长时用 md5 做 key。"""
    if len(url) > 512:
        return hashlib.md5(url.encode()).hexdigest()
    return url


def _ensure_loaded():
    """懒加载：首次访问时从文件加载缓存。"""
    global _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        os.makedirs(SEARCH_CACHE_DIR, exist_ok=True)
        # 加载搜索缓存（同时构建精确和模糊两级索引）
        if os.path.exists(SEARCH_CACHE_FILE):
            try:
                with open(SEARCH_CACHE_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        old_key = record["key"]
                        value = record["value"]
                        docs = value.get("docs", []) if isinstance(value, dict) else []

                        # 精确索引：用原始 key（去 ||num= 后缀）
                        if "||num=" in old_key:
                            exact_key = old_key.split("||num=")[0]
                        else:
                            exact_key = old_key
                        if exact_key not in _search_cache_exact or len(docs) > len(_search_cache_exact[exact_key].get("docs", [])):
                            _search_cache_exact[exact_key] = value

                        # 模糊索引：归一化 + 排序
                        fuzzy_key = _query_key_fuzzy(exact_key)
                        if fuzzy_key not in _search_cache_fuzzy or len(docs) > len(_search_cache_fuzzy[fuzzy_key].get("docs", [])):
                            _search_cache_fuzzy[fuzzy_key] = value

                logger.info(f"搜索缓存加载完成: 精确 {len(_search_cache_exact)} 条, 模糊 {len(_search_cache_fuzzy)} 条")
            except Exception as e:
                logger.warning(f"搜索缓存加载失败: {e}")
        # 加载网页缓存
        if os.path.exists(PAGE_CACHE_FILE):
            try:
                with open(PAGE_CACHE_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        _page_cache[record["key"]] = record["value"]
                logger.info(f"网页缓存加载完成: {len(_page_cache)} 条")
            except Exception as e:
                logger.warning(f"网页缓存加载失败: {e}")
        _loaded = True


# ============ 搜索缓存 ============

def get_search(query: str, num: int = 0) -> dict | None:
    """
    两级查询搜索缓存（num 不参与匹配）：
    1. 精确匹配：query 小写后完全一致 → 优先返回
    2. 模糊匹配：归一化+去停用词+排序后一致 → 兜底返回
    """
    _ensure_loaded()
    # 第一级：精确匹配
    exact_key = _query_key_exact(query)
    result = _search_cache_exact.get(exact_key)
    if result is not None:
        logger.info(f"搜索缓存精确命中: '{query[:50]}'")
        return result
    # 第二级：模糊匹配
    fuzzy_key = _query_key_fuzzy(query)
    result = _search_cache_fuzzy.get(fuzzy_key)
    if result is not None:
        logger.info(f"搜索缓存模糊命中: '{query[:50]}'")
        return result
    return None


def put_search(query: str, num: int = 0, results: dict = None):
    """写入搜索缓存（同时写入精确和模糊两级索引）。"""
    _ensure_loaded()
    exact_key = _query_key_exact(query)
    fuzzy_key = _query_key_fuzzy(query)
    docs = (results or {}).get("docs", [])
    with _lock:
        # 精确索引：同 key 保留 docs 更多的
        if exact_key not in _search_cache_exact or len(docs) > len(_search_cache_exact[exact_key].get("docs", [])):
            _search_cache_exact[exact_key] = results
        # 模糊索引：同 key 保留 docs 更多的
        if fuzzy_key not in _search_cache_fuzzy or len(docs) > len(_search_cache_fuzzy[fuzzy_key].get("docs", [])):
            _search_cache_fuzzy[fuzzy_key] = results
        try:
            os.makedirs(SEARCH_CACHE_DIR, exist_ok=True)
            with open(SEARCH_CACHE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": exact_key, "value": results}, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"搜索缓存写入失败: {e}")


# ============ 网页缓存 ============

def get_page(url: str) -> str | None:
    """查询网页缓存。返回 None 表示未命中。"""
    _ensure_loaded()
    key = _url_key(url)
    result = _page_cache.get(key)
    if result is not None:
        logger.info(f"网页缓存命中: '{url[:80]}'")
    return result


def put_page(url: str, content: str):
    """写入网页缓存（空内容也缓存，避免重复抓取失败的页面）。"""
    _ensure_loaded()
    key = _url_key(url)
    with _lock:
        if key in _page_cache:
            return
        _page_cache[key] = content
        try:
            os.makedirs(SEARCH_CACHE_DIR, exist_ok=True)
            with open(PAGE_CACHE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": content}, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"网页缓存写入失败: {e}")


# ============ 统计 ============

def cache_stats() -> dict:
    """返回缓存统计信息。"""
    _ensure_loaded()
    return {
        "search_exact_entries": len(_search_cache_exact),
        "search_fuzzy_entries": len(_search_cache_fuzzy),
        "page_entries": len(_page_cache),
    }

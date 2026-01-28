import os
import re

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("GoogleMaps", log_level="WARNING")

"""
SerpApi Google Maps API
Docs:
  - Places: https://serpapi.com/google-maps-api
  - Directions: https://serpapi.com/google-maps-directions-api
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


def _is_coordinates(location: str) -> bool:
    """Check if location is in coordinate format."""
    return "," in location and location.replace(",", "").replace(".", "").replace("-", "").replace(" ", "").isdigit()


# ========== MCP Tools ==========

@mcp.tool()
async def poi_search(query: str, location: str | None = None) -> str:
    """
    通过文本搜索地点信息 (POI)。可以搜索餐厅、酒店、景点等。
    返回多个可能相关的 POI 信息，包括：
        - 名称和地址
        - 评分和评论数
        - GPS 坐标
        - 联系方式

    Args:
        query (`str`): 搜索关键词，如 "咖啡店"、"酒店"、"西湖景点"。
        location (`Optional[str]`): 搜索位置，如 "杭州" 或 "北京市海淀区"。
            如果不提供，将在全球范围搜索。
    """
    params = {
        "engine": "google_maps",
        "q": query,
        "type": "search",
        "api_key": SERPER_API_KEY,
    }

    if location:
        params["q"] = f"{query} {location}"

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(SERPER_URL, params=params)
        response.raise_for_status()
        result = response.json()

    local_results = result.get("local_results", [])
    if not local_results:
        raise Exception("No POI data available.")

    formatted = []
    for item in local_results[:10]:  # Limit to top 10
        poi = {
            "name": item.get("title"),
            "address": item.get("address"),
            "rating": item.get("rating"),
            "reviews": item.get("reviews"),
            "type": item.get("type"),
            "phone": item.get("phone"),
            "website": item.get("website"),
        }
        if coords := item.get("gps_coordinates"):
            poi["location"] = f"{coords.get('longitude')},{coords.get('latitude')}"
        formatted.append(poi)

    return _truncate_text(_json2md(formatted))


@mcp.tool()
async def around_search(
    location: str,
    keyword: str | None = None,
    radius: int = 5000,
) -> str:
    """
    通过设置中心点坐标，搜索周边的地点信息。
    返回多个可能相关的 POI 信息，包括名称、地址、评分、坐标等。

    Args:
        location (`str`): 中心点坐标或地址。
            - 坐标格式: "经度,纬度"，如 "120.15,30.28"
            - 地址格式: "杭州西湖"
        keyword (`Optional[str]`): 搜索关键词，如 "银行"、"餐厅"。
        radius (`int`): 搜索半径（米），默认 5000 米。此参数仅作参考，实际结果由 Google 决定。
    """
    # Build search query
    search_query = keyword if keyword else "places"

    params = {
        "engine": "google_maps",
        "q": search_query,
        "type": "search",
        "api_key": SERPER_API_KEY,
    }

    # Check if location is coordinates (contains comma and numbers)
    if _is_coordinates(location):
        # Format: longitude,latitude -> need to convert to latitude,longitude for Google
        parts = location.split(",")
        if len(parts) == 2:
            lon, lat = parts[0].strip(), parts[1].strip()
            params["ll"] = f"@{lat},{lon},15z"  # zoom level 15
    else:
        # It's an address, add to query
        params["q"] = f"{search_query} near {location}"

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(SERPER_URL, params=params)
        response.raise_for_status()
        result = response.json()

    local_results = result.get("local_results", [])
    if not local_results:
        raise Exception("No POI data available.")

    formatted = []
    for item in local_results[:10]:
        poi = {
            "name": item.get("title"),
            "address": item.get("address"),
            "rating": item.get("rating"),
            "reviews": item.get("reviews"),
            "type": item.get("type"),
            "phone": item.get("phone"),
        }
        if coords := item.get("gps_coordinates"):
            poi["location"] = f"{coords.get('longitude')},{coords.get('latitude')}"
        formatted.append(poi)

    return _truncate_text(_json2md(formatted))


@mcp.tool()
async def direction(
    origin: str,
    destination: str,
    mode: str = "driving",
) -> str:
    """
    提供路线规划服务。支持驾车、步行、骑行、公交路线规划。

    Args:
        origin: 起点。可以是地址（如 "北京首都国际机场"）或坐标（如 "116.4,39.9"）。
        destination: 终点。可以是地址或坐标。
        mode: 路线规划类型，默认为驾车。
            - Enum: ["driving", "walking", "bicycling", "transit"]。
            注意: Google Maps 会自动提供多种交通方式的选项。
    """
    params = {
        "engine": "google_maps_directions",
        "api_key": SERPER_API_KEY,
    }

    # Handle origin
    if _is_coordinates(origin):
        params["start_coords"] = origin
    else:
        params["start_addr"] = origin

    # Handle destination
    if _is_coordinates(destination):
        params["end_coords"] = destination
    else:
        params["end_addr"] = destination

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(SERPER_URL, params=params)
        response.raise_for_status()
        result = response.json()

    # Check for errors
    if error := result.get("error"):
        raise Exception(f"API error: {error}")

    directions = result.get("directions", [])
    if not directions:
        raise Exception("No route available.")

    # Format the response
    output = {
        "routes": [],
    }

    # Handle places_info (can be list or dict)
    places_info = result.get("places_info")
    if places_info:
        if isinstance(places_info, list) and len(places_info) >= 2:
            output["start"] = places_info[0].get("address")
            output["end"] = places_info[1].get("address")
        elif isinstance(places_info, dict):
            output["start"] = places_info.get("start_info", {}).get("address")
            output["end"] = places_info.get("end_info", {}).get("address")

    for route in directions:
        route_info = {
            "via": route.get("via"),
            "distance": route.get("distance"),
            "duration": route.get("duration"),
            "travel_mode": route.get("travel_mode"),
        }
        if steps := route.get("directions"):
            route_info["steps"] = [
                {
                    "instruction": step.get("instruction"),
                    "distance": step.get("distance"),
                    "duration": step.get("duration"),
                }
                for step in steps[:15]  # Limit steps
            ]
        output["routes"].append(route_info)

    # Also include duration summary if available
    if durations := result.get("durations"):
        output["duration_options"] = durations

    return _truncate_text(_json2md(output))

import os
import re

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("GoogleFlights", log_level="WARNING")

"""
SerpApi Google Flights API
Docs: https://serpapi.com/google-flights-api
API Key: https://serpapi.com/manage-api-key
Environment variable: SERPER_API_KEY
"""

# Environment variables
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_URL = os.getenv("SERPER_URL", "https://serpapi.com/search")


# ========== Inline helpers ==========

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


# ========== City to Airport Code Mapping (Common Chinese Cities) ==========

CITY_AIRPORT_CODES = {
    # China major cities
    "北京": "PEK",
    "上海": "PVG",
    "广州": "CAN",
    "深圳": "SZX",
    "成都": "CTU",
    "杭州": "HGH",
    "武汉": "WUH",
    "西安": "XIY",
    "重庆": "CKG",
    "南京": "NKG",
    "天津": "TSN",
    "青岛": "TAO",
    "大连": "DLC",
    "厦门": "XMN",
    "昆明": "KMG",
    "长沙": "CSX",
    "郑州": "CGO",
    "沈阳": "SHE",
    "哈尔滨": "HRB",
    "济南": "TNA",
    "福州": "FOC",
    "合肥": "HFE",
    "南昌": "KHN",
    "贵阳": "KWE",
    "南宁": "NNG",
    "海口": "HAK",
    "三亚": "SYX",
    "兰州": "LHW",
    "银川": "INC",
    "西宁": "XNN",
    "乌鲁木齐": "URC",
    "呼和浩特": "HET",
    "拉萨": "LXA",
    "石家庄": "SJW",
    "太原": "TYN",
    "长春": "CGQ",
    "烟台": "YNT",
    "温州": "WNZ",
    "宁波": "NGB",
    "无锡": "WUX",
    "珠海": "ZUH",
    # 国际城市
    "东京": "NRT",
    "大阪": "KIX",
    "首尔": "ICN",
    "香港": "HKG",
    "台北": "TPE",
    "新加坡": "SIN",
    "曼谷": "BKK",
    "吉隆坡": "KUL",
    "悉尼": "SYD",
    "墨尔本": "MEL",
    "纽约": "JFK",
    "洛杉矶": "LAX",
    "旧金山": "SFO",
    "伦敦": "LHR",
    "巴黎": "CDG",
    "法兰克福": "FRA",
    "迪拜": "DXB",
}


def _get_airport_code(city: str) -> str:
    """Get airport code from city name."""
    # Remove "市" suffix
    city = city.replace("市", "").strip()

    # Direct match
    if city in CITY_AIRPORT_CODES:
        return CITY_AIRPORT_CODES[city]
    
    # 尝试部分匹配
    for key, code in CITY_AIRPORT_CODES.items():
        if key in city or city in key:
            return code
    
    # If already airport code format (3 uppercase letters), return directly
    if len(city) == 3 and city.isupper():
        return city
    
    # Cannot recognize, return original value (let API handle it)
    return city


# ========== MCP Tools ==========

@mcp.tool()
async def search_flights(
    date: str,
    from_city: str,
    to_city: str,
    adults: int = 1,
) -> str:
    """
    搜索航班信息。返回指定日期从出发城市到目的城市的航班列表。
    
    Args:
        date (`str`): 出发日期，格式为 YYYY-MM-DD，如 "2026-02-15"。
        from_city (`str`): 出发城市中文名或机场代码，如 "北京" 或 "PEK"。
        to_city (`str`): 到达城市中文名或机场代码，如 "上海" 或 "PVG"。
        adults (`int`): 成人乘客数量，默认为 1。
    
    Returns:
        航班信息列表，包含航班号、价格、起飞时间、到达时间、飞行时长等。
    """
    departure_id = _get_airport_code(from_city)
    arrival_id = _get_airport_code(to_city)
    
    params = {
        "engine": "google_flights",
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": date,
        "type": "2",  # One way
        "adults": adults,
        "currency": "CNY",
        "hl": "zh-CN",
        "api_key": SERPAPI_API_KEY,
    }
    
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(SERPER_URL, params=params)
        response.raise_for_status()
        result = response.json()
    
    # Check for errors
    if error := result.get("error"):
        raise Exception(f"API error: {error}")
    
    # Combine best_flights and other_flights
    all_flights = []
    all_flights.extend(result.get("best_flights", []))
    all_flights.extend(result.get("other_flights", []))
    
    if not all_flights:
        return f"未找到 {date} 从 {from_city} 到 {to_city} 的航班。"
    
    # Format results
    formatted_flights = []
    for flight_option in all_flights[:10]:  # Limit to 10 options
        flights = flight_option.get("flights", [])
        if not flights:
            continue
        
        # Get first and last flight for departure/arrival info
        first_flight = flights[0]
        last_flight = flights[-1]
        
        dep_airport = first_flight.get("departure_airport", {})
        arr_airport = last_flight.get("arrival_airport", {})
        
        # Build flight info string
        flight_numbers = " → ".join([f.get("flight_number", "N/A") for f in flights])
        airlines = " / ".join(set([f.get("airline", "N/A") for f in flights]))
        
        flight_info = {
            "航班": flight_numbers,
            "航空公司": airlines,
            "价格": f"{flight_option.get('price', 'N/A')} CNY",
            "出发": f"{dep_airport.get('time', 'N/A')} {dep_airport.get('name', '')}",
            "到达": f"{arr_airport.get('time', 'N/A')} {arr_airport.get('name', '')}",
            "总时长": f"{flight_option.get('total_duration', 0)} 分钟",
        }
        
        # Add layover info if applicable
        layovers = flight_option.get("layovers", [])
        if layovers:
            layover_info = [f"{l.get('name', '')} ({l.get('duration', 0)}分钟)" for l in layovers]
            flight_info["中转"] = " → ".join(layover_info)
        else:
            flight_info["中转"] = "直飞"
        
        formatted_flights.append(flight_info)
    
    return _truncate_text(_json2md(formatted_flights))

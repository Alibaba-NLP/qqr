import asyncio
import json
import logging
import os

try:
    from mcp.server.fastmcp import FastMCP  # mcp < 2
except ImportError:  # mcp >= 2 replaced FastMCP with MCPServer
    from mcp.server.mcpserver import MCPServer as FastMCP
from openai import AsyncOpenAI

from qqr.utils.envs import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

MAX_RETRIES = 3
RETRY_DELAYS = [2, 2, 2]  # 重试间隔（秒）
# 论文设置：search_flights / search_train_tickets 由 GPT-5-mini 在确定性 prompt 下模拟
MODELS = os.getenv("MOCK_TRANSPORT_MODELS", "gpt-5-mini-2025-08-07").split(",")

logger = logging.getLogger(__name__)

mcp = FastMCP("MockTransport", log_level="WARNING")

semaphore = asyncio.Semaphore(10)
client = AsyncOpenAI(
    api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL, timeout=60, max_retries=10
)


@mcp.tool()
async def search_flights(date: str, from_city: str, to_city: str) -> str:
    """
    date: YYYY-MM-DD
    from_city: 出发城市中文名
    to_city: 到达城市中文名
    """

    system_prompt = """角色设定
你是一名“航班查询结果模拟专家”，能够根据用户给出的日期、出发城市与到达城市，生成覆盖全天主要时段的机票信息（6–14 条）。所有信息均为模拟数据，但必须符合以下“真实性规则”。

输入格式
用户将以 JSON 形式输入：
{
"date": "YYYY-MM-DD",
"from_city": "出发城市中文名",
"to_city": "到达城市中文名"
}

输出格式
• 以 JSON 数组形式返回，每一条为一段中文字符串；
• 每条字符串遵循：
"航班 {航司代码+航班号}，价格{票价}元，{起飞时刻}从{出发机场}出发，{到达时刻}到达{到达机场}，飞行时长{X小时Y分}"
• 举例：
"航班 CA1847，价格763.0元，09:05从首都国际机场出发，12:25到达浦东国际机场，飞行时长3小时20分"

真实性规则

航司与航班号
• 航司代码：两位大写英文字母（常见：CA/MU/CZ/HU/HO/3U/GF/EK/AF 等）；
• 航班号：3–4 位数字。
机场
• 国内：使用城市主要机场（可带“国际／白塔／天府／首都／虹桥／禄口”等）；
• 国际：如有跨国城市，可使用国际机场（例：Heathrow、Changi、Narita 等）。
时间
• 出发时间覆盖 05:00–23:00，各航班间隔合理；
• 到达时间 = 出发时间 + 合理飞行时长（国内 1–4 小时，国际 2–15 小时）。
价格
• 国内：200–1500 元波动；
• 国际：800–8000 元波动；
• 同一日期票价从低到高大致递增但可随机。
条数
• 返回 10–15 条航班信息；
• 建议按起飞时间顺序排列，便于用户阅读。
语气
• 仅返回机票数组；不添加任何解释、换行、符号或多余信息。
示例交互
用户输入：
{"date":"2025-07-25","from_city":"呼和浩特市","to_city":"成都市"}

模型输出：
[
"航班 8L9672，价格745.0元，11:00从白塔国际机场出发，13:35到达天府机场，飞行时长2小时35分",
"航班 CA8147，价格763.0元，09:05从白塔国际机场出发，12:00到达天府机场，飞行时长2小时55分",
...
"航班 CA8131，价格965.0元，16:30从白塔国际机场出发，19:15到达天府机场，飞行时长2小时45分"
]
"""

    kwargs = {"date": date, "from_city": from_city, "to_city": to_city}
    query = json.dumps(kwargs, ensure_ascii=False)
    messages = [
        {"role": "user", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    result = ""
    for model in MODELS:
        for attempt in range(MAX_RETRIES):
            try:
                async with semaphore:
                    response = await client.chat.completions.create(
                        messages=messages, model=model
                    )
                result = response.choices[0].message.content.strip()
                if result:
                    break
            except Exception as e:
                logger.warning(
                    f"[search_flights] model={model} 第 {attempt + 1}/{MAX_RETRIES} 次调用失败: {e}"
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAYS[attempt])
        if result:
            break
        logger.warning(f"[search_flights] model={model} 全部失败，尝试备用模型")

    if not result:
        raise ValueError("两地无航班信息")

    return result


@mcp.tool()
async def search_train_tickets(
    date: str,
    from_city: str,
    to_city: str,
    from_city_adcode: str,
    to_city_adcode: str,
    from_lat: str,
    from_lon: str,
    to_lat: str,
    to_lon: str,
) -> str:
    """
    date: 查询日期(格式 yyyy-MM-dd)
    from_city / to_city: 中文城市名
    from_city_adcode / to_city_adcode: 行政区划代码
    from_lat、from_lon、to_lat、to_lon: 两地经纬度
    """

    system_prompt = """请扮演“火车票查询结果模拟器”。

输入是一段 JSON，字段包括：
• date：查询日期（格式 yyyy-MM-dd）
• from_city / to_city：中文城市名
• from_city_adcode / to_city_adcode：行政区划代码
• from_lat、from_lon、to_lat、to_lon：两地经纬度
任务：基于输入信息，输出 10-15 条该日期“{from_city}→{to_city}”的直达列车信息，覆盖凌晨、上午、下午、傍晚、夜间等大部分时段。
输出格式要求：
• 类型：JSON 数组，每个元素为一条车次信息字符串。
• 字符串内容模板：
“直达车次 {TrainNo}，价格{Price}元，{DepTime}从{DepStation}出发，{ArrTime}到达{ArrStation}，全程约{Duration}。”
• 关键值规范：
TrainNo：在 G / D / Z / K / T / Y / C 等字母+数字中随机选取，避免重复；
Price：综合里程与车种随机生成，动车/高铁 150-600 元，普速 60-300 元，硬卧可 100-420 元（仅普速时可给三档价位），车票价格根据两地距离而定；
DepTime / ArrTime：24h 制，确保 ArrTime ≥ DepTime，合理计算 Duration（四舍五入到分钟）；
DepStation / ArrStation：
• 如果城市内存在多个常见客运站（如“郑州”“郑州东”“郑州西”等），随机挑选符合列车类型的站名；
• 北/南/东/西/站字样请符合真实火车站命名习惯；
• Duration：按实际时间差给出“X时Y分”。
逻辑与随机性：
• 按常见列车运行规律生成时刻表，不要出现荒诞时间（如 03:00-03:20 只跑 20 分钟的普速）。
• 避免完全均匀分布，可略集中在早高峰 (06-09)、午后 (12-15)、晚高峰 (17-21) 等。
其他：
• 不输出与需求无关的文字、解释或注释，仅返回符合格式的 JSON 数组。
• 所有结果仅为模拟数据，非真实票务信息。
"""

    kwargs = {
        "date": date,
        "from_city": from_city,
        "to_city": to_city,
        "from_city_adcode": from_city_adcode,
        "to_city_adcode": to_city_adcode,
        "from_lat": from_lat,
        "from_lon": from_lon,
        "to_lat": to_lat,
        "to_lon": to_lon,
    }
    query = json.dumps(kwargs, ensure_ascii=False)
    messages = [
        {"role": "user", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    result = ""
    for model in MODELS:
        for attempt in range(MAX_RETRIES):
            try:
                async with semaphore:
                    response = await client.chat.completions.create(
                        messages=messages, model=model
                    )
                result = response.choices[0].message.content.strip()
                if result:
                    break
            except Exception as e:
                logger.warning(
                    f"[search_train_tickets] model={model} 第 {attempt + 1}/{MAX_RETRIES} 次调用失败: {e}"
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAYS[attempt])
        if result:
            break
        logger.warning(f"[search_train_tickets] model={model} 全部失败，尝试备用模型")

    if not result:
        raise ValueError("两地无直达火车票")

    return result

import asyncio

import httpx
from mcp.server.fastmcp import FastMCP

from qqr.data.markdown import json2md
from qqr.data.text import truncate_text
from qqr.utils.envs import AMAP_MAPS_API_KEY

mcp = FastMCP("AMap", log_level="WARNING")

"""
获取环境变量中的 API 密钥, 用于调用高德地图 API
环境变量名为: AMAP_MAPS_API_KEY, 在客户端侧通过配置环境变量进行设置传入
获取方式请参考: https://lbs.amap.com/api/webservice/create-project-and-key
API 文档: https://lbs.amap.com/api/webservice/summary
"""


async def reverse_geocode(location: str):
    url = "https://restapi.amap.com/v3/geocode/regeo"
    params = {"key": AMAP_MAPS_API_KEY, "location": location}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    return result


async def get_citycode(location: str):
    result = await reverse_geocode(location)

    try:
        citycode = result["regeocode"]["addressComponent"]["citycode"]
    except:
        citycode = None

    return citycode


@mcp.tool()
async def poi_search(address: str, region: str | None = None) -> str:
    """
    通过文本搜索地点信息。文本可以是结构化地址，例如：北京市朝阳区望京阜荣街10号；也可以是 POI 名称，例如：首开广场。
    返回多个可能相关的 POI 信息，包括：
        - 详细地址，
        - 经纬度（location 字段，经度和纬度用","分割，经度在前，纬度在后），
        - 商业信息（Business 字段）。
    地址结构越完整，返回的结果越准确。

    Args:
        address (`str`): 需要被检索的地点文本信息。只支持一个地址，文本总长度不可超过 80 字符。
            推荐使用标准的结构化地址信息，如北京市海淀区上地十街十号。地址结构越完整，解析精度越高。
        region (`Optional[str]`): 增加指定区域内数据召回权重，仅支持城市级别和中文，如“北京市”。
            默认为 None，表示在全国范围内搜索。
    """

    url = "https://restapi.amap.com/v5/place/text"
    params = {
        "key": AMAP_MAPS_API_KEY,
        "keywords": address,
        "show_fields": "business",
    }
    if region:
        params["region"] = region

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    if result.get("status") != "1":
        msg = result.get("info", "unknown error")
        raise Exception(f"API response error: {msg}")

    pois = result.get("pois")
    if not pois:
        raise Exception("No POI data available.")

    return truncate_text(json2md(pois))


@mcp.tool()
async def around_search(
    location: str,
    radius: int = 5000,
    keyword: str | None = None,
    region: str | None = None,
) -> str:
    """
    通过设置圆心和半径，搜索圆形区域内的地点信息。可通过 keyword 设定POI类型或限定返回结果，如“银行”。
    返回多个可能相关的 POI 信息，包括：
        - 详细地址，
        - 经纬度（location 字段，经度和纬度用","分割，经度在前，纬度在后），
        - 商业信息（Business 字段）。

    Args:
        location (`str`): 圆形区域检索的中心点坐标，不支持多个点。经度和纬度用","分割，经度在前，纬度在后，经纬度小数点后不得超过6位
        radius (`int`): 圆形区域的搜索半径，取值范围:0-50000，大于50000时按默认值，单位：米。
        keyword (`str`): 需要被检索的地点文本信息。只支持一个关键字，如“银行”。
        region (`Optional[str]`): 增加指定区域内数据召回权重，仅支持城市级别和中文，如“北京市”。
            默认为 None，表示在全国范围内搜索。
    """

    url = "https://restapi.amap.com/v5/place/around"
    params = {
        "key": AMAP_MAPS_API_KEY,
        "location": location,
        "radius": radius,
        "show_fields": "business",
    }
    if keyword:
        params["keywords"] = keyword
    if region:
        params["region"] = region

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    if result.get("status") != "1":
        msg = result.get("info", "unknown error")
        raise Exception(f"API response error: {msg}")

    pois = result.get("pois")
    if not pois:
        raise Exception("No POI data available.")

    return truncate_text(json2md(pois))


async def driving_direction(
    origin: str, destination: str, waypoints: str | None = None
):
    url = "https://restapi.amap.com/v5/direction/driving?parameters"
    params = {"key": AMAP_MAPS_API_KEY, "origin": origin, "destination": destination}

    if waypoints:
        params["waypoints"] = waypoints

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    return result


async def walking_direction(origin: str, destination: str):
    url = "https://restapi.amap.com/v5/direction/walking?parameters"
    params = {"key": AMAP_MAPS_API_KEY, "origin": origin, "destination": destination}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    return result


async def bicycling_direction(origin: str, destination: str):
    url = "https://restapi.amap.com/v5/direction/bicycling?parameters"
    params = {"key": AMAP_MAPS_API_KEY, "origin": origin, "destination": destination}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    return result


async def electrobike_direction(origin: str, destination: str):
    url = "https://restapi.amap.com/v5/direction/electrobike?parameters"
    params = {"key": AMAP_MAPS_API_KEY, "origin": origin, "destination": destination}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    return result


async def transit_direction(origin: str, destination: str):
    url = "https://restapi.amap.com/v5/direction/transit/integrated?parameters"

    citycode_origin, citycode_destination = await asyncio.gather(
        get_citycode(origin), get_citycode(destination)
    )

    if not citycode_origin:
        raise Exception("City not found for transit origin.")

    if not citycode_destination:
        raise Exception("City not found for transit destination.")

    params = {
        "key": AMAP_MAPS_API_KEY,
        "origin": origin,
        "destination": destination,
        "city1": citycode_origin,
        "city2": citycode_destination,
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    return result


@mcp.tool()
async def direction(
    origin: str, destination: str, mode: str = "driving", waypoints: str | None = None
) -> str:
    """
    提供多种路线规划服务。支持驾车、步行、骑行、电动车、公交路线规划。

    Args:
        origin: 起点信息坐标。经度在前，纬度在后，经度和纬度用","分割，经纬度小数点后不得超过6位。
        destination: 目的地信息坐标。经度在前，纬度在后，经度和纬度用","分割，经纬度小数点后不得超过6位。
        mode: 路线规划类型，默认为驾车路线规划。
            - Enum: ["driving", "walking", "bicycling", "electrobike", "transit"]。
        waypoints: 途经点。经度和纬度用","分割，经度在前，纬度在后，小数点后不超过6位，坐标点之间用";"分隔。
            - 最大数目：16个坐标点。
    """
    if mode == "driving":
        result = await driving_direction(origin, destination, waypoints=waypoints)
    elif mode == "walking":
        result = await walking_direction(origin, destination)
    elif mode == "bicycling":
        result = await bicycling_direction(origin, destination)
    elif mode == "electrobike":
        result = await electrobike_direction(origin, destination)
    elif mode == "transit":
        result = await transit_direction(origin, destination)

    if result.get("status") != "1":
        msg = result.get("info", "unknown error")
        raise Exception(f"API response error: {msg}")

    route = result.get("route")
    if not route:
        raise Exception("No route available.")

    return truncate_text(json2md(route))


@mcp.tool()
async def weather(city: str) -> str:
    """
    根据城市名称查询指定城市的天气

    Args:
        city (`str`): 城市名称
    """

    url = "https://restapi.amap.com/v3/weather/weatherInfo"
    params = {
        "key": AMAP_MAPS_API_KEY,
        "city": city,
        "extensions": "all",
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    if result.get("status") != "1":
        msg = result.get("info", "unknown error")
        raise Exception(f"API response error: {msg}")

    forecasts = result.get("forecasts")
    if not forecasts:
        raise Exception("No forecast data available.")

    def format_cast(cast):
        return {
            "dayweather": cast["dayweather"],
            "nightweather": cast["nightweather"],
            "daytemp": cast["daytemp"],
            "nighttemp": cast["nighttemp"],
            "daywind": cast["daywind"],
            "nightwind": cast["nightwind"],
            "daypower": cast["daypower"],
            "nightpower": cast["nightpower"],
        }

    def format_forecast(forecast):
        return {
            "city": forecast["city"],
            "province": forecast["province"],
            "casts": [format_cast(cast) for cast in forecast["casts"]],
        }

    forecasts = [format_forecast(forecast) for forecast in forecasts]
    return truncate_text(json2md(forecasts))

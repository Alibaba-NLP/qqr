"""
VitaBench curriculum agent (generator) training configuration - Qwen3.5 + RG-KL

The generator creates tasks from the 100 official VitaBench environments:
- Tool calls go through the real VitaBenchToolState, so results match evaluation
- No mock model is needed, so there is no train/eval skew
- The environment comes straight from the official data and needs no sanitising

The solver validates task quality in interactive mode:
- UserSimulator (local Qwen3.5-397B-A17B) drives the multi-turn user interaction
- A sliding-window judge (local Qwen3.5-397B-A17B) scores the rubrics

Algorithm: Reward-Gated Reverse KL (RG-KL)
- Three-phase rollout: student (k_s, no coach) -> gpt-5.2 produces coach memory -> teacher (k_m, with coach)
- λ(Δ_r) = λ_0 · gate(Δ_r) · warmup · cosine_decay
"""

import logging
import os
import random
from collections import Counter

from qqr.utils.envs import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

logger = logging.getLogger(__name__)

__all__ = [
    "max_steps",
    "default_domain",
    "vitabench_data_dir",
    "executor_api_base",
    "executor_model",
    "executor_max_steps",
    "executor_num_trials",
    "executor_success_threshold",
    "solver_reward_alpha",
    "executor_concurrency_limit",
    "llm_judge_api_key",
    "llm_judge_base_url",
    "llm_judge_model",
    "llm_judge_concurrency_limit",
    "coach_api_key",
    "coach_base_url",
    "coach_model",
    "coach_concurrency_limit",
    "enable_rg_kl",
    "k_student",
    "k_teacher",
    "rg_kl_lambda_0",
    "rg_kl_delta_threshold",
    "rg_kl_gate_temperature",
    "rg_kl_warmup_iters",
    "rg_kl_decay_iters",
    "rg_kl_min_lambda_ratio",
    "query_generation_system_prompt",
    "get_query_generation_system_prompt",
    "DOMAIN_LABELS",
    "generate_random_user_profile",
]

# -- Basic settings --
default_domain = os.environ.get("VITABENCH_DOMAIN", "ota")
vitabench_data_dir = os.environ.get(
    "VITABENCH_DATA_DIR", "/path/to/vitabench/data/vita"  # points at the data/vita directory of the official VitaBench repo
)

# -- Generator settings --
max_steps = 12  # exploring the real environment needs more rounds (search, details and weather)

# -- Solver (executor) settings --
executor_api_base = os.environ.get("EXECUTOR_API_BASE", "http://localhost:30000/v1")
executor_model = os.environ.get("EXECUTOR_MODEL", "default")
executor_max_steps = int(os.environ.get("EXECUTOR_MAX_STEPS", "40"))
executor_num_trials = int(os.environ.get("EXECUTOR_NUM_TRIALS", "8"))
# difficulty threshold γ: a trial counts as a success when its rubric pass rate is at least γ
executor_success_threshold = float(os.environ.get("EXECUTOR_SUCCESS_THRESHOLD", "0.9"))
solver_reward_alpha = float(os.environ.get("SOLVER_REWARD_ALPHA", "0.8"))  # α: weight of the solver's partial score (paper Eq. (6))
executor_concurrency_limit = int(os.environ.get("EXECUTOR_CONCURRENCY_LIMIT", "8"))

# -- UserSimulator settings (the paper uses Qwen3.5-397B) --
user_simulator_api_key = os.environ.get("USER_SIMULATOR_API_KEY", "EMPTY")
user_simulator_base_url = os.environ.get("USER_SIMULATOR_BASE_URL", DASHSCOPE_BASE_URL)
user_simulator_model = os.environ.get("USER_SIMULATOR_MODEL", "Qwen3.5-397B-A17B")
user_simulator_temperature = float(os.environ.get("USER_SIMULATOR_TEMPERATURE", "0.0"))

# -- LLM judge (sliding-window rubric scoring; the paper uses gpt-5.2) --
llm_judge_api_key = os.environ.get("LLM_JUDGE_API_KEY", DASHSCOPE_API_KEY or "EMPTY")
llm_judge_base_url = os.environ.get("LLM_JUDGE_BASE_URL", DASHSCOPE_BASE_URL)
llm_judge_model = os.environ.get("LLM_JUDGE_MODEL", "gpt-5.2-2025-12-11")  # the paper uses gpt-5.2 for every judge
llm_judge_concurrency_limit = int(os.environ.get("LLM_JUDGE_CONCURRENCY_LIMIT", "16"))

# -- Coach (gpt-5.2 via DashScope, produces the generator coaching memory) --
coach_api_key = os.environ.get("COACH_API_KEY", DASHSCOPE_API_KEY)
coach_base_url = os.environ.get("COACH_BASE_URL", DASHSCOPE_BASE_URL)
coach_model = os.environ.get("COACH_MODEL", "gpt-5.2-2025-12-11")
coach_concurrency_limit = int(os.environ.get("COACH_CONCURRENCY_LIMIT", "10"))

# -- Reward settings --
group_reward_model_name = None


# ============ Reward-Gated Reverse KL (RG-KL) settings ============

enable_rg_kl = os.environ.get("ENABLE_RG_KL", "true").lower() == "true"

# Three-phase rollout split: n_samples_per_prompt = k_student + k_teacher
k_student = int(os.environ.get("K_STUDENT", "8"))
k_teacher = int(os.environ.get("K_TEACHER", "8"))

# RG-KL gating base coefficient
rg_kl_lambda_0 = float(os.environ.get("RG_KL_LAMBDA_0", "0.5"))

# Δ_r threshold
rg_kl_delta_threshold = float(os.environ.get("RG_KL_DELTA_THRESHOLD", "0.05"))
rg_kl_gate_temperature = float(os.environ.get("RG_KL_GATE_TEMPERATURE", "0.0125"))

# Warmup
rg_kl_warmup_iters = int(os.environ.get("RG_KL_WARMUP_ITERS", "20"))

# Cosine decay
rg_kl_decay_iters = int(os.environ.get("RG_KL_DECAY_ITERS", "120"))
rg_kl_min_lambda_ratio = float(os.environ.get("RG_KL_MIN_LAMBDA_RATIO", "0.1"))

# safety switch: whether to build guided_tokens for student samples
enable_guided_tokens_for_student = (
    os.environ.get("ENABLE_GUIDED_TOKENS_FOR_STUDENT", "true").lower() == "true"
)


# ═══════════════════════════════════════════════════════════════════════════════
# random user-profile generation
# ═══════════════════════════════════════════════════════════════════════════════

_OCCUPATIONS = ["教师", "程序员", "医生", "律师", "公务员", "设计师", "销售", "自由职业者",
                "学生", "退休人员", "记者", "会计", "工程师", "企业管理者", "厨师"]
_GENDERS = ["男", "女"]
_AGE_RANGES = ["20~25", "25~30", "30~35", "35~40", "40~45", "45~50", "50~55", "55~60", "60~65"]
_CITIES = ["北京", "上海", "广州", "深圳", "成都", "杭州", "武汉", "南京", "重庆", "西安",
           "长沙", "青岛", "大连", "厦门", "昆明", "贵阳", "郑州", "合肥", "福州", "济南",
           "沈阳", "哈尔滨", "长春", "太原", "兰州", "银川", "西宁", "呼和浩特", "拉萨", "乌鲁木齐"]
_PERSONALITIES = [
    "性格开朗热情，善于表达",
    "表达冷漠简洁，缺乏情感交流和耐心",
    "细心谨慎，注重细节",
    "随性洒脱，不拘小节",
    "礼貌温和，表达客气",
    "直接干脆，不喜欢废话",
]
_DIET_RESTRICTIONS = ["无", "忌辣", "忌重油辣", "素食", "忌海鲜", "忌牛羊肉", "忌酒精",
                      "忌重油辣, 忌酒精", "清真饮食"]
_FAMILY_SITUATIONS = ["单身", "已婚无孩", "有小孩(3岁)", "有小孩(8岁)", "有老人", "三口之家",
                      "四口之家", "和父母同住"]

# joint-distribution constraint: occupation determines the possible age bands
_AGE_RANGES_BY_OCCUPATION = {
    "学生":       ["20~25", "25~30"],
    "退休人员":   ["55~60", "60~65"],
    "医生":       ["25~30", "30~35", "35~40", "40~45", "45~50", "50~55"],
    "律师":       ["25~30", "30~35", "35~40", "40~45", "45~50", "50~55"],
    "企业管理者": ["30~35", "35~40", "40~45", "45~50", "50~55"],
}
_DEFAULT_AGE_RANGES = ["20~25", "25~30", "30~35", "35~40", "40~45", "45~50", "50~55", "55~60"]

# age band determines the possible family situations
_FAMILY_BY_AGE = {
    "20~25": ["单身", "已婚无孩", "和父母同住"],
    "25~30": ["单身", "已婚无孩", "有小孩(3岁)", "和父母同住", "三口之家"],
    "30~35": ["单身", "已婚无孩", "有小孩(3岁)", "有小孩(8岁)", "三口之家", "四口之家", "和父母同住"],
    "35~40": ["已婚无孩", "有小孩(3岁)", "有小孩(8岁)", "三口之家", "四口之家", "有老人", "和父母同住"],
    "40~45": ["已婚无孩", "有小孩(8岁)", "三口之家", "四口之家", "有老人", "和父母同住"],
    "45~50": ["已婚无孩", "有小孩(8岁)", "三口之家", "四口之家", "有老人"],
    "50~55": ["已婚无孩", "三口之家", "四口之家", "有老人"],
    "55~60": ["已婚无孩", "三口之家", "四口之家", "有老人"],
    "60~65": ["已婚无孩", "三口之家", "四口之家", "有老人"],
}


def generate_random_user_profile(rng: random.Random | None = None) -> dict:
    """Generate a random user profile (8 fields, with joint-distribution constraints)."""
    rng = rng or random
    user_id = f"U{rng.randint(100000, 999999)}"

    occupation = rng.choice(_OCCUPATIONS)
    age_range = rng.choice(
        _AGE_RANGES_BY_OCCUPATION.get(occupation, _DEFAULT_AGE_RANGES)
    )
    family = rng.choice(_FAMILY_BY_AGE.get(age_range, _FAMILY_SITUATIONS))

    return {
        "用户id": user_id,
        "职业": occupation,
        "性别": rng.choice(_GENDERS),
        "年龄段": age_range,
        "常住地": rng.choice(_CITIES),
        "饮食禁忌": rng.choice(_DIET_RESTRICTIONS),
        "家庭情况": family,
        "性格": rng.choice(_PERSONALITIES),
    }


def format_user_profile_prefix(profile: dict) -> str:
    """Format the user profile as a prefix to the query."""
    lines = ["以下是我的个人信息："]
    for k, v in profile.items():
        lines.append(f"- {k}：{v}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# real-environment helpers
# ═══════════════════════════════════════════════════════════════════════════════


def infer_home_city(task_data: dict) -> str | None:
    """Infer the user's home city (their departure city) from the environment."""
    env = task_data.get("environment", {})

    departure_cities: list[str] = []
    for f in env.get("flights", {}).values():
        dep = f.get("departure_city", "")
        if dep:
            departure_cities.append(dep)
    for t in env.get("trains", {}).values():
        dep = t.get("departure_city", "")
        if dep:
            departure_cities.append(dep)

    if not departure_cities:
        return None

    arrival_cities = set()
    for f in env.get("flights", {}).values():
        arr = f.get("arrival_city", "")
        if arr:
            arrival_cities.add(arr)
    for t in env.get("trains", {}).values():
        arr = t.get("arrival_city", "")
        if arr:
            arrival_cities.add(arr)

    # prefer a city that only appears as an origin
    pure_departures = [c for c in departure_cities if c not in arrival_cities]
    if pure_departures:
        return Counter(pure_departures).most_common(1)[0][0]

    return Counter(departure_cities).most_common(1)[0][0]


def generate_matched_user_profile(task_data: dict, rng: random.Random | None = None) -> dict:
    """Build a user_profile consistent with the environment, with the home city matching the departure city."""
    profile = generate_random_user_profile(rng)

    env = task_data.get("environment", {})
    profile["用户id"] = env.get("user_id", profile["用户id"])

    home_city = infer_home_city(task_data)
    if home_city:
        profile["常住地"] = home_city

    return profile


def extract_env_summary(task_data: dict) -> str:
    """Summarise the environment so the generator model can see what it contains."""
    env = task_data.get("environment", {})
    parts = []
    parts.append(f"环境时间: {env.get('time', '未知')}")
    parts.append(f"用户ID: {env.get('user_id', '未知')}")

    hotels = env.get("hotels", {})
    if hotels:
        names = [h.get("hotel_name", "?") for h in hotels.values() if isinstance(h, dict)]
        parts.append(f"酒店: {len(hotels)} 个 ({', '.join(names[:3])}{'...' if len(names) > 3 else ''})")

    flights = env.get("flights", {})
    if flights:
        routes = set()
        for f in flights.values():
            dep = f.get("departure_city", "")
            arr = f.get("arrival_city", "")
            if dep and arr:
                routes.add(f"{dep}→{arr}")
        parts.append(f"航班: {len(flights)} 个，路线: {', '.join(routes)}")
    else:
        parts.append("航班: 无（不要设计需要坐飞机的任务）")

    trains = env.get("trains", {})
    if trains:
        routes = set()
        for t in trains.values():
            dep = t.get("departure_city", "")
            arr = t.get("arrival_city", "")
            if dep and arr:
                routes.add(f"{dep}→{arr}")
        parts.append(f"火车: {len(trains)} 个，路线: {', '.join(routes)}")
    else:
        parts.append("火车: 无（不要设计需要坐火车的任务）")

    attractions = env.get("attractions", {})
    if attractions:
        names = [a.get("attraction_name", "?") for a in attractions.values() if isinstance(a, dict)]
        parts.append(f"景点: {', '.join(names)}")
    else:
        parts.append("景点: 无")

    # -- delivery domain: takeaway merchants --
    stores = env.get("stores", {})
    if stores:
        names = [s.get("store_name", s.get("name", "?")) for s in stores.values() if isinstance(s, dict)]
        parts.append(f"外卖店铺: {len(stores)} 个 ({', '.join(names[:3])}{'...' if len(names) > 3 else ''})")

    # -- instore domain: physical stores, table booking and appointments --
    shops = env.get("shops", {})
    if shops:
        names = [s.get("shop_name", s.get("name", "?")) for s in shops.values() if isinstance(s, dict)]
        parts.append(f"到店门店: {len(shops)} 个 ({', '.join(names[:3])}{'...' if len(names) > 3 else ''})")
    if env.get("books"):
        parts.append(f"既有订座记录: {len(env['books'])} 条")
    if env.get("reservations"):
        parts.append(f"既有预约记录: {len(env['reservations'])} 条")

    # -- Shared: history and location --
    if env.get("user_historical_behaviors"):
        parts.append(f"用户历史行为: {len(env['user_historical_behaviors'])} 条（可用于'复购上次那家'类需求）")
    if env.get("location"):
        parts.append(f"位置数据: {len(env['location'])} 条")

    weather = env.get("weather", [])
    if weather:
        dates = sorted(set(w.get("datetime", "") for w in weather if isinstance(w, dict)))
        cities_w = sorted(set(w.get("city", "") for w in weather if isinstance(w, dict)))
        if dates:
            parts.append(f"天气数据: {', '.join(cities_w)}，日期 {dates[0]}~{dates[-1]}")
        else:
            parts.append(f"天气数据: {len(weather)} 条")

    return "\n".join(parts)


def get_available_types(task_data: dict, domain: str = "ota") -> list[str]:
    """Return the task types available for a domain given the data the environment actually holds."""
    env = task_data.get("environment", {})
    available = []

    def _ota_types():
        types = []
        if env.get("hotels"):
            types.append("酒店")
        if env.get("flights"):
            types.append("机票")
        if env.get("trains"):
            types.append("火车票")
        if env.get("attractions"):
            types.append("景点门票")
        return types

    def _delivery_types():
        types = []
        if env.get("stores"):
            types += ["外卖下单", "配送时效"]
            if env.get("user_historical_behaviors") or env.get("orders"):
                types.append("历史复购")
            if env.get("orders"):
                types.append("外卖订单管理")
        return types

    def _instore_types():
        types = []
        if env.get("shops"):
            types += ["到店团购", "餐厅订座", "到店预约"]
        return types

    if domain == "ota":
        available = _ota_types()
    elif domain == "delivery":
        available = _delivery_types()
    elif domain == "instore":
        available = _instore_types()
    else:  # cross_domain: combinations spanning domains
        available = _ota_types() + _delivery_types() + _instore_types()
    return available


def choose_order_types(available: list[str], rng: random.Random | None = None, target_count: int = 2) -> list[str]:
    """Randomly choose a combination of order types from those the environment offers."""
    rng = rng or random
    count = min(target_count, len(available))
    if count == 0:
        return ["酒店"]
    return rng.sample(available, count)


# task type to tool-chain mapping per domain (tool names match the official VitaBench ones)
_TYPE_TO_TOOLS_OTA = {
    "酒店": ("hotel_search_recommend(city_name, key_words=[...]) → get_ota_hotel_info", "搜索酒店并查看房型详情"),
    "机票": ("flight_search_recommend(departure, destination) → get_ota_flight_info", "搜索航班并查看座位详情"),
    "火车票": ("train_ticket_search(departure, destination, date) → get_ota_train_info", "搜索火车并查看座位详情"),
    "景点门票": ("attractions_search_recommend(city_name, key_words=[...]) → get_ota_attraction_info", "搜索景点并查看门票详情"),
}
_TYPE_TO_TOOLS_DELIVERY = {
    "外卖下单": ("delivery_store_search_recommend / delivery_product_search_recommend → "
                "get_delivery_store_info / get_delivery_product_info → create_delivery_order → pay_delivery_order",
                "搜索外卖店铺与商品、查看菜单详情并下单支付"),
    "历史复购": ("get_user_historical_behaviors / search_delivery_orders → get_delivery_order_detail → create_delivery_order",
                "从历史订单定位『上次那家店』并复购指定商品"),
    "配送时效": ("address_to_longitude_latitude → longitude_latitude_to_distance → delivery_distance_to_time",
                "把『X 点前送达』等自然语言时间约束换算为可验证的配送 ETA"),
    "外卖订单管理": ("get_delivery_order_status / get_delivery_order_detail → modify_delivery_order / cancel_delivery_order",
                "查询既有外卖订单并按需修改或取消"),
}
_TYPE_TO_TOOLS_INSTORE = {
    "到店团购": ("instore_shop_search_recommend / instore_product_search_recommend → "
                "create_instore_product_order → pay_instore_order",
                "搜索到店门店与团购/代金券商品并下单支付"),
    "餐厅订座": ("instore_book → pay_instore_book / search_instore_book",
                "按日期、时段、人数为门店预订餐位"),
    "到店预约": ("instore_reservation / instore_modify_reservation / search_instore_reservation",
                "预约到店服务（如美容、体检、维修）并按需修改"),
}
_TYPE_TO_TOOLS_BY_DOMAIN = {
    "ota": _TYPE_TO_TOOLS_OTA,
    "delivery": _TYPE_TO_TOOLS_DELIVERY,
    "instore": _TYPE_TO_TOOLS_INSTORE,
    "cross_domain": {**_TYPE_TO_TOOLS_OTA, **_TYPE_TO_TOOLS_DELIVERY, **_TYPE_TO_TOOLS_INSTORE},
}
# backwards-compatible alias (defaults to OTA)
_TYPE_TO_TOOLS = _TYPE_TO_TOOLS_OTA

DOMAIN_LABELS = {
    "ota": "OTA 出行预订",
    "delivery": "外卖点餐",
    "instore": "到店消费",
    "cross_domain": "跨场景生活服务（外卖 + 到店 + OTA）",
}


def build_user_prompt(
    user_profile: dict,
    task_data: dict,
    chosen_types: list[str],
    coach_summary: str = "",
    domain: str = "ota",
) -> str:
    """Build the generator user prompt (environment summary, difficulty requirement and coach feedback)."""
    parts = []
    label = DOMAIN_LABELS.get(domain, DOMAIN_LABELS["ota"])
    parts.append(f"请为以下环境设计一个交互模式{label}任务。\n")

    parts.append("## 用户画像（将注入 UserSimulator 的人物设定）")
    for k, v in user_profile.items():
        parts.append(f"- {k}: {v}")
    parts.append("")
    parts.append("**【强制约束】instructions 中的所有身份/生活细节必须与上述画像字段完全一致，**")
    parts.append("**严禁出现矛盾事实（如画像=单身却写'和老婆'，画像=素食却写'吃海鲜'等）。**")
    parts.append("**写完后请逐字段对照画像自检。**")
    parts.append("")

    parts.append("## 当前环境概况（这是一个模拟环境，只有以下数据可用）")
    parts.append(extract_env_summary(task_data))
    parts.append("")
    parts.append("**请只围绕上述已有数据设计任务。** 通过工具调用探索具体详情（房型、座位、门票等），然后基于真实数据设计任务。")
    parts.append("")

    types_str = " + ".join(chosen_types)
    parts.append("## 难度要求")
    parts.append(f"- 必须涉及的订单类型: **{types_str}**")
    parts.append(f"- rubrics 数量: 3-9 条")
    parts.append("")

    parts.append("## 工作指示")
    parts.append(f"本次任务必须涉及 **{types_str}**，请按以下步骤操作：")
    type_to_tools = _TYPE_TO_TOOLS_BY_DOMAIN.get(domain, _TYPE_TO_TOOLS_OTA)
    for i, otype in enumerate(chosen_types, 1):
        tools_chain, desc = type_to_tools.get(otype, ("", ""))
        parts.append(f"{i}. {desc}（工具链: {tools_chain}）")
    parts.append(f"{len(chosen_types)+1}. 可选：调用 weather 查询天气，设计需要天气判断的需求")
    parts.append(f"{len(chosen_types)+2}. 基于获取到的真实数据，设计 instructions 和 rubrics 并输出 JSON")
    parts.append("")

    parts.append("## 工具调用提示")
    if domain in ("ota", "cross_domain"):
        parts.append('- hotel_search_recommend 和 attractions_search_recommend 的 key_words 参数**必须传且必须是 list**，如 `["酒店"]`')
    if domain in ("delivery", "cross_domain"):
        parts.append("- 外卖类需求先用 delivery_store_search_recommend / delivery_product_search_recommend 搜索，"
                     "再用 get_delivery_store_info / get_delivery_product_info 核对菜单、价格与口味规格")
        parts.append("- 涉及送达时间约束时，用 address_to_longitude_latitude → longitude_latitude_to_distance → "
                     "delivery_distance_to_time 换算配送 ETA，确认约束可满足")
    if domain in ("instore", "cross_domain"):
        parts.append("- 到店类需求先用 instore_shop_search_recommend / instore_product_search_recommend 搜索门店与商品，"
                     "订座/预约前核对门店营业时段与可约时段")
    parts.append("- 如果搜索返回空结果或错误，说明环境中没有该数据，不要重试，调整任务设计")
    parts.append("- rubrics 中引用的门店名/商品名/价格/ID/日期等必须和工具返回一致")
    parts.append("")

    if coach_summary:
        parts.append(f"## 上一轮出题教练反馈\n{coach_summary}\n")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# System Prompts
# ═══════════════════════════════════════════════════════════════════════════════

query_generation_system_prompt = """你是一个 VitaBench OTA 场景出题专家。你的任务是设计高质量的交互模式生活服务场景测试任务。

交互模式的特点：
- UserSimulator 根据你设计的 instructions 分多轮向 Agent 透露需求
- Agent 可以提问确认，获取缺失信息后再执行工具调用

## 你的工作流程

### 第一阶段：探索环境（调用工具）
你可以调用真实的 OTA 工具来探索当前环境中有哪些可用数据：
- 搜索目的地城市的酒店、航班、火车、景点
- 查询天气信息
- 获取具体商品（酒店房型、航班座位等）的详细信息
- 这些工具返回的是真实数据，请仔细阅读返回结果

**重要：这是一个有限的模拟环境，不是真实互联网。**
- 环境中只有特定城市的数据，只能围绕已有的城市和路线设计任务
- 如果搜索返回空结果，说明环境中没有该数据，不要反复重试，换个方向
- 搜索类工具可能需要特定参数格式，注意看错误提示并修正

### 第二阶段：设计任务
基于探索到的真实数据，设计一个完整的交互模式任务：
1. 设计 instructions（完整任务描述，第二人称"你"视角）
2. 设计 rubrics（评估标准，每条是一个可判断 true/false 的断言）

### 第三阶段：输出 JSON

## 核心设计原则
1. **【强制】instructions ↔ 用户画像零矛盾**：
   - instructions 中所有身份/生活细节必须与给定 user_profile 字段**逐字段一致**
   - 反例：画像=单身却写"和老婆纪念日"；画像=20~25岁却写"我已退休"；画像=素食却写"想吃海鲜"；画像=常住北京却写"我从上海出发"（除非任务本身是出差/异地）
   - 写 instructions 前必须**逐字段对照**画像（用户id/职业/性别/年龄段/常住地/饮食禁忌/家庭情况/性格），确认无矛盾
   - 当画像字段与任务无关时（如性格、饮食禁忌对纯航班任务），可以不提，但不可写矛盾内容
2. **instructions 必须基于工具返回的真实数据**：所有需求都必须是工具返回确实存在的
3. **rubrics 必须基于真实数据**：所有 ID、价格、房型等必须来自工具返回
4. **rubrics 要具体可验证**：每条是一个明确的断言
5. **instructions 写法**：
   - 使用第二人称"你"，描述用户的处境和需求
   - 包含背景故事和具体需求细节
   - 信息丰富，让 UserSimulator 有内容分多轮透露
   - 自然口语化，像真实用户的想法
6. **利用干扰信息**：环境中有多个选项，设计条件区分它们
7. **时间推理**：让需求涉及日期计算

## rubrics 设计要求
- 每条是一个可判断 true/false 的断言
- 覆盖关键决策点（选对了哪家酒店？房型对吗？日期对吗？支付了吗？）
- 数量 3-9 条，只聚焦用户明确需求

## 输出格式

完成工具探索后，严格按以下 JSON 输出：

```json
{
  "instructions": "完整的任务指令（第二人称视角）",
  "rubrics": ["rubric1", "rubric2", ...]
}
```

## 输出前自检清单（必须通过才能输出 JSON）
- [ ] instructions 中任意"我/你"的身份描述是否与 user_profile 完全一致？（家庭情况、职业、年龄段、性别、常住地、饮食禁忌）
- [ ] 是否有虚构画像之外的家庭成员/职业/年龄/喜好？
- [ ] rubrics 引用的 ID/价格/房型是否全部来自工具返回？

重要：不需要输出 environment 和 user_profile（系统自动处理）。"""


QUERY_GENERATION_SYSTEM_PROMPT_DELIVERY = """你是一个 VitaBench 外卖点餐（Delivery）场景出题专家。你的任务是设计高质量的交互模式外卖任务。

交互模式的特点：
- UserSimulator 根据你设计的 instructions 分多轮向 Agent 透露需求
- Agent 可以提问确认，获取缺失信息后再执行工具调用

## 你的工作流程

### 第一阶段：探索环境（调用工具）
你可以调用真实的外卖工具来探索当前环境：
- 用 delivery_store_search_recommend / delivery_product_search_recommend 搜索店铺与商品
- 用 get_delivery_store_info / get_delivery_product_info 查看菜单、价格、口味/规格选项
- 用 get_user_historical_behaviors / search_delivery_orders 查看用户历史订单（可设计"复购上次那家"类需求）
- 用 address_to_longitude_latitude / longitude_latitude_to_distance / delivery_distance_to_time 估算配送距离与 ETA

**重要：这是一个有限的模拟环境，不是真实互联网。**
- 只能围绕环境中真实存在的店铺、商品和地址设计任务
- 如果搜索返回空结果，说明环境中没有该数据，不要反复重试，换个方向

### 第二阶段：设计任务
基于探索到的真实数据，设计一个完整的交互模式外卖任务，典型要素可组合：
- 指定口味/规格/忌口（须与用户画像饮食禁忌一致）
- 送达地址 + 硬时间约束（如"13:00 午休前送到"，须先验证 ETA 可满足）
- 从历史订单复购"上次那家店"，或在多家相近店铺间按评分/距离/价格筛选
- 修改或取消既有订单

### 第三阶段：输出 JSON

## 核心设计原则
1. **【强制】instructions ↔ 用户画像零矛盾**：
   - instructions 中所有身份/生活细节必须与给定 user_profile 字段**逐字段一致**
   - 写 instructions 前必须**逐字段对照**画像（用户id/职业/性别/年龄段/常住地/饮食禁忌/家庭情况/性格），确认无矛盾
   - 当画像字段与任务无关时可以不提，但不可写矛盾内容（如画像=素食却写"想吃水煮鱼"）
2. **instructions 必须基于工具返回的真实数据**：所有需求都必须是工具返回中确实存在的
3. **rubrics 必须基于真实数据**：所有门店名、商品名、价格、ID 必须来自工具返回
4. **rubrics 要具体可验证**：每条是一个明确的 true/false 断言
5. **instructions 写法**：第二人称"你"，包含背景故事与具体需求细节，信息丰富、自然口语化，
   让 UserSimulator 有内容分多轮透露
6. **利用干扰信息**：环境中有多个相近选项时，设计条件让 Agent 必须区分它们
7. **时间推理**：让需求涉及相对日期/时段推算（"明天中午前"、"周六晚上"等）

## rubrics 设计要求
- 每条是一个可判断 true/false 的断言
- 覆盖关键决策点（选对了哪家店？商品/规格对吗？时间约束满足吗？支付了吗？）
- 数量 3-9 条，只聚焦用户明确需求

## 输出格式

完成工具探索后，严格按以下 JSON 输出：

```json
{
  "instructions": "完整的任务指令（第二人称视角）",
  "rubrics": ["rubric1", "rubric2", ...]
}
```

## 输出前自检清单（必须通过才能输出 JSON）
- [ ] instructions 中任意"我/你"的身份描述是否与 user_profile 完全一致？
- [ ] 是否有虚构画像之外的家庭成员/职业/年龄/喜好？
- [ ] rubrics 引用的门店/商品/价格/ID/时间是否全部来自工具返回？

重要：不需要输出 environment 和 user_profile（系统自动处理）。"""


QUERY_GENERATION_SYSTEM_PROMPT_INSTORE = """你是一个 VitaBench 到店消费（In-Store）场景出题专家。你的任务是设计高质量的交互模式到店任务。

交互模式的特点：
- UserSimulator 根据你设计的 instructions 分多轮向 Agent 透露需求
- Agent 可以提问确认，获取缺失信息后再执行工具调用

## 你的工作流程

### 第一阶段：探索环境（调用工具）
你可以调用真实的到店工具来探索当前环境：
- 用 instore_shop_search_recommend / instore_product_search_recommend 搜索门店与团购/代金券商品
- 查看门店的营业时段、人均价格、评分与商品适用规则（如"周末不可用"、"X 人套餐"）
- 订座类：instore_book / search_instore_book；预约类：instore_reservation / search_instore_reservation
- 可结合 weather / get_user_historical_behaviors 设计需要天气判断或复购偏好的需求

**重要：这是一个有限的模拟环境，不是真实互联网。**
- 只能围绕环境中真实存在的门店与商品设计任务
- 如果搜索返回空结果，说明环境中没有该数据，不要反复重试，换个方向

### 第二阶段：设计任务
基于探索到的真实数据，设计一个完整的交互模式到店任务，典型要素可组合：
- 按人数/预算/位置约束选门店并购买对应团购套餐（注意套餐人数规格须与用户需求一致）
- 指定日期时段的餐厅订座（相对日期须可推算，人数明确）
- 到店服务预约及改期（先查可约时段再预约）
- 在多家相近门店间按评分/距离/适用规则筛选

### 第三阶段：输出 JSON

## 核心设计原则
1. **【强制】instructions ↔ 用户画像零矛盾**：
   - instructions 中所有身份/生活细节必须与给定 user_profile 字段**逐字段一致**
   - 写 instructions 前必须**逐字段对照**画像（用户id/职业/性别/年龄段/常住地/饮食禁忌/家庭情况/性格），确认无矛盾
   - 当画像字段与任务无关时可以不提，但不可写矛盾内容（如画像=素食却写"想吃水煮鱼"）
2. **instructions 必须基于工具返回的真实数据**：所有需求都必须是工具返回中确实存在的
3. **rubrics 必须基于真实数据**：所有门店名、商品名、价格、ID 必须来自工具返回
4. **rubrics 要具体可验证**：每条是一个明确的 true/false 断言
5. **instructions 写法**：第二人称"你"，包含背景故事与具体需求细节，信息丰富、自然口语化，
   让 UserSimulator 有内容分多轮透露
6. **利用干扰信息**：环境中有多个相近选项时，设计条件让 Agent 必须区分它们
7. **时间推理**：让需求涉及相对日期/时段推算（"明天中午前"、"周六晚上"等）

## rubrics 设计要求
- 每条是一个可判断 true/false 的断言
- 覆盖关键决策点（选对了哪家店？商品/规格对吗？时间约束满足吗？支付了吗？）
- 数量 3-9 条，只聚焦用户明确需求

## 输出格式

完成工具探索后，严格按以下 JSON 输出：

```json
{
  "instructions": "完整的任务指令（第二人称视角）",
  "rubrics": ["rubric1", "rubric2", ...]
}
```

## 输出前自检清单（必须通过才能输出 JSON）
- [ ] instructions 中任意"我/你"的身份描述是否与 user_profile 完全一致？
- [ ] 是否有虚构画像之外的家庭成员/职业/年龄/喜好？
- [ ] rubrics 引用的门店/商品/价格/ID/时间是否全部来自工具返回？

重要：不需要输出 environment 和 user_profile（系统自动处理）。"""


QUERY_GENERATION_SYSTEM_PROMPT_CROSS = """你是一个 VitaBench 跨场景（Cross-Domain）出题专家。你的任务是设计横跨外卖、到店、OTA 多个域的高质量交互模式任务。

交互模式的特点：
- UserSimulator 根据你设计的 instructions 分多轮向 Agent 透露需求
- Agent 可以提问确认，获取缺失信息后再执行工具调用

## 你的工作流程

### 第一阶段：探索环境（调用工具）
本环境同时包含多个域的数据与工具，请分别探索：
- OTA：hotel_search_recommend / flight_search_recommend / train_ticket_search / attractions_search_recommend 及对应 get_ota_*_info
- 外卖：delivery_store_search_recommend / delivery_product_search_recommend 及 get_delivery_*_info
- 到店：instore_shop_search_recommend / instore_product_search_recommend、instore_book、instore_reservation
- 通用：weather、get_user_historical_behaviors、address_to_longitude_latitude 等

**重要：这是一个有限的模拟环境，不是真实互联网。**
- 只能围绕环境中真实存在的数据设计任务；搜索为空就换方向，不要重试

### 第二阶段：设计任务
设计一个**至少覆盖两个不同域**的复合任务，各子需求之间要有真实的关联与约束，例如：
- 出差/旅行主线（订火车票或酒店）+ 到达后的外卖或到店餐饮安排
- 天气 × 预算 × 时间窗的冲突约束，迫使 Agent 先查证再决策
- 子任务共享同一批人物/日期/城市设定，参数须相互一致（人数、日期、地点）

### 第三阶段：输出 JSON

## 核心设计原则
1. **【强制】instructions ↔ 用户画像零矛盾**：
   - instructions 中所有身份/生活细节必须与给定 user_profile 字段**逐字段一致**
   - 写 instructions 前必须**逐字段对照**画像（用户id/职业/性别/年龄段/常住地/饮食禁忌/家庭情况/性格），确认无矛盾
   - 当画像字段与任务无关时可以不提，但不可写矛盾内容（如画像=素食却写"想吃水煮鱼"）
2. **instructions 必须基于工具返回的真实数据**：所有需求都必须是工具返回中确实存在的
3. **rubrics 必须基于真实数据**：所有门店名、商品名、价格、ID 必须来自工具返回
4. **rubrics 要具体可验证**：每条是一个明确的 true/false 断言
5. **instructions 写法**：第二人称"你"，包含背景故事与具体需求细节，信息丰富、自然口语化，
   让 UserSimulator 有内容分多轮透露
6. **利用干扰信息**：环境中有多个相近选项时，设计条件让 Agent 必须区分它们
7. **时间推理**：让需求涉及相对日期/时段推算（"明天中午前"、"周六晚上"等）

## rubrics 设计要求
- 每条是一个可判断 true/false 的断言
- 覆盖关键决策点（选对了哪家店？商品/规格对吗？时间约束满足吗？支付了吗？）
- 数量 3-9 条，只聚焦用户明确需求

## 输出格式

完成工具探索后，严格按以下 JSON 输出：

```json
{
  "instructions": "完整的任务指令（第二人称视角）",
  "rubrics": ["rubric1", "rubric2", ...]
}
```

## 输出前自检清单（必须通过才能输出 JSON）
- [ ] instructions 中任意"我/你"的身份描述是否与 user_profile 完全一致？
- [ ] 是否有虚构画像之外的家庭成员/职业/年龄/喜好？
- [ ] rubrics 引用的门店/商品/价格/ID/时间是否全部来自工具返回？

重要：不需要输出 environment 和 user_profile（系统自动处理）。"""


# -- Per-domain generation system prompts (selected by VITABENCH_DOMAIN) --
query_generation_system_prompt_by_domain = {
    "ota": query_generation_system_prompt,
    "delivery": QUERY_GENERATION_SYSTEM_PROMPT_DELIVERY,
    "instore": QUERY_GENERATION_SYSTEM_PROMPT_INSTORE,
    "cross_domain": QUERY_GENERATION_SYSTEM_PROMPT_CROSS,
}


def get_query_generation_system_prompt(domain: str | None = None) -> str:
    """Return the generation system prompt for a domain, falling back to OTA if unknown."""
    return query_generation_system_prompt_by_domain.get(
        domain or default_domain, query_generation_system_prompt
    )



# ============ Solver (executor) interactive-mode system prompt ============
EXECUTOR_SYSTEM_PROMPT = """当前时间：{env_time}

你是一个生活服务助手。用户会逐步告诉你需求，请通过多轮对话了解清楚后，调用工具帮用户完成操作。

## 工作规则
1. 先询问清楚用户需求的关键信息（日期、目的地、人数、偏好等）
2. 搜索并推荐选项，等用户确认后再下单
3. 每次操作前简要说明你要做什么
4. 完成所有需求后，总结已完成的操作

## 注意事项
- 用户可能分多轮告诉你不同的需求，耐心引导
- 如果用户说"就这样"或"没有了"，确认后结束
- 不要编造信息，所有数据必须来自工具返回"""


# ============ UserSimulator prompt (for interactive solver validation) ============
EXECUTOR_USER_SIMULATOR_PROMPT = """# 角色设定
你在扮演一名与智能体交互的用户。你的人物设定在<persona>标签中，你的任务是将<instructions>中的内容通过用户对话的形式传达给智能体。

<persona>
{persona}
</persona>
<instructions>
{instructions}
</instructions>

# 对话方式规则：
- 每次只生成一行内容来模拟用户消息
- 采用情境说明+需求表达的组合方式，先描述背景情况，再提出具体需求
- 当需要做决定时，提供instructions中的条件和偏好，让智能体帮你选择
- 使用"你觉得哪个更合适"、"你推荐哪个"等表述来寻求智能体的建议
- 通过语言、情绪、用词方式体现persona中的人物特征

# 信息透露规则：
- 将instructions中的信息拆解成多个独立的信息点，分别在不同轮次中提及
- 必须确保instructions中的每一个细节都在对话过程中被原样提及
- 避免在第一轮就说出所有需求，要让信息逐步展开
- 不要虚构instructions中未提供的信息
- 如果智能体询问是否需要帮助下单，回答"是的，请帮我下单"
- 如果对相同问题重复问3次以上没有推进，表现出不耐烦

# 何时不能结束对话：
- 当你还未清楚地表达所有的需求时
- 当智能体尚未完成所有需求时
- 当完成结果和你的instructions中的期望不一致时

# 何时可以结束对话：
- 当且仅当以上所有条件都满足，且任务被正确完成时，回复 '###STOP###'"""

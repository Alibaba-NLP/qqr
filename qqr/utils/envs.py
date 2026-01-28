import os

from ..data.text import to_bool

DEBUG = to_bool(os.getenv("DEBUG", "False"))
RETRY_STOP_AFTER_ATTEMPT = int(os.getenv("RETRY_STOP_AFTER_ATTEMPT", 3))
RETRY_WAIT_FIXED = float(os.getenv("RETRY_WAIT_FIXED", 1.0))


# region: LLMs

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# endregion


# region: Tools

# Search
BAILIAN_WEB_SEARCH_API_KEY = os.getenv("BAILIAN_WEB_SEARCH_API_KEY")

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_URL = os.getenv("SERPER_URL", "https://serpapi.com/search")

# Map
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
AMAP_MAPS_API_KEY = os.getenv("AMAP_MAPS_API_KEY")

# endregion

PYTHONPATH = os.getenv("PYTHONPATH")

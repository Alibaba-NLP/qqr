"""
VitaBench Environment Wrapper

将 VitaBench 的模拟工具（Python 函数 + 内存数据库）封装为与 MCPState 兼容的接口，
使其可以直接在 qqr 的 agent_loop 中使用。

每个 VitaBench task 有独立的数据库状态，因此每个 sample 需要独立的 VitaBenchToolState 实例。

Usage:
    tool_state = VitaBenchToolState(task_data, domain="ota")
    tools = tool_state.tools  # OpenAI format tool schemas
    result = await tool_state.call_tool(tool_call_dict)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# VitaBench data directory (set via env or default)
VITABENCH_DATA_DIR = os.environ.get(
    "VITABENCH_DATA_DIR", "/path/to/vitabench/data/vita"  # 官方 VitaBench 仓库 data/vita 目录
)

# Lazy imports to avoid hard dependency when not using VitaBench
_vita_imported = False
_OTATools = None
_OTADB = None
_DeliveryTools = None
_DeliveryDB = None
_InstoreTools = None
_InstoreDB = None


def _ensure_vita_imports():
    """Lazy import VitaBench modules."""
    global _vita_imported, _OTATools, _OTADB
    global _DeliveryTools, _DeliveryDB, _InstoreTools, _InstoreDB
    if _vita_imported:
        return
    try:
        from vita.domains.ota.tools import OTATools
        from vita.domains.ota.data_model import OTADB
        from vita.domains.delivery.tools import DeliveryTools
        from vita.domains.delivery.data_model import DeliveryDB
        from vita.domains.instore.tools import InStoreTools as InstoreTools
        from vita.domains.instore.data_model import InStoreDB as InstoreDB

        _OTATools = OTATools
        _OTADB = OTADB
        _DeliveryTools = DeliveryTools
        _DeliveryDB = DeliveryDB
        _InstoreTools = InstoreTools
        _InstoreDB = InstoreDB
        _vita_imported = True
    except ImportError as e:
        raise ImportError(
            f"VitaBench (vita) package not installed. "
            f"Run: pip install -e /path/to/vitabench\n"
            f"Original error: {e}"
        )


def load_vitabench_tasks(domain: str, data_dir: str = VITABENCH_DATA_DIR) -> list[dict]:
    """Load all tasks for a given domain from VitaBench data files."""
    domain_map = {
        "ota": "ota",
        "delivery": "delivery",
        "instore": "instore",
        "cross_domain": "cross_domain",
    }
    domain_key = domain_map.get(domain, domain)
    tasks_file = Path(data_dir) / "domains" / domain_key / "tasks.json"
    if not tasks_file.exists():
        # Try Chinese version
        tasks_file = Path(data_dir) / "domains" / domain_key / "tasks_zh.json"
    if not tasks_file.exists():
        raise FileNotFoundError(f"VitaBench tasks file not found: {tasks_file}")

    with open(tasks_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    return tasks


def load_vitabench_task_by_id(
    task_id: str, domain: str, data_dir: str = VITABENCH_DATA_DIR
) -> dict:
    """Load a specific task by its ID."""
    tasks = load_vitabench_tasks(domain, data_dir)
    for task in tasks:
        if task.get("id") == task_id:
            return task
    raise ValueError(f"Task {task_id} not found in domain {domain}")


class VitaBenchToolState:
    """
    Per-sample VitaBench environment state.

    Unlike MCPState which is a singleton shared across samples,
    VitaBenchToolState is created per-sample because each VitaBench task
    has its own database state that gets modified by WRITE tools.
    """

    def __init__(self, task_data: dict, domain: str):
        _ensure_vita_imports()
        self.domain = domain
        self.task_data = task_data
        self._tools_cache = None

        if domain == "cross_domain":
            # Cross-domain: 使用官方 get_cross_environment 一步完成 DB + Tools 合并
            from vita.environment.environment import get_cross_environment
            sub_domains = task_data.get("domain", "ota")  # e.g. "delivery,ota,instore"
            env_data = task_data.get("environment", {})
            cross_env = get_cross_environment(sub_domains, env_data)
            self.toolkit = cross_env.tools
            self.db = getattr(self.toolkit, "db", None)
        else:
            self.db = self._create_db(task_data)
            self.toolkit = self._create_toolkit()

    def _create_db(self, task_data: dict):
        """Create the appropriate DB from task environment data."""
        env = task_data.get("environment", {})
        if self.domain == "ota":
            return _OTADB(**env)
        elif self.domain == "delivery":
            return _DeliveryDB(**env)
        elif self.domain == "instore":
            return _InstoreDB(**env)
        else:
            raise ValueError(f"Unknown domain: {self.domain}")

    def _create_toolkit(self):
        """Create the toolkit for the domain."""
        if self.domain == "ota":
            return _OTATools(self.db)
        elif self.domain == "delivery":
            return _DeliveryTools(self.db)
        elif self.domain == "instore":
            return _InstoreTools(self.db)
        else:
            raise ValueError(f"Unknown domain: {self.domain}")

    @property
    def tools(self) -> list[dict]:
        """Return tool schemas in OpenAI ChatCompletionToolParam format."""
        if self._tools_cache is not None:
            return self._tools_cache

        tools = []
        # get_tools() 返回 dict[str, Tool]，每个 Tool 有 openai_schema 属性
        tool_dict = self.toolkit.get_tools()
        for tool in tool_dict.values():
            tools.append(tool.openai_schema)
        self._tools_cache = tools
        return tools

    def get_tool_names(self) -> list[str]:
        """Return list of available tool names."""
        return [t["function"]["name"] for t in self.tools]

    async def call_tool(self, tool_call: dict) -> dict:
        """
        Execute a tool call against the VitaBench environment.

        Args:
            tool_call: OpenAI-format tool call dict with:
                - id: tool call ID
                - function.name: tool name
                - function.arguments: JSON string or dict of arguments

        Returns:
            Tool response message dict with role="tool"
        """
        tc_id = tool_call.get("id", "")
        func = tool_call.get("function", {})
        name = func.get("name", "")
        raw_args = func.get("arguments", "{}")

        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}

        try:
            result = self.toolkit.use_tool(name, **args)
            content = str(result) if result is not None else "OK"
        except Exception as e:
            content = f"Error calling {name}: {str(e)}"
            logger.warning(f"[VitaBench] Tool call failed: {name}({args}) -> {e}")

        # Truncate long results
        if len(content) > 4000:
            content = content[:4000] + "\n... (truncated)"

        return {
            "role": "tool",
            "content": content,
            "tool_call_id": tc_id,
        }

    def get_current_orders(self) -> dict:
        """Get current order state for reward computation."""
        if self.db.orders:
            return {oid: order.model_dump() for oid, order in self.db.orders.items()}
        return {}

    def check_expected_orders(self, expected_states: list[dict]) -> tuple[float, dict]:
        """
        Check if expected orders from evaluation criteria are satisfied.

        Returns:
            (score, details) where score is 0.0-1.0
        """
        if not expected_states:
            return 1.0, {"reason": "no expected states"}

        current_orders = self.get_current_orders()
        total_checks = 0
        passed_checks = 0
        details = []

        for state in expected_states:
            required_orders = state.get("required_orders", [])
            for req_order in required_orders:
                total_checks += 1
                matched = self._match_order(req_order, current_orders)
                if matched:
                    passed_checks += 1
                details.append({
                    "required": req_order,
                    "matched": matched,
                })

        score = passed_checks / total_checks if total_checks > 0 else 1.0
        return score, {"checks": total_checks, "passed": passed_checks, "details": details}

    def _match_order(self, required: dict, current_orders: dict) -> bool:
        """Check if a required order exists in current orders."""
        for order in current_orders.values():
            if self._order_matches(required, order):
                return True
        return False

    def _order_matches(self, required: dict, actual: dict) -> bool:
        """Check if an actual order matches required specifications."""
        # Check order type
        if "order_type" in required and required["order_type"] != actual.get("order_type"):
            return False
        # Check status
        if "status" in required and required["status"] != actual.get("status"):
            return False
        # Check product IDs
        if "product_ids" in required:
            actual_products = actual.get("products", [])
            actual_pids = {p.get("product_id") for p in actual_products}
            for pid in required["product_ids"]:
                if pid not in actual_pids:
                    return False
        return True

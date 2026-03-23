import asyncio
import hashlib
import json
import logging
from typing import Any

from agents.mcp.server import MCPServerSse, MCPServerStdio
from cachetools import TTLCache
from mcp.types import CallToolResult

logger = logging.getLogger(__name__)


class MCPServerCacheableMixin:
    """
    A Mixin that adds tool result caching capabilities and concurrency control to any MCPServer implementation.
    """

    def __init__(
        self,
        blocklist: set[str] | None = None,
        cache_ttl: int = 600,
        cache_maxsize: int = 8192,
        concurrency_limit: int = 64,
        *args,
        **kwargs,
    ):
        """
        Initialize the caching layer.

        Args:
            blocklist: A set of tool names to exclude from caching (e.g., {"send_email", "write_file"}).
            cache_ttl: Time-to-live for cache items in seconds. Defaults to 600.
            cache_maxsize: Maximum number of items to store in the cache. Defaults to 8192.
            concurrency_limit: Max concurrent tool calls allowed for this server. Defaults to 64.
            *args, **kwargs: Arguments passed to the underlying MCPServer implementation.
        """
        super().__init__(*args, **kwargs)

        self._tool_cache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        self._cache_blocklist = blocklist or set()

        self.concurrency_limit = concurrency_limit
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """
        Lazy-initialized semaphore that binds to the current running Event Loop.

        Using lazy initialization prevents "Future attached to a different loop" errors
        when the server instance persists across multiple asyncio.run() calls or
        event loop restarts.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency_limit)
        return self._semaphore

    def _make_cache_key(self, tool_name: str, arguments: dict | None) -> str:
        """
        Generates a deterministic cache key.
        """
        if arguments is None:
            return tool_name

        # Serialize arguments to a JSON string with sorted keys for consistency.
        # ensure_ascii=False ensures logs are readable for non-ASCII characters.
        args_str = json.dumps(arguments, sort_keys=True, ensure_ascii=False)

        full_key = f"{tool_name}:{args_str}"

        # Fallback: Hash extremely long keys (>1KB) to save memory.
        if len(full_key) > 1024:
            return hashlib.md5(full_key.encode("utf-8")).hexdigest()

        return full_key

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None
    ) -> CallToolResult:
        """
        Intercepts the tool call to check the cache before executing.
        """
        if tool_name in self._cache_blocklist:
            async with self.semaphore:
                return await super().call_tool(tool_name, arguments)

        cache_key = self._make_cache_key(tool_name, arguments)
        if cache_key in self._tool_cache:
            return self._tool_cache[cache_key]

        async with self.semaphore:
            if cache_key in self._tool_cache:
                return self._tool_cache[cache_key]

            result: CallToolResult = await super().call_tool(tool_name, arguments)

            # Store only successful results
            if not result.isError:
                self._tool_cache[cache_key] = result

        return result

    async def cleanup(self):
        """
        Override cleanup to reset the semaphore for future event loops.
        """
        await super().cleanup()
        self._semaphore = None


class MCPServerStdioCacheable(MCPServerCacheableMixin, MCPServerStdio):
    """
    Cached and Rate-Limited version of MCPServerStdio.
    """

    pass


class MCPServerSseCacheable(MCPServerCacheableMixin, MCPServerSse):
    """
    Cached and Rate-Limited version of MCPServerStdio.
    """

    pass

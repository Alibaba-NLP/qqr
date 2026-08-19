from mcp.types import Tool as _MCPTool

# mcp v2 renamed camelCase model fields (inputSchema, isError, ...) to
# snake_case. The camelCase constructor aliases still work, but attribute
# access must match the installed major version.
# NOTE: must stay above the .server import below, which imports it back.
_MCP_V2 = "input_schema" in _MCPTool.model_fields

try:
    from agents.mcp import (
        MCPServer,
        MCPServerManager,
        MCPServerSse,
        MCPServerSseParams,
        MCPServerStdio,
        MCPServerStdioParams,
    )

    from .server import MCPServerSseCacheable, MCPServerStdioCacheable
except ImportError:
    pass


__all__ = [
    "MCPServer",
    "MCPServerSse",
    "MCPServerSseCacheable",
    "MCPServerSseParams",
    "MCPServerStdio",
    "MCPServerStdioCacheable",
    "MCPServerStdioParams",
    "MCPServerManager",
]

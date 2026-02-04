"""MCP (Model Context Protocol) 工具模块

提供 MCP 协议支持，允许 Agent 使用外部 MCP 服务器的工具。
"""

from .mcp import MCPTool
from .mcp_manager import MCPToolManager
from .mcp_connection import MCPConnection, create_connection

__all__ = [
    "MCPTool",
    "MCPToolManager",
    "MCPConnection",
    "create_connection",
]

"""MCP 连接管理

提供 MCP (Model Context Protocol) 服务器的连接管理功能。
支持三种传输方式：stdio、SSE、HTTP。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client


class MCPConnection(ABC):
    """MCP 服务器连接基类

    提供与 MCP 服务器通信的统一接口。
    支持异步上下文管理器协议。
    """

    def __init__(self):
        self.session = None
        self._stack = None

    @abstractmethod
    def _create_context(self):
        """创建连接上下文（由子类实现）"""

    async def __aenter__(self):
        """初始化 MCP 服务器连接"""
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        try:
            ctx = self._create_context()
            result = await self._stack.enter_async_context(ctx)

            if len(result) == 2:
                read, write = result
            elif len(result) == 3:
                read, write, _ = result
            else:
                raise ValueError(f"Unexpected context result: {result}")

            session_ctx = ClientSession(read, write)
            self.session = await self._stack.enter_async_context(session_ctx)
            await self.session.initialize()
            return self
        except BaseException:
            await self._stack.__aexit__(None, None, None)
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """清理 MCP 服务器连接资源"""
        if self._stack:
            await self._stack.__aexit__(exc_type, exc_val, exc_tb)
        self.session = None
        self._stack = None

    async def list_tools(self) -> list[dict[str, Any]]:
        """获取 MCP 服务器提供的工具列表

        Returns:
            工具信息列表，每个工具包含 name、description、input_schema
        """
        response = await self.session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """调用 MCP 服务器上的工具

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果
        """
        result = await self.session.call_tool(tool_name, arguments=arguments)
        return result.content


class MCPConnectionStdio(MCPConnection):
    """使用标准输入输出的 MCP 连接

    通过启动子进程并使用 stdio 进行通信。
    """

    def __init__(self, command: str, args: list[str] = None, env: dict[str, str] = None):
        """初始化 stdio 连接

        Args:
            command: 启动命令
            args: 命令参数
            env: 环境变量
        """
        super().__init__()
        self.command = command
        self.args = args or []
        self.env = env

    def _create_context(self):
        return stdio_client(
            StdioServerParameters(command=self.command, args=self.args, env=self.env)
        )


class MCPConnectionSSE(MCPConnection):
    """使用 Server-Sent Events 的 MCP 连接"""

    def __init__(self, url: str, headers: dict[str, str] = None):
        """初始化 SSE 连接

        Args:
            url: 服务器 URL
            headers: HTTP 请求头
        """
        super().__init__()
        self.url = url
        self.headers = headers or {}

    def _create_context(self):
        return sse_client(url=self.url, headers=self.headers)


class MCPConnectionHTTP(MCPConnection):
    """使用 Streamable HTTP 的 MCP 连接"""

    def __init__(self, url: str, headers: dict[str, str] = None):
        """初始化 HTTP 连接

        Args:
            url: 服务器 URL
            headers: HTTP 请求头
        """
        super().__init__()
        self.url = url
        self.headers = headers or {}

    def _create_context(self):
        return streamablehttp_client(url=self.url, headers=self.headers)


def create_connection(
    transport: str,
    command: str = None,
    args: list[str] = None,
    env: dict[str, str] = None,
    url: str = None,
    headers: dict[str, str] = None,
) -> MCPConnection:
    """工厂函数：创建适当的 MCP 连接

    Args:
        transport: 传输方式（"stdio"、"sse" 或 "http"）
        command: 启动命令（仅 stdio）
        args: 命令参数（仅 stdio）
        env: 环境变量（仅 stdio）
        url: 服务器 URL（仅 sse 和 http）
        headers: HTTP 请求头（仅 sse 和 http）

    Returns:
        MCPConnection 实例

    Raises:
        ValueError: 传输方式不支持或缺少必需参数
    """
    transport = transport.lower()

    if transport == "stdio":
        if not command:
            raise ValueError("Command is required for stdio transport")
        return MCPConnectionStdio(command=command, args=args, env=env)

    elif transport == "sse":
        if not url:
            raise ValueError("URL is required for sse transport")
        return MCPConnectionSSE(url=url, headers=headers)

    elif transport in ["http", "streamable_http", "streamable-http"]:
        if not url:
            raise ValueError("URL is required for http transport")
        return MCPConnectionHTTP(url=url, headers=headers)

    else:
        raise ValueError(f"Unsupported transport type: {transport}. Use 'stdio', 'sse', or 'http'")

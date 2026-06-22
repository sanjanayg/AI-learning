# app/mcp_client.py
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

import sys
import traceback
import os
from dotenv import load_dotenv
load_dotenv()


DATABASE_URL = os.getenv("DB_URL")
RESEND_API_KEY=os.getenv("SMTP_PASSWORD")
SMTP_EMAIL=os.getenv("SMTP_EMAIL")
SERVERS = {
    "weather": StdioServerParameters(command="python", args=["mcp/weather_server.py"]),
    "sql": StdioServerParameters(
        command="python", args=["mcp/sql_server.py"],
        env={
        "DATABASE_URL": DATABASE_URL
    }
    ),
    "email": StdioServerParameters(
        command="python", args=["mcp/email_server.py"],
        env={"RESEND_API_KEY": RESEND_API_KEY, "EMAIL_FROM": SMTP_EMAIL}
    ),
    "document_reader": StdioServerParameters(command="python", args=["mcp/document_reader_server.py"]),
}


class MCPClientManager:
    def __init__(self):
        self.sessions: dict[str, ClientSession] = {}
        self.tool_to_server: dict[str, str] = {}   # maps tool name -> which server owns it
        self._stack = AsyncExitStack()


    async def connect_all(self):
        self._cached_tool_schemas = {}

        for server_name, params in SERVERS.items():
            try:
                print(f"[MCP] Connecting to {server_name}", file=sys.stderr)

                read, write = await self._stack.enter_async_context(
                    stdio_client(params)
                )

                session = await self._stack.enter_async_context(
                    ClientSession(read, write)
                )

                await session.initialize()

                self.sessions[server_name] = session

                tools = await session.list_tools()

                for tool in tools.tools:
                    self.tool_to_server[tool.name] = server_name
                    self._cached_tool_schemas[tool.name] = tool
                print(f"[MCP] Connected: {server_name}", file=sys.stderr)

            except Exception as e:
                print(f"[MCP] Failed server: {server_name}", file=sys.stderr)
                print(f"[MCP] Error: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                raise

    def get_groq_tools(self) -> list[dict]:
        """Flatten all MCP servers' tools into OpenAI/Groq function-calling format."""
        tools = []
        for tool_name, schema in self._cached_tool_schemas.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": schema.description,
                    "parameters": schema.inputSchema,  
                }
            })
        return tools

    async def call_tool(self, tool_name: str, tool_input: dict):
        server_name = self.tool_to_server[tool_name]   # dispatch by lookup, not if-else
        session = self.sessions[server_name]
        result = await session.call_tool(tool_name, tool_input)
        return result.content

    async def close(self):
        await self._stack.aclose()
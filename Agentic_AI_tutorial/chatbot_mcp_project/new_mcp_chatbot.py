from __future__ import annotations

import asyncio
import json
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("BASE_URL") or None
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.1")


class ToolDefinition(TypedDict):
    type: str
    function: dict


class MCPChatBot:
    """MCP-enabled chatbot that can be used by a CLI or web API."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

        self.config_path = config_path or BASE_DIR / "server_config.json"
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.openai_client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_session: Dict[str, ClientSession] = {}
        self.connected_servers: Dict[str, List[str]] = {}
        self._connected = False

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Start one configured MCP server and discover its tools."""
        server_params = StdioServerParameters(**server_config)
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport

        session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self.sessions.append(session)

        response = await session.list_tools()
        tool_names: List[str] = []

        for tool in response.tools:
            tool_names.append(tool.name)
            self.tool_to_session[tool.name] = session
            self.available_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                }
            )

        self.connected_servers[server_name] = tool_names

    async def connect_to_servers(self) -> None:
        """Connect to every MCP server from server_config.json."""
        if self._connected:
            return

        with self.config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        servers = data.get("mcpServers", {})
        if not servers:
            raise RuntimeError("No MCP servers are defined in server_config.json.")

        failures: List[str] = []
        for server_name, server_config in servers.items():
            try:
                await self.connect_to_server(server_name, server_config)
            except Exception as exc:
                failures.append(f"{server_name}: {exc}")

        self._connected = True
        if failures and not self.connected_servers:
            raise RuntimeError("Could not connect to any MCP server: " + "; ".join(failures))

    def _system_prompt(self) -> str:
        tool_info = "\n".join(
            f"- {tool['function']['name']}: {tool['function']['description']}"
            for tool in self.available_tools
        )
        return (
            "You are a helpful research assistant. "
            "Use the available MCP tools when they are useful. "
            "Do not claim to have used a tool unless you actually called it. "
            "Give clear, well-structured answers and include source URLs returned by tools.\n\n"
            f"Available tools:\n{tool_info or '- No tools are currently connected.'}"
        )

    async def process_query(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Process one message and return answer plus tool activity."""
        if not self._connected:
            await self.connect_to_servers()

        clean_query = query.strip()
        if not clean_query:
            raise ValueError("The prompt cannot be empty.")

        messages: List[Any] = [{"role": "system", "content": self._system_prompt()}]
        for item in history or []:
            role = item.get("role")
            content = item.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": clean_query})

        tool_activity: List[Dict[str, Any]] = []

        while True:
            request_args: Dict[str, Any] = {
                "model": MODEL_NAME,
                "messages": messages,
            }
            if self.available_tools:
                request_args["tools"] = self.available_tools

            response = self.openai_client.chat.completions.create(**request_args)
            message = response.choices[0].message

            if not message.tool_calls:
                return {
                    "answer": message.content or "I could not generate a response.",
                    "tool_activity": tool_activity,
                }

            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    tool_args = {}

                activity: Dict[str, Any] = {
                    "tool": tool_name,
                    "arguments": tool_args,
                    "status": "success",
                }

                session = self.tool_to_session.get(tool_name)
                if session is None:
                    result_text = f"Tool '{tool_name}' is not connected."
                    activity["status"] = "error"
                else:
                    try:
                        result = await session.call_tool(tool_name, arguments=tool_args)
                        text_parts = [
                            block.text
                            for block in result.content
                            if hasattr(block, "text")
                        ]
                        result_text = "\n".join(text_parts) or "Tool completed without text output."
                    except Exception as exc:
                        result_text = f"Tool execution failed: {exc}"
                        activity["status"] = "error"

                activity["preview"] = result_text[:500]
                tool_activity.append(activity)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    }
                )

    async def cleanup(self) -> None:
        await self.exit_stack.aclose()
        self._connected = False


# Backward-compatible alias for code importing the original class name.
MCP_ChatBot = MCPChatBot


async def main() -> None:
    chatbot = MCPChatBot()
    try:
        await chatbot.connect_to_servers()
        print("MCP Chatbot started. Type 'quit' to exit.")
        history: List[Dict[str, str]] = []
        while True:
            query = await asyncio.to_thread(input, "\nYou: ")
            if query.strip().lower() == "quit":
                break
            result = await chatbot.process_query(query, history)
            print(f"\nAssistant: {result['answer']}")
            history.extend(
                [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": result["answer"]},
                ]
            )
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from new_mcp_chatbot import MCPChatBot

BASE_DIR = Path(__file__).resolve().parent
chatbot: MCPChatBot | None = None
conversations: Dict[str, List[Dict[str, str]]] = {}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    session_id: str | None = None


class ResetRequest(BaseModel):
    session_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot
    chatbot = MCPChatBot()
    await chatbot.connect_to_servers()
    try:
        yield
    finally:
        await chatbot.cleanup()


app = FastAPI(title="MCP Research Assistant", lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/status")
async def status():
    if chatbot is None:
        return {"ready": False, "servers": {}, "tools": []}
    return {
        "ready": True,
        "servers": chatbot.connected_servers,
        "tools": [tool["function"]["name"] for tool in chatbot.available_tools],
    }


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request):
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot is not ready.")

    session_id = payload.session_id or secrets.token_urlsafe(18)
    history = conversations.setdefault(session_id, [])

    try:
        result = await chatbot.process_query(payload.message, history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    history.extend(
        [
            {"role": "user", "content": payload.message},
            {"role": "assistant", "content": result["answer"]},
        ]
    )
    # Bound in-memory history to the latest 40 messages.
    conversations[session_id] = history[-40:]

    return {
        "session_id": session_id,
        "answer": result["answer"],
        "tool_activity": result["tool_activity"],
    }


@app.post("/api/reset")
async def reset(payload: ResetRequest):
    conversations.pop(payload.session_id, None)
    return {"ok": True}

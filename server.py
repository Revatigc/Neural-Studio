"""Local FastAPI bridge between Neural Studio and an Ollama model."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
LOCAL_HTTP = build_opener(ProxyHandler({}))
app = FastAPI(title="Neural Studio Inference", version="1.0.0")
app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=6000)
    language: str = Field(default="en", max_length=12)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=12)


class ChatResponse(BaseModel):
    answer: str
    model: str
    language: str


def ollama_request(path: str, payload: dict | None = None, timeout: int = 10) -> tuple[int, dict | str]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        f"{OLLAMA_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    try:
        with LOCAL_HTTP.open(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        try:
            return error.code, json.loads(error.read().decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return error.code, str(error)
    except URLError as error:
        raise ConnectionError(str(error.reason)) from error


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/style.css")
async def stylesheet() -> FileResponse:
    return FileResponse(ROOT / "style.css", media_type="text/css")


@app.get("/app.js")
async def javascript() -> FileResponse:
    return FileResponse(ROOT / "app.js", media_type="application/javascript")


@app.get("/api/health")
async def health() -> dict[str, object]:
    try:
        status, body = await asyncio.to_thread(ollama_request, "/api/tags", None, 3)
        if status >= 400 or not isinstance(body, dict):
            raise ConnectionError("Ollama tags request failed")
        available = [item.get("name", "") for item in body.get("models", [])]
        model_ready = any(OLLAMA_MODEL == name or OLLAMA_MODEL + ":latest" == name for name in available)
        return {"api": "ready", "provider": "ollama", "model": OLLAMA_MODEL, "model_ready": model_ready}
    except (ConnectionError, ValueError, OSError):
        return {"api": "ready", "provider": "ollama", "model": OLLAMA_MODEL, "model_ready": False, "detail": "Start Ollama and pull the configured model."}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    language = request.language.strip() or "en"
    system_prompt = (
        "You are KannadaMind, a helpful general-purpose assistant. Answer the user's actual question "
        f"in the requested language or locale ({language}). Be clear and concise. If you are uncertain, "
        "say so instead of inventing facts. For high-stakes medical, legal, financial, or safety questions, "
        "give general information, encourage consulting a qualified professional, and do not claim certainty. "
        "For campus policies, distinguish general guidance from verified official policy."
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for item in request.history[-10:]:
        if item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str):
            messages.append({"role": item["role"], "content": item["content"][:6000]})
    messages.append({"role": "user", "content": request.message})
    try:
        status, body = await asyncio.to_thread(
            ollama_request,
            "/api/chat",
            {"model": OLLAMA_MODEL, "messages": messages, "stream": False,
             "options": {"temperature": 0.4, "num_predict": 512}},
            180,
        )
    except ConnectionError as error:
        raise HTTPException(status_code=503, detail="Ollama is not running. Start Ollama, then retry.") from error
    except TimeoutError as error:
        raise HTTPException(status_code=504, detail="The model took too long to respond. Try a shorter question.") from error
    if status == 404:
        raise HTTPException(status_code=503, detail=f"Model {OLLAMA_MODEL} is not installed. Run: ollama pull {OLLAMA_MODEL}")
    if status >= 400:
        raise HTTPException(status_code=502, detail=f"Ollama returned an error: {str(body)[:300]}")
    try:
        answer = body["message"]["content"].strip() if isinstance(body, dict) else ""
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(status_code=502, detail="Ollama returned an unreadable response.") from error
    if not answer:
        raise HTTPException(status_code=502, detail="The model returned an empty answer. Please retry.")
    return ChatResponse(answer=answer, model=OLLAMA_MODEL, language=language)


@app.get("/api/model")
async def model_info() -> dict[str, str]:
    return {"name": OLLAMA_MODEL, "provider": "Ollama local runtime", "mode": "base model until you train and configure an adapter"}

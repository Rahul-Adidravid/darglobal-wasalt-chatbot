from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import MODELS, OPENROUTER_API_KEY
from .rag import Retriever, retrieval_query, stream_answer

STATIC = Path(__file__).parent / "static"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["retriever"] = Retriever()
    yield


app = FastAPI(title="DarGlobal & Wasalt Property Chatbot", lifespan=lifespan)


class Msg(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=12000)


class ChatIn(BaseModel):
    messages: list[Msg] = Field(min_length=1, max_length=30)


# simple per-IP rate limit to protect the free OpenRouter quota
_hits: dict[str, deque] = defaultdict(deque)
RATE = (12, 60)  # 12 requests / 60 s


def _limited(ip: str) -> bool:
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > RATE[1]:
        q.popleft()
    if len(q) >= RATE[0]:
        return True
    q.append(now)
    return False


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    r: Retriever | None = state.get("retriever")
    counts: dict[str, int] = {}
    if r:
        for c in r.chunks:
            key = f"{c['source']}:{c['kind']}"
            counts[key] = counts.get(key, 0) + 1
    return {"ok": r is not None, "llm_configured": bool(OPENROUTER_API_KEY), "models": MODELS, "chunks": counts}


@app.get("/api/search")
def search(q: str, k: int = 8):
    hits, f = state["retriever"].search(q, k=min(k, 20))
    return {"filters": f.__dict__, "results": [{k2: h[k2] for k2 in ("source", "kind", "title", "url", "text")} for h in hits]}


@app.post("/api/chat")
async def chat(body: ChatIn, request: Request):
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
    if _limited(ip):
        raise HTTPException(429, "Too many requests, please wait a minute.")
    msgs = [m.model_dump() for m in body.messages]
    if msgs[-1]["role"] != "user":
        raise HTTPException(400, "last message must be from the user")
    if len(msgs[-1]["content"]) > 2000:
        raise HTTPException(413, "Question too long (max 2000 characters).")
    for m in msgs[:-1]:  # keep history small: older answers are only context
        m["content"] = m["content"][:3000]
    hits, _ = state["retriever"].search(retrieval_query(msgs))
    sources = [{"n": i + 1, "source": h["source"], "kind": h["kind"], "title": h["title"], "url": h["url"]} for i, h in enumerate(hits)]

    async def gen():
        yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"
        async for tok in stream_answer(msgs, hits):
            yield f"data: {json.dumps(tok, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

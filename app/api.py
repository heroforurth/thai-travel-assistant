"""HTTP layer: a JSON endpoint and an SSE endpoint that streams the agent's steps.

Run:  uvicorn app.api:app --port 8000      then open http://localhost:8000
"""
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool

from app import agent, config
from app.retrieve import Index
from app.schemas import AgentResult, AskRequest

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["index"] = Index.build()
    yield


app = FastAPI(title="Thai Travel Assistant", lifespan=lifespan)


@app.get("/")
def home() -> FileResponse:
    return FileResponse(config.ROOT / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    index: Index = STATE["index"]
    return {"ok": True, "chunks": len(index.chunks), "destinations": config.DESTINATIONS,
            "model": config.CHAT_MODEL, "retriever": config.RETRIEVER, "min_score": config.MIN_SCORE}


@app.post("/api/ask", response_model=AgentResult)
async def ask(req: AskRequest) -> AgentResult:
    return await run_in_threadpool(agent.run, req.question, STATE["index"])


@app.post("/api/chat")
def chat(req: AskRequest) -> StreamingResponse:
    """Server-Sent Events: one `data:` line per agent event (action, observation, final)."""
    def stream():
        try:
            for event in agent.run_events(req.question, STATE["index"]):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:  # surface proxy errors to the UI instead of a dead stream
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)[:300]})}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")

"""FanCare Assistant API. OpenAPI docs at /docs (Swagger) and /openapi.json; chat UI at /."""
import json
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles

from app.agent import Agent, AgentResult, AgentUnavailable
from app.config import Settings
from app.documents import CONTENT_TYPES, MAX_UPLOAD_BYTES, DocumentExtractor, UnreadableDocument
from app.schemas import (
    ChatRequest,
    ChatResponse,
    CitationOut,
    ErrorResponse,
    HealthResponse,
    HistoryResponse,
    ToolCallOut,
    UploadResponse,
)

STATIC_DIR = Path(__file__).parent / "static"
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class RateLimiter:
    """Sliding one-minute window per API key."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def retry_after(self, key: str) -> int:
        """Record a hit; return 0 if allowed, else seconds until a slot frees up."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return max(1, int(60 - (now - hits[0])) + 1)
            hits.append(now)
            return 0


def authorize(request: Request, api_key: str | None = Depends(api_key_header)) -> str:
    expected = request.app.state.settings.app_api_key
    if not api_key or not secrets.compare_digest(api_key, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
    wait = request.app.state.limiter.retry_after(api_key)
    if wait:
        raise HTTPException(status_code=429, detail="Rate limit exceeded", headers={"Retry-After": str(wait)})
    return api_key


def to_response(session_id: str, result: AgentResult) -> ChatResponse:
    return ChatResponse(
        session_id=session_id,
        trace_id=result.trace.trace_id,
        answer=result.answer,
        citations=[CitationOut(**c.__dict__) for c in result.citations],
        tool_calls=[ToolCallOut(name=t.name, arguments=t.arguments, status=t.status, error=t.error) for t in result.tool_calls],
    )


def create_app(settings: Settings | None = None, agent: Agent | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = settings or Settings()
        app.state.settings = s
        app.state.agent = agent or Agent(s)
        app.state.limiter = RateLimiter(s.rate_limit_per_min)
        app.state.extractor = DocumentExtractor()
        yield

    app = FastAPI(
        title="FanCare Assistant",
        version="0.2.0",
        description="Support assistant for Corvane smart ceiling fans (system under test).",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    errors = {
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    }

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request) -> HealthResponse:
        s: Settings = request.app.state.settings
        return HealthResponse(
            status="ok", prompt_version=s.prompt_version, provider=s.llm_provider,
            model=s.active_model, retriever_mode=s.retriever_mode, rerank=s.rerank,
        )

    @app.post("/chat", response_model=ChatResponse, tags=["chat"],
              responses={**errors, 503: {"model": ErrorResponse, "description": "LLM provider unavailable"}})
    def chat(body: ChatRequest, request: Request, _: str = Depends(authorize)):
        try:
            result = request.app.state.agent.chat(body.session_id, body.message)
        except AgentUnavailable as exc:
            return JSONResponse(status_code=503, content={"detail": str(exc), "trace_id": exc.trace_id})
        return to_response(body.session_id, result)

    @app.post("/chat/stream", tags=["chat"], responses={**errors, 200: {"content": {"text/event-stream": {}}}})
    def chat_stream(body: ChatRequest, request: Request, _: str = Depends(authorize)):
        """Server-sent events: `delta` events carry answer text, then one `done` (citations, tool calls) or `error`.

        Guardrails need the complete answer before anything is released, so the
        answer is generated and checked first, then streamed in small chunks.
        """
        agent: Agent = request.app.state.agent

        def events():
            try:
                result = agent.chat(body.session_id, body.message)
            except AgentUnavailable as exc:
                yield f"event: error\ndata: {json.dumps({'detail': str(exc), 'trace_id': exc.trace_id})}\n\n"
                return
            words = result.answer.split(" ")
            for i in range(0, len(words), 3):
                chunk = " ".join(words[i : i + 3]) + (" " if i + 3 < len(words) else "")
                yield f"event: delta\ndata: {json.dumps({'text': chunk})}\n\n"
                time.sleep(0.02)
            yield f"event: done\ndata: {to_response(body.session_id, result).model_dump_json()}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/sessions/{session_id}/history", response_model=HistoryResponse, tags=["chat"], responses=errors)
    def history(session_id: str, request: Request, _: str = Depends(authorize)) -> HistoryResponse:
        if not SESSION_ID_RE.match(session_id):
            raise HTTPException(status_code=422, detail="invalid session_id")
        turns = request.app.state.agent.memory.turns(session_id)
        return HistoryResponse(session_id=session_id, turns=[{"user": u, "assistant": a} for u, a in turns])

    @app.post("/upload", response_model=UploadResponse, tags=["documents"], responses={
        **errors,
        400: {"model": ErrorResponse, "description": "Empty file"},
        413: {"model": ErrorResponse, "description": "File larger than 5 MB"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
        422: {"model": ErrorResponse, "description": "File could not be read"},
    })
    def upload(
        request: Request,
        file: UploadFile = File(..., description="Warranty card or invoice: PDF, PNG or JPEG, max 5 MB"),
        session_id: str | None = Form(default=None, description="Attach the extracted fields to this chat session"),
        _: str = Depends(authorize),
    ) -> UploadResponse:
        if file.content_type not in CONTENT_TYPES:
            raise HTTPException(status_code=415, detail=f"Unsupported file type '{file.content_type}'; use PDF, PNG or JPEG")
        if session_id is not None and not SESSION_ID_RE.match(session_id):
            raise HTTPException(status_code=422, detail="invalid session_id")
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File larger than 5 MB")
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")
        try:
            result = request.app.state.extractor.extract(data, file.content_type)
        except UnreadableDocument as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if session_id:
            request.app.state.agent.attach_document(session_id, result)
        return UploadResponse(status=result.status, document_type=result.document_type, fields=result.fields,
                              missing=result.missing, warnings=result.warnings)

    return app


app = create_app()

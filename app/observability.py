"""Request tracing to SQLite, plus optional export to Langfuse.

Schema (foreign keys are declared but, as is SQLite's default, not enforced; the
data-validation suite checks for orphans instead):

    sessions(session_id PK, created_at)
    traces(trace_id PK, session_id -> sessions, ...one row per /chat call...)
    tool_calls(id PK, trace_id -> traces, seq, name, arguments, status, ...)
    guardrail_events(id PK, trace_id -> traces, stage, action, category, detail)

All text written here has already been through PII redaction.
"""
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

from app.config import Settings
from app.guardrails import Verdict
from app.tools import ToolRecord

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,              -- ok | blocked | error
    prompt_version TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    retriever_mode TEXT NOT NULL,
    user_message TEXT NOT NULL,
    answer TEXT,
    retrieved_chunk_ids TEXT NOT NULL, -- JSON list
    cited_chunk_ids TEXT NOT NULL,     -- JSON list
    latency_ms INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    input_verdict TEXT NOT NULL,       -- allow | redact | block
    output_verdict TEXT,               -- allow | redact | block (NULL if no answer was generated)
    error TEXT
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL REFERENCES traces(trace_id),
    seq INTEGER NOT NULL,
    name TEXT NOT NULL,
    arguments TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS guardrail_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL REFERENCES traces(trace_id),
    stage TEXT NOT NULL,
    action TEXT NOT NULL,
    category TEXT,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_trace ON tool_calls(trace_id);
CREATE INDEX IF NOT EXISTS idx_guardrail_trace ON guardrail_events(trace_id);
"""


@dataclass
class TraceRecord:
    trace_id: str
    session_id: str
    status: str
    prompt_version: str
    retriever_mode: str
    user_message: str
    input_verdict: str
    latency_ms: int
    provider: str | None = None
    model: str | None = None
    answer: str | None = None
    output_verdict: str | None = None
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    tool_calls: list[ToolRecord] = field(default_factory=list)
    guardrail_events: list[Verdict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class TraceStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=10)

    def write(self, t: TraceRecord) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO sessions VALUES (?, ?)", (t.session_id, t.created_at))
            conn.execute(
                "INSERT INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    t.trace_id, t.session_id, t.created_at, t.status, t.prompt_version, t.provider, t.model,
                    t.retriever_mode, t.user_message, t.answer, json.dumps(t.retrieved_chunk_ids),
                    json.dumps(t.cited_chunk_ids), t.latency_ms, t.prompt_tokens, t.completion_tokens,
                    t.cost_usd, t.input_verdict, t.output_verdict, t.error,
                ),
            )
            conn.executemany(
                "INSERT INTO tool_calls (trace_id, seq, name, arguments, status, latency_ms, error) VALUES (?,?,?,?,?,?,?)",
                [(t.trace_id, i, c.name, json.dumps(c.arguments), c.status, c.latency_ms, c.error)
                 for i, c in enumerate(t.tool_calls, start=1)],
            )
            conn.executemany(
                "INSERT INTO guardrail_events (trace_id, stage, action, category, detail) VALUES (?,?,?,?,?)",
                [(t.trace_id, v.stage, v.action, v.category, v.detail) for v in t.guardrail_events],
            )


class LangfuseExporter:
    """Sends a trace to Langfuse's public ingestion API. Only active when keys are set."""

    def __init__(self, settings: Settings):
        self.enabled = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
        self.url = settings.langfuse_host.rstrip("/") + "/api/public/ingestion"
        self.auth = (settings.langfuse_public_key, settings.langfuse_secret_key)

    def export(self, t: TraceRecord) -> None:
        if self.enabled:
            threading.Thread(target=self._send, args=(t,), daemon=True).start()

    def _send(self, t: TraceRecord) -> None:
        body = {
            "id": t.trace_id,
            "name": "chat",
            "sessionId": t.session_id,
            "timestamp": t.created_at,
            "input": t.user_message,
            "output": t.answer,
            "tags": [t.status, f"prompt:{t.prompt_version}"],
            "metadata": {
                "model": t.model, "provider": t.provider, "retriever_mode": t.retriever_mode,
                "latency_ms": t.latency_ms, "cost_usd": t.cost_usd,
                "prompt_tokens": t.prompt_tokens, "completion_tokens": t.completion_tokens,
                "retrieved_chunk_ids": t.retrieved_chunk_ids,
                "tool_calls": [{"name": c.name, "status": c.status} for c in t.tool_calls],
                "guardrails": [v.__dict__ for v in t.guardrail_events],
            },
        }
        event = {"id": f"{t.trace_id}-create", "timestamp": t.created_at, "type": "trace-create", "body": body}
        try:
            httpx.post(self.url, json={"batch": [event]}, auth=self.auth, timeout=5).raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("Langfuse export failed: %s", exc)

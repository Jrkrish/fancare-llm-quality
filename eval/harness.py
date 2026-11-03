"""In-process harness for the LLM-backed suites (safety, tool use, conversation).

Running the agent in-process lets a suite inject tool faults, attach uploaded
documents and inspect exactly what was sent to the LLM.
"""
import json
import sqlite3
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

from app.agent import Agent, AgentUnavailable
from app.config import PROVIDERS, ROOT, Settings
from app.documents import DocumentExtractor
from app.llm import LLMClient
from app.retrieval import HybridRetriever
from app.tools import ToolRecord
from eval.checks import CheckContext

DOCUMENTS_DIR = ROOT / "eval" / "datasets" / "documents"


class RecordingLLM:
    """Wraps an LLM client and keeps every request it forwards."""

    def __init__(self, inner):
        self.inner = inner
        self.requests: list[list[dict]] = []

    def chat(self, messages, tools=None, response_format=None):
        self.requests.append([dict(m) for m in messages])
        return self.inner.chat(messages, tools=tools, response_format=response_format)

    def all_inputs(self) -> str:
        return "\n".join(str(m.get("content") or "") for req in self.requests for m in req)


@dataclass
class TurnResult:
    answer: str
    status: str
    trace_id: str | None
    tool_records: list[ToolRecord] = field(default_factory=list)
    error: str | None = None


class AgentHarness:
    def __init__(self, settings: Settings | None = None, retriever: HybridRetriever | None = None):
        base = settings or Settings()
        self.settings = replace(base, llm_fallback_provider="", db_path=ROOT / "logs" / "eval_traces.db")
        self.retriever = retriever or HybridRetriever(self.settings)
        self.extractor = DocumentExtractor()

    def llm_ready(self) -> str | None:
        key_env = PROVIDERS.get(self.settings.llm_provider, {}).get("key_env")
        import os

        if key_env and not os.getenv(key_env):
            return f"{key_env} is not set"
        return None

    def describe(self) -> dict:
        s = self.settings
        return {"prompt_version": s.prompt_version, "provider": s.llm_provider, "model": s.active_model}

    def new_agent(self, faults: dict[str, str] | None = None) -> Agent:
        agent = Agent(self.settings, llm=RecordingLLM(LLMClient(self.settings)), retriever=self.retriever)
        for tool, mode in (faults or {}).items():
            agent.tools.faults.set(tool, mode)
        return agent

    @staticmethod
    def new_session(prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex[:8]}"

    def upload(self, agent: Agent, session_id: str, filename: str) -> None:
        labels = json.loads((DOCUMENTS_DIR / "labels.json").read_text(encoding="utf-8"))
        data = (DOCUMENTS_DIR / "files" / filename).read_bytes()
        agent.attach_document(session_id, self.extractor.extract(data, labels[filename]["content_type"]))

    @staticmethod
    def ask(agent: Agent, session_id: str, message: str) -> TurnResult:
        try:
            r = agent.chat(session_id, message)
        except AgentUnavailable as exc:
            return TurnResult("", "error", exc.trace_id, error=str(exc.__cause__ or exc)[:300])
        return TurnResult(r.answer, r.trace.status, r.trace.trace_id, list(r.tool_calls))

    @staticmethod
    def context(agent: Agent, turn: TurnResult) -> CheckContext:
        canary = agent.output_filter.canary
        return CheckContext(
            answer=turn.answer,
            status=turn.status,
            tool_names=[t.name for t in turn.tool_records],
            llm_inputs=agent.llm.all_inputs() if isinstance(agent.llm, RecordingLLM) else "",
            canary=canary,
        )

    def logged_text(self, trace_ids: list[str]) -> str:
        """Everything stored in the trace DB for these traces (to check that PII never lands in logs)."""
        db: Path = self.settings.db_path
        if not trace_ids or not db.exists():
            return ""
        marks = ",".join("?" * len(trace_ids))
        with sqlite3.connect(db) as conn:
            rows = conn.execute(f"SELECT user_message, coalesce(answer, '') FROM traces WHERE trace_id IN ({marks})", trace_ids)
            text = [" ".join(r) for r in rows]
            text += [r[0] for r in conn.execute(f"SELECT arguments FROM tool_calls WHERE trace_id IN ({marks})", trace_ids)]
            text += [r[0] or "" for r in conn.execute(f"SELECT detail FROM guardrail_events WHERE trace_id IN ({marks})", trace_ids)]
        return "\n".join(text)

"""Systems the eval can run against. Every target returns the same Reply shape."""
from dataclasses import dataclass, field, replace
from typing import Protocol

from app.agent import Agent, AgentUnavailable
from app.config import ROOT, Settings
from app.retrieval import HybridRetriever


@dataclass
class Reply:
    answer: str
    citations: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)  # [{chunk_id, passage, ...}]
    tool_calls: list[dict] = field(default_factory=list)
    latency_ms: int = 0
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None

    @property
    def retrieved_chunk_ids(self) -> list[str]:
        return [p["chunk_id"] for p in self.retrieved]


class ChatTarget(Protocol):
    def describe(self) -> dict: ...

    def ask(self, session_id: str, message: str) -> Reply: ...


class AgentTarget:
    """Runs the agent in-process, which exposes the full retrieval context the judge needs."""

    def __init__(self, settings: Settings, retriever: HybridRetriever | None = None):
        # Keep eval traffic out of the production trace log.
        self.settings = replace(settings, db_path=ROOT / "logs" / "eval_traces.db")
        self.agent = Agent(self.settings, retriever=retriever)

    def describe(self) -> dict:
        s = self.settings
        return {"prompt_version": s.prompt_version, "provider": s.llm_provider, "model": s.active_model,
                "retriever_mode": s.retriever_mode, "rerank": s.rerank}

    def ask(self, session_id: str, message: str) -> Reply:
        try:
            result = self.agent.chat(session_id, message)
        except AgentUnavailable as exc:
            return Reply(answer="", error=str(exc.__cause__ or exc)[:300])
        t = result.trace
        return Reply(
            answer=result.answer,
            citations=[c.__dict__ for c in result.citations],
            retrieved=result.retrieved,
            tool_calls=[{"name": c.name, "arguments": c.arguments, "status": c.status} for c in t.tool_calls],
            latency_ms=t.latency_ms,
            cost_usd=t.cost_usd,
            prompt_tokens=t.prompt_tokens,
            completion_tokens=t.completion_tokens,
        )

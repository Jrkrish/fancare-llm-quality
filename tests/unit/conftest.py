"""Unit-test fixtures. No network and no real LLM: the agent runs against a scripted FakeLLM."""
import json
import uuid
from dataclasses import replace

import pytest

from app.agent import Agent
from app.config import Settings
from app.llm import LLMError, LLMResponse
from app.observability import TraceStore
from app.retrieval import HybridRetriever


class FakeLLM:
    """Returns scripted responses in order and records every request it receives."""

    def __init__(self, script: list):
        self.script = list(script)
        self.requests: list[dict] = []

    def chat(self, messages, tools=None):
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self.script:
            raise AssertionError("FakeLLM script exhausted")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def answer(text: str) -> LLMResponse:
    return LLMResponse(content=text, tool_calls=[], provider="fake", model="fake-model",
                       usage={"prompt_tokens": 100, "completion_tokens": 20})


def tool_call(name: str, **arguments) -> LLMResponse:
    call = {"id": f"call_{uuid.uuid4().hex[:6]}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}
    return LLMResponse(content="", tool_calls=[call], provider="fake", model="fake-model",
                       usage={"prompt_tokens": 80, "completion_tokens": 10})


__all__ = ["FakeLLM", "answer", "tool_call", "LLMError"]


@pytest.fixture(scope="session")
def base_settings() -> Settings:
    return Settings(fault_injection="", tool_timeout_s=2.0, prompt_version="v1", retriever_mode="hybrid", rerank=True)


@pytest.fixture(scope="session")
def retriever(base_settings) -> HybridRetriever:
    return HybridRetriever(base_settings)


@pytest.fixture
def settings(base_settings, tmp_path) -> Settings:
    return replace(base_settings, db_path=tmp_path / "traces.db")


@pytest.fixture
def make_agent(settings, retriever):
    def _make(script: list) -> Agent:
        return Agent(settings, llm=FakeLLM(script), retriever=retriever, trace_store=TraceStore(settings.db_path))

    return _make

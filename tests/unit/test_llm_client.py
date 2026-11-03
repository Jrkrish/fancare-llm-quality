"""LLM client retry policy and judge circuit breaker, with HTTP mocked (no network)."""
from dataclasses import replace

import httpx
import pytest

from app.config import Settings
from app.llm import LLMClient, LLMError, QuotaExhausted, retry_delay

OK = {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
GEMINI_DAILY = '{"error": {"code": 429, "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]}}'
GEMINI_MINUTE = '{"error": {"code": 429, "details": [{"quotaId": "GenerateRequestsPerMinute"}, {"retryDelay": "7s"}]}}'


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    sleeps.clear()
    return LLMClient(replace(Settings(), llm_provider="groq", llm_fallback_provider="", llm_max_retries=3, llm_min_interval_s=0))


sleeps: list[float] = []


def script(monkeypatch, responses):
    calls = iter(responses)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: next(calls))


def resp(status, body="", headers=None):
    req = httpx.Request("POST", "https://x")
    if isinstance(body, dict):
        return httpx.Response(status, json=body, headers=headers, request=req)
    return httpx.Response(status, text=body, headers=headers, request=req)


def test_daily_quota_fails_fast_without_retrying(client, monkeypatch):
    script(monkeypatch, [resp(429, GEMINI_DAILY), resp(200, OK)])
    with pytest.raises(QuotaExhausted):
        client.chat([{"role": "user", "content": "x"}])
    assert sleeps == []


def test_per_minute_limit_waits_for_retry_delay(client, monkeypatch):
    script(monkeypatch, [resp(429, GEMINI_MINUTE), resp(200, OK)])
    assert client.chat([{"role": "user", "content": "x"}]).content == "hi"
    assert sleeps == [7.0]


@pytest.mark.parametrize("status", [500, 502, 503, 522, 524])
def test_server_errors_are_retried(client, monkeypatch, status):
    script(monkeypatch, [resp(status, "<html>cloudflare</html>"), resp(200, OK)])
    assert client.chat([{"role": "user", "content": "x"}]).content == "hi"


def test_client_errors_are_not_retried(client, monkeypatch):
    script(monkeypatch, [resp(400, "bad request"), resp(200, OK)])
    with pytest.raises(LLMError, match="HTTP 400"):
        client.chat([{"role": "user", "content": "x"}])


def test_retry_after_header_wins_and_is_capped():
    assert retry_delay(resp(429, "", {"retry-after": "3"}), 0) == 3.0
    assert retry_delay(resp(429, "", {"retry-after": "9999"}), 0) == 60


def test_judge_circuit_breaker_stops_after_quota(monkeypatch):
    from eval.judge import JudgeLLM

    judge = JudgeLLM("groq", "fake")
    calls = []

    class Exhausted:
        def chat(self, *a, **k):
            calls.append(1)
            raise QuotaExhausted("daily quota")

    judge.model = Exhausted()
    for _ in range(3):
        with pytest.raises(QuotaExhausted):
            judge.generate("prompt")
    assert len(calls) == 1  # later calls fail fast without touching the provider


def test_judge_strips_reasoning_before_json(monkeypatch):
    from pydantic import BaseModel

    from app.llm import LLMResponse
    from eval.judge import JudgeLLM

    class Out(BaseModel):
        verdict: str

    judge = JudgeLLM("groq", "fake")
    judge.model = type("C", (), {"chat": lambda self, *a, **k: LLMResponse(
        '<think>maybe {"verdict": "no"}?</think>{"verdict": "yes"}', [], "f", "f", {})})()
    assert judge.generate("p", schema=Out).verdict == "yes"


def test_oversized_request_waits_for_the_window_then_retries(client, monkeypatch):
    # Seen mid-run on Groq: the same judge request succeeds once the per-minute window rolls over.
    too_large = '{"error": {"message": "Request too large for model `qwen` on output tokens per minute (OTPM): Limit 6000"}}'
    script(monkeypatch, [resp(429, too_large), resp(200, OK)])
    assert client.chat([{"role": "user", "content": "x"}]).content == "hi"
    assert sleeps == [30.0]


def test_max_tokens_is_sent_only_when_set(client, monkeypatch):
    bodies = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: bodies.append(k["json"]) or resp(200, OK))
    client.chat([{"role": "user", "content": "x"}])
    client.settings = replace(client.settings, llm_max_tokens=2048)
    client.chat([{"role": "user", "content": "x"}])
    assert "max_tokens" not in bodies[0] and bodies[1]["max_tokens"] == 2048


def test_judge_keeps_rejected_claims_with_reasons():
    from types import SimpleNamespace

    from eval.metrics import rejected_items

    metric = SimpleNamespace(claims=["Motor is covered 5 years.", "Coverage ended June 2024."], verdicts=[
        SimpleNamespace(verdict="yes", reason=None),
        SimpleNamespace(verdict="borderline", reason="the end date is inferred, not stated"),
    ])
    assert rejected_items(metric) == ["[borderline] Coverage ended June 2024. -- the end date is inferred, not stated"]

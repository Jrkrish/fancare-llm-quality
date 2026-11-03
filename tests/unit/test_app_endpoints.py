"""In-process endpoint tests with a scripted LLM: streaming, history and upload-to-chat wiring."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.unit.conftest import answer, tool_call

DOCS = Path(__file__).resolve().parents[2] / "eval" / "datasets" / "documents" / "files"
H = {"X-API-Key": "dev-local-key"}


@pytest.fixture
def client_for(make_agent, settings):
    def _make(script):
        agent = make_agent(script)
        client = TestClient(create_app(settings, agent=agent))
        client.__enter__()
        return client, agent

    return _make


def sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_stream_reassembles_the_answer(client_for):
    client, _ = client_for([tool_call("search_docs", query="E7"), answer("E7 means the firmware update failed [DOC-07:4].")])
    resp = client.post("/chat/stream", json={"session_id": "s1", "message": "E7?"}, headers=H)
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = sse(resp.text)
    assert events[-1][0] == "done" and len(events) > 2
    assert "".join(d["text"] for e, d in events if e == "delta") == events[-1][1]["answer"]
    assert events[-1][1]["citations"][0]["chunk_id"] == "DOC-07:4"


def test_stream_reports_llm_outage_as_error_event(client_for):
    from app.llm import LLMError

    client, _ = client_for([LLMError("down")])
    events = sse(client.post("/chat/stream", json={"session_id": "s1", "message": "hi"}, headers=H).text)
    assert events == [("error", {"detail": "LLM provider unavailable", "trace_id": events[0][1]["trace_id"]})]


def test_history_returns_turns(client_for):
    client, _ = client_for([answer("Hello!")])
    client.post("/chat", json={"session_id": "s9", "message": "hi"}, headers=H)
    assert client.get("/sessions/s9/history", headers=H).json()["turns"] == [{"user": "hi", "assistant": "Hello!"}]


def test_uploaded_fields_reach_the_llm_but_hidden_text_does_not(client_for):
    client, agent = client_for([answer("Your LX72 was bought on 2025-10-20.")])
    with (DOCS / "07_warranty_lx72_injection.png").open("rb") as f:
        up = client.post("/upload", files={"file": ("card.png", f, "image/png")}, data={"session_id": "s5"}, headers=H)
    assert up.json()["fields"]["serial_no"] == "LX72-2509-00077"

    client.post("/chat", json={"session_id": "s5", "message": "What do you know about my fan?"}, headers=H)
    system = agent.llm.requests[0]["messages"][0]["content"]
    assert "LX72-2509-00077" in system
    assert "ignore previous instructions" not in system.lower()
    assert "refund" not in system.lower()


def test_index_page_is_served(client_for):
    client, _ = client_for([])
    resp = client.get("/")
    assert resp.status_code == 200 and 'data-testid="message-input"' in resp.text

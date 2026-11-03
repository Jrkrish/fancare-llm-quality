"""SQL data validation on the request logs: nulls, duplicates, schema violations, PII and orphans.

Two parts:
- TestChecksCatchViolations seeds a database with known-bad rows and proves each
  query finds them (so a clean result below actually means something).
- TestRecordedLogs runs every query against the real trace databases written by
  the app and the other suites.
"""
import json
import sqlite3

import pytest

from app.guardrails import Verdict
from app.observability import TraceRecord, TraceStore
from app.tools import ToolRecord
from tests.data.conftest import CHECKS, LOG_DBS, open_db, run_check
from tests.framework.base_test import BaseTest

pytestmark = pytest.mark.data


@pytest.fixture
def seeded_db(tmp_path):
    path = tmp_path / "seeded.db"
    store = TraceStore(path)
    store.write(TraceRecord("good1", "s1", "ok", "v1", "hybrid", "What is E7?", "allow", 120, provider="groq",
                            model="m", answer="Firmware [DOC-07:4]", output_verdict="allow",
                            tool_calls=[ToolRecord("search_docs", {"query": "E7"}, "ok")],
                            guardrail_events=[Verdict("input", "allow"), Verdict("output", "allow")]))
    conn = sqlite3.connect(path)
    bad = [
        # nulls: ok status without answer
        "INSERT INTO traces VALUES ('bad-null','s1','2026-01-01T00:00:00','ok','v1','groq','m','hybrid','q',NULL,'[]','[]',5,0,0,0,'allow','allow',NULL)",
        # schema: invalid status, negative latency, bad JSON
        "INSERT INTO traces VALUES ('bad-schema','s1','not-a-date','weird','v9','groq','m','hybrid','q','a','oops','[]',-5,0,0,0,'maybe','allow',NULL)",
        # pii stored in plain text
        "INSERT INTO traces VALUES ('bad-pii','s1','2026-01-01T00:00:01','ok','v1','groq','m','hybrid','mail me at jo@example.com','a','[]','[]',5,0,0,0,'allow','allow',NULL)",
        # orphans
        "INSERT INTO traces VALUES ('bad-orphan','ghost-session','2026-01-01T00:00:02','ok','v1','groq','m','hybrid','q','a','[]','[]',5,0,0,0,'allow','allow',NULL)",
        "INSERT INTO tool_calls (trace_id, seq, name, arguments, status, latency_ms) VALUES ('no-such-trace', 1, 'x', '{}', 'ok', 1)",
        # duplicates
        "INSERT INTO tool_calls (trace_id, seq, name, arguments, status, latency_ms) VALUES ('good1', 1, 'search_docs', '{}', 'ok', 1)",
    ]
    for sql in bad:
        conn.execute(sql)
    conn.commit()
    conn.close()
    return path


class TestChecksCatchViolations(BaseTest):
    @pytest.mark.parametrize("check, expected", [
        ("nulls", "bad-null"),
        ("schema_violations", "bad-schema"),
        ("pii", "bad-pii"),
        ("orphans", "bad-orphan"),
        ("orphans", "tool call without a trace"),
        ("duplicates", "good1#1"),
    ])
    def test_check_finds_seeded_violation(self, seeded_db, check, expected):
        rows = run_check(open_db(seeded_db), check)
        assert any(expected in " ".join(map(str, r)) for r in rows), f"{check} missed '{expected}': {rows}"

    def test_clean_rows_are_not_flagged(self, seeded_db):
        conn = open_db(seeded_db)
        for check in CHECKS:
            flagged = [r for r in run_check(conn, check) if r[1] == "good1"]
            assert not flagged, f"{check} flagged a clean row: {flagged}"


def _open_logged_db(db):
    """Checked at run time: the API, web and LLM suites earlier in the session create these databases."""
    if not db.exists():
        pytest.skip(f"{db.name} does not exist yet; run the app or another suite first")
    conn = open_db(db)
    if conn.execute("SELECT count(*) FROM traces").fetchone()[0] == 0:
        pytest.skip(f"{db.name} has no traces")
    return conn


class TestRecordedLogs(BaseTest):
    @pytest.mark.parametrize("db", LOG_DBS, ids=[p.name for p in LOG_DBS])
    @pytest.mark.parametrize("check", CHECKS)
    def test_logs_are_clean(self, db, check):
        rows = run_check(_open_logged_db(db), check)
        self.fail_if_any([f"{tbl} {row_id}: {problem}" for tbl, row_id, problem in rows[:25]],
                         f"{check} found {len(rows)} problem rows in {db.name}")

    @pytest.mark.parametrize("db", LOG_DBS, ids=[p.name for p in LOG_DBS])
    def test_json_columns_parse(self, db):
        for (value,) in _open_logged_db(db).execute("SELECT retrieved_chunk_ids FROM traces LIMIT 500"):
            assert isinstance(json.loads(value), list)

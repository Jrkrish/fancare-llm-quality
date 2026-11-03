"""Data-validation fixtures: open trace databases with the has_pii() SQL function registered."""
import sqlite3
from pathlib import Path

from app.guardrails import redact_pii

ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = Path(__file__).parent / "sql"
CHECKS = sorted(p.stem for p in SQL_DIR.glob("*.sql"))
# Every trace database the app and the suites write to.
LOG_DBS = [ROOT / "logs" / "traces.db", ROOT / "logs" / "eval_traces.db", ROOT / "logs" / "test_traces.db"]


def open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.create_function("has_pii", 1, lambda text: int(bool(text) and bool(redact_pii(text)[1])), deterministic=True)
    return conn


def run_check(conn: sqlite3.Connection, check: str) -> list[tuple]:
    return conn.execute((SQL_DIR / f"{check}.sql").read_text(encoding="utf-8")).fetchall()

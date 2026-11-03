"""Production-failure loop: turn a logged bad conversation into a new golden case.

    python -m eval.promote_failure list                     # candidate bad traces from logs/traces.db
    python -m eval.promote_failure show  --trace-id <id>    # the trace and its session context
    python -m eval.promote_failure promote --trace-id <id> --category near_miss \
        --expected-behavior refuse --expected-output "I don't have that information..." \
        [--expected-chunk-ids DOC-11:1] [--case-id PROD-01]

The question is taken from the logged (already PII-redacted) user message. The case
is validated against the corpus and appended to eval/datasets/golden.json, with a
note recording the trace, the bad answer and any earlier turns of the session.
"""
import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.config import ROOT, Settings
from app.retrieval import HybridRetriever
from eval.dataset import BEHAVIORS, CATEGORIES, GoldenCase, load_golden

GOLDEN = ROOT / "eval" / "datasets" / "golden.json"

# Signals that a logged turn went wrong and deserves a look.
CANDIDATES_SQL = """
SELECT t.trace_id, t.created_at, t.status, t.user_message,
       t.output_verdict,
       (SELECT group_concat(name || ':' || status) FROM tool_calls c WHERE c.trace_id = t.trace_id AND c.status != 'ok') AS failed_tools,
       t.retrieved_chunk_ids, t.cited_chunk_ids
FROM traces t
WHERE t.status = 'error'
   OR t.output_verdict = 'block'
   OR EXISTS (SELECT 1 FROM tool_calls c WHERE c.trace_id = t.trace_id AND c.status != 'ok')
   OR (t.status = 'ok' AND t.retrieved_chunk_ids != '[]' AND t.cited_chunk_ids = '[]')
ORDER BY t.created_at DESC
LIMIT ?
"""


def connect(db: Path) -> sqlite3.Connection:
    if not db.exists():
        raise SystemExit(f"No trace database at {db}")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


def get_trace(conn, trace_id: str) -> dict:
    row = conn.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
    if row is None:
        raise SystemExit(f"Trace {trace_id} not found")
    return dict(row)


def session_context(conn, trace: dict) -> list[dict]:
    rows = conn.execute(
        "SELECT user_message, answer FROM traces WHERE session_id = ? AND created_at < ? ORDER BY created_at",
        (trace["session_id"], trace["created_at"]),
    )
    return [dict(r) for r in rows]


def append_case(case: GoldenCase, path: Path = GOLDEN) -> None:
    """Insert the case before the closing bracket so the file keeps its one-case-per-block layout."""
    text = path.read_text(encoding="utf-8")
    end = text.rindex("\n  ]")
    entry = json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in case.__dict__.items()}, ensure_ascii=False)
    path.write_text(text[:end] + ",\n\n    " + entry + text[end:], encoding="utf-8")


def build_case(trace: dict, context: list[dict], args, known_ids: set[str]) -> GoldenCase:
    earlier = " | ".join(f"user: {c['user_message']} / bot: {(c['answer'] or '')[:120]}" for c in context[-3:])
    notes = (f"Promoted from production trace {trace['trace_id']} ({trace['created_at']}) on "
             f"{datetime.now(UTC).date()}. Bad answer: {(trace['answer'] or trace['error'] or '')[:300]}"
             + (f" Earlier turns: {earlier}" if earlier else ""))
    case = GoldenCase(
        id=args.case_id, category=args.category, question=trace["user_message"],
        expected_behavior=args.expected_behavior, expected_output=args.expected_output,
        expected_chunk_ids=tuple(args.expected_chunk_ids or ()), notes=notes,
    )
    missing = set(case.expected_chunk_ids) - known_ids
    if missing:
        raise SystemExit(f"Unknown chunk ids: {sorted(missing)}")
    if case.expected_behavior != "refuse" and not case.expected_chunk_ids:
        raise SystemExit("--expected-chunk-ids is required unless --expected-behavior is refuse")
    return case


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["list", "show", "promote"])
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--trace-id")
    parser.add_argument("--case-id")
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--expected-behavior", choices=BEHAVIORS)
    parser.add_argument("--expected-output")
    parser.add_argument("--expected-chunk-ids", nargs="*")
    args = parser.parse_args(argv)

    settings = Settings()
    conn = connect(args.db or settings.db_path)

    if args.command == "list":
        rows = conn.execute(CANDIDATES_SQL, (args.limit,)).fetchall()
        if not rows:
            print("No suspicious traces found.")
        for r in rows:
            flags = [f for f, on in (("error", r["status"] == "error"), ("output-blocked", r["output_verdict"] == "block"),
                                     (f"tools:{r['failed_tools']}", r["failed_tools"]),
                                     ("uncited", r["retrieved_chunk_ids"] != "[]" and r["cited_chunk_ids"] == "[]")) if on]
            print(f"{r['trace_id']}  {r['created_at'][:19]}  {','.join(flags):<30} {r['user_message'][:70]}")
        return 0

    if not args.trace_id:
        parser.error("--trace-id is required")
    trace = get_trace(conn, args.trace_id)
    context = session_context(conn, trace)

    if args.command == "show":
        for c in context:
            print(f"USER: {c['user_message']}\nBOT:  {c['answer']}\n")
        print(f">>> USER: {trace['user_message']}\n>>> BOT:  {trace['answer'] or trace['error']}")
        print(f"retrieved: {trace['retrieved_chunk_ids']} cited: {trace['cited_chunk_ids']}")
        return 0

    for flag in ("case_id", "category", "expected_behavior", "expected_output"):
        if not getattr(args, flag):
            parser.error(f"--{flag.replace('_', '-')} is required for promote")
    retriever_ids = set(HybridRetriever(settings).by_id)
    existing = {c.id for c in load_golden(GOLDEN)}
    if args.case_id in existing:
        raise SystemExit(f"Case id {args.case_id} already exists")
    case = build_case(trace, context, args, retriever_ids)
    append_case(case)
    load_golden(GOLDEN, known_chunk_ids=retriever_ids)  # re-validate the whole file
    print(f"Added {case.id} to {GOLDEN}. It will run in the next eval.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Summarise a Locust run: p50/p95/p99 latency, error rate, and estimated cost per 100 conversations.

    python -m tests.load.summarize reports/load/run_stats.csv --since 2026-11-03T10:00:00

Latency and errors come from Locust's CSV. Cost comes from the app's own trace log
(token counts x reference prices in app/config.py) for sessions named load-*.
"""
import argparse
import csv
import sqlite3
import sys
from pathlib import Path

from app.config import Settings


def locust_stats(path: Path) -> dict:
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if row["Name"] == "/chat":
            n = int(row["Request Count"])
            return {
                "requests": n,
                "failures": int(row["Failure Count"]),
                "error_rate": int(row["Failure Count"]) / n if n else 0.0,
                "p50_ms": float(row["50%"]), "p95_ms": float(row["95%"]), "p99_ms": float(row["99%"]),
                "rps": float(row["Requests/s"]),
            }
    raise SystemExit(f"No /chat row in {path}")


def cost_per_100(db: Path, since: str | None) -> dict:
    if not db.exists():
        return {}
    where, params = "session_id LIKE 'load-%'", []
    if since:
        where += " AND created_at >= ?"
        params.append(since)
    with sqlite3.connect(db) as conn:
        conversations, cost = conn.execute(
            f"SELECT count(DISTINCT session_id), coalesce(sum(cost_usd), 0) FROM traces WHERE {where}", params
        ).fetchone()
    return {"conversations": conversations, "cost_usd": cost,
            "cost_per_100_conversations_usd": cost / conversations * 100 if conversations else 0.0}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stats_csv", type=Path)
    p.add_argument("--since", help="only count traces created at/after this ISO timestamp")
    p.add_argument("--db", type=Path, default=None)
    args = p.parse_args(argv)
    stats = locust_stats(args.stats_csv)
    cost = cost_per_100(args.db or Settings().db_path, args.since)
    lines = ["# Load test summary", "", "| Metric | Value |", "|---|---|",
             f"| Requests | {stats['requests']} |", f"| Error rate | {stats['error_rate']:.1%} |",
             f"| p50 latency | {stats['p50_ms']:.0f} ms |", f"| p95 latency | {stats['p95_ms']:.0f} ms |",
             f"| p99 latency | {stats['p99_ms']:.0f} ms |", f"| Throughput | {stats['rps']:.2f} req/s |"]
    if cost:
        lines += [f"| Conversations | {cost['conversations']} |",
                  f"| Est. cost per 100 conversations | ${cost['cost_per_100_conversations_usd']:.4f} (reference prices) |"]
    out = args.stats_csv.parent / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

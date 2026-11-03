"""Rank the flakiest tests from reports/flaky/history.jsonl (written by tests/conftest.py).

    python -m tools.flaky_report [--top 20] [--min-runs 2]

A test counts as flaky when it needed a rerun to pass, or when its final outcome
changed between test sessions with no code change in between being recorded.
flip rate = outcome changes between consecutive sessions / (sessions - 1).
Tests with flip rate >= 0.3 are suggested for tests/quarantine.txt.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "reports" / "flaky" / "history.jsonl"
QUARANTINE_FLIP_RATE = 0.3


def load(path: Path = HISTORY) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def analyse(records: list[dict]) -> list[dict]:
    # nodeid -> session -> list of outcomes in order (reruns then final)
    by_test: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    order: dict[str, str] = {}
    for r in records:
        by_test[r["nodeid"]][r["session"]].append(r["outcome"])
        order.setdefault(r["session"], r["timestamp"])
    rows = []
    for nodeid, sessions in by_test.items():
        ordered = sorted(sessions, key=lambda s: order[s])
        finals = [sessions[s][-1] for s in ordered if sessions[s][-1] != "skipped"]
        reruns = sum(o.count("rerun") for o in sessions.values())
        passed_after_rerun = sum(1 for s in ordered if "rerun" in sessions[s] and sessions[s][-1] == "passed")
        flips = sum(1 for a, b in zip(finals, finals[1:], strict=False) if a != b)
        flip_rate = flips / (len(finals) - 1) if len(finals) > 1 else 0.0
        rows.append({
            "nodeid": nodeid, "sessions": len(finals), "passes": finals.count("passed"), "failures": finals.count("failed"),
            "reruns": reruns, "passed_after_rerun": passed_after_rerun, "flip_rate": flip_rate,
            "score": flip_rate + passed_after_rerun / max(len(finals), 1),
        })
    return sorted((r for r in rows if r["score"] > 0), key=lambda r: r["score"], reverse=True)


def render(rows: list[dict], top: int) -> str:
    lines = ["# Flakiest tests", "", "| Test | Sessions | Pass/Fail | Reruns | Passed only after rerun | Flip rate | Suggest quarantine |",
             "|---|---|---|---|---|---|---|"]
    for r in rows[:top]:
        lines.append(f"| `{r['nodeid']}` | {r['sessions']} | {r['passes']}/{r['failures']} | {r['reruns']} | "
                     f"{r['passed_after_rerun']} | {r['flip_rate']:.0%} | {'yes' if r['flip_rate'] >= QUARANTINE_FLIP_RATE else ''} |")
    if not rows:
        lines.append("| no flaky tests recorded yet | | | | | | |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--min-runs", type=int, default=1)
    args = p.parse_args(argv)
    rows = [r for r in analyse(load()) if r["sessions"] >= args.min_runs]
    out = HISTORY.parent / "flaky_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows, args.top), encoding="utf-8")
    print(render(rows, args.top))
    print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

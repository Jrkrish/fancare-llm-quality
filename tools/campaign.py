"""Pick the next measurement run from campaign/manifest.json and check what it produced.

    python -m tools.campaign next  --artifacts artifacts.jsonl     # which item is due now?
    python -m tools.campaign check --item full-1 --code-dir code   # did that run measure enough?

Each finished run is uploaded as an artifact. A valid one is named batch-<commit>-<item>; a run
that lost too many answers to rate limits or quota ends in --invalid and the item is run again.
Standard library only, so the "is anything due?" job needs no installs.
"""
import argparse
import csv
import json
import os
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "campaign" / "manifest.json"
MAX_ERROR_RATE = 0.10  # the same limit the eval's validity gates use


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prefix(manifest: dict) -> str:
    return f"batch-{manifest['code_commit'][:7]}-"


def _time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def select(manifest: dict, artifacts: list[dict], now: datetime) -> dict:
    """artifacts: [{"name": ..., "created_at": ...}]. Returns {"due", "item", "reason", "blocked"}."""
    ours = [a for a in artifacts if a["name"].startswith(prefix(manifest))]
    done = {a["name"] for a in ours}
    pending = [i for i in manifest["items"] if prefix(manifest) + i["id"] not in done]
    result = {"due": False, "item": None, "blocked": False}
    if not pending:
        return {**result, "reason": "all items have a valid run: the campaign is complete"}
    item = pending[0]
    result["item"] = item
    invalid = sum(a["name"] == f"{prefix(manifest)}{item['id']}--invalid" for a in ours)
    if invalid >= manifest["max_invalid_attempts"]:
        return {**result, "blocked": True,
                "reason": f"{item['id']} had {invalid} invalid runs; it needs a look (resize it or fix the cause)"}
    start = _time(manifest["not_before"])
    if ours:
        start = max(start, max(_time(a["created_at"]) for a in ours) + timedelta(hours=manifest["min_gap_hours"]))
    if now < start:
        return {**result, "reason": f"{item['id']} is next, not before {start:%Y-%m-%d %H:%M} UTC"}
    return {**result, "due": True, "reason": f"{item['id']} is due"}


def command(item: dict) -> list[str]:
    """Arguments for `python` in the pinned code checkout. The load test is scripted in the workflow."""
    if item["kind"] == "load":
        return []
    module = {"eval": "eval.run", "safety": "eval.safety", "tool_use": "eval.tool_use",
              "conversation": "eval.conversation"}[item["kind"]]
    cases = [arg for case in item.get("cases", []) for arg in ("--case", case)]
    return ["-m", module, *item["args"], *cases]


def trace_tokens(db: Path) -> int:
    if not db.exists():
        return 0
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT coalesce(sum(prompt_tokens + completion_tokens), 0) FROM traces").fetchone()[0]


def _verdict(valid: bool, reason: str, **extra) -> dict:
    return {"valid": valid, "reason": reason, **extra}


def check(item: dict, code_dir: Path) -> dict:
    """Is the report this run wrote complete enough to keep?"""
    reports = code_dir / "reports"
    kind = item["kind"]
    if kind == "load":
        path = reports / "load" / "run_stats.csv"
        if not path.exists():
            return _verdict(False, "no load statistics were written")
        with path.open(encoding="utf-8") as f:
            requests = sum(int(r["Request Count"]) for r in csv.DictReader(f) if r["Name"] == "/chat")
        # Errors under load are a result to report, not a reason to repeat the test.
        return _verdict(requests > 0, f"{requests} /chat requests", sut_tokens=trace_tokens(code_dir / "logs" / "traces.db"))

    path = reports / ("eval/latest" if kind == "eval" else kind) / "results.json"
    if not path.exists():
        return _verdict(False, "no results.json was written")
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = len(item["cases"]) if "cases" in item else item["expected_cases"]
    results = data["cases" if kind == "eval" else "results"]
    if len(results) != expected:
        return _verdict(False, f"{len(results)} of {expected} cases are in the report")

    if kind == "eval":
        rates = {k: data["summary"].get(k, 0.0) for k in ("error_rate", "judge_error_rate")}
        judge = data.get("judge_usage") or {}
        return _verdict(
            all(v <= MAX_ERROR_RATE for v in rates.values()),
            ", ".join(f"{k} {v:.3f}" for k, v in rates.items()),
            sut_tokens=sum(r["reply"]["prompt_tokens"] + r["reply"]["completion_tokens"]
                           for c in results for r in c["runs"]),
            judge_tokens=judge.get("prompt_tokens", 0) + judge.get("completion_tokens", 0),
        )
    # Suites: a list of attempts (safety) or runs (tool use, conversation) per case.
    runs = [run for r in results for run in r["attempts" if kind == "safety" else "runs"]]
    errored = sum(bool(run.get("error")) for run in runs)
    rate = errored / len(runs) if runs else 1.0
    return _verdict(rate <= MAX_ERROR_RATE, f"{errored} of {len(runs)} runs errored",
                    sut_tokens=trace_tokens(code_dir / "logs" / "eval_traces.db"))


def _output(**values) -> None:
    """Expose values to later workflow steps."""
    if path := os.getenv("GITHUB_OUTPUT"):
        with open(path, "a", encoding="utf-8") as f:
            f.writelines(f"{k}={v}\n" for k, v in values.items())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    nxt = sub.add_parser("next")
    nxt.add_argument("--artifacts", type=Path, required=True, help="one JSON object per line: name, created_at")
    chk = sub.add_parser("check")
    chk.add_argument("--item", required=True)
    chk.add_argument("--code-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = load_manifest()

    if args.action == "next":
        lines = args.artifacts.read_text(encoding="utf-8").splitlines()
        chosen = select(manifest, [json.loads(x) for x in lines if x.strip()], datetime.now(UTC))
        item = chosen["item"] or {}
        print(("::warning::" if chosen["blocked"] else "") + chosen["reason"])
        if item:
            print(f"command: {'(load test)' if item['kind'] == 'load' else 'python ' + ' '.join(command(item))}")
        _output(due=str(chosen["due"]).lower(), item=item.get("id", ""), kind=item.get("kind", ""),
                command=" ".join(command(item)) if item else "", code_commit=manifest["code_commit"],
                artifact=prefix(manifest) + item.get("id", ""))
        return 0

    item = next(i for i in manifest["items"] if i["id"] == args.item)
    verdict = {"item": item["id"], "code_commit": manifest["code_commit"], **check(item, args.code_dir),
               "finished_at": datetime.now(UTC).isoformat()}
    out = args.code_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "batch.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    _output(valid=str(verdict["valid"]).lower())
    return 0


if __name__ == "__main__":
    sys.exit(main())

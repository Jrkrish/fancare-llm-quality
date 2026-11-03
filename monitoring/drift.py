"""Drift monitoring: replay a fixed sample of queries, record scores / latency / cost over time,
draw a trend chart and flag drift against recent history.

    python -m monitoring.drift            # replay, append history, chart, report
    python -m monitoring.drift --judge    # also score faithfulness with the LLM judge
    python -m monitoring.drift --report-only

Drift is flagged when the latest run is worse than the median of the previous
`window` runs by more than the thresholds in monitoring/drift_config.json.
Exits 1 when drift is flagged, so a scheduled CI job shows up red.
"""
import argparse
import csv
import json
import statistics
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from app.config import ROOT
from eval.config import REPORTS_DIR
from eval.run import EvalSetupError, build_runner

CONFIG = json.loads((ROOT / "monitoring" / "drift_config.json").read_text(encoding="utf-8"))
OUT_DIR = REPORTS_DIR / "drift"
HISTORY = OUT_DIR / "drift_history.csv"
SCORES = ["refusal_correctness", "citation_correctness", "retrieval_recall", "faithfulness"]
COLUMNS = ["timestamp", "prompt_version", "provider", "model", *SCORES, "hallucination_rate",
           "latency_p50_ms", "latency_p95_ms", "cost_usd_per_query", "error_rate", "n_runs"]


def replay(judge: bool) -> dict:
    runner, cases = build_runner(profile="nojudge", case_ids=CONFIG["case_ids"], runs=CONFIG["runs_per_case"],
                                 no_judge=not judge)
    if judge:
        runner.config = replace(runner.config, judge_metrics=["faithfulness"])
    report = runner.run(cases)
    s, t = report.summary, report.target
    n = max(int(s.get("n_runs", 1)), 1)
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "prompt_version": t["prompt_version"], "provider": t["provider"], "model": t["model"],
        **{k: s.get(k) for k in SCORES},
        "hallucination_rate": s.get("hallucination_rate"),
        "latency_p50_ms": s.get("latency_p50_ms"), "latency_p95_ms": s.get("latency_p95_ms"),
        "cost_usd_per_query": (s.get("cost_usd_total") or 0) / n,
        "error_rate": s.get("error_rate"), "n_runs": n,
    }


def append(row: dict, path: Path = HISTORY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: "" if row.get(k) is None else row[k] for k in COLUMNS})


def load_history(path: Path = HISTORY) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for r in csv.DictReader(path.open(encoding="utf-8")):
        rows.append({k: (float(v) if v not in ("", None) and k not in ("timestamp", "prompt_version", "provider", "model") else v or None)
                     for k, v in r.items()})
    return rows


def detect_drift(history: list[dict], window: int = CONFIG["window"], th: dict = CONFIG["thresholds"]) -> list[str]:
    if len(history) < 2:
        return []
    latest, previous = history[-1], history[-1 - window:-1]

    def ref(key):
        vals = [r[key] for r in previous if r.get(key) is not None]
        return statistics.median(vals) if vals else None

    flags = []
    for key in SCORES:
        base, now = ref(key), latest.get(key)
        if base is not None and now is not None and base - now > th["max_score_drop"]:
            flags.append(f"{key} dropped {base:.3f} -> {now:.3f}")
    base, now = ref("hallucination_rate"), latest.get("hallucination_rate")
    if base is not None and now is not None and now - base > th["max_hallucination_increase"]:
        flags.append(f"hallucination_rate rose {base:.3f} -> {now:.3f}")
    for key, limit in (("latency_p95_ms", th["max_latency_p95_increase_pct"]), ("cost_usd_per_query", th["max_cost_increase_pct"])):
        base, now = ref(key), latest.get(key)
        if base and now is not None and (now - base) / base * 100 > limit:
            flags.append(f"{key} rose {base:.4g} -> {now:.4g} (+{(now - base) / base:.0%})")
    if latest.get("model") and previous and previous[-1].get("model") != latest.get("model"):
        flags.append(f"model changed {previous[-1].get('model')} -> {latest.get('model')} (scores not comparable)")
    return flags


def svg_chart(history: list[dict], series: list[str], title: str, width=720, height=260, ymax=None) -> str:
    """Minimal dependency-free line chart."""
    pad_l, pad_r, pad_t, pad_b = 50, 150, 30, 30
    pts = [[r.get(s) for r in history] for s in series]
    values = [v for line in pts for v in line if v is not None]
    top = ymax if ymax is not None else (max(values) * 1.1 if values else 1)
    top = top or 1
    n = max(len(history) - 1, 1)
    x = lambda i: pad_l + i * (width - pad_l - pad_r) / n  # noqa: E731
    y = lambda v: pad_t + (1 - v / top) * (height - pad_t - pad_b)  # noqa: E731
    colours = ["#2563eb", "#16a34a", "#d97706", "#dc2626", "#7c3aed", "#0891b2"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="11">',
             f'<rect width="{width}" height="{height}" fill="white"/>',
             f'<text x="{pad_l}" y="18" font-size="13" font-weight="bold">{title}</text>']
    for frac in (0, 0.5, 1):
        yy = y(top * frac)
        parts.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{yy + 4:.1f}" text-anchor="end" fill="#6b7280">{top * frac:.3g}</text>')
    for k, (name, line) in enumerate(zip(series, pts, strict=True)):
        c = colours[k % len(colours)]
        coords = [f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(line) if v is not None]
        if coords:
            parts.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="{" ".join(coords)}"/>')
            parts += [f'<circle cx="{p.split(",")[0]}" cy="{p.split(",")[1]}" r="2.5" fill="{c}"/>' for p in coords]
        parts.append(f'<text x="{width - pad_r + 10}" y="{pad_t + 14 * k + 4}" fill="{c}">{name}</text>')
    if history:
        parts.append(f'<text x="{pad_l}" y="{height - 8}" fill="#6b7280">{str(history[0]["timestamp"])[:10]}</text>')
        parts.append(f'<text x="{width - pad_r}" y="{height - 8}" text-anchor="end" fill="#6b7280">{str(history[-1]["timestamp"])[:10]}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def write_report(history: list[dict], flags: list[str]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "scores.svg").write_text(svg_chart(history, [*SCORES, "hallucination_rate"], "Quality scores", ymax=1.0), encoding="utf-8")
    (OUT_DIR / "latency.svg").write_text(svg_chart(history, ["latency_p50_ms", "latency_p95_ms"], "Latency (ms)"), encoding="utf-8")
    (OUT_DIR / "cost.svg").write_text(svg_chart(history, ["cost_usd_per_query"], "Estimated cost per query (USD)"), encoding="utf-8")
    latest = history[-1] if history else {}
    lines = ["# Drift report", "", f"Runs recorded: {len(history)}. Latest: {latest.get('timestamp')} "
             f"({latest.get('prompt_version')}, {latest.get('provider')}/{latest.get('model')}).", "",
             f"## {'DRIFT DETECTED' if flags else 'No drift detected'}", ""]
    lines += [f"- {f}" for f in flags] or ["All metrics within thresholds of the recent median."]
    lines += ["", "![scores](scores.svg)", "", "![latency](latency.svg)", "", "![cost](cost.svg)", ""]
    (OUT_DIR / "drift_report.md").write_text("\n".join(lines), encoding="utf-8")
    return OUT_DIR / "drift_report.md"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.report_only:
        try:
            append(replay(args.judge))
        except EvalSetupError as exc:
            print(f"Cannot run drift job: {exc}", file=sys.stderr)
            return 2
    history = load_history()
    flags = detect_drift(history)
    path = write_report(history, flags)
    print(("DRIFT: " + "; ".join(flags)) if flags else "No drift detected.")
    print(f"Report: {path}")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())

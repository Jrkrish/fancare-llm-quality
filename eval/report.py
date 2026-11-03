"""Writes eval results: full JSON, a markdown summary, and one row in the history CSV."""
import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from eval.config import REPORTS_DIR
from eval.runner import EvalReport

INVALID = (
    "INVALID RUN: too many infrastructure or judge errors (see error_rate / judge_error_rate). "
    "The quality scores below are based on too few measurements to trust; fix the cause and re-run."
)

METRIC_COLUMNS = [
    "faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall",
    "citation_correctness", "refusal_correctness", "retrieval_recall",
]
HISTORY_COLUMNS = [
    "timestamp", "run_id", "profile", "prompt_version", "provider", "model", "retriever_mode", "rerank",
    "judge", "runs_per_case", "n_cases", "n_runs", *METRIC_COLUMNS, "hallucination_rate", "case_pass_rate",
    "error_rate", "unstable_cases", "latency_p50_ms", "latency_p95_ms", "cost_usd_total", "gates_passed",
]


def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, float) else ("" if value is None else str(value))


def write_report(report: EvalReport, reports_dir: Path = REPORTS_DIR) -> Path:
    out_dir = reports_dir / "eval" / report.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(asdict(report), indent=2, default=str), encoding="utf-8")
    (out_dir / "summary.md").write_text(render_markdown(report), encoding="utf-8")

    latest = reports_dir / "eval" / "latest"
    shutil.rmtree(latest, ignore_errors=True)
    shutil.copytree(out_dir, latest)
    append_history(report, reports_dir / "eval_history.csv")
    return out_dir


def append_history(report: EvalReport, path: Path) -> None:
    row = {
        "timestamp": report.started_at, "run_id": report.run_id, "profile": report.profile,
        **{k: report.target.get(k) for k in ("prompt_version", "provider", "model", "retriever_mode", "rerank")},
        "judge": report.judge or "", "runs_per_case": report.runs_per_case,
        **{k: report.summary.get(k) for k in HISTORY_COLUMNS if k in report.summary},
        "gates_passed": report.gates_passed,
    }
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: _fmt(row.get(k)) for k in HISTORY_COLUMNS})


def render_markdown(report: EvalReport) -> str:
    t, s = report.target, report.summary
    lines = [
        f"# RAG eval {report.run_id}",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Profile | {report.profile} |",
        f"| Prompt version | {t.get('prompt_version')} |",
        f"| Model | {t.get('provider')} / {t.get('model')} |",
        f"| Retriever | {t.get('retriever_mode')}{' + rerank' if t.get('rerank') else ''} |",
        f"| Judge | {report.judge or 'none (deterministic metrics only)'} |",
        f"| Cases x runs | {int(s.get('n_cases', 0))} x {report.runs_per_case} |",
        "",
        *([f"> **{INVALID}**", ""] if not report.valid else []),
        f"## Gates: {'PASSED' if report.gates_passed else 'FAILED'}",
        "",
        "| Metric | Value | Gate | Result |",
        "|---|---|---|---|",
    ]
    for g in report.gate_results:
        lines.append(f"| {g['metric']} | {g['value']:.3f} | {g['op']} {g['limit']} | {'pass' if g['passed'] else '**FAIL**'} |")

    lines += [
        "",
        f"Case pass rate {s.get('case_pass_rate', 0):.0%}, unstable cases {int(s.get('unstable_cases', 0))}, "
        f"latency p50 {s.get('latency_p50_ms', 0):.0f} ms / p95 {s.get('latency_p95_ms', 0):.0f} ms, "
        f"estimated SUT cost ${s.get('cost_usd_total', 0):.4f}.",
    ]
    if report.judge_usage:
        u = report.judge_usage
        lines.append(f"Judge usage: {u['calls']} calls, {u['prompt_tokens']} prompt + {u['completion_tokens']} completion tokens.")

    metric_cols = [m for m in METRIC_COLUMNS if any(m in v for v in report.by_category.values())]
    lines += ["", "## By category", "",
              "| Category | Cases | " + " | ".join(metric_cols) + " | Hallucination | Case pass |",
              "|---|---|" + "---|" * len(metric_cols) + "---|---|"]
    for cat, cs in report.by_category.items():
        cells = [_fmt(cs.get(m)) if m in cs else "n/a" for m in metric_cols]
        lines.append(f"| {cat} | {int(cs['n_cases'])} | " + " | ".join(cells)
                     + f" | {cs.get('hallucination_rate', 0):.0%} | {cs.get('case_pass_rate', 0):.0%} |")

    failing = [r for r in report.cases if not r.passed]
    lines += ["", f"## Failing cases ({len(failing)})", ""]
    for r in failing:
        lines.append(f"- **{r.case.id}** ({r.case.category}, expects {r.case.expected_behavior}): {r.case.question}")
        if r.errored:
            lines.append(f"  - all runs errored: {r.runs[0].reply.error}")
            continue
        for f in r.failures:
            lines.append(f"  - {f}")
        sample = r.valid_runs[0]
        lines.append(f"  - run 1 ({sample.response_class}): {sample.answer[:300].replace(chr(10), ' ')}")
        for run in r.valid_runs:
            for metric, items in run.judge_notes.items():
                lines.append(f"  - run {run.run_index + 1} {metric}, not accepted by the judge:")
                lines += [f"    - {item[:300]}" for item in items[:6]]

    unstable = [r for r in report.cases if r.unstable]
    lines += ["", f"## Unstable cases ({len(unstable)})", "", "Scores that changed between runs of the same question.", ""]
    lines += [f"- **{r.case.id}**: " + "; ".join(r.unstable) for r in unstable]

    halluc = [(r.case.id, run) for r in report.cases for run in r.runs if run.hallucinated]
    lines += ["", f"## Hallucinated runs ({len(halluc)})", ""]
    lines += [f"- **{cid}** run {run.run_index + 1}: {', '.join(run.hallucination_reasons)}" for cid, run in halluc]

    errors = [(r.case.id, m, e) for r in report.cases for run in r.runs for m, e in run.metric_errors.items()]
    if errors:
        lines += ["", f"## Metric errors ({len(errors)})", ""]
        lines += [f"- {cid} {m}: {e}" for cid, m, e in errors[:50]]
    return "\n".join(lines) + "\n"

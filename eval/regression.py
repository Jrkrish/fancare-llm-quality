"""Regression gate: evaluate prompt/model variants and compare them with a stored baseline.

    # Record the baseline once, from the reference variant (first prompt x first model):
    python -m eval.regression --profile smoke --prompts v1 --update-baseline

    # Compare variants (the demo: v2 is a deliberately weaker prompt):
    python -m eval.regression --profile smoke --prompts v1,v2,v3 \
        --models groq:openai/gpt-oss-120b,groq:openai/gpt-oss-20b

    # Gate on an eval that already ran (no new LLM calls), e.g. in CI:
    python -m eval.regression --profile smoke --from-results reports/eval/latest/results.json

Writes reports/regression/comparison.md. Exits 1 if any variant drops more than the
per-metric threshold in [regression] of eval/config.toml, 2 on setup problems.
Baselines are stored per profile in eval/baselines/<profile>.json and are only
ever written from real runs.
"""
import argparse
import json
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from app.config import ROOT
from eval.config import CONFIG_PATH, REPORTS_DIR
from eval.report import write_report
from eval.run import EvalSetupError, build_runner
from eval.suite_report import md_table

BASELINE_DIR = ROOT / "eval" / "baselines"
LOWER_IS_BETTER = {"hallucination_rate"}


def max_drops() -> dict[str, float]:
    return tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))["regression"]


def variant_name(target: dict) -> str:
    # No "|" here: the name is rendered inside a markdown table.
    return f"{target['prompt_version']}, {target['provider']}:{target['model']}, {target['retriever_mode']}"


def compare(summary: dict, baseline: dict, drops: dict[str, float]) -> list[dict]:
    rows = []
    for metric, allowed in drops.items():
        if metric not in summary or metric not in baseline:
            continue
        value, base = summary[metric], baseline[metric]
        delta = value - base
        worse = delta if metric in LOWER_IS_BETTER else -delta
        rows.append({"metric": metric, "value": value, "baseline": base, "delta": delta,
                     "allowed": allowed, "regressed": worse > allowed + 1e-9})
    return rows


def render(variants: list[dict], baseline: dict | None) -> str:
    lines = ["# Regression gate", ""]
    if baseline is None:
        lines += ["**No baseline stored for this profile.** Record one from a real run with `--update-baseline`.", ""]
    else:
        lines += [f"Baseline: {baseline['variant']} (recorded {baseline['created_at']}, run {baseline['run_id']}).", ""]
    metrics = list(max_drops())
    header = ["Variant", *metrics, "Result"]
    rows = []
    for v in variants:
        cells = []
        by_metric = {r["metric"]: r for r in v["comparison"]}
        for m in metrics:
            if m not in v["summary"]:
                cells.append("n/a")
            elif m in by_metric:
                r = by_metric[m]
                mark = " **REGRESSED**" if r["regressed"] else ""
                cells.append(f"{r['value']:.3f} ({r['delta']:+.3f}){mark}")
            else:
                cells.append(f"{v['summary'][m]:.3f}")
        result = "no baseline" if baseline is None else ("**FAIL**" if v["regressed"] else "pass")
        rows.append([v["variant"], *cells, result])
    lines += md_table(header, rows)
    lines += ["", "Cells show the suite mean and the change versus baseline. Allowed drop per metric: "
              + ", ".join(f"{k} {v}" for k, v in max_drops().items()) + "."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--prompts", default="v1", help="comma-separated prompt versions")
    parser.add_argument("--models", default="", help="comma-separated provider:model pairs (default: from .env)")
    parser.add_argument("--from-results", type=Path, action="append", help="compare existing results.json instead of running")
    parser.add_argument("--update-baseline", action="store_true", help="store the first variant's scores as the baseline")
    parser.add_argument("--require-baseline", action="store_true", help="fail if no baseline exists")
    args = parser.parse_args(argv)

    variants = []
    if args.from_results:
        for path in args.from_results:
            data = json.loads(path.read_text(encoding="utf-8"))
            variants.append({"variant": variant_name(data["target"]), "summary": data["summary"], "run_id": data["run_id"]})
    else:
        models = [m.split(":", 1) for m in args.models.split(",") if m] or [(None, None)]
        for prompt in args.prompts.split(","):
            for provider, model in models:
                try:
                    runner, cases = build_runner(profile=args.profile, prompt=prompt, provider=provider, model=model)
                except EvalSetupError as exc:
                    print(f"Cannot run regression: {exc}", file=sys.stderr)
                    return 2
                print(f"\n=== {variant_name(runner.target.describe())} ({len(cases)} cases) ===")
                report = runner.run(cases)
                write_report(report)
                variants.append({"variant": variant_name(report.target), "summary": report.summary, "run_id": report.run_id})

    baseline_path = BASELINE_DIR / f"{args.profile}.json"
    if args.update_baseline:
        ref = variants[0]
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps({
            "profile": args.profile, "variant": ref["variant"], "run_id": ref["run_id"],
            "created_at": datetime.now(UTC).isoformat(), "metrics": ref["summary"],
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Baseline written to {baseline_path} from {ref['variant']}")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
    for v in variants:
        v["comparison"] = compare(v["summary"], baseline["metrics"], max_drops()) if baseline else []
        v["regressed"] = any(r["regressed"] for r in v["comparison"])

    out = REPORTS_DIR / "regression"
    out.mkdir(parents=True, exist_ok=True)
    markdown = render(variants, baseline)
    (out / "comparison.md").write_text(markdown, encoding="utf-8")
    (out / "comparison.json").write_text(json.dumps({"baseline": baseline, "variants": variants}, indent=2), encoding="utf-8")
    print("\n" + markdown + f"\nReport: {out / 'comparison.md'}")

    if baseline is None:
        return 1 if args.require_baseline else 0
    return 1 if any(v["regressed"] for v in variants) else 0


if __name__ == "__main__":
    sys.exit(main())

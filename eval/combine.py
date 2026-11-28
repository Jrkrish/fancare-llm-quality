"""Combine RAG eval batches into one report, for when a full run does not fit in a day's quota.

    python -m eval.combine path/to/batch1/eval/latest/results.json path/to/batch2/... --profile full

Averages and percentiles are recomputed from every run, never averaged across batches. The batches
must share the system under test, judge, profile and runs per case (and the code commit, when each
has the batch.json that tools/campaign.py writes), and together cover every case exactly once.
The combined report is written like a normal run, so reports/eval/latest feeds calibration.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from eval.config import REPORTS_DIR, load_eval_config
from eval.dataset import GoldenCase, load_golden, select_cases
from eval.report import write_report
from eval.run import print_summary
from eval.runner import EvalReport, EvalRunner, RunScore
from eval.targets import Reply

SAME_IN_EVERY_BATCH = ("profile", "target", "judge", "runs_per_case")


class CombineError(ValueError):
    pass


class _Target:
    def __init__(self, description: dict):
        self.description = description

    def describe(self) -> dict:
        return self.description


class _Judge:
    """Carries the judge's name and token use into the report. It is never asked to score anything."""

    def __init__(self, name: str, usage: list[dict]):
        self.name = name
        self.calls, self.prompt_tokens, self.completion_tokens = (
            sum(u.get(k, 0) for u in usage) for k in ("calls", "prompt_tokens", "completion_tokens"))

    def get_model_name(self) -> str:
        return self.name


def _commit(results_path: Path) -> str | None:
    """The code commit from batch.json, found in the folders above results.json."""
    for folder in list(results_path.resolve().parents)[:3]:
        if (folder / "batch.json").exists():
            return json.loads((folder / "batch.json").read_text(encoding="utf-8"))["code_commit"]
    return None


def combine(paths: list[Path], profile: str, golden: list[GoldenCase] | None = None) -> EvalReport:
    batches = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    if not batches:
        raise CombineError("no batches given")
    for key in SAME_IN_EVERY_BATCH:
        values = {json.dumps(b[key], sort_keys=True) for b in batches}
        if len(values) > 1:
            raise CombineError(f"batches differ in {key}: {sorted(values)}")
    commits = {_commit(Path(p)) for p in paths}
    if len(commits) > 1:
        raise CombineError(f"batches come from different code commits: {sorted(map(str, commits))}")
    first = batches[0]
    if first["profile"] != profile:
        raise CombineError(f"batches are from profile '{first['profile']}', not '{profile}'")

    config = load_eval_config(profile, runs_per_case=first["runs_per_case"])
    expected = select_cases(golden or load_golden(config.dataset), per_category=config.cases_per_category)
    cases = [c for b in batches for c in b["cases"]]
    ids = [c["case"]["id"] for c in cases]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    missing = [c.id for c in expected if c.id not in ids]
    extra = sorted(set(ids) - {c.id for c in expected})
    if repeated or missing or extra:
        raise CombineError(f"cases must appear exactly once. repeated: {repeated} missing: {missing} unexpected: {extra}")

    judge = _Judge(first["judge"], [b.get("judge_usage") or {} for b in batches]) if first["judge"] else None
    if not judge:
        config.judge_metrics = []
    runner = EvalRunner(config, _Target(first["target"]), judge)
    by_id = {c["case"]["id"]: c for c in cases}
    results = []
    for case in expected:  # dataset order, as in a single run
        runs = [RunScore(**{**r, "reply": Reply(**r["reply"])}) for r in by_id[case.id]["runs"]]
        if len(runs) != config.runs_per_case:
            raise CombineError(f"{case.id}: {len(runs)} runs, expected {config.runs_per_case}")
        results.append(runner.aggregate_case(case, runs))
    return runner.build_report(results, min(datetime.fromisoformat(b["started_at"]) for b in batches))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", type=Path, help="results.json of each batch")
    parser.add_argument("--profile", default="full")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args(argv)
    try:
        report = combine(args.results, args.profile)
    except CombineError as exc:
        print(f"Cannot combine: {exc}", file=sys.stderr)
        return 2
    out = write_report(report, args.reports_dir)
    print_summary(report)
    print(f"\nCombined report: {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Tool-use eval: right tool, right arguments, no unnecessary calls, recovery from
injected failures, and no invented order/warranty/ticket data.

    python -m eval.tool_use --runs 3

Per run, each metric scores 1 or 0 (argument_accuracy is a fraction). A case passes
a metric if it passes in at least `min_pass_rate` of runs.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field

from app.config import ROOT
from app.tools import TOOL_SCHEMAS
from eval.checks import run_checks, ungrounded_facts
from eval.harness import AgentHarness
from eval.suite_report import md_table, pct, write_suite_report

DATASET = ROOT / "eval" / "datasets" / "tool_cases.json"
METRICS = ("tool_selection", "argument_accuracy", "no_unnecessary_calls", "recovery", "no_invented_data", "answer_checks")


@dataclass
class ToolCase:
    id: str
    message: str
    expected_calls: list[dict]
    checks: dict
    forbid_tools: list[str] = field(default_factory=list)
    max_calls: int = 3
    faults: dict[str, str] = field(default_factory=dict)
    failure_type: str = "none"


@dataclass
class ToolRun:
    answer: str
    calls: list[dict]
    scores: dict[str, float]
    problems: list[str]
    error: str | None = None


@dataclass
class ToolCaseResult:
    case: ToolCase
    runs: list[ToolRun]
    pass_rates: dict[str, float] = field(default_factory=dict)
    passed: bool = False


def load_tool_cases(path=DATASET) -> list[ToolCase]:
    return [ToolCase(**c) for c in json.loads(path.read_text(encoding="utf-8"))["cases"]]


def _arg_matches(expected, actual) -> bool:
    if isinstance(expected, list):
        return {str(x).lower() for x in expected} <= {str(x).lower() for x in (actual or [])}
    if isinstance(expected, (int, float)):
        try:
            return float(actual) == float(expected)
        except (TypeError, ValueError):
            return False
    return str(actual or "").strip().upper() == str(expected).strip().upper()


def call_matches(expected: dict, call: dict) -> bool:
    return call["name"] == expected["name"] and all(
        _arg_matches(v, call["arguments"].get(k)) for k, v in expected.get("args", {}).items()
    )


def score_run(
    case: ToolCase, answer: str, calls: list[dict], tool_results: list, check_problems: list[str], allowed: list[str] | None = None
) -> ToolRun:
    problems = list(check_problems)
    names = [c["name"] for c in calls]
    expected_names = {e["name"] for e in case.expected_calls}

    selection = 1.0 if expected_names <= set(names) else 0.0
    if not selection:
        problems.append(f"expected tools {sorted(expected_names)}, called {names}")

    if case.expected_calls:
        matched = [e for e in case.expected_calls if any(call_matches(e, c) for c in calls)]
        arg_acc = len(matched) / len(case.expected_calls)
        for e in case.expected_calls:
            if e not in matched and e["name"] in names:
                got = [c["arguments"] for c in calls if c["name"] == e["name"]]
                problems.append(f"{e['name']} expected args {e['args']}, got {got}")
    else:
        arg_acc = 1.0

    forbidden = [n for n in names if n in case.forbid_tools]
    unnecessary = 1.0 if not forbidden and len(calls) <= case.max_calls else 0.0
    if forbidden:
        problems.append(f"called forbidden tools {forbidden}")
    if len(calls) > case.max_calls:
        problems.append(f"{len(calls)} tool calls (max {case.max_calls})")

    # Facts may come from tool results, the user's own message, or the format examples in the tool descriptions.
    invented = ungrounded_facts(answer, tool_results, [case.message, *(allowed or [])])
    problems += invented

    scores = {
        "tool_selection": selection,
        "argument_accuracy": arg_acc,
        "no_unnecessary_calls": unnecessary,
        "no_invented_data": 0.0 if invented else 1.0,
        "answer_checks": 0.0 if check_problems else 1.0,
    }
    if case.faults:
        # Recovery = the failing tool was tried, the bot said the info is unavailable, and it invented nothing.
        scores["recovery"] = 1.0 if selection and not check_problems and not invented else 0.0
    return ToolRun(answer, calls, scores, problems)


class ToolUseRunner:
    def __init__(self, harness: AgentHarness, runs: int = 1, min_pass_rate: float = 0.66):
        self.harness = harness
        self.runs = runs
        self.min_pass_rate = min_pass_rate

    def run_once(self, case: ToolCase) -> ToolRun:
        agent = self.harness.new_agent(case.faults)
        turn = self.harness.ask(agent, self.harness.new_session(f"tool-{case.id}"), case.message)
        if turn.error:
            return ToolRun("", [], {}, [], error=turn.error)
        calls = [{"name": t.name, "arguments": t.arguments, "status": t.status} for t in turn.tool_records]
        results = [t.result for t in turn.tool_records if t.status == "ok"]
        check_problems = run_checks(case.checks, self.harness.context(agent, turn))
        return score_run(case, turn.answer, calls, results, check_problems, [json.dumps(TOOL_SCHEMAS)])

    def run_case(self, case: ToolCase) -> ToolCaseResult:
        runs = [self.run_once(case) for _ in range(self.runs)]
        result = ToolCaseResult(case, runs)
        valid = [r for r in runs if r.error is None]
        for m in METRICS:
            values = [r.scores[m] for r in valid if m in r.scores]
            if values:
                result.pass_rates[m] = sum(v >= 1.0 for v in values) / len(values)
        result.passed = bool(valid) and all(v >= self.min_pass_rate for v in result.pass_rates.values())
        return result


def summarise(results: list[ToolCaseResult]) -> dict:
    metric_means = {}
    for m in METRICS:
        values = [r.scores[m] for res in results for r in res.runs if r.error is None and m in r.scores]
        if values:
            metric_means[m] = statistics.fmean(values)
    by_failure: dict[str, list[float]] = defaultdict(list)
    for res in results:
        if res.case.faults:
            by_failure[res.case.failure_type] += [r.scores["recovery"] for r in res.runs if r.error is None]
    return {
        "metrics": metric_means,
        "recovery_by_failure_type": {k: statistics.fmean(v) for k, v in sorted(by_failure.items()) if v},
        "case_pass_rate": sum(r.passed for r in results) / len(results) if results else 0.0,
        "errors": sum(1 for res in results for r in res.runs if r.error),
    }


def render(results: list[ToolCaseResult], summary: dict, target: dict) -> str:
    lines = ["# Tool-use eval", "", f"Target: prompt {target['prompt_version']}, {target['provider']} / {target['model']}.", ""]
    lines += md_table(["Metric", "Mean over runs"], [[m, pct(v)] for m, v in summary["metrics"].items()])
    lines += ["", "## Recovery by injected failure type", ""]
    lines += md_table(["Failure type", "Recovery rate"], [[k, pct(v)] for k, v in summary["recovery_by_failure_type"].items()])
    lines += ["", f"Case pass rate: {pct(summary['case_pass_rate'])}. Errored runs: {summary['errors']}.", "",
              "## Failing cases", ""]
    for res in results:
        if not res.passed:
            run = next((r for r in res.runs if r.problems), res.runs[0])
            lines.append(f"- **{res.case.id}** `{res.case.message}` faults={res.case.faults or '-'}")
            lines += [f"  - {p}" for p in run.problems] or [f"  - error: {run.error}"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--case", action="append")
    args = parser.parse_args(argv)
    harness = AgentHarness()
    if harness.llm_ready():
        print(f"Cannot run tool-use eval: {harness.llm_ready()}", file=sys.stderr)
        return 2
    runner = ToolUseRunner(harness, args.runs)
    cases = [c for c in load_tool_cases() if not args.case or c.id in args.case]
    results = []
    for i, case in enumerate(cases, 1):
        res = runner.run_case(case)
        print(f"[{i}/{len(cases)}] {case.id} {'PASS' if res.passed else 'FAIL'}")
        results.append(res)
    summary = summarise(results)
    out = write_suite_report("tool_use", render(results, summary, harness.describe()),
                             {"target": harness.describe(), "summary": summary, "results": results})
    print(f"Report: {out / 'summary.md'}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

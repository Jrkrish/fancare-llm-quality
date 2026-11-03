"""Tool-use eval: right tool, right arguments, no unnecessary calls, recovery after each
injected failure type (timeout, malformed, empty), and no invented order/warranty data.

    pytest tests/tooluse --suite-runs 3
"""
import pytest

from eval.suite_report import write_suite_report
from eval.tool_use import ToolUseRunner, load_tool_cases, render, summarise
from tests.framework.base_test import BaseTest

pytestmark = [pytest.mark.llm, pytest.mark.tooluse]
CASES = load_tool_cases()


@pytest.fixture(scope="module")
def tool_session(harness, suite_runs):
    session = {"runner": ToolUseRunner(harness, suite_runs), "results": {}}
    yield session
    if session["results"]:
        results = list(session["results"].values())
        out = write_suite_report("tool_use", render(results, summarise(results), harness.describe()),
                                 {"target": harness.describe(), "summary": summarise(results), "results": results})
        print(f"\nTool-use report: {out / 'summary.md'}")


class TestToolUse(BaseTest):
    @pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
    def test_tool_case(self, case, tool_session):
        result = tool_session["runner"].run_case(case)
        tool_session["results"][case.id] = result
        if result.passed:
            return
        problems = [f"{m}: passed {rate:.0%} of runs" for m, rate in result.pass_rates.items() if rate < 0.66]
        sample = next((r for r in result.runs if r.problems or r.error), result.runs[0])
        problems += sample.problems or [f"error: {sample.error}"]
        problems.append(f"answer: {sample.answer[:300]}")
        self.fail_if_any(problems, f"{case.id} '{case.message}' faults={case.faults or '-'}")

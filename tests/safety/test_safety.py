"""Safety suite: direct and indirect prompt injection, jailbreaks, system-prompt extraction,
PII leakage (in answers and in stored logs) and blocked topics.

Each case is attacked --suite-runs times; any successful attempt fails the test.
At the end of the session the attack success rate (ASR) per category is written
to reports/safety/summary.md.

    pytest tests/safety --suite-runs 2
"""
import pytest

from eval.safety import SafetyRunner, load_safety_cases, render, summarise
from eval.suite_report import write_suite_report
from tests.framework.base_test import BaseTest

pytestmark = [pytest.mark.llm, pytest.mark.safety]
CASES = load_safety_cases()


@pytest.fixture(scope="module")
def safety_session(harness, suite_runs):
    session = {"runner": SafetyRunner(harness, suite_runs), "results": {}}
    yield session
    if session["results"]:
        results = list(session["results"].values())
        summary = summarise(results)
        out = write_suite_report("safety", render(results, summary, harness.describe()),
                                 {"target": harness.describe(), "summary": summary, "results": results})
        print(f"\nSafety ASR report: {out / 'summary.md'}")


class TestSafety(BaseTest):
    @pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
    def test_attack_is_resisted(self, case, safety_session):
        result = safety_session["runner"].run_case(case)
        safety_session["results"][case.id] = result
        if not result.valid:
            pytest.fail(f"{case.id}: every attempt errored: {result.attempts[0].error}", pytrace=False)
        problems = []
        for i, attempt in enumerate(result.attempts, 1):
            problems += [f"attempt {i}: {p}" for p in attempt.problems]
        if problems:
            problems.append(f"answer: {next(a.answer for a in result.attempts if a.problems)[:300]}")
        self.fail_if_any(problems, f"{case.id} [{case.category}] attack succeeded: {case.turns[-1][:100]}")

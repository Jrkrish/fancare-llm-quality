"""DeepEval RAG suite over the golden dataset.

Outputs are non-deterministic, so a case is not judged on one answer: it is
asked N times and must pass each metric in at least `min_case_pass_rate` of
runs. The final test applies suite-level gates to the mean scores.

    pytest tests/eval --eval-profile smoke
    pytest tests/eval --eval-profile full --prompt-version v2
"""
import pytest

from tests.framework.base_test import BaseTest

pytestmark = pytest.mark.eval


class TestRagEval(BaseTest):
    def test_golden_case(self, golden_case, eval_session):
        result = eval_session.runner.evaluate_case(golden_case)
        # A rerun (flaky retry) replaces the earlier attempt instead of double-counting it.
        eval_session.results = [r for r in eval_session.results if r.case.id != golden_case.id] + [result]

        if result.errored:
            pytest.fail(f"{golden_case.id}: every run failed to get an answer: {result.runs[0].reply.error}", pytrace=False)
        title = f"{golden_case.id} [{golden_case.category}, expects {golden_case.expected_behavior}] {golden_case.question}"
        sample = result.valid_runs[0]
        problems = result.failures + ([f"example answer ({sample.response_class}): {sample.answer[:300]}"] if result.failures else [])
        self.fail_if_any(problems, title)

    # Defined last so it runs after every golden case in this module.
    def test_suite_gates(self, eval_session):
        if not eval_session.results:
            pytest.skip("no golden cases were evaluated")
        report = eval_session.finalize()
        self.fail_if_any(report.gate_failures, f"Suite gates failed (report: {eval_session.out_dir / 'summary.md'})")

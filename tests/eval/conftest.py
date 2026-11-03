"""Fixtures for the DeepEval RAG suite.

Each golden case becomes its own pytest test (asked N times, judged on
pass-rate across runs). The session fixture collects every case result and,
at the end of the session, writes the same report and history row as
`python -m eval.run`.
"""
from dataclasses import dataclass, field

import pytest

from eval.config import load_eval_config
from eval.dataset import load_golden, select_cases
from eval.report import write_report
from eval.run import EvalSetupError, build_runner
from eval.runner import CaseResult, EvalReport, EvalRunner


@dataclass
class EvalSession:
    runner: EvalRunner
    results: list[CaseResult] = field(default_factory=list)
    report: EvalReport | None = None

    def finalize(self) -> EvalReport:
        if self.report is None:
            self.report = self.runner.build_report(self.results)
            self.out_dir = write_report(self.report)
        return self.report


def pytest_generate_tests(metafunc):
    if "golden_case" in metafunc.fixturenames:
        opts = metafunc.config.option
        cfg = load_eval_config(opts.eval_profile)
        cases = select_cases(load_golden(cfg.dataset), categories=opts.eval_category, per_category=cfg.cases_per_category)
        metafunc.parametrize("golden_case", cases, ids=[c.id for c in cases])


@pytest.fixture(scope="session")
def eval_session(request):
    opts = request.config.option
    try:
        runner, _ = build_runner(profile=opts.eval_profile, prompt=opts.prompt_version, runs=opts.eval_runs)
    except EvalSetupError as exc:
        pytest.skip(f"eval needs LLM access: {exc}")
    session = EvalSession(runner)
    yield session
    if session.results:
        session.finalize()
        print(f"\nEval report: {session.out_dir / 'summary.md'}")

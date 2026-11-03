"""Run the RAG eval from the command line.

    python -m eval.run --profile smoke
    python -m eval.run --profile full --prompt v2
    python -m eval.run --profile nojudge --category unanswerable --runs 1

Exits 1 if any suite gate fails, 2 if the target or judge is not configured.
"""
import argparse
import os
import sys
from dataclasses import replace

from app.config import PROVIDERS, Settings
from app.retrieval import HybridRetriever
from eval.config import load_eval_config
from eval.dataset import CATEGORIES, load_golden, select_cases
from eval.report import write_report
from eval.runner import EvalReport, EvalRunner
from eval.targets import AgentTarget


def provider_ready(provider: str) -> str | None:
    """Return an error message if the provider's API key is missing."""
    if provider not in PROVIDERS:
        return f"unknown provider '{provider}'"
    key_env = PROVIDERS[provider]["key_env"]
    if key_env and not os.getenv(key_env):
        return f"{key_env} is not set (needed for provider '{provider}')"
    return None


class EvalSetupError(RuntimeError):
    """The eval cannot run here (missing API key, bad option)."""


def build_runner(
    profile: str = "smoke",
    prompt: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    retriever_mode: str | None = None,
    runs: int | None = None,
    categories: list[str] | None = None,
    case_ids: list[str] | None = None,
    no_judge: bool = False,
) -> tuple[EvalRunner, list]:
    config = load_eval_config(profile, runs_per_case=runs)
    if no_judge:
        config.judge_metrics = []

    overrides = {"prompt_version": prompt, "llm_provider": provider, "llm_model": model, "retriever_mode": retriever_mode}
    settings = replace(Settings(), llm_fallback_provider="", **{k: v for k, v in overrides.items() if v})

    for name in [settings.llm_provider] + ([config.judge_provider] if config.judge_metrics else []):
        problem = provider_ready(name)
        if problem:
            raise EvalSetupError(f"{problem}. See .env.example.")

    retriever = HybridRetriever(settings)
    cases = select_cases(
        load_golden(config.dataset, known_chunk_ids=set(retriever.by_id)),
        categories=categories, ids=case_ids, per_category=config.cases_per_category,
    )
    judge = None
    if config.judge_metrics:
        from eval.judge import JudgeLLM
        judge = JudgeLLM(config.judge_provider, config.judge_model, config.judge_min_interval_s)
    return EvalRunner(config, AgentTarget(settings, retriever), judge), cases


def print_summary(report: EvalReport) -> None:
    print(f"\nGates: {'PASSED' if report.gates_passed else 'FAILED'}")
    for g in report.gate_results:
        print(f"  {'ok  ' if g['passed'] else 'FAIL'} {g['metric']:<22} {g['value']:.3f}  ({g['op']} {g['limit']})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="smoke", help="smoke | nojudge | full (see eval/config.toml)")
    parser.add_argument("--prompt", help="prompt version, e.g. v1, v2, v3")
    parser.add_argument("--provider", help="gemini | groq | ollama")
    parser.add_argument("--model", help="model name for the provider")
    parser.add_argument("--retriever", choices=["hybrid", "bm25", "vector"])
    parser.add_argument("--runs", type=int, help="runs per case (overrides profile)")
    parser.add_argument("--category", action="append", choices=CATEGORIES, help="repeatable")
    parser.add_argument("--case", action="append", help="case id, repeatable")
    parser.add_argument("--no-judge", action="store_true", help="skip LLM-judged metrics")
    args = parser.parse_args(argv)

    try:
        runner, cases = build_runner(
            args.profile, args.prompt, args.provider, args.model, args.retriever,
            args.runs, args.category, args.case, args.no_judge,
        )
    except EvalSetupError as exc:
        print(f"Cannot run eval: {exc}", file=sys.stderr)
        return 2
    est = runner.estimate_calls(cases)
    print(f"Evaluating {len(cases)} cases x {runner.config.runs_per_case} runs against {runner.target.describe()}")
    print(f"Approx. LLM calls: system under test ~{est['sut_calls_approx']}, judge ~{est['judge_calls_approx']}")

    report = runner.run(cases)
    out_dir = write_report(report)
    print_summary(report)
    print(f"\nReport: {out_dir / 'summary.md'}")
    return 0 if report.gates_passed else 1


if __name__ == "__main__":
    sys.exit(main())

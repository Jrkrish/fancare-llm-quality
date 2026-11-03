"""Eval engine: asks every case N times, scores each run, aggregates with variance, applies gates."""
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from deepeval.test_case import LLMTestCase

from eval.config import EvalConfig
from eval.dataset import GoldenCase
from eval.metrics import DETERMINISTIC_METRICS, build_judge_metric, classify_response, rejected_items
from eval.targets import ChatTarget, Reply

# Approximate LLM calls per scored run, used only for the up-front budget estimate.
JUDGE_CALLS_PER_METRIC = {"faithfulness": 3, "answer_relevancy": 2, "contextual_precision": 1, "contextual_recall": 1}


@dataclass
class RunScore:
    run_index: int
    answer: str
    response_class: str
    scores: dict[str, float | None]
    metric_errors: dict[str, str]
    hallucinated: bool
    hallucination_reasons: list[str]
    reply: Reply
    # Claims/statements a judge metric did not accept, e.g. {"faithfulness": ["[no] claim -- reason"]}.
    judge_notes: dict[str, list[str]] = field(default_factory=dict)

    @property
    def errored(self) -> bool:
        return self.reply.error is not None


@dataclass
class MetricStats:
    mean: float
    std: float
    min: float
    pass_rate: float
    n: int


@dataclass
class CaseResult:
    case: GoldenCase
    runs: list[RunScore]
    stats: dict[str, MetricStats] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    unstable: list[str] = field(default_factory=list)

    @property
    def valid_runs(self) -> list[RunScore]:
        return [r for r in self.runs if not r.errored]

    @property
    def errored(self) -> bool:
        return not self.valid_runs

    @property
    def passed(self) -> bool:
        return not self.errored and not self.failures


@dataclass
class EvalReport:
    run_id: str
    started_at: str
    profile: str
    target: dict
    judge: str | None
    runs_per_case: int
    cases: list[CaseResult]
    summary: dict[str, float] = field(default_factory=dict)
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    gate_results: list[dict] = field(default_factory=list)
    judge_usage: dict[str, int] = field(default_factory=dict)

    @property
    def gates_passed(self) -> bool:
        return all(g["passed"] for g in self.gate_results)

    @property
    def valid(self) -> bool:
        """False when infrastructure or judge errors make the scores unrepresentative."""
        return all(g["passed"] for g in self.gate_results if g.get("validity"))

    @property
    def gate_failures(self) -> list[str]:
        return [f"{g['metric']}: {g['value']:.3f} (needs {g['op']} {g['limit']})" for g in self.gate_results if not g["passed"]]


class EvalRunner:
    def __init__(self, config: EvalConfig, target: ChatTarget, judge=None):
        self.config = config
        self.target = target
        self.judge = judge
        if config.judge_metrics and judge is None:
            raise ValueError("judge metrics are enabled but no judge model was given")

    # ------------------------------------------------------------ estimation

    def estimate_calls(self, cases: list[GoldenCase]) -> dict[str, int]:
        n_runs = len(cases) * self.config.runs_per_case
        judged = sum(1 for c in cases if c.expected_behavior != "refuse") * self.config.runs_per_case
        per_run = sum(JUDGE_CALLS_PER_METRIC[m] for m in self.config.judge_metrics)
        return {"case_runs": n_runs, "sut_calls_approx": n_runs * 2, "judge_calls_approx": judged * per_run}

    # --------------------------------------------------------------- scoring

    def metrics_for(self, case: GoldenCase) -> list[str]:
        names = ["refusal_correctness", "no_hallucination"]
        if case.expected_chunk_ids:
            names.append("retrieval_recall")
        if case.expected_behavior != "refuse":
            names += ["citation_correctness", *self.config.judge_metrics]
        return names

    def score_run(self, case: GoldenCase, run_index: int, reply: Reply) -> RunScore:
        response_class = classify_response(reply.answer, reply.citations)
        meta = {
            "expected_behavior": case.expected_behavior,
            "response_class": response_class,
            "citations": reply.citations,
            "retrieved_chunk_ids": reply.retrieved_chunk_ids,
            "expected_chunk_ids": list(case.expected_chunk_ids),
        }
        test_case = LLMTestCase(
            input=case.question,
            actual_output=reply.answer,
            expected_output=case.expected_output,
            retrieval_context=[p["passage"] for p in reply.retrieved] or None,
            metadata=meta,
        )
        scores: dict[str, float | None] = {}
        errors: dict[str, str] = {}
        notes: dict[str, list[str]] = {}
        for name in self.metrics_for(case):
            if name == "no_hallucination":
                continue
            try:
                scores[name] = self._score_metric(name, test_case, response_class, notes)
            except Exception as exc:  # noqa: BLE001 - a judge failure must not abort the whole run
                scores[name], errors[name] = None, f"{type(exc).__name__}: {str(exc)[:500]}"

        reasons = []
        if case.expected_behavior == "refuse" and response_class == "answered":
            reasons.append("answered an unanswerable question")
        if any(c.get("passage") is None for c in reply.citations):
            reasons.append("cited a chunk that does not exist")
        faith = scores.get("faithfulness")
        if faith is not None and faith < self.config.hallucination_faithfulness_below:
            reasons.append(f"faithfulness {faith:.2f}")
        scores["no_hallucination"] = 0.0 if reasons else 1.0
        return RunScore(run_index, reply.answer, response_class, scores, errors, bool(reasons), reasons, reply, notes)

    def _score_metric(self, name: str, test_case: LLMTestCase, response_class: str, notes: dict) -> float | None:
        if name in DETERMINISTIC_METRICS:
            return DETERMINISTIC_METRICS[name](self.config.thresholds[name]).measure(test_case)
        has_context = bool(test_case.retrieval_context)
        if name == "faithfulness":
            if response_class == "refused":
                return None  # nothing claimed, nothing to ground
            if not has_context:
                return 0.0  # claims made with no evidence retrieved at all
        if name == "answer_relevancy" and response_class == "refused":
            return 0.0  # declining an answerable question is not a relevant answer
        if name.startswith("contextual_") and not has_context:
            return 0.0  # nothing retrieved: no gold evidence reached the model
        metric = build_judge_metric(name, self.config.thresholds[name], self.judge)
        score = metric.measure(test_case)
        rejected = rejected_items(metric)
        if rejected:
            notes[name] = rejected
        return score

    # ----------------------------------------------------------- aggregation

    def evaluate_case(self, case: GoldenCase) -> CaseResult:
        runs = []
        for i in range(self.config.runs_per_case):
            reply = self.target.ask(f"eval-{case.id}-{uuid.uuid4().hex[:8]}", case.question)
            runs.append(self.score_run(case, i, reply) if reply.error is None else
                        RunScore(i, "", "error", {}, {}, False, [], reply))
        return self.aggregate_case(case, runs)

    def aggregate_case(self, case: GoldenCase, runs: list[RunScore]) -> CaseResult:
        result = CaseResult(case, runs)
        for name in self.metrics_for(case):
            values = [r.scores[name] for r in result.valid_runs if r.scores.get(name) is not None]
            if not values:
                continue
            threshold = self.config.thresholds[name]
            stats = MetricStats(
                mean=statistics.fmean(values),
                std=statistics.pstdev(values),
                min=min(values),
                pass_rate=sum(v >= threshold for v in values) / len(values),
                n=len(values),
            )
            result.stats[name] = stats
            if stats.pass_rate < self.config.min_case_pass_rate:
                result.failures.append(f"{name}: passed {stats.pass_rate:.0%} of runs (mean {stats.mean:.2f}, threshold {threshold})")
            if stats.std > self.config.unstable_std or 0 < stats.pass_rate < 1:
                result.unstable.append(f"{name}: std {stats.std:.2f}, pass rate {stats.pass_rate:.0%}")
        return result

    def run(self, cases: list[GoldenCase], progress=print) -> EvalReport:
        started = datetime.now(UTC)
        results = []
        for i, case in enumerate(cases, start=1):
            t0 = time.perf_counter()
            result = self.evaluate_case(case)
            status = "ERROR" if result.errored else ("PASS" if result.passed else "FAIL")
            progress(f"[{i}/{len(cases)}] {case.id:<7} {case.category:<13} {status:<5} ({time.perf_counter() - t0:.1f}s)")
            results.append(result)
        return self.build_report(results, started)

    def build_report(self, results: list[CaseResult], started: datetime | None = None) -> EvalReport:
        started = started or datetime.now(UTC)
        report = EvalReport(
            run_id=started.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6],
            started_at=started.isoformat(),
            profile=self.config.profile,
            target=self.target.describe(),
            judge=self.judge.get_model_name() if self.judge else None,
            runs_per_case=self.config.runs_per_case,
            cases=results,
        )
        report.summary = summarise(results)
        categories = sorted({r.case.category for r in results})
        report.by_category = {c: summarise([r for r in results if r.case.category == c]) for c in categories}
        report.gate_results = self.apply_gates(report.summary)
        if self.judge:
            report.judge_usage = {"calls": self.judge.calls, "prompt_tokens": self.judge.prompt_tokens,
                                  "completion_tokens": self.judge.completion_tokens}
        return report

    def apply_gates(self, summary: dict[str, float]) -> list[dict]:
        gates = []
        for metric, limit in self.config.gates.items():
            if metric in summary:
                gates.append({"metric": metric, "value": summary[metric], "op": ">=", "limit": limit,
                              "passed": summary[metric] >= limit})
        if "hallucination_rate" in summary:
            value = summary["hallucination_rate"]
            gates.append({"metric": "hallucination_rate", "value": value, "op": "<=",
                          "limit": self.config.max_hallucination_rate, "passed": value <= self.config.max_hallucination_rate})
        # Validity gates: too many missing answers or missing judge scores make the averages above meaningless.
        for metric in ("error_rate", "judge_error_rate"):
            value = summary.get(metric, 0.0)
            gates.append({"metric": metric, "value": value, "op": "<=", "limit": self.config.max_error_rate,
                          "passed": value <= self.config.max_error_rate, "validity": True})
        return gates


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarise(results: list[CaseResult]) -> dict[str, float]:
    """Macro averages: each case contributes its mean over runs, so every case weighs the same."""
    summary: dict[str, float] = {}
    metric_names = sorted({m for r in results for m in r.stats})
    for name in metric_names:
        means = [r.stats[name].mean for r in results if name in r.stats]
        summary[name] = statistics.fmean(means)
    all_runs = [run for r in results for run in r.runs]
    valid = [run for run in all_runs if not run.errored]
    if valid:
        summary["hallucination_rate"] = sum(run.hallucinated for run in valid) / len(valid)
        latencies = [run.reply.latency_ms for run in valid]
        summary["latency_p50_ms"] = _percentile(latencies, 0.50)
        summary["latency_p95_ms"] = _percentile(latencies, 0.95)
        summary["cost_usd_total"] = sum(run.reply.cost_usd for run in valid)
    summary["error_rate"] = (len(all_runs) - len(valid)) / len(all_runs) if all_runs else 0.0
    # Judge measurements attempted vs failed (rate limits, quota, unparseable output). A judge metric that
    # is legitimately not applicable (e.g. faithfulness of a refusal) is None without an error and not counted.
    attempted = sum(1 for run in valid for m, v in run.scores.items() if m in JUDGE_CALLS_PER_METRIC and (v is not None or m in run.metric_errors))
    failed = sum(1 for run in valid for m in run.metric_errors if m in JUDGE_CALLS_PER_METRIC)
    summary["judge_error_rate"] = failed / attempted if attempted else 0.0
    summary["case_pass_rate"] = sum(r.passed for r in results) / len(results) if results else 0.0
    summary["unstable_cases"] = sum(bool(r.unstable) for r in results)
    summary["n_cases"] = len(results)
    summary["n_runs"] = len(all_runs)
    return summary

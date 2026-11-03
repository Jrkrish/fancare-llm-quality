"""Offline tests for the eval engine: dataset integrity, deterministic metrics,
aggregation, gates, reports and the DeepEval judge wiring. No network calls."""
import csv
import json
from collections import Counter
from dataclasses import replace

import pytest
from pydantic import BaseModel

from app.llm import LLMResponse
from eval.config import load_eval_config
from eval.dataset import CATEGORIES, DatasetError, GoldenCase, load_golden, select_cases
from eval.judge import JudgeLLM
from eval.metrics import build_judge_metric, citation_f1, classify_response, refusal_is_correct
from eval.report import write_report
from eval.runner import EvalRunner
from eval.targets import Reply


@pytest.fixture(scope="module")
def golden(retriever):
    cfg = load_eval_config("full")
    return load_golden(cfg.dataset, known_chunk_ids=set(retriever.by_id))


class TestGoldenDataset:
    def test_size_and_coverage(self, golden):
        counts = Counter(c.category for c in golden)
        assert len(golden) >= 60
        assert set(counts) == set(CATEGORIES)
        assert counts["near_miss"] >= 5

    def test_every_behaviour_is_represented(self, golden):
        assert {c.expected_behavior for c in golden} == {"answer", "partial", "refuse"}

    def test_unknown_chunk_id_is_rejected(self, tmp_path):
        bad = {"cases": [{"id": "X-1", "category": "answerable", "expected_behavior": "answer",
                          "question": "q", "expected_output": "a", "expected_chunk_ids": ["DOC-99:1"]}]}
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(DatasetError, match="DOC-99:1"):
            load_golden(path, known_chunk_ids={"DOC-01:1"})

    def test_stratified_selection(self, golden):
        subset = select_cases(golden, per_category=2)
        assert Counter(c.category for c in subset) == {c: 2 for c in CATEGORIES}


CITED = [{"chunk_id": "DOC-03:1", "doc_id": "DOC-03", "passage": "2.4 GHz only", "retrieved": True}]


@pytest.mark.parametrize(
    "answer, citations, expected",
    [
        ("It supports 2.4 GHz only [DOC-03:1].", CITED, "answered"),
        ("I don't have that information in the Corvane documentation.", [], "refused"),
        ("Sorry, I can only help with questions about Corvane ceiling fans.", [], "refused"),
        ("2.4 GHz only [DOC-03:1]. Wi-Fi 6 certification is not mentioned in the documentation.", CITED, "partial"),
        ("HomeKit isn't listed; only Alexa and Google Home are [DOC-03:1].", CITED, "partial"),
        ("Installation labour is not covered by the warranty [DOC-03:1].", CITED, "answered"),
        ("I can't follow that request. I can help with Corvane ceiling fans.", [], "refused"),
    ],
)
def test_classify_response(answer, citations, expected):
    assert classify_response(answer, citations) == expected


@pytest.mark.parametrize(
    "behavior, response_class, ok",
    [("answer", "answered", True), ("answer", "partial", True), ("answer", "refused", False),
     ("partial", "partial", True), ("partial", "answered", False), ("partial", "refused", False),
     ("refuse", "refused", True), ("refuse", "partial", True), ("refuse", "answered", False)],
)
def test_refusal_is_correct(behavior, response_class, ok):
    assert refusal_is_correct(behavior, response_class) is ok


class TestCitationF1:
    def test_perfect(self):
        assert citation_f1(CITED, {"DOC-03"}) == 1.0

    def test_no_citations_scores_zero(self):
        assert citation_f1([], {"DOC-03"}) == 0.0

    def test_unretrieved_citation_counts_as_wrong(self):
        cites = [{**CITED[0], "retrieved": False}]
        assert citation_f1(cites, {"DOC-03"}) == 0.0

    def test_partial_recall_on_multi_hop(self):
        # 1 correct citation of 1 (precision 1), 1 of 2 gold docs (recall 0.5) -> F1 0.667
        assert citation_f1(CITED, {"DOC-03", "DOC-12"}) == pytest.approx(2 / 3)


class ScriptedTarget:
    """Returns replies from a per-question queue."""

    def __init__(self, replies: dict[str, list[Reply]]):
        self.replies = {q: list(r) for q, r in replies.items()}

    def describe(self):
        return {"prompt_version": "test", "provider": "fake", "model": "fake", "retriever_mode": "hybrid", "rerank": True}

    def ask(self, session_id, message):
        return self.replies[message].pop(0)


ANSWER_CASE = GoldenCase("T-ANS", "answerable", "Wi-Fi band?", "answer", "2.4 GHz only.", ("DOC-03:1",))
REFUSE_CASE = GoldenCase("T-UNA", "unanswerable", "Matter support?", "refuse", "I don't have that information.")
GOOD = Reply("It supports 2.4 GHz only [DOC-03:1].", CITED, [{"chunk_id": "DOC-03:1", "passage": "2.4 GHz only"}], latency_ms=100)
REFUSAL = Reply("I don't have that information in the Corvane documentation.", latency_ms=50)
INVENTED = Reply("Yes, it supports Matter over Thread.", latency_ms=80)


@pytest.fixture
def config():
    return replace(load_eval_config("nojudge"), runs_per_case=3)


def test_case_passes_on_majority_and_is_flagged_unstable(config):
    target = ScriptedTarget({ANSWER_CASE.question: [GOOD, GOOD, REFUSAL]})
    result = EvalRunner(config, target).evaluate_case(ANSWER_CASE)
    assert result.passed
    assert result.stats["refusal_correctness"].pass_rate == pytest.approx(2 / 3)
    assert any(u.startswith("refusal_correctness") for u in result.unstable)


def test_answering_an_unanswerable_question_is_a_hallucination(config):
    target = ScriptedTarget({REFUSE_CASE.question: [INVENTED, INVENTED, REFUSAL]})
    result = EvalRunner(config, target).evaluate_case(REFUSE_CASE)
    assert not result.passed
    assert sum(r.hallucinated for r in result.runs) == 2
    assert result.runs[0].hallucination_reasons == ["answered an unanswerable question"]


def test_errored_runs_are_excluded_and_reported(config):
    target = ScriptedTarget({ANSWER_CASE.question: [GOOD, Reply("", error="HTTP 429"), GOOD]})
    runner = EvalRunner(config, target)
    report = runner.build_report([runner.evaluate_case(ANSWER_CASE)])
    assert report.cases[0].stats["refusal_correctness"].n == 2
    assert report.summary["error_rate"] == pytest.approx(1 / 3)
    assert "error_rate" in " ".join(report.gate_failures)  # 33% > max_error_rate


def test_report_files_and_history(config, tmp_path):
    target = ScriptedTarget({ANSWER_CASE.question: [GOOD] * 3, REFUSE_CASE.question: [INVENTED] * 3})
    runner = EvalRunner(config, target)
    report = runner.build_report([runner.evaluate_case(ANSWER_CASE), runner.evaluate_case(REFUSE_CASE)])
    assert report.summary["hallucination_rate"] == pytest.approx(0.5)
    assert not report.gates_passed

    out = write_report(report, reports_dir=tmp_path)
    summary = (out / "summary.md").read_text()
    assert "## Gates: FAILED" in summary and "T-UNA" in summary
    assert json.loads((out / "results.json").read_text())["run_id"] == report.run_id
    assert (tmp_path / "eval" / "latest" / "summary.md").exists()

    write_report(report, reports_dir=tmp_path)
    rows = list(csv.DictReader((tmp_path / "eval_history.csv").open()))
    assert len(rows) == 2 and rows[0]["hallucination_rate"] == "0.500" and rows[0]["gates_passed"] == "False"


class FakeJudgeClient:
    """Stands in for the provider HTTP call and answers DeepEval's prompts by their wording."""

    def __init__(self, verdict: str):
        self.verdict = verdict

    def chat(self, messages, tools=None, response_format=None):
        prompt = messages[0]["content"]
        if '"verdicts"' in prompt:
            content = json.dumps({"verdicts": [{"verdict": self.verdict, "reason": "scripted"}]})
        elif '"truths"' in prompt:
            content = "```json\n" + json.dumps({"truths": ["The fan supports 2.4 GHz Wi-Fi only."]}) + "\n```"
        else:
            content = json.dumps({"claims": ["The fan supports 2.4 GHz Wi-Fi."]})
        return LLMResponse(content, [], "fake", "fake", {"prompt_tokens": 10, "completion_tokens": 5})


class TestJudgeWiring:
    def test_generate_parses_into_schema(self):
        class Out(BaseModel):
            items: list[str]

        judge = JudgeLLM("groq", "fake")
        judge.model = type("C", (), {"chat": lambda self, *a, **k: LLMResponse('["a", "b"]', [], "f", "f", {})})()
        assert judge.generate("return json", schema=Out).items == ["a", "b"]  # bare list wrapped into the schema

    @pytest.mark.parametrize("verdict, expected", [("yes", 1.0), ("no", 0.0)])
    def test_deepeval_faithfulness_runs_on_custom_judge(self, verdict, expected):
        from deepeval.test_case import LLMTestCase

        judge = JudgeLLM("groq", "fake")
        judge.model = FakeJudgeClient(verdict)
        metric = build_judge_metric("faithfulness", 0.7, judge)
        case = LLMTestCase(input="Wi-Fi?", actual_output="It uses 2.4 GHz.", retrieval_context=["2.4 GHz only"])
        assert metric.measure(case) == expected
        assert judge.calls >= 3


def test_judge_errors_make_the_run_invalid(config):
    from eval.runner import RunScore

    target = ScriptedTarget({ANSWER_CASE.question: [GOOD] * 3})
    runner = EvalRunner(config, target)
    result = runner.evaluate_case(ANSWER_CASE)
    for run in result.runs:  # simulate a judge that ran out of quota on every run
        run.scores["faithfulness"] = None
        run.metric_errors["faithfulness"] = "QuotaExhausted: daily quota"
    report = runner.build_report([result])
    assert report.summary["judge_error_rate"] == 1.0
    assert not report.valid
    assert "judge_error_rate" in " ".join(report.gate_failures)
    assert isinstance(result.runs[0], RunScore)


def test_not_applicable_judge_metric_is_not_an_error(config):
    target = ScriptedTarget({REFUSE_CASE.question: [REFUSAL] * 3})
    runner = EvalRunner(config, target)
    report = runner.build_report([runner.evaluate_case(REFUSE_CASE)])
    assert report.summary["judge_error_rate"] == 0.0 and report.valid

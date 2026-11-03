"""Offline tests for the regression gate, drift detection, calibration maths,
flakiness ranking and the production-failure promotion helper."""
import json

import pytest

from eval.calibration import agreement, cohen_kappa
from eval.dataset import load_golden
from eval.promote_failure import append_case
from eval.regression import compare
from monitoring.drift import detect_drift, svg_chart
from tools.flaky_report import analyse

TH = {"max_score_drop": 0.1, "max_hallucination_increase": 0.1, "max_latency_p95_increase_pct": 50, "max_cost_increase_pct": 50}


class TestRegressionCompare:
    DROPS = {"faithfulness": 0.05, "hallucination_rate": 0.05}

    def test_within_threshold_passes(self):
        rows = compare({"faithfulness": 0.82, "hallucination_rate": 0.10}, {"faithfulness": 0.85, "hallucination_rate": 0.08}, self.DROPS)
        assert not any(r["regressed"] for r in rows)

    def test_drop_beyond_threshold_regresses(self):
        rows = compare({"faithfulness": 0.70}, {"faithfulness": 0.85}, self.DROPS)
        assert rows[0]["regressed"] and rows[0]["delta"] == pytest.approx(-0.15)

    def test_hallucination_increase_regresses(self):
        rows = compare({"hallucination_rate": 0.20}, {"hallucination_rate": 0.05}, self.DROPS)
        assert rows[0]["regressed"]

    def test_metrics_missing_from_either_side_are_ignored(self):
        assert compare({"faithfulness": 0.1}, {}, self.DROPS) == []


def row(ts, **kw):
    base = {"timestamp": ts, "model": "m", "refusal_correctness": 0.9, "citation_correctness": 0.8,
            "retrieval_recall": 0.8, "faithfulness": None, "hallucination_rate": 0.05,
            "latency_p95_ms": 2000.0, "cost_usd_per_query": 0.001}
    return {**base, **kw}


class TestDrift:
    def test_stable_history_has_no_drift(self):
        assert detect_drift([row(f"2026-11-0{i}") for i in range(1, 6)], 5, TH) == []

    def test_score_drop_latency_and_cost_are_flagged(self):
        hist = [row(f"2026-11-0{i}") for i in range(1, 6)]
        hist.append(row("2026-11-06", refusal_correctness=0.6, latency_p95_ms=4000.0, cost_usd_per_query=0.003))
        flags = detect_drift(hist, 5, TH)
        assert len(flags) == 3

    def test_model_change_is_flagged(self):
        flags = detect_drift([row("2026-11-01"), row("2026-11-02", model="other")], 5, TH)
        assert any("model changed" in f for f in flags)

    def test_svg_chart_renders(self):
        svg = svg_chart([row("2026-11-01"), row("2026-11-02")], ["refusal_correctness"], "Scores", ymax=1.0)
        assert svg.startswith("<svg") and "polyline" in svg


class TestCalibration:
    def test_kappa(self):
        assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
        assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0

    def test_unreliable_metric_is_flagged(self):
        items = [{"judge_scores": {"faithfulness": 0.9, "answer_relevancy": 0.9, "refusal_correctness": 1.0},
                  "human": {"faithful": i % 2 == 0, "relevant": True, "behaviour_correct": True}} for i in range(12)]
        result = agreement(items, {"faithfulness": 0.7, "answer_relevancy": 0.7, "refusal_correctness": 1.0})
        assert not result["faithfulness"]["reliable"]  # judge always passes, human fails half
        assert result["faithfulness"]["judge_pass_human_fail"] == 6
        assert result["answer_relevancy"]["accuracy"] == 1.0


class TestFlakyReport:
    def test_ranks_rerun_and_flipping_tests(self):
        recs = []
        for i, (a, b) in enumerate([("passed", "passed"), ("failed", "passed"), ("passed", "passed")]):
            s, ts = f"s{i}", f"2026-11-0{i + 1}"
            recs.append({"session": s, "timestamp": ts, "nodeid": "t::stable", "outcome": a if i else "passed"})
            recs.append({"session": s, "timestamp": ts, "nodeid": "t::flips", "outcome": b if i != 1 else "failed"})
        recs.append({"session": "s3", "timestamp": "2026-11-04", "nodeid": "t::rerun", "outcome": "rerun"})
        recs.append({"session": "s3", "timestamp": "2026-11-04", "nodeid": "t::rerun", "outcome": "passed"})
        ranked = {r["nodeid"]: r for r in analyse(recs)}
        assert ranked["t::flips"]["flip_rate"] == 1.0
        assert ranked["t::rerun"]["passed_after_rerun"] == 1


def test_promoted_case_keeps_file_valid(tmp_path, retriever):
    from eval.dataset import GoldenCase

    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({"cases": [{"id": "A", "category": "answerable", "expected_behavior": "answer",
                                             "question": "q", "expected_output": "a", "expected_chunk_ids": ["DOC-03:1"]}]}, indent=2)
                      .replace('"cases": [', '"cases": [\n   ').replace("}\n  ]\n}", "}\n  ]\n}"))
    append_case(GoldenCase("PROD-01", "near_miss", "Does it support HomeKit?", "refuse", "Not documented.", (), "from prod"), golden)
    cases = load_golden(golden, known_chunk_ids=set(retriever.by_id))
    assert [c.id for c in cases] == ["A", "PROD-01"]

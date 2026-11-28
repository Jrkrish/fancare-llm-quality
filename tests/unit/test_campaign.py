"""Measurement campaign: picking the due item, validating a run, and combining eval batches."""
import json
from datetime import UTC, datetime

import pytest

from eval.combine import CombineError, combine
from eval.config import load_eval_config
from eval.dataset import load_golden, select_cases
from eval.report import write_report
from eval.runner import EvalRunner, RunScore
from eval.targets import Reply
from tools.campaign import check, command, load_manifest, select

MANIFEST = load_manifest()
LATER = datetime(2026, 12, 20, tzinfo=UTC)
GOLDEN = load_golden(load_eval_config("full").dataset)


def art(name, created="2026-11-29T18:00:00Z"):
    return {"name": name, "created_at": created}


class TestManifest:
    def test_full_eval_items_cover_every_golden_case_once(self):
        ids = [c for i in MANIFEST["items"] if i["id"].startswith("full-") for c in i["cases"]]
        assert sorted(ids) == sorted(c.id for c in GOLDEN)

    def test_item_ids_are_unique(self):
        ids = [i["id"] for i in MANIFEST["items"]]
        assert len(ids) == len(set(ids))

    def test_expected_case_counts_match_the_datasets(self):
        by_id = {i["id"]: i for i in MANIFEST["items"]}
        smoke = select_cases(GOLDEN, per_category=load_eval_config("smoke").cases_per_category)
        assert by_id["regression-v1"]["expected_cases"] == len(smoke)
        for item, name, key in [("safety", "safety_cases", "cases"), ("tool-use", "tool_cases", "cases"),
                                ("conversation", "conversations", "conversations")]:
            data = json.loads((load_eval_config("full").dataset.parent / f"{name}.json").read_text(encoding="utf-8"))
            assert by_id[item]["expected_cases"] == len(data[key])

    def test_commands(self):
        assert command(MANIFEST["items"][0])[:6] == ["-m", "eval.run", "--profile", "full", "--case", "ANS-01"]
        assert command({"kind": "safety", "args": ["--runs", "1"]}) == ["-m", "eval.safety", "--runs", "1"]


class TestSelect:
    """Uses its own small manifest, so resizing the real campaign's items does not break these."""

    SMALL = {"code_commit": "abcdef1234", "not_before": "2026-11-29T17:00:00+00:00", "min_gap_hours": 24,
             "max_invalid_attempts": 2, "items": [{"id": "first"}, {"id": "second"}]}
    FIRST = "batch-abcdef1-first"

    def test_nothing_runs_before_the_start_time(self):
        chosen = select(self.SMALL, [], datetime(2026, 11, 29, 16, 59, tzinfo=UTC))
        assert not chosen["due"] and chosen["item"]["id"] == "first"

    def test_first_item_is_due_after_the_start_time(self):
        assert select(self.SMALL, [], datetime(2026, 11, 29, 17, 1, tzinfo=UTC))["due"]

    def test_waits_a_day_after_the_last_run_then_moves_on(self):
        done = [art(self.FIRST)]
        assert not select(self.SMALL, done, datetime(2026, 11, 30, 17, 59, tzinfo=UTC))["due"]
        chosen = select(self.SMALL, done, datetime(2026, 11, 30, 18, 0, tzinfo=UTC))
        assert chosen["due"] and chosen["item"]["id"] == "second"

    def test_invalid_run_is_repeated_then_blocks(self):
        once = [art(self.FIRST + "--invalid")]
        chosen = select(self.SMALL, once, LATER)
        assert chosen["due"] and chosen["item"]["id"] == "first"
        twice = select(self.SMALL, once * 2, LATER)
        assert twice["blocked"] and not twice["due"]

    def test_artifacts_from_other_code_or_workflows_are_ignored(self):
        others = [art("batch-0000000-first"), art("drift-report", "2026-12-20T00:00:00Z")]
        assert select(self.SMALL, others, LATER)["item"]["id"] == "first"

    def test_complete_campaign(self):
        done = [art(self.FIRST), art("batch-abcdef1-second")]
        chosen = select(self.SMALL, done, LATER)
        assert not chosen["due"] and chosen["item"] is None


class TestCheck:
    @staticmethod
    def write(tmp_path, folder, data):
        out = tmp_path / "reports" / folder
        out.mkdir(parents=True)
        (out / "results.json").write_text(json.dumps(data), encoding="utf-8")
        return tmp_path

    def test_missing_report_is_invalid(self, tmp_path):
        assert not check({"kind": "safety", "expected_cases": 1}, tmp_path)["valid"]

    @pytest.mark.parametrize("errored, valid", [(1, True), (2, False)])
    def test_suite_error_limit(self, tmp_path, errored, valid):
        results = [{"attempts": [{"error": "429" if i < errored else None}]} for i in range(10)]
        code = self.write(tmp_path, "safety", {"results": results})
        assert check({"kind": "safety", "expected_cases": 10}, code)["valid"] is valid

    def test_truncated_suite_report_is_invalid(self, tmp_path):
        code = self.write(tmp_path, "tool_use", {"results": [{"runs": [{"error": None}]}]})
        assert not check({"kind": "tool_use", "expected_cases": 20}, code)["valid"]

    @pytest.mark.parametrize("judge_error_rate, valid", [(0.1, True), (0.2, False)])
    def test_eval_uses_both_error_rates_and_counts_tokens(self, tmp_path, judge_error_rate, valid):
        data = {"summary": {"error_rate": 0.0, "judge_error_rate": judge_error_rate},
                "judge_usage": {"prompt_tokens": 30, "completion_tokens": 5},
                "cases": [{"runs": [{"reply": {"prompt_tokens": 100, "completion_tokens": 20}}]}]}
        verdict = check({"kind": "eval", "cases": ["ANS-01"]}, self.write(tmp_path, "eval/latest", data))
        assert verdict["valid"] is valid and verdict["sut_tokens"] == 120 and verdict["judge_tokens"] == 35


class FakeTarget:
    def describe(self):
        return {"prompt_version": "v1", "provider": "groq", "model": "m", "retriever_mode": "hybrid", "rerank": True}


class FakeJudge:
    calls, prompt_tokens, completion_tokens = 0, 0, 0

    def get_model_name(self):
        return "groq/judge"


def synthetic_report(cases):
    """A report with scores, latencies, costs and errors that differ from case to case."""
    config = load_eval_config("full")
    runner = EvalRunner(config, FakeTarget(), FakeJudge())
    results = []
    for n, case in enumerate(cases):
        runs = []
        for i in range(config.runs_per_case):
            if n % 17 == 0 and i == 0:
                runs.append(RunScore(i, "", "error", {}, {}, False, [], Reply(answer="", error="429")))
                continue
            scores = {m: ((n * 7 + i * 3) % 11) / 10 for m in runner.metrics_for(case)}
            reply = Reply(answer="a", latency_ms=100 + n * 13 + i, cost_usd=0.001 * (i + 1), prompt_tokens=100)
            runs.append(RunScore(i, "a", "answered", scores, {}, (n + i) % 5 == 0, [], reply))
        results.append(runner.aggregate_case(case, runs))
    return runner.build_report(results)


def write_batches(tmp_path, report, groups, commits=None):
    paths = []
    for k, group in enumerate(groups):
        part = synthetic_report(GOLDEN)  # fresh copy, then keep only this group's cases
        part.cases = [c for c in report.cases if c.case.id in group]
        part.judge_usage = {"calls": 10, "prompt_tokens": 100, "completion_tokens": 10}
        root = tmp_path / f"batch{k}"
        out = write_report(part, root)
        if commits:
            (root / "batch.json").write_text(json.dumps({"code_commit": commits[k]}), encoding="utf-8")
        paths.append(out / "results.json")
    return paths


class TestCombine:
    GROUPS = [[c.id for c in GOLDEN][k::3] for k in range(3)]

    def test_combined_batches_equal_the_unsplit_report(self, tmp_path):
        whole = synthetic_report(GOLDEN)
        merged = combine(write_batches(tmp_path, whole, self.GROUPS, ["abc"] * 3), "full")
        assert merged.summary == pytest.approx(whole.summary)
        assert merged.by_category.keys() == whole.by_category.keys()
        for category, summary in whole.by_category.items():
            assert merged.by_category[category] == pytest.approx(summary)
        assert [(g["metric"], g["passed"]) for g in merged.gate_results] == [
            (g["metric"], g["passed"]) for g in whole.gate_results]
        assert [c.case.id for c in merged.cases] == [c.id for c in GOLDEN]
        assert merged.judge_usage == {"calls": 30, "prompt_tokens": 300, "completion_tokens": 30}

    @pytest.mark.parametrize("groups, message", [(GROUPS[:2], "missing"), (GROUPS + GROUPS[:1], "repeated")])
    def test_incomplete_or_repeated_cases_are_refused(self, tmp_path, groups, message):
        with pytest.raises(CombineError, match=message):
            combine(write_batches(tmp_path, synthetic_report(GOLDEN), groups), "full")

    def test_different_code_commits_are_refused(self, tmp_path):
        paths = write_batches(tmp_path, synthetic_report(GOLDEN), self.GROUPS, ["abc", "abc", "def"])
        with pytest.raises(CombineError, match="commits"):
            combine(paths, "full")

    def test_different_target_is_refused(self, tmp_path):
        paths = write_batches(tmp_path, synthetic_report(GOLDEN), self.GROUPS)
        data = json.loads(paths[0].read_text(encoding="utf-8"))
        data["target"]["prompt_version"] = "v2"
        paths[0].write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(CombineError, match="target"):
            combine(paths, "full")

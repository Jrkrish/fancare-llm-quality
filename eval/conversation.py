"""Conversation eval: multi-turn context retention, topic switches, ambiguous follow-ups,
contradicting the user, and frustration handling.

    python -m eval.conversation --runs 2
    python -m eval.conversation --judge     # adds DeepEval KnowledgeRetention + ConversationalGEval

Every assistant turn is checked deterministically. A conversation passes if all
of its turns pass in at least `min_pass_rate` of runs (and judge scores clear
their threshold, when a judge is used).
"""
import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field

from app.config import ROOT
from eval.checks import run_checks
from eval.harness import AgentHarness
from eval.suite_report import md_table, pct, write_suite_report

DATASET = ROOT / "eval" / "datasets" / "conversations.json"
JUDGE_THRESHOLD = 0.7


@dataclass
class Conversation:
    id: str
    title: str
    turns: list[dict]
    judge_criteria: str = ""


@dataclass
class ConversationRun:
    transcript: list[dict]  # [{user, assistant, problems}]
    judge_scores: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def failed_turns(self) -> list[int]:
        return [i for i, t in enumerate(self.transcript) if t["problems"]]


@dataclass
class ConversationResult:
    conversation: Conversation
    runs: list[ConversationRun]
    turn_pass_rates: list[float] = field(default_factory=list)
    passed: bool = False


def load_conversations(path=DATASET) -> list[Conversation]:
    return [Conversation(**c) for c in json.loads(path.read_text(encoding="utf-8"))["conversations"]]


def judge_scores(conv: Conversation, transcript: list[dict], judge) -> dict[str, float]:
    from deepeval.metrics import ConversationalGEval, KnowledgeRetentionMetric
    from deepeval.test_case import ConversationalTestCase, Turn

    turns = []
    for t in transcript:
        turns += [Turn(role="user", content=t["user"]), Turn(role="assistant", content=t["assistant"])]
    case = ConversationalTestCase(turns=turns)
    scores = {}
    kr = KnowledgeRetentionMetric(threshold=JUDGE_THRESHOLD, model=judge, include_reason=False, async_mode=False, eval_mode="llm")
    scores["knowledge_retention"] = kr.measure(case)
    if conv.judge_criteria:
        geval = ConversationalGEval(name=conv.title, criteria=conv.judge_criteria, model=judge,
                                    threshold=JUDGE_THRESHOLD, async_mode=False)
        scores["conversation_quality"] = geval.measure(case)
    return scores


class ConversationRunner:
    def __init__(self, harness: AgentHarness, runs: int = 1, min_pass_rate: float = 0.66, judge=None):
        self.harness = harness
        self.runs = runs
        self.min_pass_rate = min_pass_rate
        self.judge = judge

    def run_once(self, conv: Conversation) -> ConversationRun:
        agent = self.harness.new_agent()
        session = self.harness.new_session(f"conv-{conv.id}")
        transcript = []
        for spec in conv.turns:
            turn = self.harness.ask(agent, session, spec["user"])
            if turn.error:
                return ConversationRun(transcript, error=turn.error)
            problems = run_checks(spec.get("checks", {}), self.harness.context(agent, turn))
            transcript.append({"user": spec["user"], "assistant": turn.answer, "problems": problems,
                               "tools": [t.name for t in turn.tool_records]})
        run = ConversationRun(transcript)
        if self.judge:
            try:
                run.judge_scores = judge_scores(conv, transcript, self.judge)
            except Exception as exc:  # noqa: BLE001 - judge failure is reported, not fatal
                run.judge_scores = {"judge_error": 0.0}
                transcript[-1]["problems"] = transcript[-1]["problems"] + [f"judge failed: {exc}"[:200]]
        return run

    def run_conversation(self, conv: Conversation) -> ConversationResult:
        runs = [self.run_once(conv) for _ in range(self.runs)]
        result = ConversationResult(conv, runs)
        valid = [r for r in runs if r.error is None]
        if not valid:
            return result
        result.turn_pass_rates = [
            sum(not r.transcript[i]["problems"] for r in valid) / len(valid) for i in range(len(conv.turns))
        ]
        judge_ok = all(
            statistics.fmean(r.judge_scores[k] for r in valid) >= JUDGE_THRESHOLD
            for k in valid[0].judge_scores
        ) if valid[0].judge_scores else True
        result.passed = judge_ok and all(p >= self.min_pass_rate for p in result.turn_pass_rates)
        return result


def render(results: list[ConversationResult], target: dict) -> str:
    lines = ["# Conversation eval", "", f"Target: prompt {target['prompt_version']}, {target['provider']} / {target['model']}.", ""]
    rows = []
    for r in results:
        valid = [run for run in r.runs if run.error is None]
        judge = valid[0].judge_scores if valid else {}
        means = {k: statistics.fmean(run.judge_scores.get(k, 0) for run in valid) for k in judge}
        rows.append([r.conversation.id, r.conversation.title, len(r.conversation.turns),
                     pct(statistics.fmean(r.turn_pass_rates)) if r.turn_pass_rates else "error",
                     ", ".join(f"{k} {v:.2f}" for k, v in means.items()) or "-",
                     "pass" if r.passed else "**FAIL**"])
    lines += md_table(["Id", "Scenario", "Turns", "Turn pass rate", "Judge", "Result"], rows)
    lines += ["", "## Failed turns", ""]
    for r in results:
        for run in r.runs:
            if run.error:
                lines.append(f"- **{r.conversation.id}**: run errored: {run.error}")
                continue
            for i in run.failed_turns:
                t = run.transcript[i]
                lines.append(f"- **{r.conversation.id}** turn {i + 1} `{t['user']}`")
                lines += [f"  - {p}" for p in t["problems"]]
                lines.append(f"  - answer: {t['assistant'][:250].replace(chr(10), ' ')}")
            break  # first run is enough to illustrate
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--judge", action="store_true", help="add DeepEval conversational metrics")
    args = parser.parse_args(argv)
    harness = AgentHarness()
    if harness.llm_ready():
        print(f"Cannot run conversation eval: {harness.llm_ready()}", file=sys.stderr)
        return 2
    judge = None
    if args.judge:
        from eval.config import load_eval_config
        from eval.judge import JudgeLLM

        cfg = load_eval_config("full")
        judge = JudgeLLM(cfg.judge_provider, cfg.judge_model, cfg.judge_min_interval_s, cfg.judge_max_tokens)
    runner = ConversationRunner(harness, args.runs, judge=judge)
    results = []
    for conv in load_conversations():
        r = runner.run_conversation(conv)
        print(f"{conv.id} {conv.title:<40} {'PASS' if r.passed else 'FAIL'}")
        results.append(r)
    out = write_suite_report("conversation", render(results, harness.describe()),
                             {"target": harness.describe(), "results": results})
    print(f"Report: {out / 'summary.md'}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

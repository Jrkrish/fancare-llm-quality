"""Conversation eval: 5-8 turn dialogues covering context retention, topic switches,
ambiguous follow-ups, contradicting the user and frustration handling.

    pytest tests/conversation --suite-runs 2
"""
import pytest

from eval.conversation import ConversationRunner, load_conversations, render
from eval.suite_report import write_suite_report
from tests.framework.base_test import BaseTest

pytestmark = [pytest.mark.llm, pytest.mark.conversation]
CONVERSATIONS = load_conversations()


@pytest.fixture(scope="module")
def conversation_session(harness, suite_runs):
    session = {"runner": ConversationRunner(harness, suite_runs), "results": {}}
    yield session
    if session["results"]:
        results = list(session["results"].values())
        out = write_suite_report("conversation", render(results, harness.describe()),
                                 {"target": harness.describe(), "results": results})
        print(f"\nConversation report: {out / 'summary.md'}")


class TestConversation(BaseTest):
    @pytest.mark.parametrize("conv", CONVERSATIONS, ids=[c.id for c in CONVERSATIONS])
    def test_conversation(self, conv, conversation_session):
        result = conversation_session["runner"].run_conversation(conv)
        conversation_session["results"][conv.id] = result
        if result.passed:
            return
        problems = []
        for run_no, run in enumerate(result.runs, 1):
            if run.error:
                problems.append(f"run {run_no} errored: {run.error}")
            for i in run.failed_turns:
                t = run.transcript[i]
                problems += [f"run {run_no}, turn {i + 1} '{t['user']}': {p}" for p in t["problems"]]
        problems.append("turn pass rates: " + ", ".join(f"{p:.0%}" for p in result.turn_pass_rates))
        self.fail_if_any(problems, f"{conv.id} {conv.title}")

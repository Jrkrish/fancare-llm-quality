"""Eval metrics.

Judge metrics (DeepEval, LLM-scored): faithfulness, answer_relevancy,
contextual_precision, contextual_recall.

Deterministic metrics (custom DeepEval BaseMetric subclasses, no LLM calls):
- refusal_correctness: did the bot answer / partially answer / decline as the case expects?
- citation_correctness: F1 of cited docs vs gold docs; citations that were not
  retrieved this turn (or do not exist) count as wrong.
- retrieval_recall: share of gold chunks that retrieval returned.

The test case metadata carries what these need: expected_behavior,
citations, retrieved_chunk_ids, expected_chunk_ids.
"""
import re

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

JUDGE_METRICS = ("faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall")

# Phrases that acknowledge missing information or decline. Kept deliberately
# explicit so the detector is auditable; judge calibration (stage 6) measures how
# often it agrees with human labels.
GAP_PATTERNS = [
    r"\bI (do not|don'?t) have (that|this|any|enough|specific)?\s*(information|details|data)\b",
    # "I don't have the current price for X in the documentation"
    r"\bI (do not|don'?t) have\b[^.\n]{0,80}\b(in|from) the (corvane )?(documentation|docs|documents|passages)\b",
    r"\b(is|are|was)? ?(not|n't|isn'?t|aren'?t)\s+(mentioned|listed|documented|specified|stated|described)\b",
    # "not covered" alone is usually a policy fact ("labour is not covered"), so require a docs reference.
    r"\bnot covered (in|by) the (corvane )?(documentation|docs|documents|passages|information)\b",
    r"\b(does|do)(n'?t| not) (say|mention|specify|state|cover|list|include|describe)\b",
    r"\bno (information|details|mention|data)\b",
    r"\b(can'?t|cannot|could ?n'?t|unable to) (find|confirm|verify|say)\b",
    r"\bI can only help with\b",
    r"\bI can'?t (help|follow|share|assist)\b",
    r"\boutside (of )?(what|the scope)\b",
    r"\bnot (something|a topic) I can\b",
]
_GAP_RE = re.compile("|".join(GAP_PATTERNS), re.IGNORECASE)


# LLMs often write typographic apostrophes, hyphens and spaces ("don’t", "non‑smart").
_PUNCT = str.maketrans({"‘": "'", "’": "'", "ʼ": "'", "‐": "-", "‑": "-",
                        " ": " ", " ": " ", " ": " "})


def acknowledges_gap(answer: str) -> bool:
    return bool(_GAP_RE.search(answer.translate(_PUNCT)))


def classify_response(answer: str, citations: list[dict]) -> str:
    """'refused' = acknowledges a gap and cites nothing; 'partial' = gap + citations; else 'answered'."""
    gap = acknowledges_gap(answer)
    cited = any(c.get("passage") for c in citations)
    if gap and not cited:
        return "refused"
    if gap:
        return "partial"
    return "answered"


def refusal_is_correct(expected_behavior: str, response_class: str) -> bool:
    if expected_behavior == "answer":
        return response_class != "refused"
    if expected_behavior == "partial":
        return response_class == "partial"
    return response_class != "answered"  # refuse: must acknowledge the gap


def citation_f1(citations: list[dict], gold_doc_ids: set[str]) -> float:
    if not citations or not gold_doc_ids:
        return 0.0
    correct = [c for c in citations if c.get("retrieved") and c.get("passage") and c["doc_id"] in gold_doc_ids]
    precision = len(correct) / len(citations)
    recall = len({c["doc_id"] for c in correct}) / len(gold_doc_ids)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


class _DeterministicMetric(BaseMetric):
    name = "deterministic"

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.async_mode = False
        self.include_reason = False
        self.strict_mode = False
        self.verbose_mode = False

    def compute(self, meta: dict) -> float:
        raise NotImplementedError

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self.score = float(self.compute(test_case.metadata or {}))
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return self.name


class RefusalCorrectnessMetric(_DeterministicMetric):
    name = "refusal_correctness"

    def compute(self, meta: dict) -> float:
        return 1.0 if refusal_is_correct(meta["expected_behavior"], meta["response_class"]) else 0.0


class CitationCorrectnessMetric(_DeterministicMetric):
    name = "citation_correctness"

    def compute(self, meta: dict) -> float:
        return citation_f1(meta["citations"], {c.split(":")[0] for c in meta["expected_chunk_ids"]})


class RetrievalRecallMetric(_DeterministicMetric):
    name = "retrieval_recall"

    def compute(self, meta: dict) -> float:
        gold = set(meta["expected_chunk_ids"])
        return len(gold & set(meta["retrieved_chunk_ids"])) / len(gold) if gold else 0.0


DETERMINISTIC_METRICS = {
    m.name: m for m in (RefusalCorrectnessMetric, CitationCorrectnessMetric, RetrievalRecallMetric)
}


def rejected_items(metric: BaseMetric) -> list[str]:
    """Claims (faithfulness) or statements (relevancy) the judge did not mark 'yes', with its reason.

    DeepEval keeps these on the metric after measure(), so collecting them costs no extra judge calls.
    """
    items = getattr(metric, "claims", None) or getattr(metric, "statements", None) or []
    out = []
    for i, v in enumerate(getattr(metric, "verdicts", None) or []):
        verdict = str(getattr(v, "verdict", "")).lower()
        if verdict == "yes":
            continue
        text = items[i] if i < len(items) and isinstance(items[i], str) else ""
        reason = getattr(v, "reason", None)
        out.append(f"[{verdict}] {text}" + (f" -- {reason}" if reason else ""))
    return out


def build_judge_metric(name: str, threshold: float, judge) -> BaseMetric:
    common = dict(threshold=threshold, model=judge, include_reason=False, async_mode=False, eval_mode="llm")
    if name == "faithfulness":
        # Penalise claims the context neither supports nor contradicts, so this
        # measures groundedness rather than only "no contradictions".
        return FaithfulnessMetric(**common, penalize_ambiguous_claims=True)
    if name == "answer_relevancy":
        return AnswerRelevancyMetric(**common)
    if name == "contextual_precision":
        return ContextualPrecisionMetric(**common)
    if name == "contextual_recall":
        return ContextualRecallMetric(**common)
    raise ValueError(f"unknown judge metric '{name}'")

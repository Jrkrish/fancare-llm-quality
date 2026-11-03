"""Document-extraction eval (no LLM): exact fields on clean documents, graceful
failure without guessing on damaged ones."""
import pytest

from eval.extraction import FIELDS, evaluate_document, load_labels, render, summarise
from eval.suite_report import write_suite_report
from tests.framework.base_test import BaseTest

pytestmark = pytest.mark.extraction
LABELS = load_labels()


@pytest.fixture(scope="module")
def extraction_session():
    from app.documents import DocumentExtractor

    session = {"extractor": DocumentExtractor(), "results": []}
    yield session
    if session["results"]:
        s = summarise(session["results"])
        write_suite_report("extraction", render(session["results"], s), {"summary": s, "results": session["results"]})


class TestExtraction(BaseTest):
    @pytest.mark.parametrize("name", list(LABELS), ids=list(LABELS))
    def test_document(self, name, extraction_session):
        label = LABELS[name]
        result = evaluate_document(extraction_session["extractor"], name, label)
        extraction_session["results"].append(result)
        if label["condition"] == "clean":
            wrong = [f"{f}: got {result.fields.get(f)!r}, expected {label['truth'][f]!r}" for f in FIELDS
                     if result.fields.get(f) != label["truth"][f]]
            self.fail_if_any(wrong + ([f"status {result.status}"] if result.status != "ok" else []), f"{name} (clean)")
        else:
            problems = [f"guessed {f}={result.fields[f]!r} (truth {label['truth'][f]!r})" for f in result.wrong_fields]
            if result.status == "ok":
                problems.append("damaged document reported status 'ok'")
            if not result.warnings:
                problems.append("no warning explains the failure")
            self.fail_if_any(problems, f"{name} ({label['condition']})")

    def test_suite_thresholds(self, extraction_session):
        if len(extraction_session["results"]) < len(LABELS):
            pytest.skip("run the whole module to check suite thresholds")
        s = summarise(extraction_session["results"])
        problems = []
        if s["field_accuracy"] < 0.95:
            problems.append(f"field accuracy {s['field_accuracy']:.0%} < 95%")
        if s["guessed_fields_on_damaged"]:
            problems.append(f"{s['guessed_fields_on_damaged']} guessed fields on damaged documents")
        self.fail_if_any(problems, "extraction suite thresholds")

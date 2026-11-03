"""Document-extraction eval over the 10 synthetic warranty cards / invoices.

    python -m eval.extraction

Clean documents: field-level accuracy and exact match (all three fields right).
Damaged documents: graceful failure, meaning no exception escapes, the status is
not "ok", and no returned field is wrong (null is the correct answer when unsure).
"""
import json
import sys
from dataclasses import dataclass

from app.config import ROOT
from app.documents import DocumentExtractor, UnreadableDocument
from eval.suite_report import md_table, pct, write_suite_report

DOCS_DIR = ROOT / "eval" / "datasets" / "documents"
FIELDS = ("serial_no", "purchase_date", "model")


@dataclass
class DocResult:
    file: str
    condition: str
    status: str  # ok | partial | failed | unreadable
    fields: dict
    truth: dict
    warnings: list[str]

    @property
    def clean(self) -> bool:
        return self.condition == "clean"

    @property
    def correct_fields(self) -> int:
        return sum(self.fields.get(f) == self.truth[f] for f in FIELDS)

    @property
    def wrong_fields(self) -> list[str]:
        """Fields that were returned with a value that is not the truth (i.e. a guess)."""
        return [f for f in FIELDS if self.fields.get(f) is not None and self.fields.get(f) != self.truth[f]]

    @property
    def passed(self) -> bool:
        if self.clean:
            return self.correct_fields == len(FIELDS)
        return not self.wrong_fields and self.status != "ok"


def load_labels() -> dict:
    return json.loads((DOCS_DIR / "labels.json").read_text(encoding="utf-8"))


def evaluate_document(extractor: DocumentExtractor, name: str, label: dict) -> DocResult:
    data = (DOCS_DIR / "files" / name).read_bytes()
    try:
        r = extractor.extract(data, label["content_type"])
        status, fields, warnings = r.status, r.fields, r.warnings
    except UnreadableDocument as exc:
        status, fields, warnings = "unreadable", dict.fromkeys(FIELDS), [str(exc)]
    return DocResult(name, label["condition"], status, fields, label["truth"], warnings)


def summarise(results: list[DocResult]) -> dict:
    clean = [r for r in results if r.clean]
    damaged = [r for r in results if not r.clean]
    return {
        "field_accuracy": sum(r.correct_fields for r in clean) / (len(clean) * len(FIELDS)) if clean else 0.0,
        "exact_match": sum(r.correct_fields == len(FIELDS) for r in clean) / len(clean) if clean else 0.0,
        "per_field_accuracy": {f: sum(r.fields.get(f) == r.truth[f] for r in clean) / len(clean) for f in FIELDS} if clean else {},
        "graceful_failure_rate": sum(r.passed for r in damaged) / len(damaged) if damaged else 1.0,
        "guessed_fields_on_damaged": sum(len(r.wrong_fields) for r in damaged),
    }


def render(results: list[DocResult], s: dict) -> str:
    lines = ["# Document extraction eval", ""]
    lines += md_table(["Metric", "Value"], [
        ["Field accuracy (clean)", pct(s["field_accuracy"])],
        ["Exact match (clean)", pct(s["exact_match"])],
        *[[f"  {f}", pct(v)] for f, v in s["per_field_accuracy"].items()],
        ["Graceful failure (damaged)", pct(s["graceful_failure_rate"])],
        ["Guessed (wrong) fields on damaged docs", s["guessed_fields_on_damaged"]],
    ])
    lines += ["", "## Per document", ""]
    lines += md_table(["File", "Condition", "Status", "Extracted", "Result"], [
        [r.file, r.condition, r.status, ", ".join(f"{k}={v}" for k, v in r.fields.items()), "pass" if r.passed else "**FAIL**"]
        for r in results
    ])
    return "\n".join(lines) + "\n"


def run() -> tuple[list[DocResult], dict]:
    extractor = DocumentExtractor()
    results = [evaluate_document(extractor, name, label) for name, label in load_labels().items()]
    return results, summarise(results)


def main() -> int:
    results, summary = run()
    out = write_suite_report("extraction", render(results, summary), {"summary": summary, "results": results})
    print(render(results, summary))
    print(f"Report: {out / 'summary.md'}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

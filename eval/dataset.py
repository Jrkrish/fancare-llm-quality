"""Golden dataset model, loading, validation and stratified selection."""
import json
from dataclasses import dataclass, field
from pathlib import Path

CATEGORIES = ("answerable", "partial", "unanswerable", "multi_hop", "conflicting", "near_miss")
BEHAVIORS = ("answer", "partial", "refuse")


@dataclass(frozen=True)
class GoldenCase:
    id: str
    category: str
    question: str
    expected_behavior: str
    expected_output: str
    expected_chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    @property
    def gold_doc_ids(self) -> set[str]:
        return {c.split(":")[0] for c in self.expected_chunk_ids}


class DatasetError(ValueError):
    pass


def load_golden(path: Path, known_chunk_ids: set[str] | None = None) -> list[GoldenCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = []
    for item in raw["cases"]:
        case = GoldenCase(**{**item, "expected_chunk_ids": tuple(item.get("expected_chunk_ids", []))})
        if case.category not in CATEGORIES:
            raise DatasetError(f"{case.id}: unknown category '{case.category}'")
        if case.expected_behavior not in BEHAVIORS:
            raise DatasetError(f"{case.id}: unknown expected_behavior '{case.expected_behavior}'")
        if case.expected_behavior != "refuse" and not case.expected_chunk_ids:
            raise DatasetError(f"{case.id}: answerable cases need expected_chunk_ids")
        if known_chunk_ids is not None:
            missing = set(case.expected_chunk_ids) - known_chunk_ids
            if missing:
                raise DatasetError(f"{case.id}: chunk ids not in corpus: {sorted(missing)}")
        cases.append(case)

    ids = [c.id for c in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise DatasetError(f"duplicate case ids: {sorted(duplicates)}")
    return cases


def select_cases(
    cases: list[GoldenCase],
    categories: list[str] | None = None,
    ids: list[str] | None = None,
    per_category: int = 0,
) -> list[GoldenCase]:
    """Filter by id/category, then optionally keep the first K cases of each category (stable subset)."""
    selected = [c for c in cases if (not ids or c.id in ids) and (not categories or c.category in categories)]
    if per_category > 0:
        counts: dict[str, int] = {}
        kept = []
        for c in selected:
            counts[c.category] = counts.get(c.category, 0) + 1
            if counts[c.category] <= per_category:
                kept.append(c)
        selected = kept
    return selected

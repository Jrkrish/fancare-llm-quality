"""Small helpers shared by the non-RAG suites to write their reports."""
import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

from eval.config import REPORTS_DIR


def _plain(obj):
    return asdict(obj) if is_dataclass(obj) else obj


def write_suite_report(suite: str, markdown: str, data: dict, reports_dir: Path = REPORTS_DIR) -> Path:
    out = reports_dir / suite
    out.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.now(UTC).isoformat(), **data}
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=_plain), encoding="utf-8")
    (out / "summary.md").write_text(markdown, encoding="utf-8")
    return out


def pct(x: float) -> str:
    return f"{x:.0%}"


def md_table(headers: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines

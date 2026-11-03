"""Field extraction from uploaded warranty cards and invoices (PDF or image).

Text comes from the PDF text layer (pypdf) or local OCR (RapidOCR, ONNX). Fields
are then pulled out with strict patterns. The extractor never guesses: a value
that is missing, ambiguous (e.g. 04/05/2025) or inconsistent with another field
is returned as null with a warning. Free text in the document is never passed
on, so instructions hidden in an upload cannot reach the LLM.
"""
import io
import re
from dataclasses import dataclass, field
from datetime import date, datetime

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
CONTENT_TYPES = {"application/pdf": "pdf", "image/png": "image", "image/jpeg": "image"}
MODEL_CODES = ("MN30", "AL36", "AL44", "HU42", "HU52", "SW44", "SW52", "SW60", "TB52", "TB60", "LX72", "BS52")

_SERIAL_SEARCH = re.compile(r"\b[A-Z]{2}\d{2}-\d{4}-\d{5}\b")
_MODEL_SEARCH = re.compile(r"\b(" + "|".join(MODEL_CODES) + r")\b")
_DATE_LABEL = re.compile(r"(purchase date|date of purchase|purchased on|invoice date|date)\s*[:\-]?\s*(.{0,30})", re.IGNORECASE)
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_TEXT_DATES = [
    (re.compile(r"\b(\d{1,2}) ([A-Za-z]{3,9}),? (\d{4})\b"), "%d %B %Y"),
    (re.compile(r"\b([A-Za-z]{3,9}) (\d{1,2}),? (\d{4})\b"), "%B %d %Y"),
]
_NUMERIC_DATE = re.compile(r"\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b")


class UnreadableDocument(ValueError):
    pass


@dataclass
class ExtractionResult:
    status: str  # ok | partial | failed
    document_type: str  # warranty_card | invoice | unknown
    fields: dict[str, str | None]
    warnings: list[str] = field(default_factory=list)

    @property
    def missing(self) -> list[str]:
        return [k for k, v in self.fields.items() if v is None]


def _parse_text_date(text: str) -> date | None:
    for pattern, fmt in _TEXT_DATES:
        m = pattern.search(text)
        if m:
            parts = " ".join(m.groups())
            for f in (fmt, fmt.replace("%B", "%b")):
                try:
                    return datetime.strptime(parts, f).date()
                except ValueError:
                    continue
    return None


def extract_purchase_date(text: str, warnings: list[str]) -> str | None:
    candidates = [m.group(2) for m in _DATE_LABEL.finditer(text)] or [text]
    found: set[date] = set()
    for chunk in candidates:
        iso = _ISO_DATE.search(chunk)
        if iso:
            try:
                found.add(date(*map(int, iso.groups())))
            except ValueError:
                warnings.append(f"invalid date '{iso.group(0)}'")
            continue
        parsed = _parse_text_date(chunk)
        if parsed:
            found.add(parsed)
        elif _NUMERIC_DATE.search(chunk):
            warnings.append(f"ambiguous date format '{_NUMERIC_DATE.search(chunk).group(0)}' (day/month order unknown)")
    if len(found) > 1:
        warnings.append("more than one purchase date found")
        return None
    if not found:
        return None
    value = found.pop()
    if value > date.today():
        warnings.append(f"purchase date {value} is in the future")
        return None
    return value.isoformat()


def extract_fields(text: str) -> ExtractionResult:
    warnings: list[str] = []
    upper = text.upper()

    serials = set(_SERIAL_SEARCH.findall(upper))
    serial = None
    if len(serials) == 1:
        serial = serials.pop()
    elif len(serials) > 1:
        warnings.append(f"several serial numbers found: {sorted(serials)}")

    models = set(_MODEL_SEARCH.findall(upper))
    model = None
    if serial and serial[:4] in MODEL_CODES:
        model = serial[:4]
        if models - {model}:
            warnings.append(f"model codes {sorted(models)} disagree with serial {serial}")
            model = None
    elif len(models) == 1:
        model = models.pop()
    elif len(models) > 1:
        warnings.append(f"several model codes found: {sorted(models)}")

    fields = {"serial_no": serial, "purchase_date": extract_purchase_date(text, warnings), "model": model}
    if serial is None and re.search(r"serial", text, re.IGNORECASE):
        warnings.append("serial number label found but no valid serial number")

    lowered = text.lower()
    doc_type = "warranty_card" if "warranty" in lowered else "invoice" if "invoice" in lowered else "unknown"
    filled = sum(v is not None for v in fields.values())
    status = "ok" if filled == 3 else "partial" if filled else "failed"
    return ExtractionResult(status, doc_type, fields, warnings)


class DocumentExtractor:
    def __init__(self):
        self._ocr = None

    def _ocr_engine(self):
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR

            self._ocr = RapidOCR()
        return self._ocr

    def read_text(self, data: bytes, kind: str) -> str:
        if kind == "pdf":
            from pypdf import PdfReader
            from pypdf.errors import PdfReadError

            try:
                reader = PdfReader(io.BytesIO(data))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except (PdfReadError, ValueError, KeyError, OSError) as exc:
                raise UnreadableDocument(f"PDF could not be read: {type(exc).__name__}") from exc

        import numpy as np
        from PIL import Image, UnidentifiedImageError

        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise UnreadableDocument("image could not be decoded") from exc
        result, _ = self._ocr_engine()(np.array(image))
        if not result:
            return ""
        # Group OCR boxes into lines by vertical position, then read left to right.
        boxes = sorted(result, key=lambda r: (r[0][0][1], r[0][0][0]))
        lines: list[list] = []
        for box in boxes:
            y = box[0][0][1]
            if lines and abs(lines[-1][0][0][0][1] - y) < 12:
                lines[-1].append(box)
            else:
                lines.append([box])
        return "\n".join(" ".join(b[1] for b in sorted(line, key=lambda b: b[0][0][0])) for line in lines)

    def extract(self, data: bytes, content_type: str) -> ExtractionResult:
        kind = CONTENT_TYPES[content_type]
        text = self.read_text(data, kind)
        if not text.strip():
            return ExtractionResult("failed", "unknown", {"serial_no": None, "purchase_date": None, "model": None},
                                    ["no readable text found"])
        return extract_fields(text)

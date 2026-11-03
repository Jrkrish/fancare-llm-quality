"""Guardrail layer, independent of the agent so it can be unit-tested on its own.

InputFilter  - blocks prompt injection, jailbreaks, system-prompt extraction and
               blocked topics; redacts PII the user types in; scans retrieved
               passages for indirect injection.
OutputFilter - redacts PII and replaces answers that leak the system prompt.

These are deliberately simple pattern rules. They are the first layer, not the
only one: the system prompt is the second, and the safety suite measures how
much gets past both.
"""
import re
from dataclasses import dataclass, field

_I = re.IGNORECASE

INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|your|system)\b[^.\n]{0,20}\b(instructions?|rules|prompts?|guidelines|directives)\b",
    r"\bnew (instructions|rules)\s*:",
    r"\b(system|developer|admin)\s+(note|message|override|command)\b[^.\n]{0,30}\b(ai|assistant|model|bot|chatbot)\b",
    r"</?\s*(system|assistant|instructions?)\s*>",
    r"\byou (are|will) now\b[^.\n]{0,40}\b(not|no longer|unrestricted|without)\b",
]
JAILBREAK_PATTERNS = [
    r"\bDAN\b",
    r"\bdo anything now\b",
    r"\bjailbr(eak|oken)\b",
    r"\bdeveloper mode\b",
    r"\b(unfiltered|uncensored|unrestricted) (mode|ai|assistant|version|model)\b",
    r"\bpretend\b[^.\n]{0,40}\b(no|without) (rules|restrictions|limits|filters|guidelines)\b",
    r"\b(act|role-?play) as\b[^.\n]{0,40}\b(evil|unfiltered|unrestricted|no rules)\b",
]
EXTRACTION_PATTERNS = [
    r"\b(system|initial|hidden|original|secret)\s+(prompt|instructions)\b",
    r"\b(repeat|print|show|reveal|output|display|dump)\b[^.\n]{0,20}\byour\b[^.\n]{0,20}\b(instructions|prompt|rules|guidelines|configuration)\b",
    r"\bwhat (were|are|have) you (been )?(told|instructed|programmed)\b",
    r"\b(everything|all (the )?text|the words) (above|before this)\b",
]
BLOCKED_TOPIC_PATTERNS = {
    "safety_bypass": (
        r"\b(bypass|bridg|disabl|defeat|jumper)\w*\b[^.\n]{0,40}\b(thermal|overheat|cut-?out|stall|protection|safety)\b"
        r"|\b(thermal|overheat|cut-?out|stall|safety) (protection|cut-?out|sensor|detector)s?\b"
        r"[^.\n]{0,30}\b(bypass|bridge|disable|defeat)\w*\b"
    ),
    "weapons": r"\b(weapons?|guns?|firearms?|explosives?|bombs?|ammunition)\b",
    "hacking": (
        r"\b(hack|exploit|brute[- ]?force|ddos)\b[^.\n]{0,40}"
        r"\b(neighbou?r|someone|other people|account|password|network)s?\b"
    ),
    "medical": r"\b(diagnos(e|is)|prescri(be|ption)|dosage|medication)\b",
    "legal": r"\b(sue|lawsuit|lawyer|attorney|legal advice)\b",
    "politics": r"\b(election|vote for|political|politics|democrats?|republicans?)\b",
    "financial_advice": r"\b(stock market|share price|crypto(currency)?|bitcoin|investing|investment advice)\b",
}

GENERIC_SCOPE = (
    "I can help with Corvane ceiling fans: setup, Wi-Fi and Bluetooth, the remote, error codes, "
    "troubleshooting, orders, warranty, returns and safety."
)
REFUSALS = {
    "prompt_injection": "I can't follow that request. " + GENERIC_SCOPE,
    "jailbreak": "I can't follow that request. " + GENERIC_SCOPE,
    "system_prompt_extraction": "I can't share my internal instructions. " + GENERIC_SCOPE,
    "system_prompt_leak": "I can't share my internal instructions. " + GENERIC_SCOPE,
    "safety_bypass": (
        "I can't help with bypassing or disabling the fan's safety protections; they prevent fire and injury. "
        "If the fan keeps stopping or showing an error, I can help you troubleshoot it or open a support ticket."
    ),
}
DEFAULT_TOPIC_REFUSAL = "That's outside what I can help with. " + GENERIC_SCOPE


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _card(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return "[REDACTED_CARD]" if 13 <= len(digits) <= 19 and _luhn_ok(digits) else match.group(0)


# Digit patterns must not start or end inside a longer run of digit groups (e.g. a non-Luhn card number),
# or inside identifiers such as serials (SW52-2403-00117) and order IDs.
_START = r"(?<![\w-])(?<!\d[ .-])"
_END = r"(?![\w-]|[ .-]\d)"

# Order matters: cards before shorter digit patterns.
PII_RULES: list[tuple[str, re.Pattern, object]] = [
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
    ("card", re.compile(_START + r"(?:\d[ -]?){12,18}\d" + _END), _card),
    ("ssn", re.compile(_START + r"\d{3}-\d{2}-\d{4}" + _END), "[REDACTED_SSN]"),
    ("national_id", re.compile(_START + r"[2-9]\d{3}[ -]\d{4}[ -]\d{4}" + _END), "[REDACTED_ID_NUMBER]"),
    ("phone", re.compile(_START + r"(?:\+91[\s-]?)?[6-9]\d{4}[\s-]\d{5}" + _END), "[REDACTED_PHONE]"),
    ("phone", re.compile(_START + r"(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}" + _END),
     "[REDACTED_PHONE]"),
]


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Return (redacted_text, sorted list of PII types found)."""
    found = set()
    for kind, pattern, repl in PII_RULES:
        new = pattern.sub(repl, text)
        if new != text:
            found.add(kind)
        text = new
    return text, sorted(found)


@dataclass
class Verdict:
    stage: str  # input | retrieval | output
    action: str  # allow | block | redact
    category: str | None = None
    detail: str | None = None


@dataclass
class FilterResult:
    text: str
    blocked: bool = False
    category: str | None = None
    verdicts: list[Verdict] = field(default_factory=list)


def _first_match(text: str, patterns: list[str]) -> str | None:
    for p in patterns:
        m = re.search(p, text, _I)
        if m:
            return m.group(0)
    return None


class InputFilter:
    def check(self, message: str) -> FilterResult:
        for category, patterns in (
            ("prompt_injection", INJECTION_PATTERNS),
            ("jailbreak", JAILBREAK_PATTERNS),
            ("system_prompt_extraction", EXTRACTION_PATTERNS),
        ):
            hit = _first_match(message, patterns)
            if hit:
                return self._block(message, category, hit)
        for topic, pattern in BLOCKED_TOPIC_PATTERNS.items():
            m = re.search(pattern, message, _I)
            if m:
                return self._block(message, topic, m.group(0))

        redacted, kinds = redact_pii(message)
        if kinds:
            return FilterResult(redacted, verdicts=[Verdict("input", "redact", "pii", ",".join(kinds))])
        return FilterResult(message, verdicts=[Verdict("input", "allow")])

    @staticmethod
    def _block(message: str, category: str, hit: str) -> FilterResult:
        redacted, _ = redact_pii(message)
        return FilterResult(redacted, blocked=True, category=category,
                            verdicts=[Verdict("input", "block", category, redact_pii(hit)[0][:100])])

    @staticmethod
    def scan_passage(text: str) -> str | None:
        """Return the matched snippet if a retrieved passage contains an injection attempt."""
        return _first_match(text, INJECTION_PATTERNS + JAILBREAK_PATTERNS)

    @staticmethod
    def refusal(category: str) -> str:
        return REFUSALS.get(category, DEFAULT_TOPIC_REFUSAL)


class OutputFilter:
    NGRAM = 8
    MIN_SHARED_NGRAMS = 2  # i.e. a run of 9+ words copied from the prompt

    def __init__(self, system_prompt: str):
        canary = re.search(r"CANARY-[A-Z0-9]+", system_prompt)
        self.canary = canary.group(0) if canary else None
        # Quoted text in the prompt is wording the bot is *meant* to say, so it is not a leak.
        unquoted = re.sub(r'"[^"]*"', " ", system_prompt)
        self.prompt_ngrams = self._ngrams(unquoted)

    @classmethod
    def _ngrams(cls, text: str) -> set[tuple[str, ...]]:
        words = re.findall(r"[a-z0-9']+", text.lower())
        return {tuple(words[i : i + cls.NGRAM]) for i in range(len(words) - cls.NGRAM + 1)}

    def leaks_prompt(self, answer: str) -> bool:
        if self.canary and self.canary in answer:
            return True
        return len(self._ngrams(answer) & self.prompt_ngrams) >= self.MIN_SHARED_NGRAMS

    def check(self, answer: str) -> FilterResult:
        if self.leaks_prompt(answer):
            return FilterResult(
                REFUSALS["system_prompt_leak"], blocked=True, category="system_prompt_leak",
                verdicts=[Verdict("output", "block", "system_prompt_leak")],
            )
        redacted, kinds = redact_pii(answer)
        if kinds:
            return FilterResult(redacted, verdicts=[Verdict("output", "redact", "pii", ",".join(kinds))])
        return FilterResult(answer, verdicts=[Verdict("output", "allow")])

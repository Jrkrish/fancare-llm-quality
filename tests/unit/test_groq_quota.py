"""Groq quota probe: header parsing and rendering with fake headers (no network)."""
from tools.groq_quota import is_low, parse_quota, render

HEADERS = {
    "x-ratelimit-limit-requests": "1000",
    "x-ratelimit-remaining-requests": "150",
    "x-ratelimit-reset-requests": "2m59.56s",
    "x-ratelimit-limit-tokens": "8000",
    "x-ratelimit-remaining-tokens": "7990",
    "x-ratelimit-reset-tokens": "7.66s",
}


def test_parse_reads_day_and_minute_fields():
    info = parse_quota(HEADERS)
    assert info["limit_requests_per_day"] == 1000
    assert info["remaining_requests_per_day"] == 150
    assert info["remaining_tokens_per_minute"] == 7990
    assert info["reset_requests"] == "2m59.56s"
    assert info["retry_after"] is None


def test_parse_tolerates_missing_and_bad_headers():
    info = parse_quota({"x-ratelimit-limit-requests": "oops"})
    assert info["limit_requests_per_day"] is None
    assert not is_low(info)


def test_low_when_under_twenty_percent_of_daily_requests():
    assert is_low(parse_quota(HEADERS))
    assert not is_low(parse_quota({**HEADERS, "x-ratelimit-remaining-requests": "500"}))


def test_render_labels_windows_and_flags_low():
    text = render(parse_quota({**HEADERS, "retry-after": "3"}), "openai/gpt-oss-120b", 429)
    assert "Requests per **day** | 150 | 1000" in text
    assert "Tokens per **minute**" in text
    assert "retry after 3s" in text
    assert "Low" in text

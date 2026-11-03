"""Base class shared by every test suite."""
import pytest


class BaseTest:
    """Common helpers so all suites report failures the same way.

    Suites that check several properties of one response collect the problems
    and fail once, so a single report shows everything that went wrong.
    """

    @staticmethod
    def fail_if_any(problems: list[str], title: str) -> None:
        if problems:
            pytest.fail(f"{title}\n  - " + "\n  - ".join(problems), pytrace=False)

    @staticmethod
    def assert_status(response, expected: int) -> None:
        assert response.status_code == expected, (
            f"{response.request.method} {response.url} -> {response.status_code}, expected {expected}: {response.text[:300]}"
        )

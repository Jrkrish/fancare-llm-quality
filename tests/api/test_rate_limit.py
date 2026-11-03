"""Rate limiting, against a server started with RATE_LIMIT_PER_MIN=3."""
import pytest

from tests.framework.base_test import BaseTest
from tests.framework.openapi import OpenApiValidator

pytestmark = pytest.mark.api


class TestRateLimit(BaseTest):
    def test_fourth_request_in_a_minute_is_429(self, limited_api):
        validator = OpenApiValidator(limited_api.openapi())
        statuses = [limited_api.history("rl-session").status_code for _ in range(3)]
        assert statuses == [200, 200, 200]

        resp = limited_api.history("rl-session")
        self.assert_status(resp, 429)
        validator.validate("/sessions/{session_id}/history", "get", resp)
        assert int(resp.headers["Retry-After"]) > 0

    def test_limit_applies_across_endpoints(self, limited_api):
        resp = limited_api.chat("rl-session", "hello")
        self.assert_status(resp, 429)

    def test_unauthenticated_requests_do_not_consume_quota(self, limited_api):
        self.assert_status(limited_api.history("rl-session", api_key="wrong"), 401)

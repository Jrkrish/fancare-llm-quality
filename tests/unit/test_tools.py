from datetime import date

import pytest

from app.tools import FaultInjector, Toolbox, ToolRunner, warranty_coverage


@pytest.fixture
def runner(settings, retriever):
    return ToolRunner(Toolbox(settings, retriever), settings)


class TestOrderStatus:
    def test_found(self, runner):
        rec = runner.run("get_order_status", '{"order_id": "ORD-10001"}')
        assert rec.status == "ok"
        assert rec.result["found"] is True
        assert rec.result["status"] == "shipped"
        assert rec.result["tracking_id"] == "UPS-7Q4K92M1"

    def test_lowercase_id_is_normalised(self, runner):
        assert runner.run("get_order_status", {"order_id": " ord-10003 "}).result["status"] == "delivered"

    def test_not_found(self, runner):
        rec = runner.run("get_order_status", {"order_id": "ORD-99999"})
        assert rec.status == "ok" and rec.result["found"] is False

    @pytest.mark.parametrize("bad", ["12345", "ORD-1", "order 10001", ""])
    def test_invalid_format(self, runner, bad):
        assert runner.run("get_order_status", {"order_id": bad}).status == "invalid_arguments"


class TestWarranty:
    def test_2025_policy_registered(self, runner):
        cov = runner.run("check_warranty", {"serial_no": "HU42-2502-00031"}).result
        assert cov["policy_doc_id"] == "DOC-12"
        assert cov["coverage"]["motor"]["until"] == "2030-03-01"
        assert cov["coverage"]["electronics"]["until"] == "2028-03-01"  # 2 years + 1 registration bonus
        assert cov["coverage"]["accessories"]["until"] == "2026-03-01"

    def test_2022_policy_for_older_purchase(self, runner):
        cov = runner.run("check_warranty", {"serial_no": "SW52-2403-00117"}).result
        assert cov["policy_doc_id"] == "DOC-13"
        assert cov["coverage"]["motor"]["until"] == "2027-04-10"
        assert cov["coverage"]["electronics"]["until"] == "2025-04-10"  # registration does not extend 2022 policy

    def test_commercial_use_is_one_year(self, runner):
        cov = runner.run("check_warranty", {"serial_no": "SW60-2508-00012"}).result
        assert {c["until"] for c in cov["coverage"].values()} == {"2026-09-01"}

    def test_void_warranty_is_never_active(self):
        entry = {"purchase_date": "2025-10-20", "registered_within_60_days": True,
                 "commercial_use": False, "void_reason": "receiver module opened"}
        cov = warranty_coverage(entry, today=date(2026, 1, 1))["coverage"]
        assert not any(c["active"] for c in cov.values())

    def test_unknown_serial(self, runner):
        assert runner.run("check_warranty", {"serial_no": "SW52-0000-00000"}).result == {
            "serial_no": "SW52-0000-00000", "found": False, "model": None, "purchase_date": None,
            "policy_doc_id": None, "registered_within_60_days": None, "commercial_use": None,
            "void_reason": None, "coverage": None,
        }

    def test_invalid_serial(self, runner):
        assert runner.run("check_warranty", {"serial_no": "ABC"}).status == "invalid_arguments"


class TestRecommendProduct:
    def test_filters_by_room_budget_and_needs(self, runner):
        rec = runner.run("recommend_product", {"room_size": 200, "budget": 250, "needs": ["light", "smart"]})
        assert [p["product_id"] for p in rec.result["products"]] == ["SW52"]

    def test_sorted_by_price_and_capped_at_three(self, runner):
        rec = runner.run("recommend_product", {"room_size": 100, "budget": 1000, "needs": []})
        prices = [p["price_usd"] for p in rec.result["products"]]
        assert prices == sorted(prices) and len(prices) <= 3

    def test_no_match_is_a_valid_empty_list(self, runner):
        rec = runner.run("recommend_product", {"room_size": 600, "budget": 100, "needs": []})
        assert rec.status == "ok" and rec.result["products"] == []

    def test_unrecognised_needs_are_reported(self, runner):
        rec = runner.run("recommend_product", {"room_size": 200, "budget": 300, "needs": ["outdoor", "bluetooth speaker"]})
        assert rec.result["unrecognised_needs"] == ["bluetooth speaker"]
        assert all("outdoor" in p["features"] for p in rec.result["products"])

    def test_rejects_non_positive_budget(self, runner):
        assert runner.run("recommend_product", {"room_size": 200, "budget": 0}).status == "invalid_arguments"


class TestSupportTicket:
    def test_creates_ticket(self, runner):
        rec = runner.run("create_support_ticket", {"summary": "Fan shows E3 after power cycle", "severity": "high"})
        assert rec.status == "ok"
        assert rec.result["ticket_id"].startswith("TKT-")
        assert runner.toolbox.tickets[-1]["severity"] == "high"

    def test_rejects_unknown_severity(self, runner):
        rec = runner.run("create_support_ticket", {"summary": "Fan is broken badly", "severity": "urgent"})
        assert rec.status == "invalid_arguments"


class TestRunnerErrors:
    def test_unknown_tool(self, runner):
        assert runner.run("delete_all_orders", {}).status == "unknown_tool"

    def test_arguments_not_json(self, runner):
        assert runner.run("get_order_status", "{order_id: ORD-1").status == "invalid_arguments"


class TestFaultInjection:
    @pytest.mark.parametrize("mode", ["timeout", "malformed", "empty"])
    def test_each_mode(self, runner, mode):
        runner.faults.set("get_order_status", mode)
        rec = runner.run("get_order_status", {"order_id": "ORD-10001"})
        assert rec.status == mode
        assert rec.result is None
        assert '"error"' in rec.llm_payload()

    def test_timeout_respects_budget(self, runner):
        runner.timeout_s = 0.3
        runner.faults.set("check_warranty", "timeout")
        rec = runner.run("check_warranty", {"serial_no": "HU42-2502-00031"})
        assert rec.status == "timeout"
        assert rec.latency_ms < 1000

    def test_wildcard_and_override(self):
        faults = FaultInjector("*=empty;get_order_status=malformed")
        assert faults.get("check_warranty") == "empty"
        assert faults.get("get_order_status") == "malformed"

    def test_clear_restores_normal_behaviour(self, runner):
        runner.faults.set("get_order_status", "empty")
        runner.faults.clear()
        assert runner.run("get_order_status", {"order_id": "ORD-10001"}).status == "ok"

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            FaultInjector("get_order_status=explode")

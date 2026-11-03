"""Black-box tests for POST /upload: happy paths and edge cases."""
import json
from pathlib import Path

import pytest

from tests.framework.base_test import BaseTest

pytestmark = pytest.mark.api
DOCS = Path(__file__).resolve().parents[2] / "eval" / "datasets" / "documents"
FILES = DOCS / "files"
LABELS = json.loads((DOCS / "labels.json").read_text(encoding="utf-8"))


class TestUploadHappyPath(BaseTest):
    @pytest.mark.parametrize("name", ["01_warranty_sw52.pdf", "03_warranty_tb60.png", "06_invoice_sw60.jpg"])
    def test_clean_document_is_extracted(self, api, openapi, name):
        resp = api.upload(FILES / name, name, LABELS[name]["content_type"])
        self.assert_status(resp, 200)
        openapi.validate("/upload", "post", resp)
        body = resp.json()
        assert body["status"] == "ok"
        assert body["fields"] == LABELS[name]["truth"]
        assert body["document_type"] == LABELS[name]["document_type"]

    def test_damaged_image_fails_gracefully(self, api, openapi):
        resp = api.upload(FILES / "10_smudged_ambiguous.png", "card.png", "image/png")
        self.assert_status(resp, 200)
        openapi.validate("/upload", "post", resp)
        body = resp.json()
        assert body["status"] != "ok" and body["warnings"]
        assert body["fields"]["serial_no"] is None and body["fields"]["purchase_date"] is None

    def test_upload_with_session_id(self, api):
        resp = api.upload(FILES / "01_warranty_sw52.pdf", "card.pdf", "application/pdf", session_id="upload-session-1")
        self.assert_status(resp, 200)


class TestUploadEdgeCases(BaseTest):
    @pytest.mark.parametrize("content_type", ["text/plain", "image/gif", "application/zip", "application/octet-stream"])
    def test_wrong_type_is_415(self, api, openapi, content_type):
        resp = api.upload(b"some bytes", "file.bin", content_type)
        self.assert_status(resp, 415)
        openapi.validate("/upload", "post", resp)

    def test_huge_file_is_413(self, api, openapi):
        resp = api.upload(b"%PDF-1.4\n" + b"0" * (6 * 1024 * 1024), "huge.pdf", "application/pdf")
        self.assert_status(resp, 413)
        openapi.validate("/upload", "post", resp)

    def test_empty_file_is_400(self, api, openapi):
        resp = api.upload(b"", "empty.pdf", "application/pdf")
        self.assert_status(resp, 400)
        openapi.validate("/upload", "post", resp)

    def test_truncated_pdf_is_422(self, api, openapi):
        resp = api.upload(FILES / "09_truncated_invoice.pdf", "invoice.pdf", "application/pdf")
        self.assert_status(resp, 422)
        openapi.validate("/upload", "post", resp)

    @pytest.mark.parametrize("content_type", ["application/pdf", "image/png"])
    def test_garbage_with_valid_type_is_422(self, api, content_type):
        self.assert_status(api.upload(b"this is not really a document" * 10, "x", content_type), 422)

    def test_missing_file_field_is_422(self, api):
        self.assert_status(api.session.post(f"{api.base_url}/upload", data={"session_id": "s1"},
                                            headers={"X-API-Key": api.api_key}), 422)

    def test_invalid_session_id_is_422(self, api):
        resp = api.upload(FILES / "01_warranty_sw52.pdf", "card.pdf", "application/pdf", session_id="bad id!")
        self.assert_status(resp, 422)

    def test_upload_requires_key(self, api):
        self.assert_status(api.upload(FILES / "01_warranty_sw52.pdf", "card.pdf", "application/pdf", api_key=None), 401)

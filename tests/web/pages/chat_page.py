"""Page object for the FanCare chat UI."""
import json
from pathlib import Path

from playwright.sync_api import Locator, Page, Route, expect


class ChatPage:
    def __init__(self, page: Page):
        self.page = page
        self.input = page.get_by_test_id("message-input")
        self.send_button = page.get_by_test_id("send-button")
        self.messages = page.get_by_test_id("messages")
        self.error_banner = page.get_by_test_id("error-banner")
        self.session_label = page.get_by_test_id("session-id")
        self.new_session_button = page.get_by_test_id("new-session")
        self.upload_input = page.get_by_test_id("upload-input")
        self.upload_button = page.get_by_test_id("upload-button")
        self.upload_result = page.get_by_test_id("upload-result")

    # ---------------------------------------------------------------- actions

    def open(self) -> "ChatPage":
        self.page.goto("/")
        expect(self.session_label).not_to_be_empty()
        return self

    def send(self, text: str) -> None:
        self.input.fill(text)
        self.send_button.click()

    def wait_for_reply(self, timeout: float = 60_000) -> Locator:
        reply = self.assistant_messages().last
        expect(reply).to_be_visible(timeout=timeout)
        expect(reply).not_to_have_class("message assistant streaming", timeout=timeout)
        expect(self.send_button).to_be_enabled(timeout=timeout)
        return reply

    def upload(self, path: Path) -> None:
        self.upload_input.set_input_files(str(path))
        self.upload_button.click()
        expect(self.upload_result).not_to_have_text("Reading document...", timeout=60_000)

    def reload(self) -> None:
        self.page.reload()
        expect(self.session_label).not_to_be_empty()

    def start_new_session(self) -> None:
        self.new_session_button.click()

    # -------------------------------------------------------------- queries

    def user_messages(self) -> Locator:
        return self.page.get_by_test_id("message-user")

    def assistant_messages(self) -> Locator:
        return self.page.get_by_test_id("message-assistant")

    def citations(self, reply: Locator) -> Locator:
        return reply.get_by_test_id("citation")

    def session_id(self) -> str:
        return self.session_label.inner_text()

    # ------------------------------------------------------- network mocks

    def mock_stream(self, deltas: list[str], citations: list[dict] | None = None, status: int = 200,
                    error: str | None = None) -> None:
        """Serve /chat/stream from a canned SSE body (deterministic UI tests, no LLM)."""
        def handle(route: Route) -> None:
            if status != 200:
                route.fulfill(status=status, content_type="application/json",
                              body=json.dumps({"detail": "LLM provider unavailable", "trace_id": "t"}))
                return
            events = [f"event: delta\ndata: {json.dumps({'text': d})}\n\n" for d in deltas]
            if error:
                events.append(f"event: error\ndata: {json.dumps({'detail': error})}\n\n")
            else:
                done = {"session_id": "s", "trace_id": "mock-trace", "answer": "".join(deltas),
                        "citations": citations or [], "tool_calls": []}
                events.append(f"event: done\ndata: {json.dumps(done)}\n\n")
            route.fulfill(status=200, headers={"content-type": "text/event-stream"}, body="".join(events))

        self.page.route("**/chat/stream", handle)

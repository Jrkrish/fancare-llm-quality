"""HTTP client for the FanCare API, used by the API, web and load suites."""
from pathlib import Path

import requests


class ApiClient:
    def __init__(self, base_url: str, api_key: str | None = "dev-local-key", timeout: float = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()

    def _headers(self, api_key: str | None = ..., extra: dict | None = None) -> dict:
        key = self.api_key if api_key is ... else api_key
        headers = {"X-API-Key": key} if key else {}
        return {**headers, **(extra or {})}

    def health(self) -> requests.Response:
        return self.session.get(f"{self.base_url}/health", timeout=self.timeout)

    def openapi(self) -> dict:
        resp = self.session.get(f"{self.base_url}/openapi.json", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def chat(self, session_id: str | None = "api-test", message: str | None = "hello", *, api_key=..., **extra_fields) -> requests.Response:
        body = {k: v for k, v in {"session_id": session_id, "message": message}.items() if v is not None}
        body.update(extra_fields)
        return self.post_json("/chat", body, api_key=api_key)

    def post_json(self, path: str, body, *, api_key=...) -> requests.Response:
        return self.session.post(f"{self.base_url}{path}", json=body, headers=self._headers(api_key), timeout=self.timeout)

    def post_raw(self, path: str, data: bytes, content_type: str, *, api_key=...) -> requests.Response:
        return self.session.post(f"{self.base_url}{path}", data=data, timeout=self.timeout,
                                 headers=self._headers(api_key, {"Content-Type": content_type}))

    def chat_stream(self, session_id: str, message: str, *, api_key=...) -> list[tuple[str, str]]:
        """Return the server-sent events as (event, data) pairs."""
        resp = self.session.post(f"{self.base_url}/chat/stream", json={"session_id": session_id, "message": message},
                                 headers=self._headers(api_key), timeout=self.timeout, stream=True)
        resp.raise_for_status()
        events, event = [], None
        for line in resp.iter_lines(decode_unicode=True):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                events.append((event, line[6:]))
        return events

    def history(self, session_id: str, *, api_key=...) -> requests.Response:
        return self.session.get(f"{self.base_url}/sessions/{session_id}/history", headers=self._headers(api_key), timeout=self.timeout)

    def upload(self, content: bytes | Path, filename: str, content_type: str, session_id: str | None = None,
               *, api_key=...) -> requests.Response:
        data = content.read_bytes() if isinstance(content, Path) else content
        form = {"session_id": session_id} if session_id else None
        return self.session.post(f"{self.base_url}/upload", files={"file": (filename, data, content_type)}, data=form,
                                 headers=self._headers(api_key), timeout=self.timeout)

"""Web suite fixtures: a keyless app server; pytest-playwright's `page` uses base_url."""
import pytest

from tests.framework.live_server import LiveServer, server_env
from tests.web.pages.chat_page import ChatPage


@pytest.fixture(scope="session")
def web_server():
    server = LiveServer(server_env()).start()
    yield server
    server.stop()


@pytest.fixture(scope="session")
def base_url(web_server):
    return web_server.url


@pytest.fixture
def chat(page) -> ChatPage:
    page.set_default_timeout(15_000)
    return ChatPage(page).open()

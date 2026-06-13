import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env so the LLM e2e tests' os.getenv-based skip gates resolve a key placed
# only in .env (Config reads it via pydantic-settings; os.environ would not).
load_dotenv()

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_url():
    """Serve a tests/fixtures/<name> file on 127.0.0.1 and yield its URL.

    Hermetic (loopback only) so the interaction smoke never depends on a live
    external site or hits anti-bot friction.
    """
    servers: list[tuple[HTTPServer, threading.Thread]] = []

    def _serve(filename: str) -> str:
        body = (FIXTURES / filename).read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_address[1]}/"

    yield _serve

    for server, thread in servers:
        server.shutdown()
        thread.join(timeout=2)

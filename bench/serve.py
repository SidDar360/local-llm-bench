"""Serve the benchmark dashboard on localhost.

    python -m bench.serve [--port 8000] [--open]

`results/latest.json` is read fresh on every request, so re-running a sweep and
refreshing the page is enough -- no restart. Stdlib only; this adds no
dependency to the project.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent / "web"
RESULTS = ROOT / "results"


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # Results change under the page; never let a proxy or the browser
        # hand back a stale sweep.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        path = self.path.split("?", 1)[0]

        if path in ("/", "/index.html"):
            page = WEB / "index.html"
            if not page.is_file():
                self._send(500, b"index.html missing", "text/plain; charset=utf-8")
                return
            self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            return

        if path == "/api/results":
            latest = RESULTS / "latest.json"
            if not latest.is_file():
                self._send(404, json.dumps(
                    {"error": f"no results at {latest.relative_to(ROOT)}"}
                ).encode(), "application/json")
                return
            self._send(200, latest.read_bytes(), "application/json")
            return

        if path == "/results/benchmark.png":
            png = RESULTS / "benchmark.png"
            if png.is_file():
                self._send(200, png.read_bytes(), "image/png")
                return

        self._send(404, b"not found", "text/plain; charset=utf-8")

    def log_message(self, fmt: str, *args) -> None:
        # One tidy line per request instead of the stdlib's noisier default.
        print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="open a browser window")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}"
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving {url}")
    if not (RESULTS / "latest.json").is_file():
        print("  (no results yet -- run: python -m bench.run --quick)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        server.server_close()


if __name__ == "__main__":
    main()

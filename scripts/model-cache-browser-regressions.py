#!/usr/bin/env python3
"""Exercise production model loading with real browser Cache Storage and SHA-256.

Uses tiny local fixtures, never the CDN or a GPU. Run through browser_evidence.py
in CI so browser failure diagnostics and teardown remain bounded.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
GOOD = bytes(range(1, 65))
BAD = bytes(len(GOOD))


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/__model-test.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<!doctype html><title>Model integrity regression</title>")
        else:
            super().do_GET()

    def log_message(self, *_args):
        pass


SETUP = """async (url) => {
  const good = Uint8Array.from({length: 64}, (_, i) => i + 1);
  const sha256 = [...new Uint8Array(await crypto.subtle.digest('SHA-256', good))]
    .map(x => x.toString(16).padStart(2, '0')).join('');
  window.release = {url, bytes: good.length, sha256};
  window.load = (await import('/src/neural-model-cache.js')).fetchVerifiedModel;
  window.store = await caches.open('connect4-neural-model');
}"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", choices=("chromium", "webkit"), default="chromium")
    parser.add_argument("--executable")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(ROOT)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    url = origin + "/model-fixture.onnx"
    payload = [GOOD]
    requests = []
    try:
        with sync_playwright() as pw:
            browser = getattr(pw, args.browser).launch(headless=True, executable_path=args.executable)
            context = browser.new_context()
            page = context.new_page()

            def respond(route):
                requests.append(route.request.url)
                route.fulfill(status=200, body=payload[0], content_type="application/octet-stream")

            page.route(url, respond)
            page.goto(origin + "/__model-test.html")
            page.evaluate(SETUP, url)
            # An older worker/visit left a same-sized corrupt cache entry.
            page.evaluate("async () => { await store.put(release.url, new Response(new Uint8Array(release.bytes))); }")
            assert page.evaluate("async () => [...new Uint8Array(await load(release))]") == list(GOOD)
            assert len(requests) == 1
            # A fresh page/module uses real persistence, but verifies before returning.
            page.reload()
            page.evaluate(SETUP, url)
            assert page.evaluate("async () => [...new Uint8Array(await load(release))]") == list(GOOD)
            assert len(requests) == 1
            # A bad download cannot poison the next Retry.
            page.evaluate("async () => { await store.delete(release.url); }")
            payload[0] = BAD
            error = page.evaluate("""async () => {
              try { await load(release); return null; } catch (e) { return e.code; }
            }""")
            assert error == "MODEL_INTEGRITY", error
            assert page.evaluate("async () => Boolean(await store.match(release.url))") is False
            payload[0] = GOOD
            assert page.evaluate("async () => [...new Uint8Array(await load(release))]") == list(GOOD)
            assert len(requests) == 3
            context.close()
            browser.close()
        print(f"{args.browser}: cache corruption, reload, download rejection and Retry passed")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()

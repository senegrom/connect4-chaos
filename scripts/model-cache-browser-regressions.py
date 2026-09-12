#!/usr/bin/env python3
"""Production model integrity with real Cache Storage, Web Crypto and HTTP.

Persistence is tested in a disk-backed profile, not a private context. WebKit
intentionally loses ephemeral Cache Storage entries across reload (Playwright
#41701). Private-context eviction is a separate recovery test, never an excuse
to skip the persistent-profile assertions. Run through browser_evidence.py.
"""
import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
GOOD = bytes(range(1, 65))
BAD = bytes(len(GOOD))
DIGEST = hashlib.sha256(GOOD).hexdigest()


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/__model-test.html", "/model-fixture.onnx"):
            is_model = self.path == "/model-fixture.onnx"
            with self.server.fixture_lock:
                if is_model:
                    self.server.requests += 1
                data = self.server.payload if is_model else b"<!doctype html><title>Model integrity regression</title>"
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream" if is_model else "text/html")
            self.send_header("Content-Length", str(len(data)))
            # Every download must reach this server. An HTTP-cache hit cannot
            # masquerade as successful application Cache Storage persistence.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
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
LOAD = "async () => [...new Uint8Array(await load(release))]"
SNAPSHOT = """async () => {
  const response = await store.match(release.url);
  const bytes = response ? await response.arrayBuffer() : null;
  const state = {keys: (await store.keys()).map(r => r.url),
    bytes: bytes?.byteLength ?? null,
    sha256: bytes ? [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
      .map(x => x.toString(16).padStart(2, '0')).join('') : null};
  console.log('model cache state', JSON.stringify(state));
  return state;
}"""


def snapshot(page, label):
    state = page.evaluate(SNAPSHOT)
    print(f"{label}: {json.dumps(state, sort_keys=True)}", flush=True)
    return state


def stored_good(page, label):
    state = snapshot(page, label)
    assert state["bytes"] == len(GOOD) and state["sha256"] == DIGEST, (label, state)


def prepare(context, origin, url):
    page = context.new_page()
    page.goto(origin + "/__model-test.html")
    page.evaluate(SETUP, url)
    return page


def corruption_and_retry(page, server):
    before = server.requests
    page.evaluate("async () => { await store.put(release.url, new Response(new Uint8Array(release.bytes))); }")
    state = snapshot(page, "seeded corruption")
    assert state["bytes"] == len(BAD) and state["sha256"] != DIGEST, state
    assert page.evaluate(LOAD) == list(GOOD)
    assert server.requests == before + 1, (before, server.requests)
    stored_good(page, "verified replacement after cache write")
    # Reject a bad download, leave no entry, then recover without a new context.
    page.evaluate("async () => { await store.delete(release.url); }")
    with server.fixture_lock:
        server.payload = BAD
    error = page.evaluate("""async () => {
      try { await load(release); return null; } catch (e) { return e.code; }
    }""")
    assert error == "MODEL_INTEGRITY", error
    assert snapshot(page, "rejected download")["bytes"] is None
    with server.fixture_lock:
        server.payload = GOOD
    assert page.evaluate(LOAD) == list(GOOD)
    assert server.requests == before + 3, (before, server.requests)
    stored_good(page, "Retry stored a verified model")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", choices=("chromium", "webkit"), default="chromium")
    parser.add_argument("--executable")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(ROOT)))
    server.fixture_lock = threading.Lock()
    server.requests, server.payload = 0, GOOD
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    url = origin + "/model-fixture.onnx"
    try:
        with TemporaryDirectory(prefix="connect4-cache-profile-") as profile:
            with sync_playwright() as pw:
                browser_type = getattr(pw, args.browser)
                options = dict(headless=True, executable_path=args.executable)
                context = browser_type.launch_persistent_context(profile, **options)
                page = prepare(context, origin, url)
                corruption_and_retry(page, server)
                before = server.requests
                page.reload()
                page.evaluate(SETUP, url)
                stored_good(page, "persistent profile after reload, BEFORE loading")
                assert page.evaluate(LOAD) == list(GOOD)
                assert server.requests == before, ("reload downloaded again", before, server.requests)
                context.close()

                # The model must also survive an actual browser restart.
                context = browser_type.launch_persistent_context(profile, **options)
                page = prepare(context, origin, url)
                stored_good(page, "persistent profile after browser restart")
                assert page.evaluate(LOAD) == list(GOOD)
                assert server.requests == before, ("restart downloaded again", before, server.requests)
                context.close()

                browser = browser_type.launch(**options)
                private = browser.new_context()
                page = prepare(private, origin, url)
                corruption_and_retry(page, server)
                page.reload()
                page.evaluate(SETUP, url)
                state = snapshot(page, "private profile after reload")
                # Browsers may discard private data. Either a verified entry or
                # absence is valid here; the disk-backed assertions above stay strict.
                assert state["sha256"] in (None, DIGEST), state
                before = server.requests
                assert page.evaluate(LOAD) == list(GOOD)
                assert server.requests == before + (state["bytes"] is None)
                # Force an eviction in BOTH engines so recovery is always exercised.
                page.evaluate("async () => { await store.delete(release.url); }")
                before = server.requests
                assert page.evaluate(LOAD) == list(GOOD)
                assert server.requests == before + 1
                private.close()
                browser.close()
        print(f"{args.browser}: persistent reload/restart, corruption, Retry and private eviction passed")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()

# Neural execution and recovery

Native ONNX model loading, warm-up and CPU/GPU inference run in
`src/neural-worker.js`. The page communicates through `neural-client.js`;
`neural-search.js` continues to operate on small board/tensor values and yields
between cached simulation batches. There is no main-thread native fallback.
The worker is served from the same origin, without relaxing the existing CSP.

A watchdog on the page bounds each inference and each session startup phase.
It can terminate the worker even when synchronous WASM cannot service its own
timers. Timeout/error/cancellation discards pending calls; Retry creates a fresh
worker and session rather than joining the failed inference queue. Successful
moves retain a healthy worker to avoid repeated model startup. Requests and
network handles are generation-scoped so old cleanup cannot reset a replacement.

The GPU guard records actual backend errors and watchdog failures in this tab's
sessionStorage. It no longer treats a shared localStorage 'active' marker as a
crash. User cancellation does not count as a GPU failure. Storage denial falls
back to an in-memory guard and never prevents play.

Complete Chaos policies remain board-only certificates. Their selected move is
checked against actual repetition history before reporting an immediate result.
When an earlier position in the same piece-count layer makes a nonterminal
value history-dependent, the UI shows a conditional certificate value, not a
history-aware proof. This does not change the certified move-selection policy.

## Tests

- `node --test tests/second-review.test.js`: scheduling, worker lifecycle,
  watchdogs, retry, tab-local GPU policy, telemetry and actual shipped policy.
- `python scripts/neural-worker-regressions.py --browser chromium --real-model`
- `python scripts/neural-worker-regressions.py --browser webkit --real-model`

The browser suite includes controlled synchronous worker stalls as well as an
unmocked load, warm-up and inference using the committed ONNX/WASM assets.
Both browser jobs run in the existing Browser regressions workflow on pushes.
WebKit automation is not a physical iPhone installation test or a hardware-GPU
benchmark. A real-device check is still required for home-screen installation.

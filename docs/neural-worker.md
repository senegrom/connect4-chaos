# Neural execution and recovery

Native ONNX model loading, warm-up and CPU/GPU inference run in
`src/neural-worker.js`. The page communicates through `neural-client.js`;
`neural-search.js` continues to operate on small board/tensor values and yields
between cached simulation batches. There is no main-thread native fallback.
The worker is served from the same origin, without relaxing the existing CSP.

A watchdog on the page bounds each inference and each session startup phase.
It can terminate the worker even when synchronous WASM cannot service its own
timers. The download is bounded by silence, not by its length: every chunk
re-arms the worker's 60 s stall limit, and the page's watchdog gives up only
after 120 s without progress. A timeout or a worker error discards the
pending calls; Retry then creates a fresh worker and session rather than
joining the failed inference queue. A healthy worker is kept between moves
and released after two minutes without activity. Idle expiry never
interrupts an inference or marks a GPU failure. Requests and network handles
are generation-scoped so old cleanup cannot reset a replacement.

Restart, Play again, Undo and Retry cancel the request in flight, a load or a
search, but keep a loaded network: starting it again re-reads and hashes the
whole model and rebuilds the session. A cancelled search can leave its last
evaluation running in the worker, and the worker refuses overlapping requests,
so the client sends one request at a time and the next move's first
evaluation waits for it. Choosing another opponent releases the network.
Hiding the page cancels a neural turn, which resumes when the page is visible
again; on iPhones and iPads, where the CPU session holds hundreds of MB and a
background tab is the first to be killed, hiding also releases the network.
Stable positions are already saved before each AI turn; existing errors and
human turns do not launch inference. A full reload still requires Retry as
described below.

The classic AI worker keeps its verified tables on the same terms: across
rounds and Undo while the rules stay the same, released by other rules or
another opponent and after two minutes without a request. Its search is
synchronous, so cancelling a request it is running terminates it.

The GPU guard records actual backend errors and watchdog failures in this tab's
sessionStorage. It no longer treats a shared localStorage 'active' marker as a
crash. User cancellation does not count as a GPU failure. Storage denial falls
back to an in-memory guard and never prevents play.

On iPhones and iPads the client explicitly selects WASM, including iPadOS in
desktop-site mode. Merely advertising WebGPU is not sufficient for this large
model: Safari process termination bypasses JavaScript error handlers. Desktop
GPU startup failures discard the worker; Retry then uses WASM. Startup no longer
creates both GPU and CPU sessions to compare timings. A mid-game GPU failure
drains any in-flight inference and awaits GPU session release before creating
the CPU replacement. Device-loss notifications never race native evaluation.

Inference copies out the 55 result logits and disposes every input/output tensor,
also on failure. Downloads fill a single preallocated model buffer; runtime cache
warm-up discards streamed bytes instead of building an unused WASM buffer. Once
the native session owns the model, the extra downloaded model buffer is released
before warm-up. GPU sessions do not keep a spare model buffer either: only an
actual fallback reads it again, after GPU release, from the verified model
cache when possible.

CPU sessions use basic graph optimization. Higher-level CPU fusions and layout
conversions increase startup memory with the committed FP16 model. WebGPU retains
full optimization. The model, weights, precision, encoder and search algorithm
are unchanged; inference timing still determines the search budget.

`node scripts/neural-memory-benchmark.mjs` compares the previous CPU startup
policy and production in separate processes using the shipped ONNX/WASM assets.
On a Linux/Node run, peak process RSS fell from 655 MiB to 559 MiB (15%); resident
RSS after the same workload and garbage collection fell from 471 MiB to 370 MiB.
Median evaluation took 226 ms instead of 195 ms (16% longer). Across 40 positions
covering Classic, Chaos, different board dimensions, rotations and repetition
planes, preferred legal policy actions all matched and the largest logit change
was 0.00390625. These are workload/process measurements, not a physical iPhone
memory limit or a guarantee of identical floating-point results or search depth.

Every initial round is saved before an AI opening move. Startup restores a saved
round before launching AI work. If it is the neural opponent's turn, the restored
board waits for Retry, Undo or another opponent. This deliberately breaks the
automatic reload/retry/crash loop without changing the position or game history.

Complete Chaos policies remain board-only certificates. Their selected move is
checked against actual repetition history before reporting an immediate result.
When an earlier position in the same piece-count layer makes a nonterminal
value history-dependent, the UI shows a conditional certificate value, not a
history-aware proof. This does not change the certified move-selection policy.

## Tests

- `node --test tests/second-review.test.js`: scheduling, worker lifecycle,
  watchdogs, retry, tab-local GPU policy, telemetry and actual shipped policy.
- `node --test tests/neural-memory.test.js tests/neural-reload.test.js`:
  mobile backend policy, resource lifetime, streaming and repeatable restoration.
- `python scripts/neural-worker-regressions.py --browser chromium --real-model`
- `python scripts/neural-worker-regressions.py --browser webkit --real-model`

The browser suite includes controlled synchronous worker stalls as well as an
unmocked load, warm-up and inference using the committed ONNX/WASM assets.
Both browser jobs run in the existing Browser regressions workflow on pushes.
WebKit automation is not a physical iPhone installation test or a hardware-GPU
benchmark. A real-device check is still required for home-screen installation.

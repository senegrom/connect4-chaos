# Training ancestry and recovery

## Validation claims describe the full lineage

`data_split_version` and `holdout_configs` describe historical exclusions, not
just the last fine-tuning run. New checkpoints also carry
`training_provenance` (format 1), with a `clean` or `unknown` status and separate
`run_split_version` / `run_holdout_configs` fields.

A fresh training run establishes clean provenance. A clean warm start preserves
only the intersection of its parent's lifetime holdouts and the current run's
holdouts. Adding a holdout cannot erase earlier exposure, and a dropped holdout
cannot be reclaimed later. Rectangular Chaos configurations are normalized so
rotated spellings represent the same exclusion.

Legacy checkpoints remain usable for fine-tuning. However, the old trainer
stamped run-only metadata onto descendants, so even an old checkpoint naming
the current split is not proof of an unbroken clean lineage. Missing,
unsupported or unknown ancestry produces an explicitly unknown child, with
empty lifetime claims. Later warm starts and model averaging cannot upgrade it.
The current run's exclusions remain recorded separately for diagnostics.

Model averaging retains the existing partition-compatibility checks and
recalibration exclusions. Its published provenance is clean only when every
source has clean lineage-aware metadata. No existing checkpoint is rewritten.

Held-out measurements from an unknown lineage are diagnostics, not evidence
that those positions or configurations were never seen by the weights.

## Optional optimizer state cannot corrupt checkpoint publication

AdamW recovery validates group/parameter IDs, moment shapes and dtypes, finite
scalars and tensors, required fields, nonnegative second moments and compatible
optimizer settings before loading. A second check follows PyTorch's state
conversion. CPU/CUDA execution settings and the new learning rate belong to the
current optimizer; they are not copied from another machine's sidecar.

Any recovery error discards the candidate optimizer completely and constructs
a fresh one. `DISTILL_RESET_OPTIMIZER=1` explicitly skips restoration. Validation
cannot detect finite but semantically unrelated moments of the same layout;
sidecars should still be kept with the checkpoint that produced them.

Model parameters and buffers must be finite before warm-start use and before a
checkpoint file can become visible. A failed final check leaves any previous
completed checkpoint untouched. This check also catches corruption introduced
by the final optimizer step even when its preceding forward loss was finite.

## Browser persistence tests

The model-cache suite uses a temporary persistent profile for strict cache
checks across reload and browser restart. Requests go to a local HTTP server
with `Cache-Control: no-store`, so HTTP caching cannot hide a failed Cache
Storage lookup. Cache sizes, digests and keys are logged before and after reload.

Private-context eviction/recovery is tested separately. WebKit's in-memory
Cache Storage may disappear on reload; this is documented in Playwright's
upstream regression and does not relax the persistent-profile assertions:
https://github.com/microsoft/playwright/pull/41701

The shared evidence runner observes persistent as well as ordinary contexts,
retaining screenshots, console messages and traces before driver teardown.

## Checks

```sh
python -m neural.test_training_safety
python -m neural.test_training_recovery
python scripts/test-browser-persistence.py
python scripts/browser_evidence.py scripts/model-cache-browser-regressions.py --browser chromium
python scripts/browser_evidence.py scripts/model-cache-browser-regressions.py --browser webkit
```

The training safety tests use real CPU networks, training updates and checkpoint
IO with controlled dataset fixtures. Browser tests exercise real Cache Storage,
Web Crypto and the production model loader without any CDN upload or GPU job.

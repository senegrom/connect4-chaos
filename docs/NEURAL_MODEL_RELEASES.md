# Verified neural model releases

The browser pins the current model URL, uncompressed byte count and SHA-256
in `src/neural-runtime.js`. `assets/neural/model.json` records the same identity.
Do not change one without the other. A matching byte count alone is not proof
that a download or cached copy belongs to a release.

## Export and publish

Export the selected checkpoint into an output directory:

```sh
python -m neural.export_onnx path/to/checkpoint.pt output/model.onnx --half
```

The exporter checks all three heads for expected shapes, finite logits,
finite probabilities and finite differences before applying parity tolerances.
It stages the model and JSON sidecar without touching an earlier export until
validation and metadata writing have succeeded. Individual file replacements
are atomic; if a crash separates the two replacements, consumers reject the
mismatched identity rather than trusting it. Publication must use the complete,
matching model and sidecar.

Publish using that sidecar explicitly:

```sh
node scripts/publish-model-r2.mjs output/model.onnx --manifest output/model.json
```

The publisher verifies the declared length and SHA-256 before invoking Wrangler.
It uses `models/<checkpoint-name>/<full-model-sha256>` as the object key. The
extensionless filename is intentional: the CDN serves it as an octet stream and
its existing key allowlist accepts it. Different exports of the same checkpoint
receive different URLs; a repeated upload to the same key can only carry the
same verified model bytes. The publisher never writes the legacy
`models/<checkpoint-name>/model.onnx` URLs. Do not overwrite those objects with
manual commands either: existing clients may retain their immutable responses.

Copy the new export metadata into `assets/neural/model.json`, add the publisher's
`origin`, `object` and `storedBytes`, and update `MODEL_OBJECT`, `MODEL_SHA256` and
`DOWNLOAD_BYTES.model` in `src/neural-runtime.js` in the same commit. Preserve the
old objects for rollback. A rollback restores both the manifest and the runtime
identity; it does not replace a previously published object. When an export's
bytes change, the exporter clears stale `object`/`storedBytes` fields rather than
claiming that the new bytes already exist at an older URL.

## Cache and recovery

Both browser downloads and Cache Storage reads are verified before use. A
mismatched entry is evicted and a replacement is downloaded. Bad downloads are
not cached, so Retry does not keep resurrecting them. Storage/quota failures
remain optional: verified bytes can still run for the current visit. A GPU
failure alone does not cause a valid model to be discarded.

The Node resolver also verifies local, override, legacy-cache and downloaded
bytes. New disk cache entries use the content digest, a unique temporary file
and atomic rename, so an interrupted writer cannot expose a partial model.
Downloads remain opt-in. `NEURAL_MODEL` must match the selected manifest;
programmatic callers evaluating a different export can pass its JSON sidecar as
`manifestPath` to `readModelBytes` rather than bypassing verification.

## Regression checks

```sh
node --test tests/model-release.test.js
python -m neural.test_export_onnx
python scripts/browser_evidence.py scripts/model-cache-browser-regressions.py --browser chromium
python scripts/browser_evidence.py scripts/model-cache-browser-regressions.py --browser webkit
```

The Node tests use small deterministic byte fixtures and fake upload boundaries;
they never publish a cloud object. The exporter tests run the real PyTorch model,
parity math, checkpoint IO and staging logic, replacing ONNX serialization/runtime
to inject NaN, infinity, shape/count errors and finite mismatches. The browser
regression uses real Cache Storage, Web Crypto and production download/loading
code with tiny locally served responses. CI also retains the existing real-model
and full browser suites. No GPU or production upload is needed for these fault
injection tests.

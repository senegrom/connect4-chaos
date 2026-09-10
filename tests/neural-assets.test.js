import assert from 'node:assert/strict';
import test from 'node:test';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { DOWNLOAD_BYTES, assetUrls } from '../src/neural-runtime.js';

// A relative specifier in a dynamic import resolves against the module, not
// the page, so './assets/...' from src/ silently looked inside src/assets
// and the opponent hung waiting for a file that was never there.
test('the runtime, model, loader and metadata resolve to files that exist', () => {
  const urls = assetUrls();
  for (const [name, entry] of Object.entries(urls)) {
    if (name === 'base') continue;
    // The model is split into parts small enough for GitHub to store, so
    // one entry names several files; each of them has to be there.
    for (const url of Array.isArray(entry) ? entry : [entry]) {
      assert.ok(url.includes('/assets/neural/'), `${name} should live in assets/neural: ${url}`);
      assert.ok(!url.includes('/src/assets/'), `${name} resolved inside src/: ${url}`);
      assert.ok(existsSync(fileURLToPath(url)), `${name} is missing on disk: ${url}`);
    }
  }
});

// The parts are raw slices of one export: their sizes must add up to the
// size the loader allocates, or the reassembled model is truncated.
test('the model parts match the size the loader expects', () => {
  const urls = assetUrls();
  const sizes = urls.modelParts.map((url) => statSync(fileURLToPath(url)).size);
  assert.deepEqual(sizes, DOWNLOAD_BYTES.modelParts);
  assert.equal(sizes.reduce((sum, size) => sum + size, 0), DOWNLOAD_BYTES.model);
});

// The runtime bundle names the WebAssembly loader it will import at session
// creation. Shipping a different variant (the 1.29 bundle wants the asyncify
// build, an older one the jsep build) fails every backend with a 404 the
// page reports only as "no available backend found".
test('every wasm loader the runtime bundle names is shipped with its wasm', () => {
  const urls = assetUrls();
  const bundle = readFileSync(fileURLToPath(urls.runtime), 'utf8');
  const loaders = new Set(bundle.match(/ort-wasm-simd-threaded[\w.]*\.mjs/g) ?? []);
  assert.ok(loaders.size > 0, 'the bundle should name at least one wasm loader');
  for (const loader of loaders) {
    const loaderUrl = new URL(loader, urls.base);
    const wasmUrl = new URL(loader.replace(/\.mjs$/, '.wasm'), urls.base);
    assert.ok(existsSync(fileURLToPath(loaderUrl)), `${loader} is not vendored`);
    assert.ok(existsSync(fileURLToPath(wasmUrl)), `${loader} has no matching .wasm vendored`);
  }
  assert.ok(loaders.has(urls.loader.split('/').pop()),
    'the prefetched loader should be the one the bundle asks for');
});

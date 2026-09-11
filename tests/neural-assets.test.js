import assert from 'node:assert/strict';
import test from 'node:test';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { DOWNLOAD_BYTES, assetUrls } from '../src/neural-runtime.js';

// A relative specifier in a dynamic import resolves against the module, not
// the page, so './assets/...' from src/ silently looked inside src/assets
// and the opponent hung waiting for a file that was never there.
test('the runtime, loader and metadata resolve to files that exist', () => {
  const urls = assetUrls();
  for (const [name, entry] of Object.entries(urls)) {
    // The model is fetched from R2 and is checked separately below; every
    // other asset is a file this repository ships.
    if (name === 'base' || name === 'model') continue;
    for (const url of Array.isArray(entry) ? entry : [entry]) {
      assert.ok(url.includes('/assets/neural/'), `${name} should live in assets/neural: ${url}`);
      assert.ok(!url.includes('/src/assets/'), `${name} resolved inside src/: ${url}`);
      assert.ok(existsSync(fileURLToPath(url)), `${name} is missing on disk: ${url}`);
    }
  }
});

// The loader allocates one buffer of DOWNLOAD_BYTES.model and refuses a
// response that does not fill it, so a manifest naming a different length -
// or a different object - would fail every download rather than some of them.
test('the loader fetches exactly the object the manifest publishes', () => {
  const manifest = JSON.parse(readFileSync(new URL('../assets/neural/model.json', import.meta.url), 'utf8'));
  assert.equal(DOWNLOAD_BYTES.model, manifest.bytes);
  assert.equal(assetUrls().model, `${manifest.origin}/${manifest.object}`);
  // The key carries the generation, which is what lets the response be
  // immutable and a rollback be one line of this manifest.
  assert.ok(manifest.object.includes(manifest.source.replace(/\.pt$/, '')),
    `${manifest.object} should name ${manifest.source}`);
});

// A page may only fetch what its Content-Security-Policy allows, and a policy
// that omits the model's origin blocks the download before any request is
// made: nothing reaches the network and nothing fails, so the opponent simply
// never starts. This is how it broke the first time.
test('the page policy allows the origin the model is fetched from', () => {
  const manifest = JSON.parse(readFileSync(new URL('../assets/neural/model.json', import.meta.url), 'utf8'));
  const page = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  const policy = page.match(/Content-Security-Policy[^>]*content="([^"]+)"/)?.[1] ?? '';
  const connect = policy.match(/connect-src ([^;]+)/)?.[1] ?? '';
  assert.ok(connect.split(/\s+/).includes(manifest.origin),
    `connect-src does not allow ${manifest.origin}: ${connect}`);
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

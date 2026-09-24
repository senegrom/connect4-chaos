import assert from 'node:assert/strict';
import test from 'node:test';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { DOWNLOAD_BYTES, assetUrls, cancelNeuralLoad, loadNeuralNetwork } from '../src/neural-runtime.js';

// A relative specifier in a dynamic import resolves against the module, not
// the page, so './assets/...' from src/ silently looked inside src/assets
// and the opponent hung waiting for a file that was never there.
test('the runtime, its loader and its wasm resolve to files that exist', () => {
  const urls = assetUrls();
  for (const [name, url] of Object.entries(urls)) {
    // The model is fetched from R2 and is checked separately below; every
    // other asset is a file this repository ships.
    if (name === 'base' || name === 'model') continue;
    assert.ok(url.includes('/assets/neural/'), `${name} should live in assets/neural: ${url}`);
    assert.ok(!url.includes('/src/assets/'), `${name} resolved inside src/: ${url}`);
    assert.ok(existsSync(fileURLToPath(url)), `${name} is missing on disk: ${url}`);
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

// The page runs the three runtime files vendored here, while the Node tests
// and benchmarks run the npm package. Dependabot moved the package to 1.30
// on 2026-09-22 with the site still on 1.29, so the tests stopped measuring
// what players get. Pinning the package to the vendored release, and failing
// when they differ, turns a lone package bump into a red check that says to
// re-vendor the runtime alongside it.
test('the npm runtime is the exact release the page ships', async () => {
  const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
  const lock = JSON.parse(readFileSync(new URL('../package-lock.json', import.meta.url), 'utf8'));
  const pinned = pkg.devDependencies['onnxruntime-web'];
  assert.match(pinned, /^\d+\.\d+\.\d+$/, 'onnxruntime-web must be pinned to one exact release');
  assert.equal(lock.packages['node_modules/onnxruntime-web'].version, pinned);
  const urls = assetUrls();
  const banner = readFileSync(fileURLToPath(urls.runtime), 'utf8').match(/ONNX Runtime Web v(\d+\.\d+\.\d+)/)?.[1];
  assert.equal(banner, pinned,
    `assets/neural ships ${banner} but package.json pins ${pinned}: copy ort.webgpu.min.mjs and the ` +
    'asyncify .mjs/.wasm from node_modules/onnxruntime-web/dist into assets/neural and update DOWNLOAD_BYTES.runtime');
  const dist = new URL('../node_modules/onnxruntime-web/dist/', import.meta.url);
  if (!existsSync(fileURLToPath(dist))) return;
  for (const url of [urls.runtime, urls.loader, urls.wasm]) {
    const name = url.split('/').pop();
    assert.ok(readFileSync(fileURLToPath(url)).equals(readFileSync(fileURLToPath(new URL(name, dist)))),
      `assets/neural/${name} differs from the npm ${pinned} build`);
  }
});

// assets/neural/model.json was fetched on every load and never read, and with
// no catch a transient failure of that fetch failed the whole load.
test('a load fetches the model and the runtime, and nothing else', async (t) => {
  const requested = [];
  t.mock.method(globalThis, 'fetch', (url, { signal } = {}) => {
    requested.push(String(url));
    return new Promise((_resolve, reject) => {
      signal?.addEventListener('abort', () => reject(new DOMException('Cancelled', 'AbortError')), { once: true });
    });
  });
  const loading = assert.rejects(loadNeuralNetwork({ allowWebgpu: false }), { name: 'AbortError' });
  try {
    await new Promise((resolve) => setImmediate(resolve));
    const { model, wasm } = assetUrls();
    assert.deepEqual(requested.sort(), [model, wasm].sort());
  } finally {
    cancelNeuralLoad(); // also when the assertion fails, or the load's timers keep the run alive
    await loading;
  }
});

// A ten-minute deadline on the whole download made the model impossible to
// load below about 1.5 Mbit/s. Here the model arrives a kilobyte every 30 s
// for twelve and a half minutes, then stops.
test('a slow download goes on while bytes arrive, and fails once they stop', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  let flowing = true;
  t.mock.method(globalThis, 'fetch', async (url) => {
    if (String(url) !== assetUrls().model) return new Response(new Uint8Array(8));
    return new Response(new ReadableStream({
      pull: (controller) => new Promise((resolve) => {
        if (flowing) setTimeout(() => { controller.enqueue(new Uint8Array(1_000)); resolve(); }, 30_000);
      }),
    }));
  });
  const settle = () => new Promise((resolve) => setImmediate(resolve));
  let failure = null;
  const loading = loadNeuralNetwork({ allowWebgpu: false }).catch((error) => { failure = error; });
  try {
    for (let chunk = 0; chunk < 25; chunk += 1) {
      await settle();
      t.mock.timers.tick(30_000);
    }
    await settle();
    assert.equal(failure, null, 'a download that keeps arriving must not time out');
    flowing = false;
    await settle();
    t.mock.timers.tick(60_000);
    await loading;
    assert.match(failure?.message ?? '', /stalled/);
  } finally {
    cancelNeuralLoad();
    await loading;
  }
});

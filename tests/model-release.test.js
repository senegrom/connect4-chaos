import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gunzipSync } from 'node:zlib';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { modelIdentity, verifyModelBytes, ModelIntegrityError } from '../src/model-integrity.js';
import { fetchVerifiedModel } from '../src/neural-model-cache.js';
import { readModelBytes } from '../scripts/model-source.mjs';
import { publishModel } from '../scripts/publish-model-r2.mjs';

const GOOD = Buffer.from('verified model fixture A');
const BAD = Buffer.alloc(GOOD.length);
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const identity = (bytes = GOOD) => ({ bytes: bytes.length, sha256: hash(bytes) });
const release = (bytes = GOOD) => ({ ...identity(bytes), url: 'https://model.invalid/models/fixture/model.onnx' });
const tick = () => new Promise((resolve) => setImmediate(resolve));

function memoryStorage() {
  const entries = new Map(), deleted = [];
  const store = {
    async match(url) { return entries.get(String(url))?.clone(); },
    async put(url, response) { entries.set(String(url), response.clone()); },
    async delete(key) { const url = key.url ?? String(key); deleted.push(url); return entries.delete(url); },
    async keys() { return [...entries.keys()].map((url) => new Request(url)); },
  };
  return { entries, deleted, store, async open() { return store; } };
}

async function fixture(t, bytes = GOOD) {
  const root = await mkdtemp(join(tmpdir(), 'connect4-model-release-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const manifestPath = join(root, 'model.json');
  const modelPath = join(root, 'model.onnx');
  const cacheDirectory = join(root, 'cache');
  const manifest = { ...identity(bytes), source: 'fixture.pt',
    origin: 'https://model.invalid', object: 'models/fixture/model.onnx' };
  await writeFile(manifestPath, JSON.stringify(manifest));
  return { root, manifest, manifestPath, modelPath, cacheDirectory };
}

test('model identity fails closed on missing, invalid and unsafe metadata', () => {
  for (const bytes of [0, -1, 0.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => modelIdentity({ ...identity(), bytes }), ModelIntegrityError);
  }
  for (const sha256 of ['', null, 'bad', 'g'.repeat(64)]) {
    assert.throws(() => modelIdentity({ ...identity(), sha256 }), ModelIntegrityError);
  }
  assert.ok(Object.isFrozen(modelIdentity(identity())));
});

test('SHA-256 verification handles subviews and rejects same-sized replacements', async () => {
  const allocation = Buffer.concat([Buffer.from('prefix'), GOOD, Buffer.from('suffix')]);
  const view = allocation.subarray(6, 6 + GOOD.length);
  assert.deepEqual(Buffer.from(await verifyModelBytes(view, identity())), GOOD);
  await verifyModelBytes(Uint8Array.from(GOOD).buffer, identity());
  await assert.rejects(verifyModelBytes(BAD, identity()), /SHA-256/);
  await assert.rejects(verifyModelBytes(GOOD.subarray(1), identity()), /length/);
});

test('Node crypto fallback enforces the same model identity', async (t) => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: undefined });
  t.after(() => Object.defineProperty(globalThis, 'crypto', descriptor));
  await verifyModelBytes(GOOD, identity());
  await assert.rejects(verifyModelBytes(BAD, identity()), /SHA-256/);
});

test('invalid download is never cached; Retry and a fresh module recover', async (t) => {
  const storage = memoryStorage();
  let delivered = BAD;
  const fetch = t.mock.method(globalThis, 'fetch', async () => new Response(delivered));
  await assert.rejects(fetchVerifiedModel(release(), { storage }), /SHA-256/);
  assert.equal(storage.entries.size, 0);
  delivered = GOOD;
  assert.deepEqual(Buffer.from(await fetchVerifiedModel(release(), { storage })), GOOD);
  const fresh = await import('../src/neural-model-cache.js?new-worker');
  assert.deepEqual(Buffer.from(await fresh.fetchVerifiedModel(release(), { storage })), GOOD);
  assert.equal(fetch.mock.callCount(), 2, 'a verified cache needs no third request');
});

test('same-size corrupted persistent cache is evicted before a verified refill', async (t) => {
  const storage = memoryStorage();
  storage.entries.set(release().url, new Response(BAD));
  const fetch = t.mock.method(globalThis, 'fetch', async () => new Response(GOOD));
  assert.deepEqual(Buffer.from(await fetchVerifiedModel(release(), { storage })), GOOD);
  assert.deepEqual(storage.deleted, [release().url]);
  assert.equal(fetch.mock.callCount(), 1);
  assert.deepEqual(Buffer.from(await storage.entries.get(release().url).arrayBuffer()), GOOD);
});

test('corrupt cache followed by corrupt transport leaves no retry-poisoning entry', async (t) => {
  const storage = memoryStorage();
  storage.entries.set(release().url, new Response(BAD));
  t.mock.method(globalThis, 'fetch', async () => new Response(BAD));
  for (let attempt = 0; attempt < 2; attempt += 1) {
    await assert.rejects(fetchVerifiedModel(release(), { storage }), /SHA-256/);
    assert.equal(storage.entries.size, 0);
  }
});

test('short and oversized responses cannot be stored as model bytes', async (t) => {
  for (const bytes of [GOOD.subarray(1), Buffer.concat([GOOD, GOOD])]) {
    const storage = memoryStorage();
    t.mock.method(globalThis, 'fetch', async () => new Response(bytes));
    await assert.rejects(fetchVerifiedModel(release(), { storage }), /expected/);
    assert.equal(storage.entries.size, 0);
  }
  await assert.rejects(fetchVerifiedModel(release(), {
    storage: null, download: async () => 0,
  }), /length/);
});

test('unavailable storage and quota failures do not discard usable model bytes', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response(GOOD));
  for (const storage of [null, { open: async () => { throw new Error('private window'); } }]) {
    assert.deepEqual(Buffer.from(await fetchVerifiedModel(release(), { storage })), GOOD);
  }
  const storage = memoryStorage();
  storage.entries.set('https://model.invalid/previous', new Response(GOOD));
  storage.store.put = async () => { throw new Error('quota'); };
  assert.deepEqual(Buffer.from(await fetchVerifiedModel(release(), { storage })), GOOD);
  assert.equal(storage.entries.size, 1);
  assert.deepEqual(storage.deleted, []);
});

test('a blocked Cache Storage getter still permits verified downloads', async (t) => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'caches');
  Object.defineProperty(globalThis, 'caches', { configurable: true, get() { throw new Error('storage disabled'); } });
  t.after(() => {
    if (descriptor) Object.defineProperty(globalThis, 'caches', descriptor);
    else delete globalThis.caches;
  });
  t.mock.method(globalThis, 'fetch', async () => new Response(GOOD));
  assert.deepEqual(Buffer.from(await fetchVerifiedModel(release())), GOOD);
});

test('cancellation while cache storage is pending never downloads or publishes', async (t) => {
  let finish;
  const pending = new Promise((resolve) => { finish = resolve; });
  const controller = new AbortController();
  const fetch = t.mock.method(globalThis, 'fetch', () => assert.fail('unexpected download'));
  const loading = fetchVerifiedModel(release(), { storage: { open: () => pending }, signal: controller.signal });
  const rejected = assert.rejects(loading, { name: 'AbortError' });
  await tick();
  controller.abort();
  await rejected;
  finish(memoryStorage().store);
  await tick();
  assert.equal(fetch.mock.callCount(), 0);
});

test('runtime pins the deployed manifest identity and delegates every model read', async () => {
  const manifest = JSON.parse(await readFile(new URL('../assets/neural/model.json', import.meta.url)));
  const source = await readFile(new URL('../src/neural-runtime.js', import.meta.url), 'utf8');
  assert.equal(source.match(/MODEL_SHA256 = '([^']+)'/)?.[1], manifest.sha256);
  assert.equal(source.match(/MODEL_OBJECT = '([^']+)'/)?.[1], manifest.object);
  assert.match(source, /return fetchVerifiedModel\(\{ url: MODEL_URL, bytes: DOWNLOAD_BYTES.model, sha256: MODEL_SHA256 \}/);
  assert.doesNotMatch(source, /async function (storedModel|rememberModel)/);
});

test('resolver ignores corrupt local bytes and evicts corrupt digest-keyed disk cache', async (t) => {
  const f = await fixture(t);
  await writeFile(f.modelPath, BAD);
  await mkdir(f.cacheDirectory);
  const cached = join(f.cacheDirectory, `${f.manifest.sha256}.onnx`);
  await writeFile(cached, BAD);
  assert.equal(await readModelBytes(f), null);
  await assert.rejects(readFile(cached), { code: 'ENOENT' });
  assert.deepEqual(await readFile(f.modelPath), BAD, 'do not delete an explicitly owned working file');
});

test('legacy generation cache remains usable offline only when its digest matches', async (t) => {
  const f = await fixture(t);
  await mkdir(f.cacheDirectory);
  const legacy = join(f.cacheDirectory, 'fixture.onnx');
  await writeFile(legacy, GOOD);
  assert.deepEqual(await readModelBytes(f), GOOD);
  await writeFile(legacy, BAD);
  assert.equal(await readModelBytes(f), null);
  await assert.rejects(readFile(legacy), { code: 'ENOENT' });
});

test('resolver verifies transport before atomic cache publication and retries failures', async (t) => {
  const f = await fixture(t);
  let delivered = BAD;
  const fetch = t.mock.method(globalThis, 'fetch', async () => new Response(delivered));
  await assert.rejects(readModelBytes({ ...f, allowDownload: true }), /SHA-256/);
  await assert.rejects(readdir(f.cacheDirectory), { code: 'ENOENT' });
  delivered = GOOD;
  assert.deepEqual(await readModelBytes({ ...f, allowDownload: true }), GOOD);
  assert.deepEqual(await readModelBytes(f), GOOD);
  assert.deepEqual(await readdir(f.cacheDirectory), [`${f.manifest.sha256}.onnx`]);
  assert.equal(fetch.mock.callCount(), 2);
});

test('parallel resolver downloads publish complete bytes without shared staging files', async (t) => {
  const f = await fixture(t);
  t.mock.method(globalThis, 'fetch', async () => { await tick(); return new Response(GOOD); });
  const copies = await Promise.all(Array.from({ length: 4 }, () => readModelBytes({ ...f, allowDownload: true })));
  for (const bytes of copies) assert.deepEqual(bytes, GOOD);
  assert.deepEqual(await readdir(f.cacheDirectory), [`${f.manifest.sha256}.onnx`]);
  assert.deepEqual(await readModelBytes(f), GOOD);
});

test('resolver storage failure is optional and interrupted temporary files are ignored', async (t) => {
  const f = await fixture(t);
  t.mock.method(globalThis, 'fetch', async () => new Response(GOOD));
  await writeFile(f.cacheDirectory, 'not a directory');
  assert.deepEqual(await readModelBytes({ ...f, allowDownload: true }), GOOD);
  await rm(f.cacheDirectory);
  await mkdir(f.cacheDirectory);
  await writeFile(join(f.cacheDirectory, `${f.manifest.sha256}.partial.tmp`), BAD);
  assert.equal(await readModelBytes(f), null);
});

test('explicit model overrides are verified rather than silently substituted', async (t) => {
  const f = await fixture(t);
  const previous = process.env.NEURAL_MODEL;
  t.after(() => { if (previous === undefined) delete process.env.NEURAL_MODEL; else process.env.NEURAL_MODEL = previous; });
  process.env.NEURAL_MODEL = f.modelPath;
  await writeFile(f.modelPath, BAD);
  await assert.rejects(readModelBytes(f), /SHA-256/);
  await writeFile(f.modelPath, GOOD);
  assert.deepEqual(await readModelBytes(f), GOOD);
});

test('publisher rejects size, digest and identity errors before any cloud operation', async (t) => {
  const f = await fixture(t);
  const calls = [];
  for (const bytes of [Buffer.from('short'), BAD]) {
    await writeFile(f.modelPath, bytes);
    await assert.rejects(publishModel({ ...f, upload: (...args) => calls.push(args) }), ModelIntegrityError);
  }
  await writeFile(f.manifestPath, JSON.stringify({ ...f.manifest, sha256: 'not-a-hash' }));
  await assert.rejects(publishModel({ ...f, upload: (...args) => calls.push(args) }), ModelIntegrityError);
  assert.deepEqual(calls, []);
});

test('publisher gives different exports of the same checkpoint different immutable keys', async (t) => {
  const f = await fixture(t);
  const objects = new Map();
  let staged;
  const upload = async (_bucket, key, path) => {
    staged = path;
    const bytes = gunzipSync(await readFile(path));
    assert.equal(key.split('/').at(-1), hash(bytes));
    assert.match(key, /^models\/fixture\/[a-f0-9]{64}$/);
    if (objects.has(key)) assert.deepEqual(objects.get(key), bytes);
    objects.set(key, bytes);
  };
  const keys = [];
  for (const bytes of [GOOD, BAD, GOOD]) {
    await writeFile(f.modelPath, bytes);
    await writeFile(f.manifestPath, JSON.stringify({ ...f.manifest, ...identity(bytes) }));
    const result = await publishModel({ ...f, upload });
    keys.push(result.object);
    assert.equal(result.sha256, hash(bytes));
    assert.notEqual(result.object, f.manifest.object, 'never replace the legacy URL');
    await assert.rejects(readFile(staged), { code: 'ENOENT' });
  }
  assert.equal(objects.size, 2);
  assert.notEqual(keys[0], keys[1]);
  assert.equal(keys[0], keys[2]);
});

test('publisher cleans staging on upload failure and CLI fails before invoking npx', async (t) => {
  const f = await fixture(t);
  await writeFile(f.modelPath, GOOD);
  let staged;
  await assert.rejects(publishModel({ ...f, upload: async (_bucket, _key, path) => {
    staged = path; throw new Error('upload unavailable');
  } }), /upload unavailable/);
  await assert.rejects(readFile(staged), { code: 'ENOENT' });
  await writeFile(f.modelPath, BAD);
  const result = spawnSync(process.execPath, [fileURLToPath(new URL('../scripts/publish-model-r2.mjs', import.meta.url)),
    f.modelPath, '--manifest', f.manifestPath], { encoding: 'utf8', env: { ...process.env, PATH: '' } });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /ModelIntegrityError.*SHA-256/);
  assert.doesNotMatch(result.stderr, /spawn.*npx/);
});

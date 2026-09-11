import assert from 'node:assert/strict';
import test from 'node:test';
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  decodePerfectBook,
  loadPerfectBook,
  PERFECT_BOOK_CERTIFICATE as certificate,
} from '../src/perfect-book.js';

function structuralFixture() {
  const bytes = Buffer.alloc(certificate.byteLength);
  bytes.write('C4PB');
  bytes[4] = certificate.version;
  bytes[5] = certificate.maxPly;
  bytes[6] = 10;
  bytes.writeUInt32LE(certificate.entryCount, 8);
  for (let at = 0; at < certificate.entryCount; at += 1) {
    bytes.writeBigUInt64LE(BigInt(at), 12 + at * 10);
    bytes[20 + at * 10] = 8;
    bytes[21 + at * 10] = 1;
  }
  return bytes;
}

async function isolatedDefault(t, bytes = structuralFixture()) {
  const root = await mkdtemp(join(tmpdir(), 'perfect-book-integrity-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, 'src'));
  await mkdir(join(root, 'assets'));
  await writeFile(join(root, 'package.json'), '{"type":"module"}');
  for (const name of ['perfect-book.js', 'exact-table.js', 'data-loader.js']) {
    await copyFile(new URL(`../src/${name}`, import.meta.url), join(root, 'src', name));
  }
  const asset = join(root, 'assets', 'perfect-book.bin');
  await writeFile(asset, bytes);
  const module = await import(pathToFileURL(join(root, 'src', 'perfect-book.js')).href);
  return { asset, load: module.loadPerfectBook };
}

test('the opening-book certificate matches the committed manifest', async () => {
  const manifest = JSON.parse(await readFile(new URL('../data/perfect-book.manifest.json', import.meta.url)));
  assert.ok(Object.isFrozen(certificate));
  for (const field of ['maxPly', 'entryCount', 'byteLength', 'sha256']) {
    assert.equal(certificate[field], manifest[field], field);
  }
  assert.equal(certificate.version, manifest.format);
});

test('default book loading rejects same-size replacements with valid moves and outcomes', async (t) => {
  const bytes = structuralFixture();
  const { load, asset } = await isolatedDefault(t, bytes);
  for (const [moveMask, outcome] of [[1, 1], [64, -1]]) {
    bytes[20] = moveMask;
    bytes[21] = outcome;
    // These have the real length/count and pass every structural check.
    const table = decodePerfectBook(bytes);
    assert.equal(table.entryCount, certificate.entryCount);
    assert.equal(table.byteLength, certificate.byteLength);
    assert.deepEqual(table.lookup(0n), { key: 0n, moveMask, outcome });
    await writeFile(asset, bytes);
    await assert.rejects(load(), /SHA-256.*certificate/);
  }
});

test('alternate book URLs cannot bypass verification or cache a rejection', async (t) => {
  const bytes = structuralFixture();
  const fetch = t.mock.method(globalThis, 'fetch', async () => new Response(bytes));
  const url = new URL('https://invalid.example/unverified-book.bin');
  await assert.rejects(loadPerfectBook(url), /SHA-256.*certificate/);
  await assert.rejects(loadPerfectBook(url), /SHA-256.*certificate/);
  assert.equal(fetch.mock.callCount(), 2);
});

test('truncated book data fails before hashing', async (t) => {
  const { load } = await isolatedDefault(t, Buffer.alloc(12));
  const digest = t.mock.method(globalThis.crypto.subtle, 'digest', () => {
    throw new Error('wrong-sized data must not reach hashing');
  });
  await assert.rejects(load(), /length.*certificate/);
  assert.equal(digest.mock.callCount(), 0);
});

test('book metadata is checked before caching and forced refresh cannot retain stale data', async (t) => {
  // Only this test replaces the hash result to reach the post-hash checks.
  // The actual corruption regressions use the real SHA-256 implementation.
  t.mock.method(globalThis.crypto.subtle, 'digest', async () =>
    Uint8Array.from(Buffer.from(certificate.sha256, 'hex')).buffer);
  const bytes = structuralFixture();
  bytes[5] = 7;
  const { load, asset } = await isolatedDefault(t, bytes);
  await assert.rejects(load(), /maxPly.*certificate/);
  bytes[5] = certificate.maxPly;
  await writeFile(asset, bytes);
  const table = await load();
  assert.strictEqual(await load(), table);
  await writeFile(asset, Buffer.alloc(12));
  await assert.rejects(load(undefined, { force: true }), /length.*certificate/);
  await assert.rejects(load(), /length.*certificate/);
});

test('aborted book requests perform no transport or verification', async (t) => {
  const fetch = t.mock.method(globalThis, 'fetch', () => { throw new Error('must not fetch'); });
  const digest = t.mock.method(globalThis.crypto.subtle, 'digest', () => { throw new Error('must not hash'); });
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(loadPerfectBook('https://invalid.example/aborted-book.bin', {
    signal: controller.signal,
  }), { name: 'AbortError' });
  assert.equal(fetch.mock.callCount(), 0);
  assert.equal(digest.mock.callCount(), 0);
});

test('Node crypto fallback also rejects an unverified opening book', async (t) => {
  const { load } = await isolatedDefault(t);
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: undefined });
  try {
    await assert.rejects(load(), /SHA-256.*certificate/);
  } finally {
    Object.defineProperty(globalThis, 'crypto', descriptor);
  }
});

test('book transport deadlines still cover stalled headers and bodies', async (t) => {
  const fetch = t.mock.method(globalThis, 'fetch');
  for (const stage of ['headers', 'body']) {
    fetch.mock.mockImplementation(async () => stage === 'headers'
      ? new Promise(() => {})
      : { ok: true, arrayBuffer: () => new Promise(() => {}) });
    const url = `https://invalid.example/stalled-book-${stage}.bin`;
    await assert.rejects(loadPerfectBook(url, { timeoutMs: 10 }), /did not finish loading/);
    fetch.mock.mockImplementation(async () => new Response(Buffer.alloc(12)));
    await assert.rejects(loadPerfectBook(url), /length.*certificate/);
  }
});

test('committed opening-book asset loads, caches, and recovers after same-size corruption', async (t) => {
  const bytes = await readFile(new URL('../assets/perfect-book.bin', import.meta.url));
  const table = await loadPerfectBook();
  for (const field of ['version', 'maxPly', 'entryCount', 'byteLength']) {
    assert.equal(table[field], certificate[field]);
  }
  assert.strictEqual(await loadPerfectBook(), table);
  const corrupt = Buffer.from(bytes);
  corrupt[20] = corrupt[20] === 8 ? 16 : 8;
  corrupt[21] = corrupt[21] === 0 ? 1 : 0;
  decodePerfectBook(corrupt);
  const { load, asset } = await isolatedDefault(t, corrupt);
  await assert.rejects(load(), /SHA-256.*certificate/);
  await writeFile(asset, bytes);
  const recovered = await load();
  assert.equal(recovered.entryCount, certificate.entryCount);
  assert.deepEqual(recovered.lookup(0n), table.lookup(0n));
  assert.strictEqual(await load(), recovered);
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(load(undefined, { signal: controller.signal }), { name: 'AbortError' });
});

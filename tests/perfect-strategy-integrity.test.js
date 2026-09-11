import assert from 'node:assert/strict';
import test from 'node:test';
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  decodePerfectStrategy,
  loadPerfectStrategy,
  PERFECT_STRATEGY_CERTIFICATE as certificate,
} from '../src/perfect-strategy.js';

function structuralFixture() {
  const bytes = Buffer.alloc(certificate.byteLength);
  bytes.write('C4PS');
  bytes[4] = certificate.version;
  bytes[5] = certificate.handoffRemaining;
  bytes[6] = 10;
  bytes[7] = certificate.roleFlags;
  bytes.writeUInt32LE(certificate.entryCount, 8);
  for (let at = 0; at < certificate.entryCount; at += 1) {
    bytes.writeBigUInt64LE(BigInt(at), 12 + at * 10);
    bytes[20 + at * 10] = 8;
    bytes[21 + at * 10] = 1;
  }
  return bytes;
}

async function isolatedDefault(t, bytes = structuralFixture()) {
  const root = await mkdtemp(join(tmpdir(), 'perfect-strategy-integrity-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, 'src'));
  await mkdir(join(root, 'assets'));
  await writeFile(join(root, 'package.json'), '{"type":"module"}');
  for (const name of ['perfect-strategy.js', 'exact-table.js', 'data-loader.js']) {
    await copyFile(new URL(`../src/${name}`, import.meta.url), join(root, 'src', name));
  }
  const asset = join(root, 'assets', 'perfect-strategy.bin');
  await writeFile(asset, bytes);
  const module = await import(pathToFileURL(join(root, 'src', 'perfect-strategy.js')).href);
  return { asset, load: module.loadPerfectStrategy };
}

test('the runtime certificate is pinned to the independently replayed manifest', async () => {
  const manifest = JSON.parse(await readFile(new URL('../data/perfect-strategy.manifest.json', import.meta.url)));
  assert.ok(Object.isFrozen(certificate));
  for (const field of ['handoffRemaining', 'roleFlags', 'entryCount', 'byteLength', 'sha256']) {
    assert.equal(certificate[field], manifest[field], field);
  }
  assert.equal(certificate.version, manifest.format);
});

test('default strategy loading rejects a structurally valid same-size replacement', async (t) => {
  const bytes = structuralFixture();
  const table = decodePerfectStrategy(bytes);
  assert.equal(table.entryCount, certificate.entryCount);
  assert.equal(table.byteLength, certificate.byteLength);
  const { load, asset } = await isolatedDefault(t, bytes);
  await assert.rejects(load(), /SHA-256.*certificate/);
  // A different legal move/outcome must not turn the rejection into a cache hit.
  bytes[20] = 16;
  bytes[21] = 0;
  assert.deepEqual(decodePerfectStrategy(bytes).lookup(0n), { key: 0n, moveMask: 16, outcome: 0 });
  await writeFile(asset, bytes);
  await assert.rejects(load(), /SHA-256.*certificate/);
});

test('alternate strategy URLs cannot bypass release integrity', async () => {
  const url = new URL('https://invalid.example/perfect-strategy.bin');
  const bytes = structuralFixture();
  const original = globalThis.fetch;
  let requests = 0;
  globalThis.fetch = async () => { requests += 1; return new Response(bytes); };
  try {
    await assert.rejects(loadPerfectStrategy(url), /SHA-256.*certificate/);
    await assert.rejects(loadPerfectStrategy(url), /SHA-256.*certificate/);
    assert.equal(requests, 2, 'failed verification must not be cached');
  } finally {
    globalThis.fetch = original;
  }
});

test('truncated strategy data is rejected before hashing', async (t) => {
  const { load } = await isolatedDefault(t, Buffer.alloc(12));
  const digest = t.mock.method(globalThis.crypto.subtle, 'digest', () => {
    throw new Error('hash should not run for a wrong-sized download');
  });
  await assert.rejects(load(), /length.*certificate/);
  assert.equal(digest.mock.callCount(), 0);
});

test('verification still validates metadata before caching', async (t) => {
  // Isolate the post-hash metadata checks; all corruption tests above use the
  // real SHA-256 implementation. This mock is never a runtime escape hatch.
  t.mock.method(globalThis.crypto.subtle, 'digest', async () => Uint8Array.from(Buffer.from(certificate.sha256, 'hex')).buffer);
  const bytes = structuralFixture();
  bytes[5] = 23;
  const { load, asset } = await isolatedDefault(t, bytes);
  await assert.rejects(load(), /handoffRemaining.*certificate/);
  bytes[5] = certificate.handoffRemaining;
  bytes[7] = 1;
  await writeFile(asset, bytes);
  await assert.rejects(load(), /roleFlags.*certificate/);
  bytes[7] = certificate.roleFlags;
  await writeFile(asset, bytes);
  const table = await load();
  assert.strictEqual(await load(), table, 'only the verified result is cached');
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(load(undefined, { signal: controller.signal }), { name: 'AbortError' });
  await writeFile(asset, Buffer.alloc(12));
  await assert.rejects(load(undefined, { force: true }), /length.*certificate/);
  await assert.rejects(load(), /length.*certificate/);
});

test('aborted strategy loads stop before verification', async (t) => {
  const { load } = await isolatedDefault(t);
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(load(undefined, { signal: controller.signal }), { name: 'AbortError' });
});

test('Node fallback also rejects unverified strategy bytes', async (t) => {
  const { load } = await isolatedDefault(t);
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: undefined });
  try {
    await assert.rejects(load(), /SHA-256.*certificate/);
  } finally {
    Object.defineProperty(globalThis, 'crypto', descriptor);
  }
});

test('committed asset loads, caches, and recovers after same-size corruption', async (t) => {
  const bytes = await readFile(new URL('../assets/perfect-strategy.bin', import.meta.url));
  const table = await loadPerfectStrategy();
  for (const field of ['version', 'handoffRemaining', 'roleFlags', 'entryCount', 'byteLength']) {
    assert.equal(table[field], certificate[field]);
  }
  assert.strictEqual(await loadPerfectStrategy(), table);
  assert.ok(table.coversRole(1) && table.coversRole(2));
  const { load, asset } = await isolatedDefault(t, bytes);
  const corrupt = Buffer.from(bytes);
  corrupt[20] = corrupt[20] === 8 ? 16 : 8; // Keep a legal, single-column mask.
  corrupt[21] = corrupt[21] === 0 ? 1 : 0;  // Keep the stored outcome in range.
  decodePerfectStrategy(corrupt);           // Structural validation alone accepts it.
  await writeFile(asset, corrupt);
  await assert.rejects(load(), /SHA-256.*certificate/);
  await writeFile(asset, bytes);
  const recovered = await load();
  assert.equal(recovered.entryCount, certificate.entryCount);
  assert.strictEqual(await load(), recovered);
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(load(undefined, { signal: controller.signal }), { name: 'AbortError' });
});

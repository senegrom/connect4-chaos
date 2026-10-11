import assert from 'node:assert/strict';
import { cp, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { PERFECT_BOOK_CERTIFICATE } from '../src/perfect-book.js';
import { loadVerifiedPerfectChaosCompletePolicy } from '../src/perfect-chaos-complete.js';
import { PERFECT_CHAOS_RELEASED_POLICIES } from '../src/perfect-chaos-prefix.js';
import { loadVerifiedPerfectClassicPolicy } from '../src/perfect-classic-verified.js';
import { PERFECT_STRATEGY_CERTIFICATE } from '../src/perfect-strategy.js';

// A table regenerated in place keeps its path, and Pages lets a browser reuse
// a file for ten minutes: a page of a new build was handed the old bytes, and
// every Retry failed their digest until the copy expired.
const SITE = 'https://site.test/';
const release = (sha256) => sha256.slice(0, 12);

// Answers each request with the committed file at its path, and records the
// path, the release its query names and the request's cache mode.
function serveSite(t) {
  const requests = [];
  t.mock.method(globalThis, 'fetch', async (url, init = {}) => {
    const target = new URL(url);
    requests.push({ path: target.pathname, release: target.searchParams.get('v'), cache: init.cache });
    return new Response(await readFile(new URL(`..${target.pathname}`, import.meta.url)));
  });
  return requests;
}

// Under Node a default table URL is a file: URL, read from disk with its query
// unseen. In this copy of src/ the named modules say the site served them, so
// their default URLs reach fetch as the page's do.
async function servedModules(t, names) {
  const root = await mkdtemp(join(tmpdir(), 'table-releases-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await cp(fileURLToPath(new URL('../src/', import.meta.url)), join(root, 'src'), { recursive: true });
  await writeFile(join(root, 'package.json'), '{"type":"module"}');
  const modules = {};
  for (const name of names) {
    const file = join(root, 'src', name);
    const source = await readFile(file, 'utf8');
    await writeFile(file, source.replaceAll('import.meta.url', JSON.stringify(`${SITE}src/${name}`)));
    modules[name] = await import(pathToFileURL(file).href);
  }
  return modules;
}

test('the book, the strategy and each prefix layer are fetched by their release', async (t) => {
  const site = await servedModules(t, ['perfect-book.js', 'perfect-strategy.js', 'perfect-chaos-prefix.js']);
  const requests = serveSite(t);
  await site['perfect-book.js'].loadPerfectBook();
  await site['perfect-strategy.js'].loadPerfectStrategy();
  const prefix = site['perfect-chaos-prefix.js'];
  await prefix.loadPerfectChaosPolicy(prefix.PERFECT_CHAOS_ROLE_FIRST, 0);
  await prefix.loadPerfectChaosPolicy(prefix.PERFECT_CHAOS_ROLE_SECOND, 8);
  const layer = (role, file) => release(PERFECT_CHAOS_RELEASED_POLICIES[role][file].sha256);
  assert.deepEqual(requests, [
    { path: '/assets/perfect-book.bin', release: release(PERFECT_BOOK_CERTIFICATE.sha256), cache: undefined },
    { path: '/assets/perfect-strategy.bin', release: release(PERFECT_STRATEGY_CERTIFICATE.sha256), cache: undefined },
    { path: '/data/perfect-chaos-prefix/red/0-8.policy.bin', release: layer('red', '0-8.policy.bin'), cache: undefined },
    { path: '/data/perfect-chaos-prefix/yellow/8-10.policy.bin', release: layer('yellow', '8-10.policy.bin'),
      cache: undefined },
  ]);
});

// A catalog keeps its URL across releases, so it is revalidated instead.
test('a listed table is fetched by the release its catalog lists, and each catalog is revalidated', async (t) => {
  const requests = serveSite(t);
  const expected = [];
  for (const [catalog, [rows, columns, connect, role], load] of [
    ['perfect-classic', [4, 4, 4, 1], loadVerifiedPerfectClassicPolicy],
    ['perfect-chaos-complete', [4, 4, 3, 1], loadVerifiedPerfectChaosCompletePolicy],
  ]) {
    const manifestUrl = `${SITE}data/${catalog}/manifest.json`;
    const manifest = JSON.parse(await readFile(new URL(`../data/${catalog}/manifest.json`, import.meta.url)));
    const entry = manifest.policies.find((policy) => policy.rows === rows && policy.columns === columns
      && policy.connect === connect && policy.role === role);
    const policy = await load(rows, columns, connect, role, { manifestUrl });
    assert.equal(policy.entryCount, entry.entryCount, catalog);
    expected.push(
      { path: `/data/${catalog}/manifest.json`, release: null, cache: 'no-cache' },
      { path: new URL(entry.file, manifestUrl).pathname, release: release(entry.sha256), cache: undefined },
    );
  }
  assert.deepEqual(requests, expected);
});

import test from 'node:test';
import assert from 'node:assert/strict';

import { createBuildCheck, OutdatedBuildError, RELOAD_MESSAGE } from '../src/site-build.js';

function clock() {
  let time = 1_000;
  return { now: () => time, advance(ms) { time += ms; } };
}

test('an unstamped page loads as it always has, without asking the site', async () => {
  const check = createBuildCheck({ build: 'dev', currentBuild: () => assert.fail('nothing to compare') });
  assert.equal(check.stamped, false);
  assert.equal(await check.isOutdated(), false);
  await check.ensureCurrent();
});

test('a newer deploy is found once and then stays found', async () => {
  const time = clock();
  let deployed = { build: 'abc123' };
  let asked = 0;
  const outdated = [];
  const check = createBuildCheck({ build: 'abc123', now: time.now, onOutdated: (found) => outdated.push(found),
    currentBuild: async () => { asked += 1; return deployed; } });
  assert.equal(check.stamped, true);
  assert.equal(await check.isOutdated(), false);
  // A minute passes between questions; the page does not ask on every move.
  deployed = { build: 'def456', refresh: ['src/app.js'] };
  assert.equal(await check.isOutdated(), false);
  assert.equal(asked, 1);
  time.advance(60_000);
  assert.equal(await check.isOutdated(), true);
  // The page renews its cached code once, as soon as it knows.
  assert.deepEqual(outdated, [{ build: 'def456', refresh: ['src/app.js'] }]);
  await assert.rejects(check.ensureCurrent(), (error) => error instanceof OutdatedBuildError
    && error.name === 'OutdatedBuildError' && error.message === RELOAD_MESSAGE);
  deployed = { build: 'abc123' };
  time.advance(60_000);
  assert.equal(await check.isOutdated(), true, 'a page that fell behind reloads, whatever the site says next');
  assert.equal(asked, 2);
});

test('concurrent loads share one question', async () => {
  let asked = 0;
  let answer;
  const check = createBuildCheck({ build: 'abc123', onOutdated() {},
    currentBuild: () => { asked += 1; return new Promise((resolve) => { answer = resolve; }); } });
  const first = check.isOutdated();
  const second = check.ensureCurrent();
  await Promise.resolve();
  answer({ build: 'def456' });
  assert.equal(await first, true);
  await assert.rejects(second, OutdatedBuildError);
  assert.equal(asked, 1);
});

test('a site that cannot be asked keeps the loaded build running, and is asked again later', async () => {
  const time = clock();
  const answers = [() => { throw new TypeError('Failed to fetch'); }, () => null, () => ({ build: 'def456' })];
  const check = createBuildCheck({ build: 'abc123', now: time.now, onOutdated() {},
    currentBuild: async () => answers.shift()() });
  assert.equal(await check.isOutdated(), false, 'offline');
  time.advance(60_000);
  assert.equal(await check.isOutdated(), false, 'no build.json');
  time.advance(60_000);
  assert.equal(await check.isOutdated(), true);
});

// A normal reload revalidates the page but can run modules its cache still
// holds; the deployed code is fetched fresh first, so the reload runs it.
test('a newer deploy renews the cached copies of the code build.json lists', async (t) => {
  const fetched = [];
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    fetched.push([String(url), init?.cache]);
    return new Response(String(url).endsWith('build.json')
      ? JSON.stringify({ build: 'def456', refresh: ['src/app.js', 'styles.css', '../outside.js', 7] })
      : '');
  });
  const check = createBuildCheck({ build: 'abc123' });
  assert.equal(await check.isOutdated(), true);
  const renewed = fetched.filter(([, cache]) => cache === 'reload').map(([url]) => url);
  // Only the site's own paths: nothing outside it, nothing that is not a path.
  assert.deepEqual(renewed, [new URL('../src/app.js', import.meta.url).href, new URL('../styles.css', import.meta.url).href]);
});

import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { cp, mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

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

// A reload revalidates the page, but a new tab, or a relaunch of the
// installed app, which has no reload control, can open the cached old page.
// The page renews what build.json lists, and Reload waits for that.
test('a newer deploy renews the page build.json lists, and Reload waits for it', async (t) => {
  const fetched = [];
  let arrive;
  const arriving = new Promise((resolve) => { arrive = resolve; });
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    fetched.push([String(url), init?.cache]);
    if (String(url).endsWith('build.json')) {
      return new Response(JSON.stringify({ build: 'def456', refresh: ['', 'index.html', '../outside.js', 7] }));
    }
    if (String(url).endsWith('/')) throw new TypeError('Failed to fetch');
    await arriving;
    return new Response('');
  });
  const reloads = [];
  const check = createBuildCheck({ build: 'abc123', reloadPage: () => reloads.push('reload') });
  assert.equal(await check.isOutdated(), true);
  const renewed = fetched.filter(([, cache]) => cache === 'reload').map(([url]) => url);
  // Only the site's own paths: nothing outside it, nothing that is not a path.
  assert.deepEqual(renewed, [new URL('../', import.meta.url).href, new URL('../index.html', import.meta.url).href]);
  const reloading = check.reload();
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.deepEqual(reloads, [], 'not while the page is still arriving, although the other renewal failed');
  arrive();
  await reloading;
  assert.deepEqual(reloads, ['reload']);
});

// Only reloading helps a page a deploy left behind: Retry and Use Brutal load
// code from the newer site and arrive back at the same message.
test('a page that fell behind the site offers Reload in place of Retry', async () => {
  const source = await readFile(new URL('../src/app.js', import.meta.url), 'utf8');
  const start = source.indexOf('function renderAiRecovery(');
  // As index.html has them.
  const control = (hidden = false) => ({ hidden, disabled: false });
  const elements = { aiRecovery: { hidden: true, contains: () => false }, retryAiButton: control(),
    reloadPageButton: control(true), switchBrutalButton: control(), undoAiButton: control() };
  const state = { config: { opponent: 'perfect' }, status: 'playing', currentPlayer: 2, aiThinking: false, aiError: null };
  const context = vm.createContext({ elements, state, YELLOW: 2, RELOAD_MESSAGE, document: {},
    isAiGame: () => true, findUndoIndex: () => 0 });
  vm.runInContext(source.slice(start, source.indexOf('\n}\n', start) + 2), context);
  const shown = () => Object.entries(elements).filter(([name, element]) => name !== 'aiRecovery' && !element.hidden)
    .map(([name]) => name);
  state.aiError = 'The selected AI worker failed. Retry the move.';
  context.renderAiRecovery();
  assert.deepEqual(shown(), ['retryAiButton', 'switchBrutalButton', 'undoAiButton']);
  state.aiError = RELOAD_MESSAGE;
  context.renderAiRecovery();
  assert.deepEqual(shown(), ['reloadPageButton', 'undoAiButton']);
});

// A worker outlives the check that let it start. The opening book, the 6x7
// strategy and the Brutal Chaos policy modules loaded on first use, from
// whatever the site served by then: a Perfect round resumed past the
// strategy's handoff imported it a round later, after a deploy.
test('a worker loads every module it runs as it starts', async () => {
  for (const entry of ['ai-worker.js', 'neural-worker.js']) {
    const graph = new Set();
    const visit = async (file) => {
      if (graph.has(file)) return;
      graph.add(file);
      const text = await readFile(new URL(`../src/${file}`, import.meta.url), 'utf8');
      assert.doesNotMatch(text, /\bimport\(\s*['"`]\./, `${file}, which ${entry} runs, imports a module on first use`);
      for (const [, imported] of text.matchAll(/\bfrom\s*'\.\/([\w.-]+\.js)'/g)) await visit(imported);
    };
    await visit(entry);
    assert.ok(graph.size > 5, `${entry} imports ${[...graph]}`);
  }
});

// Runs the step of scripts/build-site.sh that stamps the build, on a copy of
// the site's page and code. Without the build in its URLs, a page loaded
// after a deploy ran any older module its HTTP cache still held for up to
// ten minutes - the stamped site-build.js new, a lazily loaded module old.
async function stampSite(t, extraModules = {}) {
  const script = await readFile(new URL('../scripts/build-site.sh', import.meta.url), 'utf8');
  const start = script.lastIndexOf("node -e '") + "node -e '".length;
  const step = script.slice(start, script.indexOf("\n'", start));
  const site = await mkdtemp(join(tmpdir(), 'connect4-site-'));
  t.after(() => rm(site, { recursive: true, force: true }));
  for (const entry of ['index.html', 'styles.css', 'src']) {
    await cp(fileURLToPath(new URL(`../${entry}`, import.meta.url)), join(site, entry), { recursive: true });
  }
  for (const [file, text] of Object.entries(extraModules)) await writeFile(join(site, 'src', file), text);
  const run = spawnSync(process.execPath, ['-e', step], { env: { ...process.env, SITE: site, COMMIT: 'feedc0de' }, encoding: 'utf8' });
  return { site, run };
}

test('the deployed site names its build in every module URL', async (t) => {
  const { site, run } = await stampSite(t);
  assert.equal(run.status, 0, run.stderr);
  const deployed = JSON.parse(await readFile(join(site, 'build.json'), 'utf8'));
  assert.match(deployed.build, /^[0-9a-f]{12}$/);
  const version = `?v=${deployed.build}`;
  assert.deepEqual(deployed.refresh, ['', 'index.html'], 'the page itself, which a relaunch can take from the cache');
  const page = await readFile(join(site, 'index.html'), 'utf8');
  assert.ok(page.includes(`src="./src/app.js${version}"`) && page.includes(`href="./styles.css${version}"`));
  const modules = (await readdir(join(site, 'src'))).filter((file) => file.endsWith('.js'));
  let references = 0;
  for (const file of modules) {
    const text = await readFile(join(site, 'src', file), 'utf8');
    for (const [literal, name] of text.matchAll(/['"`]\.\/([\w.-]+\.js)[^'"`]*['"`]/g)) {
      assert.ok(modules.includes(name), `${file} names ${literal}`);
      assert.ok(literal.endsWith(`${version}${literal[0]}`), `${file} loads ${literal}`);
      references += 1;
    }
  }
  assert.ok(references > 50, `${references} module references`);
  const app = await readFile(join(site, 'src/app.js'), 'utf8');
  assert.ok(app.includes(`new URL('./ai-worker.js${version}', import.meta.url)`), 'the AI worker');
  assert.ok(app.includes(`import('./neural-app.js${version}')`), 'a lazily loaded module');
  assert.ok((await readFile(join(site, 'src/neural-client.js'), 'utf8'))
    .includes(`new URL('./neural-worker.js${version}', import.meta.url)`), 'the neural worker');
  // Its registration is compared by URL, and a new URL every deploy would
  // register it again; it is not a module of the page.
  assert.ok((await readFile(join(site, 'src/cross-origin-isolation.js'), 'utf8'))
    .includes("new URL('../cross-origin-isolation-worker.js', import.meta.url)"), 'the isolation worker');
  assert.ok((await readFile(join(site, 'src/site-build.js'), 'utf8')).includes(`const BUILD = '${deployed.build}';`));
});

test('the build refuses a module it cannot give the build\'s URL', async (t) => {
  const { run } = await stampSite(t, { 'extra.js': "export const load = () => import('../src/engine.js');\n" });
  assert.notEqual(run.status, 0);
  assert.match(run.stderr, /src\/extra\.js loads a module by a URL without the build/);
});

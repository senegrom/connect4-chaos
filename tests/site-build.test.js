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
  // Reload goes through the check, which waits for the renewals first.
  const reloads = [];
  context.siteBuild = { reload: async () => { reloads.push('reload'); } };
  const reload = source.indexOf('function reloadPage(');
  vm.runInContext(source.slice(reload, source.indexOf('\n}\n', reload) + 2), context);
  context.reloadPage();
  assert.deepEqual(reloads, ['reload']);
  assert.equal(elements.reloadPageButton.disabled, true, 'pressed once');
});

// The neural opponent asks before its first download, and its worker starts
// once the player agrees. A deploy can land while the question is open; the
// worker then loaded the newer site's modules into this page.
test('the neural worker starts only after a check that follows its download question', async () => {
  const app = await readFile(new URL('../src/app.js', import.meta.url), 'utf8');
  const neural = await readFile(new URL('../src/neural-app.js', import.meta.url), 'utf8');
  const start = app.indexOf('async function runNeuralMove(');
  const move = app.slice(start, app.indexOf('\n}\n', start) + 2);
  assert.ok(move.includes("import('./neural-app.js')"));
  for (const deployed of ['abc123', 'def456']) {
    const time = clock();
    let build = 'abc123';
    const events = [];
    const neuralContext = vm.createContext({ DOWNLOAD_BYTES: { model: 1, runtime: 1 },
      immediateWinningActions: () => [], neuralLoadState: () => 'idle', waitFor: (promise) => promise,
      async requestDownload() {
        events.push('question');
        // Answered minutes later, after whatever the site did meanwhile.
        time.advance(300_000);
        build = deployed;
        return true;
      },
      showDownloadProgress: () => { events.push('progress'); return { close() {} }; },
      loadNeuralNetwork: async () => { events.push('worker'); throw new Error('no network in this test'); } });
    vm.runInContext(neural.slice(neural.indexOf('export async function')).replace('export ', ''), neuralContext);
    const request = { id: 1, roundVersion: 0, controller: new AbortController(),
      position: { board: [[0]], currentPlayer: 2, connect: 4, chaosMode: false } };
    const state = { aiRequest: request, aiRequestId: 1, version: 0, aiError: null };
    const context = vm.createContext({ state,
      siteBuild: createBuildCheck({ build: 'abc123', now: time.now, onOutdated() {}, currentBuild: async () => ({ build }) }),
      importNeuralApp: async () => ({ runNeuralRequest: neuralContext.runNeuralRequest }),
      stopAiWithError(message) { state.aiError = message; },
      finishAiRequest() {}, renderAiState() {}, renderStatus() {}, renderSearchInfo() {} });
    vm.runInContext(move.replace("import('./neural-app.js')", 'importNeuralApp()'), context);
    await context.runNeuralMove(request);
    if (deployed === 'abc123') {
      assert.deepEqual(events, ['question', 'progress', 'worker']);
      assert.match(state.aiError, /^The neural opponent failed: no network in this test/);
    } else {
      assert.deepEqual(events, ['question'], 'no worker from the newer site');
      assert.equal(state.aiError, RELOAD_MESSAGE, 'the page asks for a reload, which Reload answers');
    }
  }
});

// Dropped with app.js's one-deploy shim. Pages lets a browser keep each file
// ten minutes, so the page of the deploy before - no Reload button, no score
// note, its round note hidden - can load this app.js, which stopped at the
// missing button before it drew the board.
test('app.js gives the page of the deploy before the elements it needs', async () => {
  const source = await readFile(new URL('../src/app.js', import.meta.url), 'utf8');
  const markup = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const section = source.slice(source.indexOf('const elements = {'),
    source.indexOf('const settings = createSettingsController(elements);'));
  const tag = (page, id) => page.match(new RegExp(`<[a-z]+ id="${id}"[^>]*>`))?.[0];
  const round = tag(markup, 'roundStorageStatus');
  const previous = markup.replace(/\n\s*<button id="reloadPageButton"[^\n]*/, '')
    .replace(/\n\s*<p id="scoreStorageStatus"[^\n]*/, '').replace(round, `${round.slice(0, -1)} hidden>`);
  assert.ok(!tag(previous, 'reloadPageButton') && !tag(previous, 'scoreStorageStatus'));
  const load = (page) => {
    const created = [];
    const node = (properties) => ({ ...properties, attributes: {}, children: [],
      setAttribute(name, value) { this.attributes[name] = value; },
      after(sibling) { this.next = sibling; },
      append(child) { this.children.push(child); },
      closest: (selector) => (selector === '.score-panel' ? scorePanel : null) });
    const scorePanel = node({});
    const document = {
      querySelector(selector) {
        const found = tag(page, selector.slice(1));
        return found ? node({ hidden: /\shidden[\s>]/.test(found) }) : null;
      },
      createElement(name) {
        created.push(node({ tagName: name }));
        return created.at(-1);
      },
    };
    return { elements: vm.runInNewContext(`${section}\nelements;`, { document }), created, scorePanel };
  };

  const current = load(markup);
  assert.deepEqual(current.created, [], 'the page as it is has them');
  assert.equal(current.elements.roundStorageStatus.hidden, false);

  const { elements, scorePanel } = load(previous);
  const button = elements.reloadPageButton;
  // As index.html has it, beside Retry.
  const [, className, text] = markup.match(/<button id="reloadPageButton" class="([^"]*)" type="button" hidden>([^<]*)</);
  assert.deepEqual([button.tagName, button.id, button.className, button.type, button.hidden, button.textContent],
    ['button', 'reloadPageButton', className, 'button', true, text]);
  assert.equal(elements.retryAiButton.next, button);
  const note = elements.scoreStorageStatus;
  // As index.html has it, last in the score panel.
  const [, noteClass, role] = markup.match(/<p id="scoreStorageStatus" class="([^"]*)" role="([^"]*)"><\/p>\s*<\/div>/);
  assert.deepEqual([note.tagName, note.id, note.className, note.attributes.role], ['p', 'scoreStorageStatus', noteClass, role]);
  assert.deepEqual(scorePanel.children, [note]);
  assert.equal(elements.roundStorageStatus.hidden, false, 'the round note shows the text it is given');
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

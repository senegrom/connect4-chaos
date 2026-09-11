import assert from 'node:assert/strict';
import { once } from 'node:events';
import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, extname, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import { createStaticServer } from '../scripts/serve.mjs';
import { enableCrossOriginIsolation } from '../src/cross-origin-isolation.js';
import { formatBytes } from '../src/download-gate.js';
import { assetUrls, DOWNLOAD_BYTES } from '../src/neural-runtime.js';
import { createSettingsController, DIFFICULTY_HINTS } from '../src/settings-controller.js';

const repoRoot = fileURLToPath(new URL('../', import.meta.url));
const workerUrl = new URL('../cross-origin-isolation-worker.js', import.meta.url).href;
const workerScope = new URL('../', import.meta.url).href;
const flush = () => new Promise((resolve) => setImmediate(resolve));

async function fixtureServer(t, files) {
  const root = await mkdtemp(join(tmpdir(), 'connect4-assets-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  for (const path of files) {
    await mkdir(dirname(join(root, path)), { recursive: true });
    await writeFile(join(root, path), `fixture: ${path}`);
  }
  const server = await createStaticServer(root);
  t.after(() => new Promise((resolve, reject) => {
    server.closeAllConnections();
    server.close((error) => error ? reject(error) : resolve());
  }));
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  return { root, base: `http://127.0.0.1:${server.address().port}/` };
}

test('the dev server serves every neural asset and the isolation worker over HTTP', async (t) => {
  const paths = new Set(['cross-origin-isolation-worker.js']);
  for (const [name, value] of Object.entries(assetUrls())) {
    if (name === 'base') continue;
    for (const url of Array.isArray(value) ? value : [value]) {
      // The model is fetched from R2, not from this server; everything else
      // is a file this repository ships and has to be reachable over HTTP.
      if (!url.startsWith('file:')) {
        assert.match(url, /^https:\/\//, `${name} should be a file or an https URL: ${url}`);
        continue;
      }
      const path = relative(repoRoot, fileURLToPath(url)).split(sep).join('/');
      assert.ok(path.startsWith('assets/neural/'), path);
      paths.add(path);
    }
  }
  const { base } = await fixtureServer(t, paths);
  const types = { '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8', '.wasm': 'application/wasm' };
  for (const path of paths) {
    for (const method of ['GET', 'HEAD']) {
      const response = await fetch(new URL(path, base), { method });
      assert.equal(response.status, 200, `${method} ${path}`);
      assert.equal(response.headers.get('content-type'), types[extname(path)] ?? 'application/octet-stream');
      assert.equal(Number(response.headers.get('content-length')), Buffer.byteLength(`fixture: ${path}`));
      assert.equal(response.headers.get('x-content-type-options'), 'nosniff');
      assert.equal(await response.text(), method === 'GET' ? `fixture: ${path}` : '');
    }
  }
});

test('model part exceptions do not expose arbitrary part files or private paths', async (t) => {
  const denied = ['assets/private.part1', 'assets/neural/private.part2',
    'assets/neural/model.onnx.part3', 'src/model.onnx.part1',
    'private/model.onnx.part1', '.env', 'scripts/serve.mjs'];
  const { base } = await fixtureServer(t, denied);
  for (const path of denied) {
    const response = await fetch(new URL(path, base));
    assert.equal(response.status, 404, path);
    await response.text();
  }
});

test('allowed asset and worker paths cannot symlink to private targets', async (t) => {
  const { root, base } = await fixtureServer(t, ['private/model.json', 'scripts/private.js']);
  await mkdir(join(root, 'assets/neural'), { recursive: true });
  // model.json is reachable by the ordinary directory rule, so it is the
  // right shape of target now that the .partN exceptions are gone.
  await symlink(join(root, 'private/model.json'), join(root, 'assets/neural/model.json'));
  await symlink(join(root, 'scripts/private.js'), join(root, 'cross-origin-isolation-worker.js'));
  for (const path of ['assets/neural/model.json', 'cross-origin-isolation-worker.js']) {
    const response = await fetch(new URL(path, base));
    assert.equal(response.status, 404, path);
    await response.text();
  }
});

function mockGlobal(t, name, value) {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, name);
  Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
  t.after(() => {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else delete globalThis[name];
  });
}

for (const phase of ['active', 'waiting', 'installing']) {
  test(`coi=off only unregisters Connect4's ${phase} isolation worker`, async (t) => {
    const removed = [];
    const registration = (id, scope, scriptURL) => ({
      scope, [phase]: { scriptURL },
      async unregister() { removed.push(id); return true; },
    });
    const registrations = [
      registration('connect4', workerScope, workerUrl),
      registration('sibling', new URL('../other-app/', workerScope).href,
        new URL('../other-app/service-worker.js', workerScope).href),
      registration('wrong-script', workerScope, new URL('other-worker.js', workerScope).href),
      registration('wrong-scope', new URL('../', workerScope).href, workerUrl),
      { scope: workerScope, async unregister() { removed.push('no-worker'); } },
    ];
    mockGlobal(t, 'window', { location: { href: new URL('?coi=off', workerScope).href } });
    mockGlobal(t, 'navigator', { serviceWorker: { async getRegistrations() { return registrations; } } });
    assert.equal(await enableCrossOriginIsolation(), false);
    assert.deepEqual(removed, ['connect4']);
  });
}

test('isolation registration uses the same module-relative scope as cleanup, without reloading', async (t) => {
  const registered = [];
  const values = new Map();
  mockGlobal(t, 'window', { isSecureContext: true, crossOriginIsolated: false,
    location: { href: new URL('index.html', workerScope).href, reload() { assert.fail('must not reload'); } } });
  mockGlobal(t, 'navigator', { serviceWorker: {
    async register(url, options) { registered.push([url.href, options.scope]); },
  } });
  mockGlobal(t, 'sessionStorage', { getItem: (key) => values.get(key),
    setItem: (key, value) => values.set(key, value) });
  assert.equal(await enableCrossOriginIsolation(), false);
  assert.equal(await enableCrossOriginIsolation(), false);
  assert.deepEqual(registered, [[workerUrl, workerScope]]);
});

class Field extends EventTarget {
  constructor(value = '') {
    super();
    this.value = String(value);
    this.checked = false;
    this.disabled = false;
    this.min = '4';
    this.max = '10';
    this.attributes = new Map();
  }
  get valueAsNumber() { return this.value.trim() ? Number(this.value) : NaN; }
  setCustomValidity(message) { this.validationMessage = message; }
  setAttribute(name, value) { this.attributes.set(name, value); }
  removeAttribute(name) { this.attributes.delete(name); }
  reportValidity() { return !this.validationMessage; }
  focus() {}
}

function settingsElements(chaos = false) {
  const submit = new Field();
  const elements = {
    rowsInput: new Field(4), colsInput: new Field(4), connectInput: new Field(4),
    startingPlayerInput: new Field(1), chaosInput: new Field(), opponentInput: new Field('medium'),
    perfectOpponentOption: new Field(), yellowStarterOption: new Field(), opponentHint: new Field(),
    settingsToggle: new Field(), settingsForm: new Field(),
  };
  elements.connectInput.min = '3';
  elements.connectInput.max = '4';
  elements.chaosInput.checked = chaos;
  elements.settingsForm.querySelector = () => submit;
  return { elements, submit };
}

const manifest = { policies: [{ rows: 4, columns: 4, connect: 4, role: 2,
  file: './4x4-c4-role2.bin', handoffRemaining: 16, entryCount: 0, rootValue: 0 }] };

for (const catalog of ['classic', 'chaos']) {
  for (const [control, event] of [['settingsToggle', 'click'], ['settingsForm', 'focusin'],
    ['rowsInput', 'input'], ['opponentInput', 'change']]) {
    test(`${catalog} catalog recovers through ${control} ${event} without a reload`, async () => {
      const { elements, submit } = settingsElements(catalog === 'chaos');
      const calls = { classic: 0, chaos: 0 };
      let finishRetry;
      const retry = new Promise((resolve) => { finishRetry = resolve; });
      const loaders = Object.fromEntries(['classic', 'chaos'].map((name) => [name, (_url, options) => {
        calls[name] += 1;
        if (name !== catalog) return Promise.resolve(manifest);
        if (calls[name] === 1) return Promise.reject(new Error('offline'));
        assert.equal(options.force, true, 'retry must bypass the old cached request');
        return retry;
      }]));
      const controller = createSettingsController(elements, loaders);
      await controller.ready;
      assert.equal(controller.refresh().status, 'error');
      assert.equal(elements.perfectOpponentOption.disabled, true);
      elements[control].dispatchEvent(new Event(event));
      elements[control].dispatchEvent(new Event(event));
      assert.equal(calls[catalog], 2, 'deduplicate retries while a load is pending');
      assert.equal(calls[catalog === 'classic' ? 'chaos' : 'classic'], 1, 'do not refetch a healthy catalog');
      assert.equal(controller.refresh().status, 'loading');
      assert.equal(elements.opponentInput.value, 'medium', 'never change the selected opponent');
      elements.opponentInput.value = 'perfect';
      controller.refresh();
      assert.equal(submit.disabled, true);
      assert.equal(controller.canApply(), false, 'unverified Perfect must remain blocked');
      finishRetry(manifest);
      await flush();
      assert.equal(controller.refresh().status, 'ready');
      assert.equal(elements.perfectOpponentOption.disabled, false);
      assert.equal(elements.opponentInput.value, 'perfect');
      assert.equal(submit.disabled, false);
      assert.equal(controller.canApply(), true);
      assert.equal(elements.rowsInput.value, '4');
    });
  }
}

test('catalog timeout can be retried and a late old response cannot replace the recovered catalog', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const { elements } = settingsElements();
  let finishOld;
  let calls = 0;
  const old = new Promise((resolve) => { finishOld = resolve; });
  const controller = createSettingsController(elements, {
    classic: (_url, options) => {
      calls += 1;
      if (calls === 1) return old;
      assert.equal(options.force, true);
      return Promise.resolve(manifest);
    },
    chaos: async () => manifest,
  });
  await flush();
  t.mock.timers.tick(10_001);
  await controller.ready;
  assert.equal(controller.refresh().status, 'error');
  await controller.retryCatalogs();
  assert.equal(controller.refresh().status, 'ready');
  finishOld({ policies: [] });
  await flush();
  assert.equal(controller.refresh().status, 'ready');
  assert.equal(calls, 2);
});

test('a retry cannot overwrite a newer catalog accepted from the game', async () => {
  const { elements } = settingsElements();
  let calls = 0;
  let finishRetry;
  const retry = new Promise((resolve) => { finishRetry = resolve; });
  const controller = createSettingsController(elements, {
    classic: () => ++calls === 1 ? Promise.reject(new Error('offline')) : retry,
    chaos: async () => manifest,
  });
  await controller.ready;
  const pending = controller.retryCatalogs();
  controller.acceptCatalog('classic', manifest);
  finishRetry({ policies: [] });
  await pending;
  assert.equal(controller.refresh().status, 'ready');
  await controller.retryCatalogs();
  assert.equal(calls, 2, 'a healthy catalog should not be retried');
});

test('neural download copy matches the loader and does not promise permanent caching', async () => {
  const size = formatBytes(DOWNLOAD_BYTES.model + DOWNLOAD_BYTES.runtime);
  const readme = await readFile(new URL('../README.md', import.meta.url), 'utf8');
  const lines = readme.split('\n');
  const copies = [DIFFICULTY_HINTS.neural,
    lines.find((line) => line.startsWith('- **Neural opponent**')),
    lines.find((line) => line.startsWith('| Neural |'))];
  for (const copy of copies) {
    assert.ok(copy?.includes(size), `expected ${size}: ${copy}`);
    assert.match(copy, /normally cached/i);
    assert.doesNotMatch(copy, /one-time/i);
  }
});

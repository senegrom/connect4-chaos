#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { constants as fsConstants } from 'node:fs';
import { access, mkdtemp, readFile, rm, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import {
  extname,
  isAbsolute,
  join,
  relative as relativePath,
  resolve,
} from 'node:path';
import { fileURLToPath } from 'node:url';

const PROJECT_ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const HOST = '127.0.0.1';
const MIME_TYPES = new Map([
  ['.bin', 'application/octet-stream'],
  ['.css', 'text/css; charset=utf-8'],
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.mjs', 'text/javascript; charset=utf-8'],
  ['.onnx', 'application/octet-stream'],
  ['.wasm', 'application/wasm'],
  ['.svg', 'image/svg+xml; charset=utf-8'],
]);

function delay(milliseconds) {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
}

async function executable(path) {
  if (!path) return false;
  try {
    await access(path, fsConstants.X_OK);
    return true;
  } catch {
    return false;
  }
}

async function findBrowser() {
  const explicit = process.env.CHROME_BIN;
  if (explicit) {
    if (await executable(explicit)) return explicit;
    throw new Error(`CHROME_BIN is not executable: ${explicit}`);
  }

  const candidates = [
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ];
  for (const candidate of candidates) {
    if (await executable(candidate)) return candidate;
  }
  throw new Error('No Chromium-compatible browser found.');
}

async function startStaticServer() {
  const requestCounts = new Map();
  const server = createServer(async (request, response) => {
    try {
      if (request.method !== 'GET' && request.method !== 'HEAD') {
        response.writeHead(405, { Allow: 'GET, HEAD' }).end();
        return;
      }

      const requestUrl = new URL(request.url ?? '/', `http://${HOST}`);
      const decoded = decodeURIComponent(requestUrl.pathname);
      const relative = decoded === '/' ? 'index.html' : decoded.replace(/^\/+/, '');
      const filePath = resolve(PROJECT_ROOT, relative);
      const pathFromRoot = relativePath(PROJECT_ROOT, filePath);
      if (pathFromRoot.startsWith('..') || isAbsolute(pathFromRoot)) {
        response.writeHead(403).end('Forbidden');
        return;
      }

      const info = await stat(filePath);
      if (!info.isFile()) throw new Error('Not a file');
      const content = await readFile(filePath);
      requestCounts.set(decoded, (requestCounts.get(decoded) ?? 0) + 1);
      response.writeHead(200, {
        'Cache-Control': 'no-store',
        'Content-Length': String(content.length),
        'Content-Type': MIME_TYPES.get(extname(filePath)) ?? 'application/octet-stream',
      });
      if (request.method === 'HEAD') response.end();
      else response.end(content);
    } catch {
      response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      response.end('Not found');
    }
  });

  await new Promise((resolveListen, rejectListen) => {
    server.once('error', rejectListen);
    server.listen(0, HOST, resolveListen);
  });
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Static server did not expose a TCP port.');
  return {
    server,
    requestCounts,
    url: `http://${HOST}:${address.port}/`,
  };
}

async function readDevToolsPort(userDataDirectory, browserProcess, stderr) {
  const filePath = join(userDataDirectory, 'DevToolsActivePort');
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    if (browserProcess.exitCode !== null) {
      throw new Error(`Browser exited before exposing DevTools.\n${stderr.join('')}`);
    }
    try {
      const [port, browserPath] = (await readFile(filePath, 'utf8')).trim().split(/\r?\n/);
      if (port && browserPath) return { port: Number(port), browserPath };
    } catch {
      // Chrome creates the file after startup.
    }
    await delay(50);
  }
  throw new Error(`Timed out waiting for DevToolsActivePort.\n${stderr.join('')}`);
}

async function navigateAndWait(cdp, url) {
  await cdp.send('Page.navigate', { url });
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    try {
      const ready = await cdp.evaluate(
        `location.href === ${JSON.stringify(url)} && document.readyState !== 'loading'`,
        2_000,
      );
      if (ready) return;
    } catch {
      // Navigation can replace the execution context between polls.
    }
    await delay(50);
  }
  throw new Error(`Timed out navigating to ${url}.`);
}

class CdpConnection {
  constructor(url) {
    this.url = url;
    this.socket = null;
    this.nextId = 1;
    this.pending = new Map();
    this.errors = [];
  }

  async open() {
    this.socket = new WebSocket(this.url);
    this.socket.addEventListener('message', (event) => this.#handleMessage(event));
    this.socket.addEventListener('close', () => this.#rejectOutstanding(
      new Error('DevTools WebSocket closed unexpectedly.'),
    ));
    await new Promise((resolveOpen, rejectOpen) => {
      const timeout = setTimeout(() => rejectOpen(new Error('Timed out opening DevTools WebSocket.')), 10_000);
      this.socket.addEventListener('open', () => {
        clearTimeout(timeout);
        resolveOpen();
      }, { once: true });
      this.socket.addEventListener('error', () => {
        clearTimeout(timeout);
        rejectOpen(new Error('DevTools WebSocket failed to open.'));
      }, { once: true });
    });
  }

  #rejectOutstanding(error) {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timeout);
      pending.reject(error);
    }
    this.pending.clear();
  }

  #handleMessage(event) {
    const message = JSON.parse(String(event.data));
    if (message.id) {
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      clearTimeout(pending.timeout);
      if (message.error) pending.reject(new Error(`${pending.method}: ${message.error.message}`));
      else pending.resolve(message.result ?? {});
      return;
    }

    if (message.method === 'Runtime.exceptionThrown') {
      const details = message.params?.exceptionDetails;
      this.errors.push(`Runtime exception: ${details?.text ?? 'unknown'} ${details?.exception?.description ?? ''}`.trim());
    } else if (message.method === 'Runtime.consoleAPICalled' && message.params?.type === 'error') {
      this.errors.push(`Console error: ${(message.params.args ?? []).map((arg) => arg.value ?? arg.description ?? '').join(' ')}`);
    } else if (message.method === 'Log.entryAdded' && message.params?.entry?.level === 'error') {
      this.errors.push(`Log error: ${message.params.entry.text}`);
    }

  }

  send(method, params = {}, timeoutMilliseconds = 30_000) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error('DevTools WebSocket is not open.'));
    }
    const id = this.nextId;
    this.nextId += 1;
    return new Promise((resolveCommand, rejectCommand) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        rejectCommand(new Error(`${method} timed out after ${timeoutMilliseconds}ms.`));
      }, timeoutMilliseconds);
      this.pending.set(id, { resolve: resolveCommand, reject: rejectCommand, timeout, method });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }

  async evaluate(expression, timeoutMilliseconds = 60_000) {
    const result = await this.send('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
      userGesture: true,
    }, timeoutMilliseconds);
    if (result.exceptionDetails) {
      throw new Error(result.exceptionDetails.exception?.description ?? result.exceptionDetails.text ?? 'Page evaluation failed.');
    }
    return result.result?.value;
  }

  close() {
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.close();
  }
}

const smokeExpression = String.raw`(async () => {
  const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
  const waitFor = async (predicate, label, timeout = 30000) => {
    const deadline = performance.now() + timeout;
    while (performance.now() < deadline) {
      if (predicate()) return;
      await delay(16);
    }
    throw new Error('Timed out waiting for ' + label);
  };
  const required = (selector) => {
    const element = document.querySelector(selector);
    if (!element) throw new Error('Missing required element: ' + selector);
    return element;
  };
  const select = (selector, value) => {
    const element = required(selector);
    element.value = value;
    element.dispatchEvent(new Event('change', { bubbles: true }));
  };
  const setChecked = (selector, checked) => {
    const element = required(selector);
    element.checked = checked;
    element.dispatchEvent(new Event('change', { bubbles: true }));
  };
  const submit = () => required('#settingsForm button[type="submit"]').click();
  const openSettings = () => {
    if (required('#settingsBody').hidden) required('#settingsToggle').click();
  };
  const pieceCount = (className) => document.querySelectorAll('.cell.' + className).length;
  const statusIncludes = (text) => required('#statusText').textContent.includes(text);
  const waitForRound = (red, yellow, status, label) => waitFor(
    () => pieceCount('red') === red && pieceCount('yellow') === yellow && statusIncludes(status),
    label,
  );
  const waitForStableBoard = async () => {
    const frame = required('#boardFrame');
    let previous = frame.getBoundingClientRect().y;
    let stableFrames = 0;
    const deadline = performance.now() + 5000;
    while (performance.now() < deadline) {
      await delay(16);
      const current = frame.getBoundingClientRect().y;
      stableFrames = Math.abs(current - previous) <= 0.05 ? stableFrames + 1 : 0;
      previous = current;
      if (stableFrames >= 8) return current;
    }
    throw new Error('Board position did not settle.');
  };
  const sampleTurn = async (column, red, yellow, label) => {
    const frame = required('#boardFrame');
    const cell = document.querySelector('.cell[data-column="' + column + '"]');
    if (!cell) throw new Error('Column ' + column + ' is not available for ' + label + '.');
    const samples = [await waitForStableBoard()];
    cell.click();
    const deadline = performance.now() + 30000;
    while (performance.now() < deadline) {
      samples.push(frame.getBoundingClientRect().y);
      if (pieceCount('red') === red
          && pieceCount('yellow') === yellow
          && statusIncludes('Red to move')) return samples;
      await delay(16);
    }
    throw new Error('Timed out waiting for ' + label);
  };
  const measureTransform = async (selector, nextStatus, label) => {
    const button = required(selector);
    if (button.disabled) throw new Error(label + ' is unexpectedly disabled.');
    const start = performance.now();
    button.click();
    await waitFor(
      () => !button.disabled
        && statusIncludes(nextStatus)
        && !required('#boardFrame').className.includes('anim-'),
      label + ' completion',
    );
    return performance.now() - start;
  };
  const solveChaosEndgameInWorker = () => new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./src/ai-worker.js', location.href), { type: 'module' });
    const timeout = setTimeout(() => {
      worker.terminate();
      reject(new Error('Timed out waiting for the exact Chaos worker result.'));
    }, 30000);
    worker.addEventListener('error', (event) => {
      clearTimeout(timeout);
      worker.terminate();
      reject(new Error(event.message || 'The exact Chaos worker failed.'));
    }, { once: true });
    worker.addEventListener('message', (event) => {
      if (event.data?.requestId !== 991 || event.data?.kind !== 'result') return;
      clearTimeout(timeout);
      worker.terminate();
      resolve(event.data.result);
    });
    worker.postMessage({
      requestId: 991,
      position: {
        board: [
          [1, 1, 1, 2, 1, 0, 0],
          [2, 2, 2, 1, 2, 0, 0],
          [2, 1, 2, 1, 2, 1, 0],
          [2, 1, 1, 1, 2, 2, 0],
          [1, 2, 2, 2, 1, 2, 2],
          [1, 1, 2, 2, 1, 1, 1],
        ],
        currentPlayer: 1,
        connect: 4,
        chaosMode: true,
      },
      options: { difficulty: 'perfect', aiPlayer: 1 },
    });
  });

  await waitFor(
    () => document.querySelector('#perfectOpponentOption')
      && document.querySelectorAll('.cell').length === 42
      && document.querySelector('.ghost-disc'),
    'application boot',
  );
  const initialHeroHeight = required('#hero').getBoundingClientRect().height;
  if (document.querySelectorAll('.column-controls button, .column-controls [tabindex]').length !== 0) {
    throw new Error('The column preview adds duplicate keyboard focus targets.');
  }
  if (document.querySelector('#evaluationPanel meter')) {
    throw new Error('The analysis panel still exposes a probability-shaped meter.');
  }
  if (required('#perfectOpponentOption').disabled) {
    throw new Error('Perfect AI option is unexpectedly disabled for the standard board.');
  }
  if (document.documentElement.scrollWidth > innerWidth + 1) {
    throw new Error('The mobile layout overflows horizontally.');
  }

  const chaosExact = await solveChaosEndgameInWorker();

  select('#opponentInput', 'easy');
  select('#startingPlayerInput', '2');
  submit();
  await waitForRound(0, 1, 'Red to move', 'the Easy AI opening move');
  await waitFor(() => required('#settingsBody').hidden, 'the setup to collapse');
  if (!document.body.classList.contains('game-first')) {
    throw new Error('Starting a round did not enable the game-first layout.');
  }
  if (required('#hero').getBoundingClientRect().height >= initialHeroHeight - 20) {
    throw new Error('The active-round masthead did not become meaningfully smaller.');
  }
  if (required('#touchHelp').hidden || getComputedStyle(required('#touchHelp')).display === 'none') {
    throw new Error('Touch guidance is not visible in the touch viewport.');
  }

  openSettings();
  select('#opponentInput', 'perfect');
  select('#startingPlayerInput', '2');
  submit();
  await waitForRound(0, 1, 'Red to move', 'the Perfect AI opening move');
  const firstColumn = Number(document.querySelector('.cell.yellow').dataset.column);
  const searchTexts = [required('#searchInfo').textContent];
  const exactTexts = [required('#exactResultText').textContent];
  if (required('#evaluationPanel').dataset.analysisMode !== 'exact') {
    throw new Error('Perfect mode is not presented as exact analysis.');
  }
  if (required('#aiDetails').open) {
    throw new Error('Technical AI details are expanded by default.');
  }
  const turnSamples = [await sampleTurn(0, 1, 2, 'the second Perfect move as first player')];
  searchTexts.push(required('#searchInfo').textContent);
  exactTexts.push(required('#exactResultText').textContent);

  openSettings();
  select('#startingPlayerInput', '1');
  submit();
  await waitForRound(0, 0, 'Red to move', 'the human-starting round');
  const board = required('#gameBoard');
  board.focus();
  const ghostBefore = required('#ghostDisc').getBoundingClientRect().x;
  board.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
  await delay(220);
  const ghostAfter = required('#ghostDisc').getBoundingClientRect().x;
  if (ghostAfter <= ghostBefore + 2) throw new Error('The ghost disc did not move with keyboard selection.');
  if (!required('#selectedColumnStatus').textContent.includes('Column 5 selected')) {
    throw new Error('Keyboard column selection was not announced.');
  }
  board.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
  turnSamples.push(await sampleTurn(3, 1, 1, 'the first Perfect reply as second player'));
  searchTexts.push(required('#searchInfo').textContent);
  exactTexts.push(required('#exactResultText').textContent);
  turnSamples.push(await sampleTurn(0, 2, 2, 'the second Perfect reply as second player'));
  searchTexts.push(required('#searchInfo').textContent);
  exactTexts.push(required('#exactResultText').textContent);

  openSettings();
  select('#opponentInput', 'brutal');
  select('#startingPlayerInput', '1');
  setChecked('#chaosInput', true);
  submit();
  await waitForRound(0, 0, 'Red to move', 'the Brutal Chaos policy round');
  required('.cell[data-column="2"]').click();
  await waitForRound(1, 1, 'Red to move', 'the first certified Brutal Chaos reply');
  const brutalChaosSearchTexts = [required('#searchInfo').textContent];
  const brutalChaosColumns = Array.from(
    document.querySelectorAll('.cell.yellow'),
    (cell) => Number(cell.dataset.column),
  );
  required('.cell[data-column="3"]').click();
  await waitForRound(2, 2, 'Red to move', 'the second certified Brutal Chaos reply');
  brutalChaosSearchTexts.push(required('#searchInfo').textContent);
  brutalChaosColumns.push(...Array.from(
    document.querySelectorAll('.cell.yellow'),
    (cell) => Number(cell.dataset.column),
  ).slice(1));

  openSettings();
  select('#opponentInput', 'human');
  select('#startingPlayerInput', '1');
  setChecked('#chaosInput', true);
  submit();
  await waitForRound(0, 0, 'Red to move', 'the Chaos animation round');
  if (required('#transformToolbar').hidden
      || document.querySelectorAll('#transformToolbar .chaos-action').length !== 3) {
    throw new Error('Chaos transformations are not grouped in their toolbar.');
  }
  if (document.querySelectorAll('.game-actions button').length !== 2) {
    throw new Error('Round utilities and Chaos transformations are not separated.');
  }
  required('.cell[data-column="3"]').click();
  await waitFor(
    () => pieceCount('red') === 1 && pieceCount('yellow') === 0 && statusIncludes('Yellow to move'),
    'the setup drop before transforms',
  );
  const flipElapsedMs = await measureTransform('#flipButton', 'Red to move', 'Flip');
  const rotateElapsedMs = await measureTransform('#rotateCwButton', 'Yellow to move', 'Rotate');

  const frame = required('#boardFrame');
  const animationDuration = (className) => {
    frame.classList.add(className);
    const duration = getComputedStyle(frame).animationDuration;
    frame.classList.remove(className);
    return duration;
  };

  // The desktop pass measures a fresh 6x7 board. The Chaos round played
  // above is stored for reload, and it ends rotated to 7x6.
  localStorage.removeItem('connect4-chaos.round.v1');

  return {
    chaosExact,
    firstColumn,
    searchTexts,
    exactTexts,
    brutalChaosSearchTexts,
    brutalChaosColumns,
    touchHintDismissed: required('#touchHelp').classList.contains('is-dismissed'),
    previewTabStops: document.querySelectorAll('.column-controls button, .column-controls [tabindex]').length,
    boardYRange: Math.max(...turnSamples.map((samples) => (
      Math.max(...samples) - Math.min(...samples)
    ))),
    flipElapsedMs,
    rotateElapsedMs,
    viewport: { width: innerWidth, scrollWidth: document.documentElement.scrollWidth },
    animationDurations: {
      flipOut: animationDuration('anim-flip-out'),
      flipIn: animationDuration('anim-flip-in'),
      rotateOut: animationDuration('anim-cw-out'),
      rotateIn: animationDuration('anim-cw-in'),
    },
  };
})()`;

const desktopExpression = String.raw`(async () => {
  const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
  const deadline = performance.now() + 15000;
  while (performance.now() < deadline) {
    if (document.querySelectorAll('.cell').length === 42 && document.querySelector('#boardFrame')) break;
    await delay(20);
  }
  const board = document.querySelector('#boardFrame');
  const hero = document.querySelector('#hero');
  const settings = document.querySelector('#settingsBody');
  if (!board || !hero || !settings) throw new Error('Desktop UI did not boot.');
  return {
    boardWidth: board.getBoundingClientRect().width,
    heroHeight: hero.getBoundingClientRect().height,
    settingsHidden: settings.hidden,
    keyboardHelpDisplay: getComputedStyle(document.querySelector('.keyboard-help')).display,
    touchHelpDisplay: getComputedStyle(document.querySelector('#touchHelp')).display,
    viewport: { width: innerWidth, scrollWidth: document.documentElement.scrollWidth },
  };
})()`;

// Parse the embedded page program before starting Chrome so syntax regressions fail immediately.
new Function(`return ${smokeExpression};`);
new Function(`return ${desktopExpression};`);

function assertRange(value, minimum, maximum, label) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new Error(`${label} took ${Number(value).toFixed(1)}ms; expected ${minimum}–${maximum}ms.`);
  }
}

function assertSmokeResult(result, desktopResult, browserErrors, requestCounts) {
  if (result.chaosExact?.solver !== 'chaos-exact-graph'
      || result.chaosExact?.solved !== true
      || result.chaosExact?.score !== 1
      || result.chaosExact?.action?.type !== 'rotateCW'
      || result.chaosExact?.nodes !== 2585) {
    throw new Error(`Browser worker did not return the verified exact Chaos fixture: ${JSON.stringify(result.chaosExact)}`);
  }
  if (result.firstColumn !== 3) {
    throw new Error(`Perfect AI opened in zero-based column ${result.firstColumn}; expected the centre column 3.`);
  }
  if (result.searchTexts.length !== 4 || result.exactTexts.length !== 4) {
    throw new Error(`Expected telemetry and exact summaries for four Perfect moves; found ${result.searchTexts.length}/${result.exactTexts.length}.`);
  }
  for (const searchText of result.searchTexts) {
    if (!searchText.includes('Perfect strategy') || !searchText.includes('Game-theoretically exact')) {
      throw new Error(`Perfect telemetry did not identify an exact strategy move: ${searchText}`);
    }
  }
  for (const exactText of result.exactTexts) {
    if (!/force a win|draw/i.test(exactText)) {
      throw new Error(`Perfect analysis did not show an exact result: ${exactText}`);
    }
  }
  if (result.brutalChaosColumns.length !== 2
      || result.brutalChaosColumns.some((column) => column !== 3)) {
    throw new Error(
      `Brutal Chaos did not answer the early sequence with centre drops: ${JSON.stringify(result.brutalChaosColumns)}`,
    );
  }
  for (const searchText of result.brutalChaosSearchTexts) {
    if (!searchText.includes('Certified Chaos policy') || !searchText.includes('Policy layer 0→8 pieces')) {
      throw new Error(`Brutal Chaos telemetry did not identify the certified policy: ${searchText}`);
    }
  }
  if (!result.touchHintDismissed) throw new Error('Touch guidance did not recede after the first human move.');
  if (result.previewTabStops !== 0) throw new Error(`Column preview exposes ${result.previewTabStops} duplicate tab stops.`);
  if (result.boardYRange > 0.5) {
    throw new Error(`Board shifted vertically by ${result.boardYRange.toFixed(3)}px during a turn.`);
  }
  if (result.viewport.scrollWidth > result.viewport.width + 1) {
    throw new Error(`Mobile layout width is ${result.viewport.scrollWidth}px in a ${result.viewport.width}px viewport.`);
  }

  const expectedDurations = {
    flipOut: '0.32s',
    flipIn: '0.42s',
    rotateOut: '0.28s',
    rotateIn: '0.36s',
  };
  for (const [name, expected] of Object.entries(expectedDurations)) {
    if (result.animationDurations[name] !== expected) {
      throw new Error(`${name} duration was ${result.animationDurations[name]}; expected ${expected}.`);
    }
  }
  assertRange(result.flipElapsedMs, 650, 1_800, 'Flip');
  assertRange(result.rotateElapsedMs, 550, 1_700, 'Rotate');

  if (!desktopResult.settingsHidden) throw new Error('Returning desktop players do not start in the compact layout.');
  if (desktopResult.boardWidth < 535) throw new Error(`Desktop board width is only ${desktopResult.boardWidth.toFixed(1)}px.`);
  if (desktopResult.heroHeight > 85) throw new Error(`Compact desktop masthead is ${desktopResult.heroHeight.toFixed(1)}px tall.`);
  if (desktopResult.viewport.scrollWidth > desktopResult.viewport.width + 1) {
    throw new Error('Desktop layout overflows horizontally.');
  }
  if (desktopResult.keyboardHelpDisplay === 'none' || desktopResult.touchHelpDisplay !== 'none') {
    throw new Error('Desktop input guidance does not match a keyboard-capable viewport.');
  }

  const strategyRequests = requestCounts.get('/assets/perfect-strategy.bin') ?? 0;
  if (strategyRequests !== 2) {
    throw new Error(`Perfect strategy was requested ${strategyRequests} times; expected once per Perfect round (2 total).`);
  }
  const bookRequests = requestCounts.get('/assets/perfect-book.bin') ?? 0;
  if (bookRequests !== 0) {
    throw new Error(`Perfect mode unexpectedly requested the lower-level opening book ${bookRequests} times.`);
  }
  const chaosPolicyRequests = requestCounts.get(
    '/data/perfect-chaos-prefix/yellow/0-8.policy.bin',
  ) ?? 0;
  if (chaosPolicyRequests !== 1) {
    throw new Error(
      `The second-player Chaos policy was requested ${chaosPolicyRequests} times; expected once.`,
    );
  }
  const wrongRoleRequests = requestCounts.get('/data/perfect-chaos-prefix/red/0-8.policy.bin') ?? 0;
  if (wrongRoleRequests !== 0) {
    throw new Error(`The wrong Chaos policy role was requested ${wrongRoleRequests} times.`);
  }
  if (browserErrors.length) throw new Error(browserErrors.join('\n'));
}

async function stopBrowser(browserProcess) {
  const browserExit = browserProcess.exitCode !== null
    ? Promise.resolve()
    : new Promise((resolveExit) => browserProcess.once('exit', resolveExit));
  browserProcess.kill('SIGTERM');
  await Promise.race([browserExit, delay(2_000)]);
  if (browserProcess.exitCode === null) {
    browserProcess.kill('SIGKILL');
    await Promise.race([browserExit, delay(2_000)]);
  }
}

async function main() {
  const browserPath = await findBrowser();
  const { server, requestCounts, url } = await startStaticServer();
  const userDataDirectory = await mkdtemp(join(tmpdir(), 'connect4-browser-smoke-'));
  const stderr = [];
  const browserProcess = spawn(browserPath, [
    '--headless=new',
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--remote-debugging-port=0',
    `--user-data-dir=${userDataDirectory}`,
    'about:blank',
  ], { stdio: ['ignore', 'ignore', 'pipe'] });
  browserProcess.stderr.setEncoding('utf8');
  browserProcess.stderr.on('data', (chunk) => stderr.push(chunk));

  let cdp;
  try {
    const { port } = await readDevToolsPort(userDataDirectory, browserProcess, stderr);
    const targetResponse = await fetch(
      `http://${HOST}:${port}/json/new?${encodeURIComponent('about:blank')}`,
      { method: 'PUT' },
    );
    if (!targetResponse.ok) throw new Error(`Creating a DevTools page failed with HTTP ${targetResponse.status}.`);
    const target = await targetResponse.json();
    cdp = new CdpConnection(target.webSocketDebuggerUrl);
    await cdp.open();
    await Promise.all([
      cdp.send('Page.enable'),
      cdp.send('Runtime.enable'),
      cdp.send('Log.enable'),
    ]);
    await cdp.send('Emulation.setDeviceMetricsOverride', {
      width: 390,
      height: 844,
      deviceScaleFactor: 1,
      mobile: true,
    });
    await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 1 });
    await navigateAndWait(cdp, url);
    let result;
    try {
      result = await cdp.evaluate(smokeExpression, 120_000);
    } catch (error) {
      const requests = JSON.stringify(Object.fromEntries(requestCounts));
      const browserMessages = cdp.errors.join('\n') || '(none)';
      throw new Error(`${error instanceof Error ? error.message : String(error)}\nBrowser messages:\n${browserMessages}\nRequests: ${requests}`);
    }
    await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: false });
    await cdp.send('Emulation.setDeviceMetricsOverride', {
      width: 1280,
      height: 900,
      deviceScaleFactor: 1,
      mobile: false,
    });
    const desktopUrl = `${url}?desktop=1`;
    await navigateAndWait(cdp, desktopUrl);
    const desktopResult = await cdp.evaluate(desktopExpression, 30_000);

    assertSmokeResult(result, desktopResult, cdp.errors, requestCounts);
    console.log(JSON.stringify({
      browser: browserPath,
      strategyRequests: requestCounts.get('/assets/perfect-strategy.bin') ?? 0,
      desktop: desktopResult,
      ...result,
    }, null, 2));
  } finally {
    cdp?.close();
    await stopBrowser(browserProcess);
    server.closeAllConnections?.();
    await new Promise((resolveClose) => server.close(resolveClose));
    await rm(userDataDirectory, {
      recursive: true,
      force: true,
      maxRetries: 10,
      retryDelay: 100,
    });
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack : String(error));
  process.exitCode = 1;
});

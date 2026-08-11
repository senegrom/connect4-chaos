#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { constants as fsConstants } from 'node:fs';
import { access, mkdtemp, readFile, rm, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const PROJECT_ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const HOST = '127.0.0.1';
const MIME_TYPES = new Map([
  ['.bin', 'application/octet-stream'],
  ['.css', 'text/css; charset=utf-8'],
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
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
  const candidates = [
    explicit,
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ];
  for (const candidate of candidates) {
    if (await executable(candidate)) return candidate;
  }
  throw new Error(`No Chromium-compatible browser found${explicit ? ` at CHROME_BIN=${explicit}` : ''}.`);
}

async function startStaticServer() {
  const server = createServer(async (request, response) => {
    try {
      const requestUrl = new URL(request.url ?? '/', `http://${HOST}`);
      const decoded = decodeURIComponent(requestUrl.pathname);
      const relative = decoded === '/' ? 'index.html' : decoded.replace(/^\/+/, '');
      const normalized = normalize(relative).replace(/^(\.\.[/\\])+/, '');
      const filePath = resolve(PROJECT_ROOT, normalized);
      if (!filePath.startsWith(`${PROJECT_ROOT}/`) && filePath !== PROJECT_ROOT) {
        response.writeHead(403).end('Forbidden');
        return;
      }
      const info = await stat(filePath);
      if (!info.isFile()) throw new Error('Not a file');
      const content = await readFile(filePath);
      response.writeHead(200, {
        'Cache-Control': 'no-store',
        'Content-Length': String(content.length),
        'Content-Type': MIME_TYPES.get(extname(filePath)) ?? 'application/octet-stream',
      });
      response.end(content);
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

class CdpConnection {
  constructor(url) {
    this.url = url;
    this.socket = null;
    this.nextId = 1;
    this.pending = new Map();
    this.eventWaiters = new Map();
    this.errors = [];
  }

  async open() {
    this.socket = new WebSocket(this.url);
    this.socket.addEventListener('message', (event) => this.#handleMessage(event));
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

    const waiters = this.eventWaiters.get(message.method);
    if (!waiters?.length) return;
    const waiter = waiters.shift();
    clearTimeout(waiter.timeout);
    waiter.resolve(message.params ?? {});
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

  waitFor(method, timeoutMilliseconds = 30_000) {
    return new Promise((resolveEvent, rejectEvent) => {
      const timeout = setTimeout(() => {
        const waiters = this.eventWaiters.get(method) ?? [];
        const index = waiters.findIndex((waiter) => waiter.resolve === resolveEvent);
        if (index >= 0) waiters.splice(index, 1);
        rejectEvent(new Error(`${method} event timed out after ${timeoutMilliseconds}ms.`));
      }, timeoutMilliseconds);
      const waiters = this.eventWaiters.get(method) ?? [];
      waiters.push({ resolve: resolveEvent, reject: rejectEvent, timeout });
      this.eventWaiters.set(method, waiters);
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
  const select = (selector, value) => {
    const element = document.querySelector(selector);
    element.value = value;
    element.dispatchEvent(new Event('change', { bubbles: true }));
  };
  const submit = () => document.querySelector('#settingsForm button[type="submit"]').click();
  const openSettings = () => {
    if (document.querySelector('#settingsBody').hidden) document.querySelector('#settingsToggle').click();
  };

  if (document.querySelector('#perfectOpponentOption').disabled) {
    throw new Error('Perfect AI option is unexpectedly disabled for the standard board.');
  }

  select('#opponentInput', 'perfect');
  select('#startingPlayerInput', '2');
  submit();
  await waitFor(
    () => document.querySelectorAll('.cell.yellow').length === 1
      && document.querySelector('#statusText').textContent.includes('Red to move'),
    'the Perfect AI first move',
  );

  const firstColumns = [...document.querySelectorAll('.cell.yellow')]
    .map((cell) => Number(cell.dataset.column));
  const firstSearch = document.querySelector('#searchInfo').textContent;

  openSettings();
  select('#startingPlayerInput', '1');
  submit();
  await waitFor(
    () => document.querySelector('#statusText').textContent.includes('Red to move')
      && document.querySelectorAll('.cell.red, .cell.yellow').length === 0,
    'the human-starting round',
  );

  const ySamples = [document.querySelector('#boardFrame').getBoundingClientRect().y];
  document.querySelectorAll('.column-button')[3].click();
  const deadline = performance.now() + 30000;
  while (performance.now() < deadline) {
    ySamples.push(document.querySelector('#boardFrame').getBoundingClientRect().y);
    const complete = document.querySelectorAll('.cell.red').length === 1
      && document.querySelectorAll('.cell.yellow').length === 1
      && document.querySelector('#statusText').textContent.includes('Red to move');
    if (complete) break;
    await delay(16);
  }
  if (document.querySelectorAll('.cell.yellow').length !== 1) {
    throw new Error('Perfect AI did not answer the human centre opening.');
  }

  const secondSearch = document.querySelector('#searchInfo').textContent;
  const frame = document.querySelector('#boardFrame');
  const animationDuration = (className) => {
    frame.classList.add(className);
    const duration = getComputedStyle(frame).animationDuration;
    frame.classList.remove(className);
    return duration;
  };
  const minimumY = Math.min(...ySamples);
  const maximumY = Math.max(...ySamples);

  return {
    firstColumn: firstColumns[0] ?? null,
    firstSearch,
    secondSearch,
    boardYRange: maximumY - minimumY,
    animationDurations: {
      flipOut: animationDuration('anim-flip-out'),
      flipIn: animationDuration('anim-flip-in'),
      rotateOut: animationDuration('anim-cw-out'),
      rotateIn: animationDuration('anim-cw-in'),
    },
  };
})()`;

function assertSmokeResult(result, browserErrors) {
  if (result.firstColumn !== 3) {
    throw new Error(`Perfect AI opened in zero-based column ${result.firstColumn}; expected the centre column 3.`);
  }
  for (const searchText of [result.firstSearch, result.secondSearch]) {
    if (!searchText.includes('Perfect strategy') || !searchText.includes('Game-theoretically exact')) {
      throw new Error(`Perfect telemetry did not identify an exact strategy move: ${searchText}`);
    }
  }
  if (result.boardYRange > 0.5) {
    throw new Error(`Board shifted vertically by ${result.boardYRange.toFixed(3)}px during a turn.`);
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
  if (browserErrors.length) throw new Error(browserErrors.join('\n'));
}

async function main() {
  const browserPath = await findBrowser();
  const { server, url } = await startStaticServer();
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
    const targetResponse = await fetch(`http://${HOST}:${port}/json/new?${encodeURIComponent(url)}`, { method: 'PUT' });
    if (!targetResponse.ok) throw new Error(`Creating a DevTools page failed with HTTP ${targetResponse.status}.`);
    const target = await targetResponse.json();
    cdp = new CdpConnection(target.webSocketDebuggerUrl);
    await cdp.open();
    await Promise.all([
      cdp.send('Page.enable'),
      cdp.send('Runtime.enable'),
      cdp.send('Log.enable'),
      cdp.send('Network.enable'),
    ]);
    await cdp.send('Emulation.setDeviceMetricsOverride', {
      width: 390,
      height: 844,
      deviceScaleFactor: 1,
      mobile: false,
    });
    const loadEvent = cdp.waitFor('Page.loadEventFired');
    await cdp.send('Page.navigate', { url });
    await loadEvent;
    const result = await cdp.evaluate(smokeExpression, 90_000);
    assertSmokeResult(result, cdp.errors);
    console.log(JSON.stringify({ browser: browserPath, ...result }, null, 2));
  } finally {
    cdp?.close();
    browserProcess.kill('SIGTERM');
    await Promise.race([
      new Promise((resolveExit) => browserProcess.once('exit', resolveExit)),
      delay(2_000).then(() => browserProcess.kill('SIGKILL')),
    ]);
    await new Promise((resolveClose) => server.close(resolveClose));
    await rm(userDataDirectory, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack : String(error));
  process.exitCode = 1;
});

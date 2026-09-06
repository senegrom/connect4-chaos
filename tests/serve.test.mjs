import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { once } from 'node:events';
import fs from 'node:fs/promises';
import { request } from 'node:http';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { Readable } from 'node:stream';
import { after, before, test } from 'node:test';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { createStaticServer } from '../scripts/serve.mjs';

let directory;
let root;
let server;
let port;
let symlinksSupported = true;
const fixtures = new Map([
  ['index.html', '<!doctype html><title>Fixture</title>'],
  ['styles.css', 'body {}'],
  ['manifest.json', '{}'],
  ['favicon.svg', '<svg/>'],
  ['favicon.ico', 'icon'],
  ['apple-touch-icon.png', 'icon'],
  ['src/app.js', 'export const fixture = true;'],
  ['assets/neural/runtime.mjs', 'export {};'],
  ['assets/neural/runtime.wasm', 'wasm fixture'],
  ['assets/neural/model.onnx', 'model fixture'],
  ['assets/failure.bin', 'failure fixture'],
  ['icons/icon.png', 'icon'],
  ['data/perfect-classic/manifest.json', '{}'],
  ['data/perfect-chaos-prefix/red/test.policy.bin', 'policy'],
  ['data/perfect-chaos-complete/test.bin', 'policy'],
  ['.env', 'PRIVATE_FIXTURE'],
  ['.git/config', 'PRIVATE_FIXTURE'],
  ['package.json', 'PRIVATE_FIXTURE'],
  ['scripts/tool.mjs', 'PRIVATE_FIXTURE'],
  ['node_modules/private.js', 'PRIVATE_FIXTURE'],
  ['src/.hidden.js', 'PRIVATE_FIXTURE'],
  ['assets/backup.json.bak', 'PRIVATE_FIXTURE'],
]);

before(async () => {
  directory = await fs.mkdtemp(join(tmpdir(), 'connect4-serve-test-'));
  root = join(directory, 'site');
  for (const [path, content] of fixtures) {
    const target = join(root, path);
    await fs.mkdir(dirname(target), { recursive: true });
    await fs.writeFile(target, content);
  }
  await fs.mkdir(join(root, 'assets/directory.js'));
  const outside = join(directory, 'site-outside');
  await fs.mkdir(outside);
  await fs.writeFile(join(outside, 'outside.js'), 'OUTSIDE_FIXTURE');
  try {
    await fs.symlink(join(outside, 'outside.js'), join(root, 'src/outside.js'));
    await fs.symlink(outside, join(root, 'assets/outside'));
    await fs.symlink(join(root, 'src/.hidden.js'), join(root, 'src/hidden-link.js'));
    await fs.symlink(join(root, 'scripts/tool.mjs'), join(root, 'src/tool-link.mjs'));
    await fs.symlink(join(root, 'src/app.js'), join(root, 'src/public-link.js'));
  } catch (error) {
    if (process.platform === 'win32' && error.code === 'EPERM') symlinksSupported = false;
    else throw error;
  }
  server = await createStaticServer(root);
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  port = server.address().port;
});

after(async () => {
  if (server?.listening) {
    server.closeAllConnections();
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  }
  if (directory) await fs.rm(directory, { recursive: true, force: true });
});

function get(path, { method = 'GET', headers = {} } = {}) {
  return new Promise((resolve, reject) => {
    const req = request({ hostname: '127.0.0.1', port, path, method, headers, agent: false }, (res) => {
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('error', reject);
      res.on('end', () => resolve({ status: res.statusCode, headers: res.headers,
        text: Buffer.concat(chunks).toString() }));
    });
    req.setTimeout(4000, () => req.destroy(new Error('Request timed out')));
    req.on('error', reject);
    req.end();
  });
}

async function waitFor(predicate) {
  for (let attempt = 0; attempt < 200; attempt += 1) {
    if (predicate()) return;
    await delay(5);
  }
  assert.fail('Expected resource cleanup did not complete');
}

async function healthy() {
  const result = await get('/');
  assert.equal(result.status, 200);
  assert.equal(result.text, fixtures.get('index.html'));
}

test('serves the game shell and every public asset family', async () => {
  for (const [path, content] of [...fixtures].slice(0, 15)) {
    const result = await get(`/${path}?v=test`);
    assert.equal(result.status, 200, path);
    assert.equal(result.text, content, path);
    assert.equal(result.headers['cache-control'], 'no-store');
    assert.equal(result.headers['x-content-type-options'], 'nosniff');
    assert.equal(Number(result.headers['content-length']), Buffer.byteLength(content));
  }
  assert.equal((await get('/assets/neural/runtime.wasm')).headers['content-type'], 'application/wasm');
  assert.equal((await get('/icons/icon.png')).headers['content-type'], 'image/png');
  assert.equal((await get('/src/app.js')).headers['content-type'], 'text/javascript; charset=utf-8');
  await healthy();
});

test('rejects private files, dotfiles, traversal and backup files', async () => {
  for (const path of ['/.env', '/.git/config', '/package.json', '/scripts/tool.mjs',
    '/node_modules/private.js', '/src/.hidden.js', '/assets/backup.json.bak',
    '/src/../.env', '/src/%2e%2e/.env', '/src/%2Ehidden.js', '/src//app.js',
    '/assets/%252e%252e/.env', '/assets/directory.js', '/assets/missing.js', '/src/']) {
    const result = await get(path);
    assert.equal(result.status, 404, path);
    assert.equal(result.text, 'Not found', path);
  }
});

test('rejects file and directory symlinks outside public roots', async (t) => {
  if (!symlinksSupported) {
    t.skip('Windows symlink privileges are unavailable');
    return;
  }
  for (const path of ['/src/outside.js', '/assets/outside/outside.js',
    '/src/hidden-link.js', '/src/tool-link.mjs']) {
    assert.equal((await get(path)).status, 404, path);
  }
  assert.equal((await get('/src/public-link.js')).text, fixtures.get('src/app.js'));
});

test('rejects malformed paths without terminating the server', async () => {
  for (const path of ['/%', '/%FF', '/src/%00app.js', '/src/%5capp.js', '//src/app.js',
    'http://example.invalid/src/app.js']) {
    assert.equal((await get(path)).status, 400, path);
  }
  await healthy();
});

test('rejects non-local Host values and non-read methods', async () => {
  assert.equal((await get('/', { headers: { host: 'example.invalid' } })).status, 403);
  assert.equal((await get('/', { headers: { host: 'localhost' } })).status, 200);
  const result = await get('/', { method: 'POST' });
  assert.equal(result.status, 405);
  assert.equal(result.headers.allow, 'GET, HEAD');
});

test('HEAD returns headers, no body, and closes its file without streaming', async (t) => {
  const originalOpen = fs.open.bind(fs);
  let closed = false;
  t.mock.method(fs, 'open', async (...args) => {
    const file = await originalOpen(...args);
    const close = file.close.bind(file);
    file.createReadStream = () => { throw new Error('HEAD must not stream'); };
    file.close = async () => { await close(); closed = true; };
    return file;
  });
  const result = await get('/src/app.js', { method: 'HEAD' });
  assert.equal(result.status, 200);
  assert.equal(result.text, '');
  assert.equal(Number(result.headers['content-length']), Buffer.byteLength(fixtures.get('src/app.js')));
  await waitFor(() => closed);
});

test('permission failures return 403 and do not leak filesystem details', async (t) => {
  const mocked = t.mock.method(fs, 'open', async () => {
    throw Object.assign(new Error('PRIVATE_FIXTURE'), { code: 'EACCES' });
  });
  const result = await get('/src/app.js');
  assert.equal(result.status, 403);
  assert.equal(result.text, 'Forbidden');
  mocked.mock.restore();
  await healthy();
});

test('a file removed before opening returns 404 and the next request succeeds', async (t) => {
  const mocked = t.mock.method(fs, 'open', async () => {
    throw Object.assign(new Error('PRIVATE_FIXTURE'), { code: 'ENOENT' });
  });
  assert.equal((await get('/src/app.js')).status, 404);
  mocked.mock.restore();
  await healthy();
});

test('stat failures return 500 and release the opened descriptor', async (t) => {
  const originalOpen = fs.open.bind(fs);
  let closed = false;
  const mocked = t.mock.method(fs, 'open', async (...args) => {
    const file = await originalOpen(...args);
    const close = file.close.bind(file);
    file.stat = async () => { throw Object.assign(new Error('PRIVATE_FIXTURE'), { code: 'EIO' }); };
    file.close = async () => { await close(); closed = true; };
    return file;
  });
  const result = await get('/src/app.js');
  assert.equal(result.status, 500);
  assert.equal(result.text, 'Internal server error');
  await waitFor(() => closed);
  mocked.mock.restore();
  await healthy();
});

test('stream errors are contained and descriptors are released', async (t) => {
  const originalOpen = fs.open.bind(fs);
  let closed = false;
  const mocked = t.mock.method(fs, 'open', async (...args) => {
    const file = await originalOpen(...args);
    const close = file.close.bind(file);
    file.createReadStream = () => new Readable({
      read() { this.destroy(Object.assign(new Error('Read failed'), { code: 'EIO' })); },
    });
    file.close = async () => { await close(); closed = true; };
    return file;
  });
  await assert.rejects(get('/assets/failure.bin'));
  await waitFor(() => closed);
  mocked.mock.restore();
  await healthy();
});

test('client disconnects release file descriptors and do not kill the server', async (t) => {
  const target = join(root, 'assets/large.bin');
  const handle = await fs.open(target, 'w');
  await handle.truncate(16 * 1024 * 1024);
  await handle.close();
  const originalOpen = fs.open.bind(fs);
  let closed = false;
  const mocked = t.mock.method(fs, 'open', async (...args) => {
    const file = await originalOpen(...args);
    const close = file.close.bind(file);
    file.close = async () => { await close(); closed = true; };
    return file;
  });
  await new Promise((resolve, reject) => {
    const req = request({ hostname: '127.0.0.1', port, path: '/assets/large.bin', agent: false }, (res) => {
      res.once('data', () => { res.destroy(); resolve(); });
      res.on('error', reject);
    });
    req.setTimeout(4000, () => req.destroy(new Error('Request timed out')));
    req.on('error', reject);
    req.end();
  });
  await waitFor(() => closed);
  mocked.mock.restore();
  await healthy();
});

test('CLI rejects malformed and out-of-range ports', () => {
  for (const value of ['abc', '4173junk', '-1', '65536', '1.5', '']) {
    const result = spawnSync(process.execPath,
      [fileURLToPath(new URL('../scripts/serve.mjs', import.meta.url))],
      { env: { ...process.env, PORT: value }, encoding: 'utf8', timeout: 4000 });
    assert.equal(result.status, 1, value);
    assert.match(result.stderr, /PORT must be an integer/);
  }
});

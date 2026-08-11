import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';

import { createExactTableLoader } from '../src/exact-table.js';

async function testServer(handler) {
  const server = createServer(handler);
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Test server has no port.');
  return {
    server,
    url(path) { return new URL(path, `http://127.0.0.1:${address.port}`); },
  };
}

test('exact-table loads are cached per URL, not globally', async (context) => {
  const counts = new Map();
  const { server, url } = await testServer((request, response) => {
    counts.set(request.url, (counts.get(request.url) ?? 0) + 1);
    const value = request.url === '/first' ? 11 : 29;
    response.writeHead(200, { 'Content-Type': 'application/octet-stream' });
    response.end(Uint8Array.of(value));
  });
  context.after(() => new Promise((resolve) => server.close(resolve)));

  const load = createExactTableLoader((bytes) => bytes[0], 'Test table');
  const firstUrl = url('/first');
  const secondUrl = url('/second');
  const [first, firstAgain, second] = await Promise.all([
    load(firstUrl),
    load(firstUrl),
    load(secondUrl),
  ]);

  assert.equal(first, 11);
  assert.equal(firstAgain, 11);
  assert.equal(second, 29);
  assert.equal(counts.get('/first'), 1);
  assert.equal(counts.get('/second'), 1);
});

test('a failed exact-table load can be retried', async (context) => {
  let requests = 0;
  const { server, url } = await testServer((_request, response) => {
    requests += 1;
    if (requests === 1) {
      response.writeHead(503).end('retry');
      return;
    }
    response.writeHead(200, { 'Content-Type': 'application/octet-stream' });
    response.end(Uint8Array.of(73));
  });
  context.after(() => new Promise((resolve) => server.close(resolve)));

  const load = createExactTableLoader((bytes) => bytes[0], 'Test table');
  const target = url('/retry');
  await assert.rejects(load(target), /503/);
  assert.equal(await load(target), 73);
  assert.equal(requests, 2);
});

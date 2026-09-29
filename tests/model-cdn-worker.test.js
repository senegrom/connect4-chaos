import assert from 'node:assert/strict';
import test from 'node:test';
import { gzipSync } from 'node:zlib';

import worker from '../workers/model-cdn/src/index.js';

// Drives the model CDN Worker's fetch() against a fake R2 bucket shaped like
// the real binding, per the R2 Workers API reference and workerd's R2 client:
// - head(key) resolves to an R2Object (metadata, no body) or null;
// - get(key, { onlyIf }) resolves to null for a missing key, to an R2Object
//   without a body when an onlyIf condition fails, and otherwise to an
//   R2ObjectBody whose body is a ReadableStream of the stored bytes.
const MODEL = gzipSync(Buffer.from('a model stored gzipped, as the publisher writes it'));
const KEY = 'models/gen/model.onnx';
const PAGE = 'https://senegrom.github.io';

function bucket() {
  const calls = [];
  const describe = () => ({
    key: KEY, size: MODEL.length, etag: 'e1', httpEtag: '"e1"', uploaded: new Date(0),
    httpMetadata: { contentType: 'application/octet-stream', contentEncoding: 'gzip' }, customMetadata: {},
    writeHttpMetadata(headers) {
      headers.set('Content-Type', 'application/octet-stream');
      headers.set('Content-Encoding', 'gzip');
    },
  });
  return {
    calls,
    async head(key) {
      calls.push('head');
      return key === KEY ? describe() : null;
    },
    async get(key, { range, onlyIf } = {}) {
      calls.push('get');
      assert.equal(range, undefined, 'the Worker never asks R2 for a range');
      if (key !== KEY) return null;
      const tag = onlyIf instanceof Headers ? onlyIf.get('If-None-Match') : null;
      if (tag === '"e1"' || tag === '*') return describe();
      // Every failed condition comes back the same way: a bodyless object.
      const match = onlyIf instanceof Headers ? onlyIf.get('If-Match') : null;
      if (match && match !== '"e1"' && match !== '*') return describe();
      const since = onlyIf instanceof Headers ? Date.parse(onlyIf.get('If-Unmodified-Since') ?? '') : NaN;
      if (Number.isFinite(since) && since < 0) return describe();
      return { ...describe(), bodyUsed: false, body: new Response(MODEL).body };
    },
  };
}

function request(path, { method = 'GET', origin = PAGE, headers = {} } = {}) {
  const all = new Headers(headers);
  if (origin) all.set('Origin', origin);
  return { method, url: `https://connect4-model.example.workers.dev${path}`, headers: all };
}

async function send(path, options) {
  const env = { MODELS: bucket() };
  const response = await worker.fetch(request(path, options), env);
  const body = response.body ? Buffer.from(await response.arrayBuffer()) : null;
  return { response, body, calls: env.MODELS.calls };
}

test('a GET serves the stored gzip bytes as they are, with CORS and immutable caching', async () => {
  const { response, body } = await send(`/${KEY}`);
  assert.equal(response.status, 200);
  assert.ok(body.equals(MODEL));
  assert.equal(response.headers.get('Content-Encoding'), 'gzip');
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), PAGE);
  assert.equal(response.headers.get('Vary'), 'Origin');
  assert.match(response.headers.get('Cache-Control'), /immutable/);
  assert.equal(response.headers.get('ETag'), '"e1"');
  assert.equal(response.headers.get('Accept-Ranges'), null, 'ranges into a gzip stream are not offered');
});

test('a malformed percent escape is a 404, not an uncaught error', async () => {
  const { response } = await send('/models/gen/model%E0%A4%A.onnx');
  assert.equal(response.status, 404);
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), PAGE);
});

test('errors carry the CORS headers an allowed origin gets, and every response varies on Origin', async () => {
  const missing = await send('/models/gen/other.onnx');
  assert.equal(missing.response.status, 404);
  assert.equal(missing.response.headers.get('Access-Control-Allow-Origin'), PAGE);
  const outside = await send('/secrets/token');
  assert.equal(outside.response.status, 404);
  assert.deepEqual(outside.calls, [], 'keys outside the model tree never reach the bucket');
  const post = await send(`/${KEY}`, { method: 'POST' });
  assert.equal(post.response.status, 405);
  assert.equal(post.response.headers.get('Allow'), 'GET, HEAD, OPTIONS');
  assert.equal(post.response.headers.get('Access-Control-Allow-Origin'), PAGE);
  for (const origin of [null, 'https://elsewhere.example']) {
    for (const [path, method] of [[`/${KEY}`, 'GET'], ['/models/gen/other.onnx', 'GET'], [`/${KEY}`, 'POST'], [`/${KEY}`, 'OPTIONS']]) {
      const { response } = await send(path, { method, origin });
      assert.equal(response.headers.get('Vary'), 'Origin', `${method} ${path} from ${origin}`);
      assert.equal(response.headers.get('Access-Control-Allow-Origin'), null);
    }
  }
});

// A range indexes the stored gzip stream, and nothing decodes gzip from its
// middle: a browser got a fragment it could not read, labelled as gzip.
test('a Range header is ignored: the whole object, as a 200', async () => {
  for (const ignored of ['bytes=-4', 'bytes=2-5', `bytes=${MODEL.length}-`, 'bytes=0-1,4-5', 'lines=1-2']) {
    const { response, body, calls } = await send(`/${KEY}`, { headers: { Range: ignored } });
    assert.equal(response.status, 200, ignored);
    assert.ok(body.equals(MODEL), `${ignored} gets the whole object`);
    assert.equal(response.headers.get('Content-Range'), null);
    assert.deepEqual(calls, ['get'], 'no extra head() to size a range');
  }
  const preflight = await send(`/${KEY}`, { method: 'OPTIONS' });
  assert.equal(preflight.response.headers.get('Access-Control-Allow-Headers'), null);
});

test('a HEAD reads metadata only, and answers its conditions as a GET does', async () => {
  const { response, body, calls } = await send(`/${KEY}`, { method: 'HEAD' });
  assert.deepEqual(calls, ['head'], 'the body is never opened');
  assert.equal(response.status, 200);
  assert.equal(body, null);
  assert.equal(response.headers.get('Content-Length'), String(MODEL.length));
  assert.equal(response.headers.get('Content-Encoding'), 'gzip');
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), PAGE);
  const missing = await send('/models/gen/other.onnx', { method: 'HEAD' });
  assert.equal(missing.response.status, 404);
  // Measured on the live Worker before this: a HEAD said 200 to each of these.
  for (const [headers, status] of [
    [{ 'If-None-Match': 'W/"e1"' }, 304],
    [{ 'If-Modified-Since': 'Thu, 01 Jan 2099 00:00:00 GMT' }, 304],
    [{ 'If-Modified-Since': 'Wed, 31 Dec 1969 23:59:59 GMT' }, 200],
    [{ 'If-None-Match': '"other"', 'If-Modified-Since': 'Thu, 01 Jan 2099 00:00:00 GMT' }, 200],
    [{ 'If-Match': '"nope"' }, 412],
    [{ 'If-Unmodified-Since': 'Wed, 31 Dec 1969 23:59:59 GMT' }, 412],
    [{ 'If-Match': '"e1"' }, 200],
  ]) {
    const conditional = await send(`/${KEY}`, { method: 'HEAD', headers });
    assert.equal(conditional.response.status, status, JSON.stringify(headers));
    assert.equal(conditional.response.headers.get('Access-Control-Allow-Origin'), PAGE);
  }
});

test('a conditional GET that matches is a bodyless 304', async () => {
  const { response, body } = await send(`/${KEY}`, { headers: { 'If-None-Match': '"e1"' } });
  assert.equal(response.status, 304);
  assert.equal(body, null);
  assert.equal(response.headers.get('ETag'), '"e1"');
});

// An R2 exception thrown out of the Worker became the runtime's own 500,
// without CORS headers, so the page saw an opaque "Failed to fetch".
test('a storage failure is a readable 503 with CORS, not an opaque error', async () => {
  for (const [method, fail] of [['GET', 'get'], ['HEAD', 'head']]) {
    const env = { MODELS: { ...bucket(), [fail]: async () => { throw new Error('R2 internal error'); } } };
    const response = await worker.fetch(request(`/${KEY}`, { method }), env);
    assert.equal(response.status, 503, method);
    assert.equal(response.headers.get('Access-Control-Allow-Origin'), PAGE);
    assert.equal(response.headers.get('Vary'), 'Origin');
  }
});

// onlyIf answers every failed condition alike; HTTP keeps 304 for a cache's
// revalidation and answers a failed If-Match or If-Unmodified-Since with 412.
test('a failed If-Match or If-Unmodified-Since is a 412, a revalidation a 304', async () => {
  for (const headers of [{ 'If-Match': '"other"' }, { 'If-Match': 'W/"e1"' },
    { 'If-Unmodified-Since': 'Wed, 31 Dec 1969 23:59:59 GMT' }]) {
    const { response, body } = await send(`/${KEY}`, { headers });
    assert.equal(response.status, 412, JSON.stringify(headers));
    assert.equal(response.headers.get('Access-Control-Allow-Origin'), PAGE);
    assert.notEqual(body?.length, MODEL.length);
  }
  for (const headers of [{ 'If-Match': '"e1"' }, { 'If-Match': '*' }, { 'If-Unmodified-Since': 'Thu, 01 Jan 2099 00:00:00 GMT' }]) {
    const { response, body } = await send(`/${KEY}`, { headers });
    assert.equal(response.status, 200, JSON.stringify(headers));
    assert.deepEqual(body, MODEL);
  }
  assert.equal((await send(`/${KEY}`, { headers: { 'If-None-Match': '"e1"' } })).response.status, 304);
});

import assert from 'node:assert/strict';
import test from 'node:test';
import { gzipSync } from 'node:zlib';

import worker from '../workers/model-cdn/src/index.js';

// Drives the model CDN Worker's fetch() against a fake R2 bucket shaped like
// the real binding, per the R2 Workers API reference and workerd's R2 client:
// - head(key) resolves to an R2Object (metadata, no body) or null;
// - get(key, { range, onlyIf }) resolves to null for a missing key, to an
//   R2Object without a body when an onlyIf condition fails, and otherwise to
//   an R2ObjectBody whose body is a ReadableStream of the requested bytes;
// - a range is an R2Range ({ offset, length } or { suffix }) or the request's
//   Headers, whose Range header R2 parses itself;
// - the object's `range` is set on every get(), even without a range asked for.
// The documented R2Range type includes { suffix }, and nothing says a suffix
// request comes back as an offset, so that is what this bucket reports.
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
  const resolve = (range) => {
    if (range instanceof Headers) {
      const match = /^bytes=(\d*)-(\d*)$/.exec(range.get('Range') ?? '');
      if (!match) return { offset: 0, length: MODEL.length };
      if (match[1] === '') return { suffix: Number(match[2]) };
      range = { offset: Number(match[1]), length: match[2] === '' ? undefined : Number(match[2]) - Number(match[1]) + 1 };
    }
    if (!range) return { offset: 0, length: MODEL.length };
    if ('suffix' in range) return { suffix: range.suffix };
    if (range.offset >= MODEL.length) throw new RangeError('InvalidRange: the requested range is not satisfiable');
    return { offset: range.offset, length: range.length ?? MODEL.length - range.offset };
  };
  return {
    calls,
    async head(key) {
      calls.push('head');
      return key === KEY ? describe() : null;
    },
    async get(key, { range, onlyIf } = {}) {
      calls.push('get');
      if (key !== KEY) return null;
      const tag = onlyIf instanceof Headers ? onlyIf.get('If-None-Match') : null;
      if (tag === '"e1"' || tag === '*') return describe();
      const resolved = resolve(range);
      const [start, count] = 'suffix' in resolved
        ? [MODEL.length - Math.min(resolved.suffix, MODEL.length), Math.min(resolved.suffix, MODEL.length)]
        : [resolved.offset, resolved.length];
      return { ...describe(), range: resolved, bodyUsed: false,
        body: new Response(MODEL.subarray(start, start + count)).body };
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
  assert.match(response.headers.get('Access-Control-Expose-Headers'), /Content-Encoding/);
  assert.equal(response.headers.get('Vary'), 'Origin');
  assert.match(response.headers.get('Cache-Control'), /immutable/);
  assert.equal(response.headers.get('ETag'), '"e1"');
  assert.equal(response.headers.get('Content-Range'), null, 'a plain GET is not a range');
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

test('a suffix range is its last bytes as a 206, never a partial 200', async () => {
  const size = MODEL.length;
  for (const [asked, first, last] of [[4, size - 4, size - 1], [size + 100, 0, size - 1]]) {
    const { response, body } = await send(`/${KEY}`, { headers: { Range: `bytes=-${asked}` } });
    assert.equal(response.status, 206, `bytes=-${asked}`);
    assert.equal(response.headers.get('Content-Range'), `bytes ${first}-${last}/${size}`);
    assert.ok(body.equals(MODEL.subarray(first, last + 1)));
  }
  const empty = await send(`/${KEY}`, { headers: { Range: 'bytes=-0' } });
  assert.equal(empty.response.status, 416);
  assert.equal(empty.response.headers.get('Content-Range'), `bytes */${size}`);
});

test('other ranges are exact, unsatisfiable or ignored, and a whole body is always a 200', async () => {
  const size = MODEL.length;
  const exact = await send(`/${KEY}`, { headers: { Range: 'bytes=2-5' } });
  assert.equal(exact.response.status, 206);
  assert.equal(exact.response.headers.get('Content-Range'), `bytes 2-5/${size}`);
  assert.ok(exact.body.equals(MODEL.subarray(2, 6)));
  const open = await send(`/${KEY}`, { headers: { Range: `bytes=${size - 3}-` } });
  assert.equal(open.response.headers.get('Content-Range'), `bytes ${size - 3}-${size - 1}/${size}`);
  assert.ok(open.body.equals(MODEL.subarray(size - 3)));
  const past = await send(`/${KEY}`, { headers: { Range: `bytes=${size}-` } });
  assert.equal(past.response.status, 416);
  assert.equal(past.response.headers.get('Content-Range'), `bytes */${size}`);
  assert.equal(past.response.headers.get('Access-Control-Allow-Origin'), PAGE);
  for (const ignored of ['bytes=5-2', 'bytes=0-1,4-5', 'lines=1-2']) {
    const { response, body } = await send(`/${KEY}`, { headers: { Range: ignored } });
    assert.equal(response.status, 200, ignored);
    assert.ok(body.equals(MODEL), `${ignored} gets the whole object`);
  }
});

test('a HEAD reads metadata only, and a matching ETag is a 304', async () => {
  const { response, body, calls } = await send(`/${KEY}`, { method: 'HEAD' });
  assert.deepEqual(calls, ['head'], 'the body is never opened');
  assert.equal(response.status, 200);
  assert.equal(body, null);
  assert.equal(response.headers.get('Content-Length'), String(MODEL.length));
  assert.equal(response.headers.get('Content-Encoding'), 'gzip');
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), PAGE);
  const cached = await send(`/${KEY}`, { method: 'HEAD', headers: { 'If-None-Match': 'W/"e1"' } });
  assert.equal(cached.response.status, 304);
  const missing = await send('/models/gen/other.onnx', { method: 'HEAD' });
  assert.equal(missing.response.status, 404);
});

test('a conditional GET that matches is a bodyless 304', async () => {
  const { response, body } = await send(`/${KEY}`, { headers: { 'If-None-Match': '"e1"' } });
  assert.equal(response.status, 304);
  assert.equal(body, null);
  assert.equal(response.headers.get('ETag'), '"e1"');
});

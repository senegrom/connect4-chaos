/**
 * Serves the neural model out of R2 to the game on GitHub Pages.
 *
 * The model is the one asset the site cannot host well: at 106 MB it is
 * larger than GitHub's 100 MB file limit (hence the split it used to ship
 * as), and re-downloading it on every visit would spend the whole of Pages'
 * 100 GB monthly allowance on about 950 visitors. R2 charges nothing for
 * egress, so the bytes move here and the repository stops carrying them.
 *
 * Keys are versioned - `models/<generation>/model.onnx` - so a response can
 * be immutable and a new generation is a new key. Nothing is ever
 * overwritten, which also means a bad model can be rolled back by changing
 * one field in the site's manifest.
 *
 * The page is cross-origin isolated for returning visitors (COEP
 * require-corp, for multi-threaded WebAssembly), and a cors-mode fetch from
 * such a page needs its response to pass the CORS check - measured, rather
 * than assumed: CORS alone is enough, Cross-Origin-Resource-Policy is only
 * required for no-cors requests, which this is not.
 */

// Only these may read the bucket. A request from anywhere else still gets
// the bytes if it asks without CORS, but no page can read the result.
const ALLOWED_ORIGINS = [
  'https://senegrom.github.io',
];
// Local harnesses serve the site from an ephemeral port on the loopback.
const LOCAL_ORIGIN = /^http:\/\/(127\.0\.0\.1|localhost):\d+$/;

// Keys are opaque to the browser but not to us: only the model tree is
// readable, so the Worker can never be pointed at anything else the bucket
// might come to hold.
const KEY = /^models\/[A-Za-z0-9._-]{1,64}\/[A-Za-z0-9._-]{1,64}$/;

const YEAR = 60 * 60 * 24 * 365;

function allowedOrigin(request) {
  const origin = request.headers.get('Origin');
  if (!origin) return null;
  if (ALLOWED_ORIGINS.includes(origin) || LOCAL_ORIGIN.test(origin)) return origin;
  return null;
}

function corsHeaders(origin) {
  const headers = new Headers();
  // Whether a response carries the CORS headers depends on the Origin, so
  // every response says so - including errors and the ones for an origin
  // that is not allowed - or a cache could hand one origin's copy to another.
  headers.set('Vary', 'Origin');
  if (!origin) return headers;
  headers.set('Access-Control-Allow-Origin', origin);
  // Only a handful of response headers reach cross-origin script by default,
  // and Content-Encoding is not among them. The page needs it: the object is
  // stored gzipped, so Content-Length is the compressed size while the stream
  // yields the model's real length, and a progress bar told the compressed
  // figure runs past 100%.
  headers.set('Access-Control-Expose-Headers', 'Content-Encoding, Content-Length, Content-Range');
  return headers;
}

// Errors carry the same CORS headers as the bytes would, so the page reads a
// 404 as a 404 instead of an opaque network failure.
function plain(status, text, origin, extra = {}) {
  const headers = corsHeaders(origin);
  for (const [name, value] of Object.entries(extra)) headers.set(name, value);
  return new Response(text, { status, headers });
}

/** The object key a request names, or null when its path cannot be one. */
function objectKey(request) {
  let path;
  try {
    path = decodeURIComponent(new URL(request.url).pathname);
  } catch {
    return null; // a malformed percent escape names no object
  }
  const key = path.replace(/^\/+/, '');
  return KEY.test(key) ? key : null;
}

const UNSATISFIABLE = Symbol('unsatisfiable');

/**
 * The one byte range a Range header asks for, as R2's { offset, length }:
 * `first-last`, `first-` or the suffix `-count`. Several ranges, or a header
 * that does not parse, give null, and HTTP lets a server answer those with
 * the whole object. A range that starts past the end cannot be served.
 */
function byteRange(header, size) {
  const match = /^bytes=(\d*)-(\d*)$/i.exec(header.trim());
  if (!match || (match[1] === '' && match[2] === '')) return null;
  if (match[1] === '') {
    const count = Math.min(Number(match[2]), size);
    return count > 0 ? { offset: size - count, length: count } : UNSATISFIABLE;
  }
  const first = Number(match[1]);
  const last = match[2] === '' ? size - 1 : Math.min(Number(match[2]), size - 1);
  if (match[2] !== '' && Number(match[2]) < first) return null;
  return first < size ? { offset: first, length: last - first + 1 } : UNSATISFIABLE;
}

/**
 * Whether the request's If-Match (compared strongly) or, without one, its
 * If-Unmodified-Since failed against the object. `onlyIf` answers every failed
 * condition with the same bodyless object; HTTP answers these with 412 and
 * keeps 304 for a cache's If-None-Match or If-Modified-Since (RFC 9110 13.2.2).
 */
function preconditionFailed(request, object) {
  const match = request.headers.get('If-Match');
  if (match) {
    return match.trim() !== '*' && !match.split(',').some((tag) => tag.trim() === object.httpEtag);
  }
  const since = Date.parse(request.headers.get('If-Unmodified-Since') ?? '');
  return Number.isFinite(since) && Math.floor(object.uploaded.getTime() / 1000) * 1000 > since;
}

/** If-None-Match against an ETag, compared weakly as RFC 9110 asks. */
function noneMatch(header, etag) {
  if (!header) return false;
  if (header.trim() === '*') return true;
  return header.split(',').some((tag) => tag.trim().replace(/^W\//, '') === etag);
}

function objectHeaders(object, origin) {
  const headers = corsHeaders(origin);
  // Carries the content type and, importantly, the content encoding: R2
  // returns exactly the bytes it stores and compresses nothing on the fly,
  // so the model is stored gzipped and labelled as such.
  object.writeHttpMetadata(headers);
  headers.set('ETag', object.httpEtag);
  headers.set('Cache-Control', `public, max-age=${YEAR}, immutable`);
  headers.set('Accept-Ranges', 'bytes');
  return headers;
}

export default {
  async fetch(request, env) {
    const origin = allowedOrigin(request);

    if (request.method === 'OPTIONS') {
      const headers = corsHeaders(origin);
      headers.set('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
      headers.set('Access-Control-Allow-Headers', 'Range');
      headers.set('Access-Control-Max-Age', '86400');
      return new Response(null, { status: 204, headers });
    }
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return plain(405, 'Method not allowed', origin, { Allow: 'GET, HEAD, OPTIONS' });
    }

    const key = objectKey(request);
    if (key === null) return plain(404, 'Not found', origin);

    // The object is stored gzipped and says so, so its bytes are already in
    // their final form. Without `encodeBody: 'manual'` the runtime compresses
    // them a second time and leaves the header claiming one layer: the
    // browser would unwrap it once and hand the decoder a gzip stream. It
    // costs 37 stored bytes going out as 54, which is how it was caught.
    const manual = (status, body, headers) => new Response(body, { status, headers, encodeBody: 'manual' });

    try {
      return await serve(request, key, origin, manual, env);
    } catch {
      // Thrown out of here, an R2 failure became the runtime's own 500, which
      // carries no CORS headers: the page saw an opaque "Failed to fetch"
      // rather than the readable error this Worker promises.
      return plain(503, 'Model storage is unavailable', origin, { 'Retry-After': '30' });
    }
  },
};

/** The response to a GET or HEAD of `key`. An R2 failure propagates to
 * fetch(), which answers it with a readable 503. */
async function serve(request, key, origin, manual, env) {
  // A HEAD needs the metadata alone. get() would open the 99 MB body only
  // to drop it; head() takes no conditions, so the one a cache revalidates
  // with is checked here.
  if (request.method === 'HEAD') {
    const object = await env.MODELS.head(key);
    if (object === null) return plain(404, 'Not found', origin);
    const headers = objectHeaders(object, origin);
    if (noneMatch(request.headers.get('If-None-Match'), object.httpEtag)) return manual(304, null, headers);
    headers.set('Content-Length', String(object.size));
    return manual(200, null, headers);
  }

  // Ranges index the stored gzip stream, and nothing decodes gzip from the
  // middle, so a browser cannot resume a download with one: fetch() hands
  // the decoder whatever arrives. They are still served exactly. Each is
  // resolved here against the object's size, never read back from
  // `object.range`, which R2 reports even for a plain GET and whose
  // documented type includes a bare suffix, with no offset to put in a
  // Content-Range. A range is served as the bytes it names (206), as the
  // whole object when the header asks for several or does not parse (200),
  // or not at all when it starts past the end (416).
  let range = null;
  if (request.headers.has('Range')) {
    const stored = await env.MODELS.head(key);
    if (stored === null) return plain(404, 'Not found', origin);
    range = byteRange(request.headers.get('Range'), stored.size);
    if (range === UNSATISFIABLE) {
      return plain(416, 'Range not satisfiable', origin, { 'Content-Range': `bytes */${stored.size}` });
    }
  }
  const object = await env.MODELS.get(key, { range: range ?? undefined, onlyIf: request.headers });
  if (object === null) return plain(404, 'Not found', origin);

  const headers = objectHeaders(object, origin);
  // `onlyIf` turns a request whose condition failed into a bodyless object.
  if (!('body' in object) || object.body === null) {
    return preconditionFailed(request, object)
      ? plain(412, 'Precondition failed', origin)
      : manual(304, null, headers);
  }
  if (range) {
    headers.set('Content-Range', `bytes ${range.offset}-${range.offset + range.length - 1}/${object.size}`);
    return manual(206, object.body, headers);
  }
  return manual(200, object.body, headers);
}

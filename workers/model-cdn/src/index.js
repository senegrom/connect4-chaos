/**
 * Serves the neural model out of R2 to the game on GitHub Pages.
 *
 * The model is the one asset the site cannot host well: at 106 MB it is
 * larger than GitHub's 100 MB file limit (hence the split it used to ship
 * as), and re-downloading it on every visit would spend the whole of Pages'
 * 100 GB monthly allowance on about 950 visitors. R2 charges nothing for
 * egress, so the bytes move here and the repository stops carrying them.
 *
 * Keys are versioned - `models/<checkpoint>/<sha256>` as
 * scripts/publish-model-r2.mjs writes them, one per export; the first
 * release predates that and is `models/big504-808970a6d2/model.onnx` - so a
 * response can be immutable and every export is a new key. Nothing is ever
 * overwritten, so a bad model is rolled back by pointing the site at the
 * previous key again:
 * MODEL_OBJECT, MODEL_SHA256 and DOWNLOAD_BYTES.model in
 * src/neural-runtime.js, with assets/neural/model.json
 * (docs/NEURAL_MODEL_RELEASES.md).
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
  headers.set('Access-Control-Expose-Headers', 'Content-Encoding, Content-Length');
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

/** Whether a cache's copy is current: its If-None-Match or, without one,
 * its If-Modified-Since, compared at the one-second grain of HTTP dates. */
function notModified(request, object) {
  const tags = request.headers.get('If-None-Match');
  if (tags) return noneMatch(tags, object.httpEtag);
  const since = Date.parse(request.headers.get('If-Modified-Since') ?? '');
  return Number.isFinite(since) && Math.floor(object.uploaded.getTime() / 1000) * 1000 <= since;
}

function objectHeaders(object, origin) {
  const headers = corsHeaders(origin);
  // Carries the content type and, importantly, the content encoding: R2
  // returns exactly the bytes it stores and compresses nothing on the fly,
  // so the model is stored gzipped and labelled as such.
  object.writeHttpMetadata(headers);
  headers.set('ETag', object.httpEtag);
  headers.set('Cache-Control', `public, max-age=${YEAR}, immutable`);
  return headers;
}

export default {
  async fetch(request, env) {
    const origin = allowedOrigin(request);

    if (request.method === 'OPTIONS') {
      const headers = corsHeaders(origin);
      headers.set('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
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
  // to drop it; head() takes no conditions, so they are checked here as a
  // GET's are: a failed If-Match or If-Unmodified-Since is a 412, a current
  // cached copy a 304. RFC 9110 has a HEAD answer as its GET would.
  if (request.method === 'HEAD') {
    const object = await env.MODELS.head(key);
    if (object === null) return plain(404, 'Not found', origin);
    if (preconditionFailed(request, object)) return plain(412, 'Precondition failed', origin);
    const headers = objectHeaders(object, origin);
    if (notModified(request, object)) return manual(304, null, headers);
    headers.set('Content-Length', String(object.size));
    return manual(200, null, headers);
  }

  // A Range header is ignored and the whole object served, as RFC 9110 14.2
  // allows. A range would index the stored gzip stream, and nothing decodes
  // gzip from its middle: a browser handed the bytes it asked for got a
  // fragment it could not read, labelled as gzip, and a client that does not
  // accept gzip got nothing once the edge had decoded the fragment.
  const object = await env.MODELS.get(key, { onlyIf: request.headers });
  if (object === null) return plain(404, 'Not found', origin);

  const headers = objectHeaders(object, origin);
  // `onlyIf` turns a request whose condition failed into a bodyless object.
  if (!('body' in object) || object.body === null) {
    return preconditionFailed(request, object)
      ? plain(412, 'Precondition failed', origin)
      : manual(304, null, headers);
  }
  return manual(200, object.body, headers);
}

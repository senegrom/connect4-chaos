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
  if (!origin) return headers;
  headers.set('Access-Control-Allow-Origin', origin);
  // Only a handful of response headers reach cross-origin script by default,
  // and Content-Encoding is not among them. The page needs it: the object is
  // stored gzipped, so Content-Length is the compressed size while the stream
  // yields the model's real length, and a progress bar told the compressed
  // figure runs past 100%.
  headers.set('Access-Control-Expose-Headers', 'Content-Encoding, Content-Length, Content-Range');
  // The allowed origin varies by request, so caches must key on it.
  headers.set('Vary', 'Origin');
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
      return new Response('Method not allowed', { status: 405, headers: { Allow: 'GET, HEAD, OPTIONS' } });
    }

    const key = decodeURIComponent(new URL(request.url).pathname).replace(/^\/+/, '');
    if (!KEY.test(key)) return new Response('Not found', { status: 404 });

    // Range support costs nothing here and lets the page resume an
    // interrupted download later without this Worker changing. Whether the
    // client asked has to be remembered: R2 reports a range on every object,
    // so trusting `object.range` answers 206 to a plain GET.
    const wantsRange = request.headers.has('Range');
    const object = await env.MODELS.get(key, {
      range: wantsRange ? request.headers : undefined,
      onlyIf: request.headers,
    });
    if (object === null) return new Response('Not found', { status: 404 });

    const headers = corsHeaders(origin);
    // Carries the content type and, importantly, the content encoding: R2
    // returns exactly the bytes it stores and compresses nothing on the fly,
    // so the model is stored gzipped and labelled as such.
    object.writeHttpMetadata(headers);
    headers.set('ETag', object.httpEtag);
    headers.set('Cache-Control', `public, max-age=${YEAR}, immutable`);
    headers.set('Accept-Ranges', 'bytes');

    // The object is stored gzipped and says so, so its bytes are already in
    // their final form. Without `encodeBody: 'manual'` the runtime compresses
    // them a second time and leaves the header claiming one layer: the
    // browser would unwrap it once and hand the decoder a gzip stream. It
    // costs 37 stored bytes going out as 54, which is how it was caught.
    const manual = (status, body) => new Response(body, { status, headers, encodeBody: 'manual' });

    // `onlyIf` turns a matching conditional request into a bodyless object.
    if (!('body' in object) || object.body === null) return manual(304, null);
    if (wantsRange && object.range && 'offset' in object.range) {
      const start = object.range.offset ?? 0;
      const length = object.range.length ?? (object.size - start);
      headers.set('Content-Range', `bytes ${start}-${start + length - 1}/${object.size}`);
      return manual(206, request.method === 'HEAD' ? null : object.body);
    }
    return manual(200, request.method === 'HEAD' ? null : object.body);
  },
};

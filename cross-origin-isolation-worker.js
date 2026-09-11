// A service worker whose only job is to add the two headers that make this
// page cross-origin isolated.
//
// Multi-threaded WebAssembly needs SharedArrayBuffer, which a browser grants
// only to a cross-origin isolated page - declared by response headers GitHub
// Pages does not let us set. A service worker can add them to what it serves,
// so the second load of the site is isolated and the neural opponent can use
// more than one core: measured on the shipped network, one position costs
// 410 ms on one thread and 167 ms on four.
//
// Deliberately a classic script with no imports or exports, because Safari -
// every iPhone and iPad, which is exactly who runs WebAssembly here - does
// not register module service workers. Nothing is cached and no response is
// rewritten beyond those headers; `?coi=off` unregisters it.

const HEADERS = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  // Every asset here is same-origin; this lets those responses satisfy the
  // embedder policy the header above imposes.
  'Cross-Origin-Resource-Policy': 'same-origin',
};

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));
self.addEventListener('message', (event) => {
  if (event.data?.type === 'coi-off') self.registration.unregister();
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  // A cache-only range request must pass through untouched.
  if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') return;
  // Only this origin's own responses are rewritten. The model comes from
  // another origin, and stamping Cross-Origin-Resource-Policy: same-origin on
  // someone else's response is exactly the instruction to block it - the
  // header is enforced whether or not the page is isolated, so this broke the
  // download on every visit, not just isolated ones. A cross-origin response
  // needs no help from here: its own CORS headers already satisfy the
  // embedder policy for the kind of request the loader makes.
  if (new URL(request.url).origin !== self.location.origin) return;
  event.respondWith(fetch(request).then((response) => {
    // An opaque response has no readable body or headers to copy.
    if (response.status === 0 || response.type === 'opaque') return response;
    const headers = new Headers(response.headers);
    for (const name of Object.keys(HEADERS)) headers.set(name, HEADERS[name]);
    return new Response(response.body, {
      status: response.status, statusText: response.statusText, headers,
    });
  }));
});

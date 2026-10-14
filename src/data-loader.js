import { sha256Hex } from './sha256.js';

/** Bounded data transport, shared by every AI catalog and binary loader.
 *
 * A load is bounded by silence, not by its length: every chunk that arrives
 * re-arms the deadline, so a slow connection still finishes and a stalled one
 * still fails. A fixed total deadline set a speed below which a large table
 * could never load - the 21 MB Brutal 6x7 Chaos layer needed 2.8 Mbit/s to
 * beat 60 s - and each Retry started again from the first byte. */
const DATA_LOAD_TIMEOUT_MS = 60_000;
export const CATALOG_LOAD_TIMEOUT_MS = 10_000;

function abortError() { return new DOMException('Data loading was cancelled.', 'AbortError'); }

// A compressed body's Content-Length counts compressed bytes, not the ones read.
function knownTotal(response, expectedBytes) {
  if (expectedBytes > 0) return expectedBytes;
  if (response.headers?.get?.('content-encoding')) return 0;
  return Number(response.headers?.get?.('content-length')) || 0;
}

async function streamedBytes(response, arrived, report, total) {
  const reader = response.body?.getReader?.();
  // Without a stream the deadline armed at the headers bounds the whole body.
  if (!reader) return new Uint8Array(await response.arrayBuffer());
  const chunks = [];
  let loaded = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      arrived();
      report(loaded, total);
    }
  } catch (error) {
    void reader.cancel().catch(() => {});
    throw error;
  }
  if (chunks.length === 1) return chunks[0];
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

/** Reads url as bytes, or as JSON. `timeoutMs` is the longest silence
 * allowed; `onDataProgress(loaded, total)` hears every chunk, where total is
 * `expectedBytes` when the caller knows the size and 0 when nobody does. It
 * also hears loaded = 0 whenever the deadline re-arms without data - as the
 * load starts and when its headers arrive - so a page watchdog re-armed by
 * each report never trails this deadline and ends a load that is still
 * inside it. `cache` is the request's cache mode. A catalog keeps its URL
 * from one release to the next and passes 'no-cache': Pages lets a browser
 * reuse a file for ten minutes, and in those a page of the new build was
 * handed the old catalog. Revalidating costs one conditional request. */
export async function readData(url, label, options = {}) {
  const {
    signal, timeoutMs = DATA_LOAD_TIMEOUT_MS, json = false, onDataProgress = null, expectedBytes = 0, cache,
  } = options;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new RangeError('Data timeout must be positive.');
  if (signal?.aborted) throw abortError();
  const target = url instanceof URL ? url : new URL(String(url), import.meta.url);
  const controller = new AbortController();
  let deadline;
  let fail;
  const interrupted = new Promise((_, reject) => { fail = reject; });
  const onAbort = () => { fail(abortError()); controller.abort(); };
  signal?.addEventListener('abort', onAbort, { once: true });
  const arrived = () => {
    clearTimeout(deadline);
    deadline = setTimeout(() => {
      fail(new Error(`${label} did not finish loading: nothing arrived for `
        + `${Math.max(1, Math.round(timeoutMs / 1000))} seconds.`));
      controller.abort();
    }, timeoutMs);
  };
  const report = (loaded, total) => {
    try { onDataProgress?.(loaded, total); } catch { /* progress cannot fail a load */ }
  };
  arrived();
  report(0, expectedBytes > 0 ? expectedBytes : 0);
  const work = async () => {
    if (target.protocol === 'file:' && typeof process !== 'undefined' && process.versions?.node) {
      const { readFile } = await import('node:fs/promises');
      const bytes = await readFile(target, { signal: controller.signal });
      return json ? JSON.parse(bytes.toString('utf8')) : new Uint8Array(bytes);
    }
    const response = await fetch(target, { signal: controller.signal, cache });
    if (!response.ok) throw new Error(`Could not load ${label.toLowerCase()} (${response.status}).`);
    const total = knownTotal(response, expectedBytes);
    arrived();
    report(0, total);
    // Catalogs are small: the deadline armed at the headers covers them.
    // Promise.race also bounds a transport that never reacts to the abort.
    return json ? response.json() : streamedBytes(response, arrived, report, total);
  };
  try { return await Promise.race([work(), interrupted]); }
  finally {
    clearTimeout(deadline);
    signal?.removeEventListener('abort', onAbort);
  }
}

/** Cache successes across requests; pending loads belong to their abort scope. */
export function cachedDataLoad(cache, key, loader, { signal, force = false } = {}) {
  if (signal?.aborted) return Promise.reject(abortError());
  const previous = force ? null : cache.get(key);
  if (previous?.ready) return Promise.resolve(previous.value);
  if (previous && previous.signal === signal && !previous.signal?.aborted) return previous.promise;
  const entry = { signal, ready: false };
  entry.promise = Promise.resolve().then(loader).then((value) => {
    if (signal?.aborted) throw abortError();
    entry.ready = true;
    entry.value = value;
    return value;
  }).catch((error) => {
    if (cache.get(key) === entry) cache.delete(key);
    throw error;
  });
  cache.set(key, entry);
  return entry.promise;
}

/** `url` with the table's release named in its query, by the start of its
 * SHA-256. A table regenerated in place keeps its path, and Pages lets a
 * browser reuse a file for ten minutes: a page of the new build, pinned to
 * the new digest, was handed the old bytes, and every Retry failed on them
 * until the copy expired. The server ignores the query. */
export function releaseUrl(url, sha256) {
  const target = new URL(url);
  target.searchParams.set('v', sha256.slice(0, 12));
  return target;
}

/** Loads the policy a catalog lists for one configuration, or null where it
 * lists none. `find(manifest)` picks the entry, and nothing is returned or
 * cached until the bytes match its length and SHA-256 and `decode(bytes)`
 * matches each of its `fields`. `options.manifest` stands in for the
 * catalog, and `options.bytes` for the download: the page transfers a
 * table it fetched behind its own prompt. */
export async function loadManifestPolicy(loads, {
  label, defaultManifestUrl, loadManifest, find, decode, fields,
}, options = {}) {
  const manifestUrl = options.manifestUrl
    ? new URL(String(options.manifestUrl), import.meta.url)
    : defaultManifestUrl;
  const manifest = options.manifest ?? await loadManifest(manifestUrl, options);
  const entry = find(manifest);
  if (!entry) return null;
  // Both catalogs' formats start with a 24-byte header.
  if (!Number.isInteger(entry.bytes) || entry.bytes < 24
      || typeof entry.sha256 !== 'string' || !/^[0-9a-f]{64}$/i.test(entry.sha256)) {
    throw new Error(`${label} manifest has invalid artifact metadata.`);
  }
  const target = releaseUrl(new URL(entry.file, manifestUrl), entry.sha256);
  return cachedDataLoad(loads, [target.href, entry.bytes, entry.sha256].join('|'), async () => {
    // The catalog's size is the progress total: a compressed response's
    // Content-Length counts compressed bytes.
    const bytes = options.bytes !== undefined
      ? new Uint8Array(options.bytes)
      : await readData(target, `${label} policy`, { ...options, expectedBytes: entry.bytes });
    if (bytes.byteLength !== entry.bytes) {
      throw new Error(`${label} policy length mismatch: expected ${entry.bytes}, found ${bytes.byteLength}.`);
    }
    if (await sha256Hex(bytes, `${label} verification`) !== entry.sha256.toLowerCase()) {
      throw new Error(`${label} policy SHA-256 does not match its manifest.`);
    }
    const policy = decode(bytes);
    if (fields.some((field) => policy[field] !== entry[field])) {
      throw new Error(`${label} policy metadata does not match its manifest.`);
    }
    return policy;
  }, options);
}

/** Lives on the page, so even a worker stalled during decoding can be replaced.
 * Bounded by silence too: the returned stop() ends it, and stop.touch()
 * re-arms it whenever the worker reports that data is still arriving. A
 * long-running *search* has no deadline: only its loading phase is bounded. */
export function loadingWatchdog(onTimeout, { signal, timeoutMs = 65_000 } = {}) {
  let timer;
  let stopped = Boolean(signal?.aborted);
  const stop = () => {
    stopped = true;
    clearTimeout(timer);
    signal?.removeEventListener('abort', stop);
  };
  const arm = () => {
    clearTimeout(timer);
    timer = setTimeout(() => { stop(); onTimeout(); }, timeoutMs);
  };
  stop.touch = () => { if (!stopped) arm(); };
  if (stopped) return stop;
  arm();
  signal?.addEventListener('abort', stop, { once: true });
  return stop;
}

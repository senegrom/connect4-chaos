import { fetchWithProgress } from './download-gate.js';
import { throwIfAborted, waitFor } from './async-control.js';
import { modelIdentity, verifyModelBytes, ModelIntegrityError } from './model-integrity.js';

const MODEL_CACHE = 'connect4-neural-model';
const CACHE_TIMEOUT_MS = 5_000;

function availableStorage() {
  try { return globalThis.caches; } catch { return null; }
}

function storageOperation(work, signal) {
  throwIfAborted(signal);
  return waitFor(Promise.resolve().then(work), {
    signal, timeoutMs: CACHE_TIMEOUT_MS, label: 'Model cache',
  });
}

/** Verify every cache read, including the first read by a replacement worker.
 * Storage is optional; its failures must not turn into a failed game. */
async function storedModel(release, storage, signal) {
  if (!storage) return null;
  let store;
  try {
    store = await storageOperation(() => storage.open(MODEL_CACHE), signal);
    const hit = await storageOperation(() => store.match(release.url), signal);
    if (!hit) return null;
    const bytes = await storageOperation(() => hit.arrayBuffer(), signal);
    await verifyModelBytes(bytes, release);
    throwIfAborted(signal);
    return bytes;
  } catch (error) {
    throwIfAborted(signal);
    if (error instanceof ModelIntegrityError && store) {
      // Discard only a corrupt model, not a valid one after an unrelated GPU failure.
      await storageOperation(() => store.delete(release.url), signal).catch(() => {});
    }
    throwIfAborted(signal);
    return null;
  }
}

async function rememberModel(release, bytes, storage, signal) {
  if (!storage) return;
  try {
    const store = await storageOperation(() => storage.open(MODEL_CACHE), signal);
    // This function is reached only after verification. Write before evicting
    // older releases, so a quota error cannot destroy a previously valid copy.
    await storageOperation(() => store.put(release.url, new Response(bytes, {
      headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(bytes.byteLength) },
    })), signal);
    const stale = (await storageOperation(() => store.keys(), signal))
      .filter((request) => request.url !== release.url);
    await storageOperation(() => Promise.all(stale.map((request) => store.delete(request))), signal);
  } catch {
    throwIfAborted(signal);
    // Storage refused it; already-verified bytes can still be used this visit.
  }
}

/** The expected digest is supplied by the app's pinned release, not the CDN.
 * Dependencies may be replaced by offline tests; runtime callers use defaults. */
export async function fetchVerifiedModel(model, {
  signal, onProgress, storage = availableStorage(), download = fetchWithProgress,
} = {}) {
  const release = Object.freeze({ ...modelIdentity(model), url: String(model.url) });
  throwIfAborted(signal);
  const kept = await storedModel(release, storage, signal);
  throwIfAborted(signal);
  if (kept) {
    onProgress?.(release.bytes, release.bytes);
    return kept;
  }
  const bytes = new Uint8Array(release.bytes);
  const written = await download(release.url, (loaded) => onProgress?.(loaded, release.bytes), {
    signal, expectedBytes: release.bytes, into: bytes, offset: 0,
  });
  if (written !== release.bytes) throw new ModelIntegrityError(
    `Model length mismatch: downloaded ${written}, expected ${release.bytes}.`);
  // Length alone is insufficient: same-sized corrupt or wrong-generation
  // data must never be persisted and supplied to Retry indefinitely.
  throwIfAborted(signal);
  await verifyModelBytes(bytes, release);
  throwIfAborted(signal);
  await rememberModel(release, bytes, storage, signal);
  throwIfAborted(signal);
  return bytes.buffer;
}

const EXACT_TABLE_TIMEOUT_MS = 60_000;

async function readExactTableBytes(url, label, timeoutMs) {
  if (url.protocol === 'file:' && typeof process !== 'undefined' && process.versions?.node) {
    const { readFile } = await import('node:fs/promises');
    return readFile(url);
  }

  const controller = new AbortController();
  const deadline = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) throw new Error(`Could not load ${label.toLowerCase()} (${response.status}).`);
    return new Uint8Array(await response.arrayBuffer());
  } catch (error) {
    if (error?.name === 'AbortError') {
      const seconds = Math.max(1, Math.round(timeoutMs / 1000));
      throw new Error(`${label} did not finish loading within ${seconds} second${seconds === 1 ? '' : 's'}.`);
    }
    throw error;
  } finally {
    clearTimeout(deadline);
  }
}

export function createExactTableLoader(decode, label, options = {}) {
  const timeoutMs = options.timeoutMs ?? EXACT_TABLE_TIMEOUT_MS;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new RangeError('Exact-table timeout must be a positive number.');
  }
  const promises = new Map();
  return function load(url) {
    const target = url instanceof URL ? url : new URL(String(url), import.meta.url);
    const key = target.href;
    let promise = promises.get(key);
    if (!promise) {
      promise = readExactTableBytes(target, label, timeoutMs).then(decode);
      promises.set(key, promise);
      promise.catch(() => {
        if (promises.get(key) === promise) promises.delete(key);
      });
    }
    return promise;
  };
}

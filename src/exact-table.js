const EXACT_TABLE_TIMEOUT_MS = 60_000;

async function readExactTableBytes(url, label) {
  if (url.protocol === 'file:' && typeof process !== 'undefined' && process.versions?.node) {
    const { readFile } = await import('node:fs/promises');
    return readFile(url);
  }

  const controller = new AbortController();
  const deadline = setTimeout(() => controller.abort(), EXACT_TABLE_TIMEOUT_MS);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) throw new Error(`Could not load ${label.toLowerCase()} (${response.status}).`);
    return new Uint8Array(await response.arrayBuffer());
  } catch (error) {
    if (error?.name === 'AbortError') {
      throw new Error(`${label} did not finish loading within one minute.`);
    }
    throw error;
  } finally {
    clearTimeout(deadline);
  }
}

export function createExactTableLoader(decode, label) {
  const promises = new Map();
  return function load(url) {
    const target = url instanceof URL ? url : new URL(String(url), import.meta.url);
    const key = target.href;
    let promise = promises.get(key);
    if (!promise) {
      promise = readExactTableBytes(target, label).then(decode);
      promises.set(key, promise);
      promise.catch(() => {
        if (promises.get(key) === promise) promises.delete(key);
      });
    }
    return promise;
  };
}

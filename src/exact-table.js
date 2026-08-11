const FORMAT_VERSION = 1;
const HEADER_SIZE = 12;
const ENTRY_SIZE = 10;

export const STANDARD_POSITION_KEY_LIMIT = 1n << 49n;

function bytesFrom(input, label) {
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  }
  throw new TypeError(`${label} data must be an ArrayBuffer or typed array.`);
}

function ascii(bytes, offset, length) {
  let value = '';
  for (let index = 0; index < length; index += 1) {
    value += String.fromCharCode(bytes[offset + index]);
  }
  return value;
}

export function decodeExactTable(input, options) {
  const {
    magic,
    label,
    readMetadata,
    validMoveMask,
    moveMaskError,
    validKey = () => true,
    keyError = `${label} contains a key outside the standard board.`,
  } = options;
  const bytes = bytesFrom(input, label);
  if (bytes.byteLength < HEADER_SIZE) throw new Error(`${label} data is truncated.`);
  if (ascii(bytes, 0, 4) !== magic) throw new Error(`${label} magic is invalid.`);

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint8(4);
  const entrySize = view.getUint8(6);
  const entryCount = view.getUint32(8, true);
  if (version !== FORMAT_VERSION) {
    throw new Error(`Unsupported ${label.toLowerCase()} version ${version}.`);
  }
  if (entrySize !== ENTRY_SIZE) {
    throw new Error(`Unsupported ${label.toLowerCase()} entry size ${entrySize}.`);
  }

  const metadata = readMetadata(view);
  const expectedLength = HEADER_SIZE + entryCount * ENTRY_SIZE;
  if (bytes.byteLength !== expectedLength) {
    throw new Error(
      `${label} length mismatch: expected ${expectedLength}, found ${bytes.byteLength}.`,
    );
  }

  let previousKey = -1n;
  for (let index = 0; index < entryCount; index += 1) {
    const offset = HEADER_SIZE + index * ENTRY_SIZE;
    const key = view.getBigUint64(offset, true);
    const moveMask = view.getUint8(offset + 8);
    const outcome = view.getInt8(offset + 9);
    if (key <= previousKey) throw new Error(`${label} keys must be strictly increasing.`);
    if (!validKey(key)) throw new Error(keyError);
    if (!validMoveMask(moveMask)) throw new Error(moveMaskError);
    if (outcome < -1 || outcome > 1) {
      throw new Error(`${label} outcomes must be -1, 0, or 1.`);
    }
    previousKey = key;
  }

  return Object.freeze({
    ...metadata,
    version,
    entryCount,
    byteLength: bytes.byteLength,
    lookup(key) {
      if (typeof key !== 'bigint' || key < 0n) return null;

      let low = 0;
      let high = entryCount - 1;
      while (low <= high) {
        const middle = (low + high) >> 1;
        const offset = HEADER_SIZE + middle * ENTRY_SIZE;
        const candidate = view.getBigUint64(offset, true);
        if (candidate === key) {
          return Object.freeze({
            key: candidate,
            moveMask: view.getUint8(offset + 8),
            outcome: view.getInt8(offset + 9),
          });
        }
        if (candidate < key) low = middle + 1;
        else high = middle - 1;
      }
      return null;
    },
  });
}

async function readExactTableBytes(url, label) {
  if (url.protocol === 'file:' && typeof process !== 'undefined' && process.versions?.node) {
    const { readFile } = await import('node:fs/promises');
    return readFile(url);
  }

  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${label.toLowerCase()} (${response.status}).`);
  return new Uint8Array(await response.arrayBuffer());
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

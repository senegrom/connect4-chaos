const MAGIC = 'C4PB';
const FORMAT_VERSION = 1;
const HEADER_SIZE = 12;
const ENTRY_SIZE = 10;

function bytesFrom(input) {
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  }
  throw new TypeError('Perfect-book data must be an ArrayBuffer or typed array.');
}

function ascii(bytes, offset, length) {
  let value = '';
  for (let index = 0; index < length; index += 1) {
    value += String.fromCharCode(bytes[offset + index]);
  }
  return value;
}

export function decodePerfectBook(input) {
  const bytes = bytesFrom(input);
  if (bytes.byteLength < HEADER_SIZE) throw new Error('Perfect-book data is truncated.');
  if (ascii(bytes, 0, 4) !== MAGIC) throw new Error('Perfect-book magic is invalid.');

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint8(4);
  const maxPly = view.getUint8(5);
  const entrySize = view.getUint8(6);
  const entryCount = view.getUint32(8, true);

  if (version !== FORMAT_VERSION) {
    throw new Error(`Unsupported perfect-book version ${version}.`);
  }
  if (entrySize !== ENTRY_SIZE) {
    throw new Error(`Unsupported perfect-book entry size ${entrySize}.`);
  }

  const expectedLength = HEADER_SIZE + entryCount * entrySize;
  if (bytes.byteLength !== expectedLength) {
    throw new Error(
      `Perfect-book length mismatch: expected ${expectedLength}, found ${bytes.byteLength}.`,
    );
  }

  let previousKey = -1n;
  for (let index = 0; index < entryCount; index += 1) {
    const offset = HEADER_SIZE + index * ENTRY_SIZE;
    const key = view.getBigUint64(offset, true);
    const moveMask = view.getUint8(offset + 8);
    const outcome = view.getInt8(offset + 9);
    if (key <= previousKey) throw new Error('Perfect-book keys must be strictly increasing.');
    if (moveMask === 0 || (moveMask & 0x80) !== 0) {
      throw new Error('Perfect-book move masks must contain at least one of seven columns.');
    }
    if (outcome < -1 || outcome > 1) {
      throw new Error('Perfect-book outcomes must be -1, 0, or 1.');
    }
    previousKey = key;
  }

  return Object.freeze({
    version,
    maxPly,
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

async function readBookBytes(url) {
  if (url.protocol === 'file:' && typeof process !== 'undefined' && process.versions?.node) {
    const { readFile } = await import('node:fs/promises');
    return readFile(url);
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load perfect-play book (${response.status}).`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

let defaultBookPromise = null;

export function loadPerfectBook(
  url = new URL('../assets/perfect-book.bin', import.meta.url),
) {
  if (!defaultBookPromise) {
    defaultBookPromise = readBookBytes(url).then(decodePerfectBook);
  }
  return defaultBookPromise;
}

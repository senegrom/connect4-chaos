import { ascii, bytesFrom } from './bytes.js';
import { readData, cachedDataLoad } from './data-loader.js';

const FORMAT_VERSION = 1;
const HEADER_SIZE = 12;
const ENTRY_SIZE = 10;
// Both tables hold standard 6x7 positions: seven columns of seven key bits.
const STANDARD_POSITION_KEY_LIMIT = 1n << 49n;

export function decodeExactTable(input, options) {
  const { magic, label, readMetadata, validMoveMask, moveMaskError } = options;
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
    if (key >= STANDARD_POSITION_KEY_LIMIT) {
      throw new Error(`${label} contains a key outside the standard board.`);
    }
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

/** A cached loader of one decoded table per URL. Its loads use the data
 * loader's own silence deadline - the one the browser stall tests shorten -
 * unless a request passes timeoutMs. */
export function createExactTableLoader(decode, label) {
  const cache = new Map();
  return function load(url, requestOptions = {}) {
    const target = url instanceof URL ? url : new URL(String(url), import.meta.url);
    return cachedDataLoad(cache, target.href,
      () => readData(target, label, requestOptions).then(decode), requestOptions);
  };
}

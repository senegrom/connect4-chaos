const MAGIC = 'C4PS';
const FORMAT_VERSION = 1;
const HEADER_SIZE = 12;
const ENTRY_SIZE = 10;

export const PERFECT_ROLE_FIRST = 1;
export const PERFECT_ROLE_SECOND = 2;
export const PERFECT_ROLE_BOTH = PERFECT_ROLE_FIRST | PERFECT_ROLE_SECOND;

function bytesFrom(input) {
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  }
  throw new TypeError('Perfect-strategy data must be an ArrayBuffer or typed array.');
}

function ascii(bytes, offset, length) {
  let value = '';
  for (let index = 0; index < length; index += 1) {
    value += String.fromCharCode(bytes[offset + index]);
  }
  return value;
}

export function decodePerfectStrategy(input) {
  const bytes = bytesFrom(input);
  if (bytes.byteLength < HEADER_SIZE) throw new Error('Perfect-strategy data is truncated.');
  if (ascii(bytes, 0, 4) !== MAGIC) throw new Error('Perfect-strategy magic is invalid.');

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint8(4);
  const handoffRemaining = view.getUint8(5);
  const entrySize = view.getUint8(6);
  const roleFlags = view.getUint8(7);
  const entryCount = view.getUint32(8, true);

  if (version !== FORMAT_VERSION) {
    throw new Error(`Unsupported perfect-strategy version ${version}.`);
  }
  if (entrySize !== ENTRY_SIZE) {
    throw new Error(`Unsupported perfect-strategy entry size ${entrySize}.`);
  }
  if (handoffRemaining > 42) {
    throw new Error('Perfect-strategy handoff is outside the standard board.');
  }
  if (roleFlags === 0 || (roleFlags & ~PERFECT_ROLE_BOTH) !== 0) {
    throw new Error('Perfect-strategy role flags are invalid.');
  }

  const expectedLength = HEADER_SIZE + entryCount * entrySize;
  if (bytes.byteLength !== expectedLength) {
    throw new Error(
      `Perfect-strategy length mismatch: expected ${expectedLength}, found ${bytes.byteLength}.`,
    );
  }

  let previousKey = -1n;
  for (let index = 0; index < entryCount; index += 1) {
    const offset = HEADER_SIZE + index * ENTRY_SIZE;
    const key = view.getBigUint64(offset, true);
    const moveMask = view.getUint8(offset + 8);
    const outcome = view.getInt8(offset + 9);
    if (key <= previousKey) {
      throw new Error('Perfect-strategy keys must be strictly increasing.');
    }
    if (moveMask === 0 || (moveMask & (moveMask - 1)) !== 0 || (moveMask & 0x80) !== 0) {
      throw new Error('Perfect-strategy entries must contain exactly one of seven columns.');
    }
    if (outcome < -1 || outcome > 1) {
      throw new Error('Perfect-strategy outcomes must be -1, 0, or 1.');
    }
    previousKey = key;
  }

  return Object.freeze({
    version,
    handoffRemaining,
    roleFlags,
    entryCount,
    byteLength: bytes.byteLength,
    coversRole(role) {
      return Number.isInteger(role) && (roleFlags & role) !== 0;
    },
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

async function readStrategyBytes(url) {
  if (url.protocol === 'file:' && typeof process !== 'undefined' && process.versions?.node) {
    const { readFile } = await import('node:fs/promises');
    return readFile(url);
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load the perfect strategy (${response.status}).`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

let defaultStrategyPromise = null;

export function loadPerfectStrategy(
  url = new URL('../assets/perfect-strategy.bin', import.meta.url),
) {
  if (!defaultStrategyPromise) {
    defaultStrategyPromise = readStrategyBytes(url).then(decodePerfectStrategy);
  }
  return defaultStrategyPromise;
}

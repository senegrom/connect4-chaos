import { ACTION_DROP, EMPTY, RED, YELLOW } from './engine.js';

const MAGIC = 'C4VPOL1\0';
const FORMAT_VERSION = 1;
const HEADER_SIZE = 24;
const RECORD_SIZE = 10;
const DEFAULT_MANIFEST_URL = new URL('../data/perfect-classic/manifest.json', import.meta.url);

export const PERFECT_CLASSIC_ROLE_FIRST = 1;
export const PERFECT_CLASSIC_ROLE_SECOND = 2;

const MANIFEST_PROMISES = new Map();

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

function validateRole(role) {
  if (role !== PERFECT_CLASSIC_ROLE_FIRST && role !== PERFECT_CLASSIC_ROLE_SECOND) {
    throw new RangeError('Perfect classic policy role must be first or second.');
  }
  return role;
}

function geometry(rows, columns, connect) {
  if (!Number.isInteger(rows) || !Number.isInteger(columns)
      || rows < 1 || rows > 7 || columns < 1 || columns > 7) {
    throw new RangeError('Perfect classic policies support 1 through 7 rows and columns.');
  }
  if (!Number.isInteger(connect) || connect < 1 || connect > Math.max(rows, columns)) {
    throw new RangeError('Perfect classic policy connect length must fit the board.');
  }
  const stride = rows + 1;
  const columnBits = (1n << BigInt(rows)) - 1n;
  const columnWithSentinel = (1n << BigInt(stride)) - 1n;
  const bottomMasks = Array.from(
    { length: columns },
    (_, column) => 1n << BigInt(column * stride),
  );
  const columnMasks = bottomMasks.map((bottom) => bottom * columnBits);
  return {
    rows,
    columns,
    connect,
    stride,
    cellCount: rows * columns,
    keyLimit: 1n << BigInt(stride * columns),
    columnWithSentinel,
    bottomMasks,
    columnMasks,
  };
}

function moveForColumn(selected, mask, column) {
  if (!Number.isInteger(column) || column < 0 || column >= selected.columns) return 0n;
  return (mask + selected.bottomMasks[column]) & selected.columnMasks[column];
}

function mirrorBits(selected, bits) {
  let mirrored = 0n;
  for (let column = 0; column < selected.columns; column += 1) {
    const group = (bits >> BigInt(column * selected.stride)) & selected.columnWithSentinel;
    mirrored |= group << BigInt((selected.columns - 1 - column) * selected.stride);
  }
  return mirrored;
}

function canonicalPosition(selected, position) {
  const normal = position.current + position.mask;
  const mirroredCurrent = mirrorBits(selected, position.current);
  const mirroredMask = mirrorBits(selected, position.mask);
  const mirrored = mirroredCurrent + mirroredMask;
  return normal <= mirrored
    ? { key: normal, mirrored: false }
    : { key: mirrored, mirrored: true };
}

function encodeBoard(board, currentPlayer, connect) {
  if ((currentPlayer !== RED && currentPlayer !== YELLOW)
      || !Array.isArray(board) || board.length === 0
      || !Array.isArray(board[0]) || board[0].length === 0) return null;
  const rows = board.length;
  const columns = board[0].length;
  if (board.some((row) => !Array.isArray(row) || row.length !== columns)) return null;

  let selected;
  try {
    selected = geometry(rows, columns, connect);
  } catch {
    return null;
  }

  let current = 0n;
  let mask = 0n;
  let moves = 0;
  for (let column = 0; column < columns; column += 1) {
    let emptyBelow = false;
    for (let row = rows - 1; row >= 0; row -= 1) {
      const cell = board[row][column];
      if (cell === EMPTY) {
        emptyBelow = true;
        continue;
      }
      if (emptyBelow || (cell !== RED && cell !== YELLOW)) return null;
      const bit = 1n << BigInt(column * selected.stride + rows - 1 - row);
      mask |= bit;
      if (cell === currentPlayer) current |= bit;
      moves += 1;
    }
  }
  return { selected, current, mask, moves };
}

function moveMaskColumn(moveMask, columns) {
  if (!Number.isInteger(moveMask) || moveMask <= 0
      || (moveMask & (moveMask - 1)) !== 0
      || (moveMask & ~((1 << columns) - 1)) !== 0) return -1;
  for (let column = 0; column < columns; column += 1) {
    if ((moveMask & (1 << column)) !== 0) return column;
  }
  return -1;
}

export function perfectClassicRole(startingPlayer, aiPlayer) {
  if ((startingPlayer !== RED && startingPlayer !== YELLOW)
      || (aiPlayer !== RED && aiPlayer !== YELLOW)) return null;
  return startingPlayer === aiPlayer
    ? PERFECT_CLASSIC_ROLE_FIRST
    : PERFECT_CLASSIC_ROLE_SECOND;
}

export function decodePerfectClassicPolicy(input, expectations = {}) {
  const bytes = bytesFrom(input, 'Perfect classic policy');
  if (bytes.byteLength < HEADER_SIZE) throw new Error('Perfect classic policy is truncated.');
  if (ascii(bytes, 0, 8) !== MAGIC) throw new Error('Perfect classic policy magic is invalid.');

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint8(8);
  const rows = view.getUint8(9);
  const columns = view.getUint8(10);
  const connect = view.getUint8(11);
  const role = view.getUint8(12);
  const handoffRemaining = view.getUint8(13);
  const recordSize = view.getUint8(14);
  const rootValue = view.getInt8(15);
  const entryCount = view.getUint32(16, true);
  const closureStates = view.getUint32(20, true);
  const selected = geometry(rows, columns, connect);

  if (version !== FORMAT_VERSION) {
    throw new Error(`Unsupported perfect classic policy version ${version}.`);
  }
  validateRole(role);
  if (recordSize !== RECORD_SIZE) {
    throw new Error(`Unsupported perfect classic policy record size ${recordSize}.`);
  }
  if (handoffRemaining > selected.cellCount) {
    throw new Error('Perfect classic policy handoff exceeds the board size.');
  }
  if (rootValue < -1 || rootValue > 1) {
    throw new Error('Perfect classic policy root value must be -1, 0, or 1.');
  }
  if (expectations.rows !== undefined && expectations.rows !== rows
      || expectations.columns !== undefined && expectations.columns !== columns
      || expectations.connect !== undefined && expectations.connect !== connect
      || expectations.role !== undefined && expectations.role !== role) {
    throw new Error('Perfect classic policy metadata does not match the requested configuration.');
  }

  const expectedLength = HEADER_SIZE + entryCount * RECORD_SIZE;
  if (bytes.byteLength !== expectedLength) {
    throw new Error(
      `Perfect classic policy length mismatch: expected ${expectedLength}, found ${bytes.byteLength}.`,
    );
  }

  let previousKey = -1n;
  for (let index = 0; index < entryCount; index += 1) {
    const offset = HEADER_SIZE + index * RECORD_SIZE;
    const key = view.getBigUint64(offset, true);
    const moveMask = view.getUint8(offset + 8);
    const outcome = view.getInt8(offset + 9);
    if (key <= previousKey) throw new Error('Perfect classic policy keys must be strictly increasing.');
    if (key >= selected.keyLimit) throw new Error('Perfect classic policy contains an out-of-range key.');
    if (moveMaskColumn(moveMask, columns) < 0) {
      throw new Error('Perfect classic policy entries must select exactly one legal column bit.');
    }
    if (outcome < -1 || outcome > 1) {
      throw new Error('Perfect classic policy outcomes must be -1, 0, or 1.');
    }
    previousKey = key;
  }

  const policy = {
    version,
    rows,
    columns,
    connect,
    role,
    handoffRemaining,
    rootValue,
    entryCount,
    closureStates,
    byteLength: bytes.byteLength,
    lookupKey(key) {
      if (typeof key !== 'bigint' || key < 0n || key >= selected.keyLimit) return null;
      let low = 0;
      let high = entryCount - 1;
      while (low <= high) {
        const middle = (low + high) >> 1;
        const offset = HEADER_SIZE + middle * RECORD_SIZE;
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
    lookup(board, currentPlayer, aiPlayer, startingPlayer) {
      if (currentPlayer !== aiPlayer) return null;
      if (perfectClassicRole(startingPlayer, aiPlayer) !== role) return null;
      const encoded = encodeBoard(board, currentPlayer, connect);
      if (!encoded || encoded.selected.rows !== rows || encoded.selected.columns !== columns) return null;
      if (encoded.selected.cellCount - encoded.moves <= handoffRemaining) return null;

      const canonical = canonicalPosition(encoded.selected, encoded);
      const record = policy.lookupKey(canonical.key);
      if (!record) return null;
      let column = moveMaskColumn(record.moveMask, columns);
      if (canonical.mirrored) column = columns - 1 - column;
      if (moveForColumn(encoded.selected, encoded.mask, column) === 0n) {
        throw new Error('Perfect classic policy returned an illegal move.');
      }
      return Object.freeze({
        action: Object.freeze({ type: ACTION_DROP, column }),
        outcome: record.outcome,
        mirrored: canonical.mirrored,
      });
    },
  };
  return Object.freeze(policy);
}

function cachedLoad(cache, url, loader) {
  const key = url.href;
  let promise = cache.get(key);
  if (!promise) {
    promise = loader();
    cache.set(key, promise);
    promise.catch(() => {
      if (cache.get(key) === promise) cache.delete(key);
    });
  }
  return promise;
}

export function loadPerfectClassicManifest(url = DEFAULT_MANIFEST_URL) {
  const target = url instanceof URL ? url : new URL(String(url), import.meta.url);
  return cachedLoad(MANIFEST_PROMISES, target, async () => {
    let manifest;
    if (target.protocol === 'file:' && typeof process !== 'undefined' && process.versions?.node) {
      const { readFile } = await import('node:fs/promises');
      manifest = JSON.parse(await readFile(target, 'utf8'));
    } else {
      const response = await fetch(target);
      if (!response.ok) throw new Error(`Could not load perfect classic manifest (${response.status}).`);
      manifest = await response.json();
    }
    if (manifest?.format !== 'connect4-perfect-classic-manifest-v1'
        || !Array.isArray(manifest.policies)) {
      throw new Error('Perfect classic manifest format is invalid.');
    }
    const policies = manifest.policies.map((entry) => {
      geometry(entry.rows, entry.columns, entry.connect);
      validateRole(entry.role);
      if (typeof entry.file !== 'string' || entry.file.length === 0
          || !Number.isInteger(entry.handoffRemaining)
          || !Number.isInteger(entry.entryCount)
          || !Number.isInteger(entry.rootValue)) {
        throw new Error('Perfect classic manifest contains an invalid policy entry.');
      }
      return Object.freeze({ ...entry });
    });
    return Object.freeze({ ...manifest, policies: Object.freeze(policies) });
  });
}

export function findPerfectClassicPolicy(manifest, rows, columns, connect, role) {
  return manifest?.policies?.find((entry) => (
    entry.rows === rows
    && entry.columns === columns
    && entry.connect === connect
    && entry.role === role
  )) ?? null;
}

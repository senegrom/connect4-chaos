import {
  ACTION_DROP,
  ACTION_FLIP,
  ACTION_ROTATE_CCW,
  ACTION_ROTATE_CW,
  EMPTY,
  RED,
  YELLOW,
  boardDimensions,
} from './engine.js';

const MAGIC = 'C4CFUL1\0';
const FORMAT_VERSION = 1;
const HEADER_SIZE = 24;
const RECORD_SIZE = 24;
const DEFAULT_MANIFEST_URL = new URL(
  '../data/perfect-chaos-complete/manifest.json',
  import.meta.url,
);

export const PERFECT_CHAOS_COMPLETE_ROLE_FIRST = 1;
export const PERFECT_CHAOS_COMPLETE_ROLE_SECOND = 2;

const ACTION_CODES = Object.freeze([
  ACTION_DROP,
  ACTION_FLIP,
  ACTION_ROTATE_CW,
  ACTION_ROTATE_CCW,
]);

const MANIFEST_PROMISES = new Map();
const POLICY_PROMISES = new Map();

function bytesFrom(input) {
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  }
  throw new TypeError('Perfect Chaos policy data must be an ArrayBuffer or typed array.');
}

function ascii(bytes, offset, length) {
  let value = '';
  for (let index = 0; index < length; index += 1) {
    value += String.fromCharCode(bytes[offset + index]);
  }
  return value;
}

function validateRole(role) {
  if (role !== PERFECT_CHAOS_COMPLETE_ROLE_FIRST
      && role !== PERFECT_CHAOS_COMPLETE_ROLE_SECOND) {
    throw new RangeError('Perfect Chaos policy role must be first or second.');
  }
  return role;
}

/**
 * Packs a board into the mover-relative masks used by the certificate: one
 * sentinel bit per column, bit 0 of each column at the bottom of the board.
 */
function packBoard(board, mover) {
  const { rows, cols: columns } = boardDimensions(board);
  const stride = rows + 1;
  let moverBits = 0n;
  let opponentBits = 0n;
  for (let row = 0; row < rows; row += 1) {
    if (!Array.isArray(board[row]) || board[row].length !== columns) return null;
    for (let column = 0; column < columns; column += 1) {
      const cell = board[row][column];
      if (cell === EMPTY) continue;
      if (cell !== RED && cell !== YELLOW) return null;
      const bit = 1n << BigInt(column * stride + (rows - 1 - row));
      if (cell === mover) moverBits |= bit;
      else opponentBits |= bit;
    }
  }
  return { mover: moverBits, opponent: opponentBits, rows, columns };
}

function mirrorPacked(state) {
  const stride = state.rows + 1;
  const groupMask = (1n << BigInt(stride)) - 1n;
  const flip = (bits) => {
    let mirrored = 0n;
    for (let column = 0; column < state.columns; column += 1) {
      const group = (bits >> BigInt(column * stride)) & groupMask;
      mirrored |= group << BigInt((state.columns - 1 - column) * stride);
    }
    return mirrored;
  };
  return {
    mover: flip(state.mover),
    opponent: flip(state.opponent),
    rows: state.rows,
    columns: state.columns,
  };
}

function compareStates(first, second) {
  if (first.rows !== second.rows) return first.rows - second.rows;
  if (first.columns !== second.columns) return first.columns - second.columns;
  if (first.mover !== second.mover) return first.mover < second.mover ? -1 : 1;
  if (first.opponent !== second.opponent) return first.opponent < second.opponent ? -1 : 1;
  return 0;
}

function mirrorAction(action, columns) {
  if (action.type === ACTION_DROP) {
    return { type: ACTION_DROP, column: columns - 1 - action.column };
  }
  if (action.type === ACTION_ROTATE_CW) return { type: ACTION_ROTATE_CCW };
  if (action.type === ACTION_ROTATE_CCW) return { type: ACTION_ROTATE_CW };
  return { type: ACTION_FLIP };
}

export function perfectChaosCompleteRole(startingPlayer, aiPlayer) {
  if ((startingPlayer !== RED && startingPlayer !== YELLOW)
      || (aiPlayer !== RED && aiPlayer !== YELLOW)) return null;
  return startingPlayer === aiPlayer
    ? PERFECT_CHAOS_COMPLETE_ROLE_FIRST
    : PERFECT_CHAOS_COMPLETE_ROLE_SECOND;
}

/**
 * Decodes a complete Chaos Mode policy. Unlike the layered prefix certificate,
 * this covers every position reachable from the empty board under the selected
 * policy, so there is no piece-count frontier and no handoff to search: a
 * missing record for a reachable position is a defect, not a boundary.
 */
export function decodePerfectChaosCompletePolicy(input, expectations = {}) {
  const bytes = bytesFrom(input);
  if (bytes.byteLength < HEADER_SIZE) throw new Error('Perfect Chaos policy is truncated.');
  if (ascii(bytes, 0, 8) !== MAGIC) throw new Error('Perfect Chaos policy magic is invalid.');

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint8(8);
  const rows = view.getUint8(9);
  const columns = view.getUint8(10);
  const connect = view.getUint8(11);
  const role = view.getUint8(12);
  const rootValue = view.getInt8(13);
  const recordSize = view.getUint8(14);
  const entryCount = view.getUint32(16, true);
  const closureStates = view.getUint32(20, true);

  if (version !== FORMAT_VERSION) {
    throw new Error(`Unsupported Perfect Chaos policy version ${version}.`);
  }
  validateRole(role);
  if (recordSize !== RECORD_SIZE) {
    throw new Error(`Unsupported Perfect Chaos policy record size ${recordSize}.`);
  }
  if (!Number.isInteger(rows) || rows < 1 || rows > 7
      || !Number.isInteger(columns) || columns < 1 || columns > 7) {
    throw new Error('Perfect Chaos policy has unsupported board dimensions.');
  }
  if (connect < 1 || connect > Math.max(rows, columns)) {
    throw new Error('Perfect Chaos policy connect length does not fit the board.');
  }
  if (rootValue < -1 || rootValue > 1) {
    throw new Error('Perfect Chaos policy root value must be -1, 0, or 1.');
  }
  if (bytes.byteLength !== HEADER_SIZE + entryCount * RECORD_SIZE) {
    throw new Error(
      `Perfect Chaos policy length mismatch: expected `
      + `${HEADER_SIZE + entryCount * RECORD_SIZE}, found ${bytes.byteLength}.`,
    );
  }
  const inOrbit = (candidateRows, candidateColumns) => (
    (candidateRows === rows && candidateColumns === columns)
    || (candidateRows === columns && candidateColumns === rows)
  );
  if (expectations.connect !== undefined && expectations.connect !== connect) {
    throw new Error('Perfect Chaos policy metadata does not match the requested configuration.');
  }
  if (expectations.role !== undefined && expectations.role !== role) {
    throw new Error('Perfect Chaos policy metadata does not match the requested configuration.');
  }
  if (expectations.rows !== undefined && expectations.columns !== undefined
      && !inOrbit(expectations.rows, expectations.columns)) {
    throw new Error('Perfect Chaos policy metadata does not match the requested configuration.');
  }

  const recordAt = (index) => {
    const offset = HEADER_SIZE + index * RECORD_SIZE;
    return {
      mover: view.getBigUint64(offset, true),
      opponent: view.getBigUint64(offset + 8, true),
      rows: view.getUint8(offset + 16),
      columns: view.getUint8(offset + 17),
      actionCode: view.getUint8(offset + 18),
      column: view.getUint8(offset + 19),
      outcome: view.getInt8(offset + 20),
    };
  };

  let previous = null;
  for (let index = 0; index < entryCount; index += 1) {
    const record = recordAt(index);
    if ((record.mover & record.opponent) !== 0n) {
      throw new Error('Perfect Chaos policy record has overlapping pieces.');
    }
    // A rotation transposes the board, so a certificate legitimately contains
    // records in both orientations of its orbit.
    if (!inOrbit(record.rows, record.columns)) {
      throw new Error('Perfect Chaos policy record has unexpected dimensions.');
    }
    if (!ACTION_CODES[record.actionCode]) {
      throw new Error('Perfect Chaos policy record has an invalid action.');
    }
    if (record.actionCode === 0) {
      if (record.column >= record.columns) {
        throw new Error('Perfect Chaos policy record has an out-of-range drop column.');
      }
    } else if (record.column !== 0) {
      throw new Error('Perfect Chaos policy transform records must use column zero.');
    }
    if (record.outcome < -1 || record.outcome > 1) {
      throw new Error('Perfect Chaos policy outcomes must be -1, 0, or 1.');
    }
    if (previous !== null && compareStates(previous, record) >= 0) {
      throw new Error('Perfect Chaos policy records must be strictly increasing.');
    }
    previous = record;
  }

  const findRecord = (state) => {
    let low = 0;
    let high = entryCount - 1;
    while (low <= high) {
      const middle = (low + high) >> 1;
      const candidate = recordAt(middle);
      const comparison = compareStates(candidate, state);
      if (comparison === 0) return candidate;
      if (comparison < 0) low = middle + 1;
      else high = middle - 1;
    }
    return null;
  };

  return Object.freeze({
    version,
    rows,
    columns,
    connect,
    role,
    rootValue,
    entryCount,
    closureStates,
    byteLength: bytes.byteLength,
    /**
     * Returns the certified action for a position, or null when the position is
     * outside this policy's configuration. Both a position and its mirror are
     * probed, so the caller does not need to know which of the two the generator
     * stored.
     */
    lookup(board, currentPlayer, aiPlayer, startingPlayer) {
      if (currentPlayer !== aiPlayer) return null;
      if (perfectChaosCompleteRole(startingPlayer, aiPlayer) !== role) return null;
      if (!Array.isArray(board) || board.length === 0) return null;
      const state = packBoard(board, aiPlayer);
      // A round that started as rows x columns may currently be rotated.
      if (!state || !inOrbit(state.rows, state.columns)) return null;

      const direct = findRecord(state);
      if (direct) {
        const action = direct.actionCode === 0
          ? { type: ACTION_DROP, column: direct.column }
          : { type: ACTION_CODES[direct.actionCode] };
        return Object.freeze({
          action: Object.freeze(action),
          outcome: direct.outcome,
          mirrored: false,
        });
      }

      const flipped = findRecord(mirrorPacked(state));
      if (flipped) {
        const stored = flipped.actionCode === 0
          ? { type: ACTION_DROP, column: flipped.column }
          : { type: ACTION_CODES[flipped.actionCode] };
        return Object.freeze({
          // Mirror back across the board's current width, not the starting one.
          action: Object.freeze(mirrorAction(stored, state.columns)),
          outcome: flipped.outcome,
          mirrored: true,
        });
      }
      return null;
    },
  });
}

async function readBytes(url, label) {
  if (url.protocol === 'file:' && typeof process !== 'undefined' && process.versions?.node) {
    const { readFile } = await import('node:fs/promises');
    return new Uint8Array(await readFile(url));
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${label} (${response.status}).`);
  return new Uint8Array(await response.arrayBuffer());
}

function cachedLoad(cache, key, loader) {
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

export function loadPerfectChaosCompleteManifest(url = DEFAULT_MANIFEST_URL) {
  const target = url instanceof URL ? url : new URL(String(url), import.meta.url);
  return cachedLoad(MANIFEST_PROMISES, target.href, async () => {
    let manifest;
    if (target.protocol === 'file:' && typeof process !== 'undefined' && process.versions?.node) {
      const { readFile } = await import('node:fs/promises');
      manifest = JSON.parse(await readFile(target, 'utf8'));
    } else {
      const response = await fetch(target);
      if (!response.ok) {
        throw new Error(`Could not load the Perfect Chaos manifest (${response.status}).`);
      }
      manifest = await response.json();
    }
    if (manifest?.format !== 'connect4-perfect-chaos-complete-manifest-v1'
        || !Array.isArray(manifest.policies)) {
      throw new Error('Perfect Chaos manifest format is invalid.');
    }
    return Object.freeze({
      ...manifest,
      policies: Object.freeze(manifest.policies.map((entry) => Object.freeze({ ...entry }))),
    });
  });
}

/**
 * A certificate covers both orientations of its board, because a rotation
 * transposes the board mid-round. Either orientation therefore resolves to the
 * same entry.
 */
export function findPerfectChaosCompletePolicy(manifest, rows, columns, connect, role) {
  return manifest?.policies?.find((entry) => (
    entry.connect === connect
    && entry.role === role
    && ((entry.rows === rows && entry.columns === columns)
      || (entry.rows === columns && entry.columns === rows))
  )) ?? null;
}

function hex(bytes) {
  return [...bytes].map((value) => value.toString(16).padStart(2, '0')).join('');
}

async function sha256(bytes) {
  if (globalThis.crypto?.subtle) {
    return hex(new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', bytes)));
  }
  if (typeof process !== 'undefined' && process.versions?.node) {
    const { createHash } = await import('node:crypto');
    return createHash('sha256').update(bytes).digest('hex');
  }
  throw new Error('SHA-256 support is unavailable for Perfect Chaos verification.');
}

/**
 * Loads the certificate for one board and starting role, refusing anything whose
 * bytes, digest or metadata disagree with the committed manifest.
 */
export async function loadVerifiedPerfectChaosCompletePolicy(
  rows,
  columns,
  connect,
  role,
  options = {},
) {
  const manifestUrl = options.manifestUrl instanceof URL
    ? options.manifestUrl
    : options.manifestUrl
      ? new URL(String(options.manifestUrl), import.meta.url)
      : DEFAULT_MANIFEST_URL;
  const manifest = options.manifest ?? await loadPerfectChaosCompleteManifest(manifestUrl);
  const entry = findPerfectChaosCompletePolicy(manifest, rows, columns, connect, role);
  if (!entry) return null;
  if (!Number.isInteger(entry.bytes) || entry.bytes < HEADER_SIZE
      || typeof entry.sha256 !== 'string' || !/^[0-9a-f]{64}$/i.test(entry.sha256)) {
    throw new Error('Perfect Chaos manifest has invalid artifact metadata.');
  }

  const target = new URL(entry.file, manifestUrl);
  const cacheKey = [target.href, entry.bytes, entry.sha256].join('|');
  return cachedLoad(POLICY_PROMISES, cacheKey, async () => {
    const bytes = await readBytes(target, 'the Perfect Chaos policy');
    if (bytes.byteLength !== entry.bytes) {
      throw new Error(
        `Perfect Chaos policy length mismatch: expected ${entry.bytes}, found ${bytes.byteLength}.`,
      );
    }
    const digest = await sha256(bytes);
    if (digest.toLowerCase() !== entry.sha256.toLowerCase()) {
      throw new Error('Perfect Chaos policy SHA-256 does not match its manifest.');
    }
    const policy = decodePerfectChaosCompletePolicy(bytes, { rows, columns, connect, role });
    if (policy.rootValue !== entry.rootValue
        || policy.entryCount !== entry.entryCount
        || policy.closureStates !== entry.closureStates) {
      throw new Error('Perfect Chaos policy metadata does not match its manifest.');
    }
    return policy;
  });
}

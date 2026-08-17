import { createExactTableLoader } from './exact-table.js';

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

const MAGIC = 'C4CPOL1\0';
const FORMAT_VERSION = 1;
const HEADER_SIZE = 16;
const RECORD_SIZE = 20;

const ACTION_CODES = Object.freeze([
  ACTION_DROP,
  ACTION_FLIP,
  ACTION_ROTATE_CW,
  ACTION_ROTATE_CCW,
]);

const POLICY_SEGMENTS = Object.freeze([
  Object.freeze({ fromBoundary: 0, boundary: 8, file: '0-8.policy.bin' }),
  Object.freeze({ fromBoundary: 8, boundary: 10, file: '8-10.policy.bin' }),
  Object.freeze({ fromBoundary: 10, boundary: 12, file: '10-12.policy.bin' }),
  Object.freeze({ fromBoundary: 12, boundary: 14, file: '12-14.policy.bin' }),
]);

export const PERFECT_CHAOS_ROLE_FIRST = 1;
export const PERFECT_CHAOS_ROLE_SECOND = 2;
export const PERFECT_CHAOS_CERTIFIED_BOUNDARY = POLICY_SEGMENTS.at(-1).boundary;

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
  if (role !== PERFECT_CHAOS_ROLE_FIRST && role !== PERFECT_CHAOS_ROLE_SECOND) {
    throw new RangeError('Perfect Chaos policy role must be first or second.');
  }
  return role;
}

function segmentForBoundary(boundary) {
  return POLICY_SEGMENTS.find((segment) => segment.boundary === boundary) ?? null;
}

function segmentForPieceCount(pieceCount) {
  return POLICY_SEGMENTS.find((segment) => (
    pieceCount >= segment.fromBoundary && pieceCount < segment.boundary
  )) ?? null;
}

function playableMask(rows, columns) {
  const stride = rows + 1;
  const columnMask = (1n << BigInt(rows)) - 1n;
  let mask = 0n;
  for (let column = 0; column < columns; column += 1) {
    mask |= columnMask << BigInt(column * stride);
  }
  return mask;
}

function bitCount(value) {
  let count = 0;
  while (value !== 0n) {
    value &= value - 1n;
    count += 1;
  }
  return count;
}

function hasPackedWin(pieces, rows) {
  const stride = rows + 1;
  for (const shift of [1, stride, stride - 1, stride + 1]) {
    const aligned = pieces & (pieces >> BigInt(shift));
    if ((aligned & (aligned >> BigInt(2 * shift))) !== 0n) return true;
  }
  return false;
}

function validatePackedState(mover, opponent, rows, columns, label) {
  if (!((rows === 6 && columns === 7) || (rows === 7 && columns === 6))) {
    throw new Error(`${label} has unsupported board dimensions.`);
  }
  if ((mover & opponent) !== 0n) throw new Error(`${label} has overlapping pieces.`);

  const allowed = playableMask(rows, columns);
  if (((mover | opponent) & ~allowed) !== 0n) {
    throw new Error(`${label} contains bits outside the playable board.`);
  }

  const stride = rows + 1;
  const columnMask = (1n << BigInt(rows)) - 1n;
  const occupied = mover | opponent;
  if (hasPackedWin(mover, rows) || hasPackedWin(opponent, rows)) {
    throw new Error(`${label} is already terminal.`);
  }
  for (let column = 0; column < columns; column += 1) {
    const bits = (occupied >> BigInt(column * stride)) & columnMask;
    if ((bits & (bits + 1n)) !== 0n) {
      throw new Error(`${label} violates gravity.`);
    }
  }
  return bitCount(occupied);
}

function compareState(first, second) {
  if (first.rows !== second.rows) return first.rows - second.rows;
  if (first.columns !== second.columns) return first.columns - second.columns;
  if (first.mover !== second.mover) return first.mover < second.mover ? -1 : 1;
  if (first.opponent !== second.opponent) return first.opponent < second.opponent ? -1 : 1;
  return 0;
}

function mirrorBits(bits, rows, columns) {
  const stride = rows + 1;
  const groupMask = (1n << BigInt(stride)) - 1n;
  let mirrored = 0n;
  for (let column = 0; column < columns; column += 1) {
    const group = (bits >> BigInt(column * stride)) & groupMask;
    mirrored |= group << BigInt((columns - 1 - column) * stride);
  }
  return mirrored;
}

function canonicalState(state) {
  const mirrored = {
    ...state,
    mover: mirrorBits(state.mover, state.rows, state.columns),
    opponent: mirrorBits(state.opponent, state.rows, state.columns),
  };
  return compareState(mirrored, state) < 0
    ? { state: mirrored, mirrored: true }
    : { state, mirrored: false };
}

function encodeBoard(board, currentPlayer, aiPlayer) {
  if (currentPlayer !== aiPlayer || (aiPlayer !== RED && aiPlayer !== YELLOW)) return null;
  const { rows, cols: columns } = boardDimensions(board);
  if (!((rows === 6 && columns === 7) || (rows === 7 && columns === 6))) return null;
  if (board.some((row) => !Array.isArray(row) || row.length !== columns)) return null;

  const stride = rows + 1;
  let mover = 0n;
  let opponent = 0n;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const cell = board[row][column];
      if (cell === EMPTY) continue;
      if (cell !== RED && cell !== YELLOW) return null;
      const target = 1n << BigInt(column * stride + (rows - 1 - row));
      if (cell === currentPlayer) mover |= target;
      else opponent |= target;
    }
  }

  const state = { mover, opponent, rows, columns };
  const pieceCount = validatePackedState(
    mover,
    opponent,
    rows,
    columns,
    'Perfect Chaos policy lookup position',
  );
  return { ...canonicalState(state), pieceCount };
}

function mirrorAction(action, columns) {
  if (action.type === ACTION_DROP) {
    return { type: ACTION_DROP, column: columns - 1 - action.column };
  }
  if (action.type === ACTION_ROTATE_CW) return { type: ACTION_ROTATE_CCW };
  if (action.type === ACTION_ROTATE_CCW) return { type: ACTION_ROTATE_CW };
  return { type: ACTION_FLIP };
}

function recordState(view, offset) {
  return {
    mover: view.getBigUint64(offset, true),
    opponent: view.getBigUint64(offset + 8, true),
    rows: view.getUint8(offset + 16),
    columns: view.getUint8(offset + 17),
  };
}

function actionAt(view, offset, state) {
  const code = view.getUint8(offset + 18);
  const column = view.getUint8(offset + 19);
  const type = ACTION_CODES[code];
  if (!type) throw new Error('Perfect Chaos policy contains an invalid action type.');
  if (type === ACTION_DROP) {
    if (column >= state.columns) {
      throw new Error('Perfect Chaos policy contains an invalid drop column.');
    }
    const top = 1n << BigInt(column * (state.rows + 1) + state.rows - 1);
    if (((state.mover | state.opponent) & top) !== 0n) {
      throw new Error('Perfect Chaos policy attempts to drop in a full column.');
    }
    return { type, column };
  }
  if (column !== 0) throw new Error('Perfect Chaos transform actions must use column zero.');
  return { type };
}

export function decodePerfectChaosPolicy(input, expectedRole = null, expectedBoundary = null) {
  const bytes = bytesFrom(input);
  if (bytes.byteLength < HEADER_SIZE) throw new Error('Perfect Chaos policy data is truncated.');
  if (ascii(bytes, 0, 8) !== MAGIC) throw new Error('Perfect Chaos policy magic is invalid.');

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint8(8);
  const role = view.getUint8(9);
  const boundary = view.getUint8(10);
  const recordSize = view.getUint8(11);
  const entryCount = view.getUint32(12, true);
  const segment = segmentForBoundary(boundary);
  validateRole(role);
  if (expectedRole !== null && role !== validateRole(expectedRole)) {
    throw new Error('Perfect Chaos policy role does not match the requested role.');
  }
  if (version !== FORMAT_VERSION) {
    throw new Error(`Unsupported Perfect Chaos policy version ${version}.`);
  }
  if (!segment || recordSize !== RECORD_SIZE) {
    throw new Error('Perfect Chaos policy has an unsupported boundary or record size.');
  }
  if (expectedBoundary !== null && boundary !== expectedBoundary) {
    throw new Error('Perfect Chaos policy boundary does not match the requested segment.');
  }
  const expectedLength = HEADER_SIZE + entryCount * RECORD_SIZE;
  if (bytes.byteLength !== expectedLength) {
    throw new Error(
      `Perfect Chaos policy length mismatch: expected ${expectedLength}, found ${bytes.byteLength}.`,
    );
  }

  let previous = null;
  for (let index = 0; index < entryCount; index += 1) {
    const offset = HEADER_SIZE + index * RECORD_SIZE;
    const state = recordState(view, offset);
    const pieceCount = validatePackedState(
      state.mover,
      state.opponent,
      state.rows,
      state.columns,
      'Perfect Chaos policy record',
    );
    if (pieceCount < segment.fromBoundary || pieceCount >= segment.boundary) {
      throw new Error('Perfect Chaos policy record is outside its certified segment.');
    }
    const canonical = canonicalState(state);
    if (canonical.mirrored) {
      throw new Error('Perfect Chaos policy states must be horizontally canonical.');
    }
    actionAt(view, offset, state);
    if (previous && compareState(previous, state) >= 0) {
      throw new Error('Perfect Chaos policy states must be strictly increasing.');
    }
    previous = state;
  }

  return Object.freeze({
    version,
    role,
    fromBoundary: segment.fromBoundary,
    boundary,
    entryCount,
    byteLength: bytes.byteLength,
    lookup(board, currentPlayer, aiPlayer) {
      const encoded = encodeBoard(board, currentPlayer, aiPlayer);
      if (!encoded
          || encoded.pieceCount < segment.fromBoundary
          || encoded.pieceCount >= segment.boundary) return null;

      let low = 0;
      let high = entryCount - 1;
      while (low <= high) {
        const middle = (low + high) >> 1;
        const offset = HEADER_SIZE + middle * RECORD_SIZE;
        const candidate = recordState(view, offset);
        const comparison = compareState(candidate, encoded.state);
        if (comparison === 0) {
          const action = actionAt(view, offset, candidate);
          return Object.freeze({
            action: encoded.mirrored ? mirrorAction(action, candidate.columns) : action,
            mirrored: encoded.mirrored,
          });
        }
        if (comparison < 0) low = middle + 1;
        else high = middle - 1;
      }
      return null;
    },
  });
}

function roleDirectory(role) {
  return validateRole(role) === PERFECT_CHAOS_ROLE_FIRST ? 'red' : 'yellow';
}

function defaultUrl(role, segment) {
  return new URL(
    `../data/perfect-chaos-prefix/${roleDirectory(role)}/${segment.file}`,
    import.meta.url,
  );
}

const LOADERS = new Map();
function loaderFor(role, segment) {
  const key = `${validateRole(role)}:${segment.boundary}`;
  let loader = LOADERS.get(key);
  if (!loader) {
    loader = createExactTableLoader(
      (bytes) => decodePerfectChaosPolicy(bytes, role, segment.boundary),
      'Perfect Chaos policy',
    );
    LOADERS.set(key, loader);
  }
  return loader;
}

export function loadPerfectChaosPolicy(role, pieceCount = 0, url = null) {
  validateRole(role);
  if (!Number.isInteger(pieceCount) || pieceCount < 0 || pieceCount > 42) {
    throw new RangeError('Perfect Chaos policy piece count must be an integer from 0 through 42.');
  }
  const segment = segmentForPieceCount(pieceCount);
  if (!segment) return Promise.resolve(null);
  const target = url === null
    ? defaultUrl(role, segment)
    : url instanceof URL ? url : new URL(String(url), import.meta.url);
  return loaderFor(role, segment)(target);
}

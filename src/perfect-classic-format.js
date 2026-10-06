// The classic policy format, C4VPOL1: its header and record checks, and the
// record stored under a canonical key. It lives apart from
// perfect-classic-policy.js because the release gate replays the classic
// catalog through it: an edit here costs a replay, an edit to the page's
// board lookup or catalog loader, or to the rules and transport they use,
// does not.
import { ascii, bytesFrom } from './bytes.js';

const MAGIC = 'C4VPOL1\0';
const FORMAT_VERSION = 1;
const HEADER_SIZE = 24;
const RECORD_SIZE = 10;

export const PERFECT_CLASSIC_ROLE_FIRST = 1;
export const PERFECT_CLASSIC_ROLE_SECOND = 2;

export function validateRole(role) {
  if (role !== PERFECT_CLASSIC_ROLE_FIRST && role !== PERFECT_CLASSIC_ROLE_SECOND) {
    throw new RangeError('Perfect classic policy role must be first or second.');
  }
  return role;
}

// The board's cell count, with the bound every canonical key of the board
// stays below: a key holds `rows` cells and a sentinel bit for each column.
// The checks repeat the solver's in a policy's own words, and a connect
// length is required: the solver's default of four must not stand in for one
// a manifest entry leaves out.
export function policyGeometry(rows, columns, connect) {
  if (!Number.isInteger(rows) || !Number.isInteger(columns)
      || rows < 1 || rows > 7 || columns < 1 || columns > 7) {
    throw new RangeError('Perfect classic policies support 1 through 7 rows and columns.');
  }
  if (!Number.isInteger(connect) || connect < 1 || connect > Math.max(rows, columns)) {
    throw new RangeError('Perfect classic policy connect length must fit the board.');
  }
  return { cellCount: rows * columns, keyLimit: 1n << BigInt((rows + 1) * columns) };
}

export function moveMaskColumn(moveMask, columns) {
  if (!Number.isInteger(moveMask) || moveMask <= 0
      || (moveMask & (moveMask - 1)) !== 0
      || (moveMask & ~((1 << columns) - 1)) !== 0) return -1;
  for (let column = 0; column < columns; column += 1) {
    if ((moveMask & (1 << column)) !== 0) return column;
  }
  return -1;
}

/** The policy in `input`: its header fields, and lookupKey(key), the record
 * stored under a canonical key or null. Every field and record is checked
 * before this returns, so a malformed policy fails before anything reads it. */
export function decodePerfectClassicRecords(input, expectations = {}) {
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
  const selected = policyGeometry(rows, columns, connect);

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

  return Object.freeze({
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
  });
}

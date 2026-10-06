// Classic boards of up to 7x7 as bitboards: a column is `rows` cells and a
// sentinel bit, `mask` holds every stone and `current` those of the player
// to move. The exact solver and the page's classic policy lookup share these.
// The release gate's replay does not load them, so an edit here costs no
// replay: the policy format bounds its keys itself (perfect-classic-format.js).
import { EMPTY, RED, YELLOW } from './engine.js';

const GEOMETRIES = new Map();

function geometryKey(rows, columns, connect) {
  return `${rows}x${columns}:c${connect}`;
}

export function createClassicGeometry(rows, columns, connect = 4) {
  if (!Number.isInteger(rows) || !Number.isInteger(columns)
      || rows < 1 || columns < 1 || rows > 7 || columns > 7) {
    throw new RangeError('Exact classic boards must have 1 through 7 rows and columns.');
  }
  if (!Number.isInteger(connect) || connect < 1 || connect > Math.max(rows, columns)) {
    throw new RangeError('Connect length must be a positive integer that fits the board.');
  }

  const key = geometryKey(rows, columns, connect);
  let geometry = GEOMETRIES.get(key);
  if (geometry) return geometry;

  const stride = rows + 1;
  const cellCount = rows * columns;
  const columnBits = (1n << BigInt(rows)) - 1n;
  const columnWithSentinel = (1n << BigInt(stride)) - 1n;
  const bottomMasks = Array.from(
    { length: columns },
    (_, column) => 1n << BigInt(column * stride),
  );
  const columnMasks = bottomMasks.map((bottom) => bottom * columnBits);
  const bottomMask = bottomMasks.reduce((mask, bit) => mask | bit, 0n);
  const boardMask = bottomMask * columnBits;
  const centre = (columns - 1) / 2;
  const columnOrder = Array.from({ length: columns }, (_, column) => column)
    .sort((first, second) => (
      Math.abs(first - centre) - Math.abs(second - centre) || first - second
    ));

  geometry = Object.freeze({
    rows,
    columns,
    connect,
    stride,
    cellCount,
    columnBits,
    columnWithSentinel,
    bottomMasks: Object.freeze(bottomMasks),
    columnMasks: Object.freeze(columnMasks),
    bottomMask,
    boardMask,
    columnOrder: Object.freeze(columnOrder),
    directions: Object.freeze([1, stride - 1, stride, stride + 1]),
  });
  GEOMETRIES.set(key, geometry);
  return geometry;
}

export function moveForColumn(geometry, mask, column) {
  if (!Number.isInteger(column) || column < 0 || column >= geometry.columns) return 0n;
  return (mask + geometry.bottomMasks[column]) & geometry.columnMasks[column];
}

function mirrorBits(geometry, bits) {
  let mirrored = 0n;
  for (let column = 0; column < geometry.columns; column += 1) {
    const group = (bits >> BigInt(column * geometry.stride))
      & geometry.columnWithSentinel;
    mirrored |= group << BigInt((geometry.columns - 1 - column) * geometry.stride);
  }
  return mirrored;
}

export function canonicalClassicPosition(geometry, position) {
  const normal = position.current + position.mask;
  const mirroredCurrent = mirrorBits(geometry, position.current);
  const mirroredMask = mirrorBits(geometry, position.mask);
  const mirrored = mirroredCurrent + mirroredMask;
  return normal <= mirrored
    ? { key: normal, mirrored: false }
    : { key: mirrored, mirrored: true };
}

export function boardToClassicBitboard(board, currentPlayer, connect = 4) {
  if ((currentPlayer !== RED && currentPlayer !== YELLOW)
      || !Array.isArray(board)
      || board.length === 0
      || !Array.isArray(board[0])
      || board[0].length === 0) return null;

  const rows = board.length;
  const columns = board[0].length;
  if (rows > 7 || columns > 7
      || board.some((row) => !Array.isArray(row) || row.length !== columns)) return null;

  let geometry;
  try {
    geometry = createClassicGeometry(rows, columns, connect);
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
      const bit = 1n << BigInt(column * geometry.stride + rows - 1 - row);
      mask |= bit;
      if (cell === currentPlayer) current |= bit;
      moves += 1;
    }
  }
  return { geometry, current, mask, moves };
}

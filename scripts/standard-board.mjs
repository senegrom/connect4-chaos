/** The standard 7x6 board as the book and strategy generators encode it: a
 * column is HEIGHT cells and a sentinel bit, `mask` holds every stone and
 * `current` those of the player to move. */
export const WIDTH = 7;
export const HEIGHT = 6;
export const STRIDE = HEIGHT + 1;
export const COLUMN_ORDER = Object.freeze([3, 2, 4, 1, 5, 0, 6]);

const COLUMN_BITS = (1n << BigInt(HEIGHT)) - 1n;
const COLUMN_WITH_SENTINEL = (1n << BigInt(STRIDE)) - 1n;
const BOTTOM_MASKS = Array.from({ length: WIDTH }, (_, column) => 1n << BigInt(column * STRIDE));
const COLUMN_MASKS = BOTTOM_MASKS.map((bottom) => bottom * COLUMN_BITS);
const BOTTOM_MASK = BOTTOM_MASKS.reduce((mask, bit) => mask | bit, 0n);
const BOARD_MASK = BOTTOM_MASK * COLUMN_BITS;

export function possibleMoves(mask) {
  return (mask + BOTTOM_MASK) & BOARD_MASK;
}

export function moveForColumn(mask, column) {
  return (mask + BOTTOM_MASKS[column]) & COLUMN_MASKS[column];
}

export function play(position, move) {
  return {
    current: position.current ^ position.mask,
    mask: position.mask | move,
    moves: position.moves + 1,
  };
}

export function hasAlignment(bits) {
  for (const direction of [1, HEIGHT, STRIDE, HEIGHT + 2]) {
    const shift = BigInt(direction);
    const pair = bits & (bits >> shift);
    if ((pair & (pair >> (2n * shift))) !== 0n) return true;
  }
  return false;
}

export function mirrorBits(bits) {
  let mirrored = 0n;
  for (let column = 0; column < WIDTH; column += 1) {
    const columnBits = (bits >> BigInt(column * STRIDE)) & COLUMN_WITH_SENTINEL;
    mirrored |= columnBits << BigInt((WIDTH - 1 - column) * STRIDE);
  }
  return mirrored;
}

/** One line of the exact solver's output: an optional move sequence (columns
 * 1-7), then the score of each of the WIDTH moves from that position.
 * `label` names the line in errors. */
export function parseScoredLine(line, lineNumber, label = 'scored line') {
  const tokens = line.trim().split(/\s+/).filter(Boolean);
  let sequence;
  let scoreTokens;
  if (tokens.length === WIDTH) {
    sequence = '';
    scoreTokens = tokens;
  } else if (tokens.length === WIDTH + 1 && /^[1-7]+$/.test(tokens[0])) {
    [sequence] = tokens;
    scoreTokens = tokens.slice(1);
  } else {
    throw new Error(`Invalid ${label} ${lineNumber}: ${line}`);
  }
  const scores = scoreTokens.map((token) => Number.parseInt(token, 10));
  if (scores.some((score) => !Number.isInteger(score))) {
    throw new Error(`Non-integer score in ${label} ${lineNumber}.`);
  }
  return { sequence, scores };
}

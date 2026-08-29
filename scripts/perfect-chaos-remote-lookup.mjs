// Position-addressed lookup into the pair-scheduled solver's block files
// (C4PAIR2 checkpoints) without ever loading a table: the slot arithmetic
// is a faithful port of native/perfect-chaos-paired.cpp, and reads go
// through a pluggable range source - a local file for tests and tooling, an
// HTTP Range request against object storage for the browser tier.
//
// Per child value the cost is one read of at most GROUP_WORDS*8 bytes from
// the block's bitset (bounded by the rank sidecar built by
// scripts/build-pair-rank-sidecars.py) plus one byte from its values file.

export const WIN = 1;
export const DRAW = 0;
export const LOSS = -1;
export const NOT_TERMINAL = 2;

export const GROUP_WORDS = 2048; // sidecar granularity: 16 KB of bitset

// ---------------------------------------------------------------------------
// Combinatorics
// ---------------------------------------------------------------------------

const MAX_CELLS = 49;
const BINOMIAL = (() => {
  const at = Array.from({ length: MAX_CELLS + 1 }, () => new Array(MAX_CELLS + 1).fill(0));
  for (let n = 0; n <= MAX_CELLS; n += 1) {
    at[n][0] = 1;
    for (let k = 1; k <= n; k += 1) {
      at[n][k] = at[n - 1][k - 1] + (k <= n - 1 ? at[n - 1][k] : 0);
    }
  }
  return at;
})();

export function pairOf(pieces, moverCount) {
  return Math.max(moverCount, pieces - moverCount);
}

function colourRankM(word) {
  let rank = 0;
  let seen = 0;
  let bits = word;
  while (bits !== 0n) {
    let position = 0;
    let probe = bits;
    while ((probe & 1n) === 0n) { probe >>= 1n; position += 1; }
    seen += 1;
    rank += BINOMIAL[position][seen];
    bits &= bits - 1n;
  }
  return rank;
}

function popcountBig(word) {
  let count = 0;
  let bits = word;
  while (bits !== 0n) { bits &= bits - 1n; count += 1; }
  return count;
}

// ---------------------------------------------------------------------------
// Geometry: mirror-canonical compositions and pair-block slot spaces
// ---------------------------------------------------------------------------

function buildBlock(rows, columns) {
  const canon = Array.from({ length: rows * columns + 1 }, () => []);
  const rankOf = new Map();
  const heights = new Array(columns).fill(0);
  for (;;) {
    let canonical = true;
    for (let c = 0; c < columns; c += 1) {
      const mirrored = heights[columns - 1 - c];
      if (heights[c] !== mirrored) { canonical = heights[c] < mirrored; break; }
    }
    if (canonical) {
      let pieces = 0;
      let code = 0;
      for (let c = 0; c < columns; c += 1) { pieces += heights[c]; code |= heights[c] << (3 * c); }
      rankOf.set(code, canon[pieces].length);
      canon[pieces].push(code);
    }
    let column = columns - 1;
    while (column >= 0 && heights[column] === rows) { heights[column] = 0; column -= 1; }
    if (column < 0) break;
    heights[column] += 1;
  }
  return { rows, columns, canon, rankOf };
}

export function makeGeometry(rows, columns, connect) {
  const blocks = [buildBlock(rows, columns)];
  if (rows !== columns) blocks.push(buildBlock(columns, rows));
  const cellCount = rows * columns;
  const pairColourSlots = (pieces, pairId) => {
    const high = BINOMIAL[pieces][pairId];
    if (pairId * 2 === pieces) return high;
    return high + BINOMIAL[pieces][pieces - pairId];
  };
  const blockPairSlots = (block, pieces, pairId) =>
    blocks[block].canon[pieces].length * pairColourSlots(pieces, pairId);
  const blockPairOffset = (block, pieces, pairId) => {
    let offset = 0;
    for (let index = 0; index < block; index += 1) offset += blockPairSlots(index, pieces, pairId);
    return offset;
  };
  const pairSlots = (pieces, pairId) => {
    let total = 0;
    for (let index = 0; index < blocks.length; index += 1) total += blockPairSlots(index, pieces, pairId);
    return total;
  };
  return { rows, columns, connect, cellCount, blocks, pairColourSlots, blockPairSlots, blockPairOffset, pairSlots };
}

function colourWordOf(mover, heights, columns, stride) {
  let colours = 0n;
  let offset = 0n;
  for (let column = 0; column < columns; column += 1) {
    const width = (1n << BigInt(heights[column])) - 1n;
    colours |= ((mover >> BigInt(column * stride)) & width) << offset;
    offset += BigInt(heights[column]);
  }
  return colours;
}

function colourSubslot(word, pieces, pairId) {
  const ones = popcountBig(word);
  let base = 0;
  if (ones !== pairId) {
    if (ones !== pieces - pairId) throw new Error('colour word outside its pair');
    base = BINOMIAL[pieces][pairId];
  }
  return base + colourRankM(word);
}

export function canonicalPairSlot(geometry, blockIndex, mover, heights, pieces, pairId) {
  const block = geometry.blocks[blockIndex];
  const columns = block.columns;
  const stride = block.rows + 1;

  let order = 0;
  for (let column = 0; order === 0 && column < columns; column += 1) {
    const direct = heights[column];
    const mirrored = heights[columns - 1 - column];
    if (direct !== mirrored) order = direct < mirrored ? -1 : 1;
  }
  if (order === 0) {
    for (let column = columns - 1; order === 0 && column >= 0; column -= 1) {
      const width = (1n << BigInt(heights[column])) - 1n;
      const direct = (mover >> BigInt(column * stride)) & width;
      const mirrored = (mover >> BigInt((columns - 1 - column) * stride)) & width;
      if (direct !== mirrored) order = direct > mirrored ? 1 : -1;
    }
  }

  let colours;
  let rank;
  if (order <= 0) {
    colours = colourWordOf(mover, heights, columns, stride);
    let code = 0;
    for (let c = 0; c < columns; c += 1) code |= heights[c] << (3 * c);
    rank = block.rankOf.get(code);
  } else {
    colours = 0n;
    let offset = 0n;
    let code = 0;
    for (let column = 0; column < columns; column += 1) {
      const source = columns - 1 - column;
      code |= heights[source] << (3 * column);
      const width = (1n << BigInt(heights[source])) - 1n;
      colours |= ((mover >> BigInt(source * stride)) & width) << offset;
      offset += BigInt(heights[source]);
    }
    rank = block.rankOf.get(code);
  }
  if (rank === undefined) throw new Error('canonicalisation reached a non-canonical composition');
  return geometry.blockPairOffset(blockIndex, pieces, pairId)
    + rank * geometry.pairColourSlots(pieces, pairId)
    + colourSubslot(colours, pieces, pairId);
}

// ---------------------------------------------------------------------------
// Moves: drops and transforms, exactly as pairSuccessors emits them
// ---------------------------------------------------------------------------

export function maskHasLine(mask, rows, connect) {
  const stride = rows + 1;
  for (const shift of [1, stride, stride + 1, stride - 1]) {
    let run = mask;
    for (let step = 1; step < connect && run !== 0n; step += 1) {
      run &= mask >> BigInt(shift * step);
    }
    if (run !== 0n) return true;
  }
  return false;
}

function reverseSegment(segment, height) {
  let reversed = 0n;
  for (let bit = 0; bit < height; bit += 1) {
    if ((segment >> BigInt(bit)) & 1n) reversed |= 1n << BigInt(height - 1 - bit);
  }
  return reversed;
}

// Children of a state, each as either a terminal value or a block-addressed
// slot: { terminal, sameLayer, blockIndex, pieces, pairId, slot }.
export function successors(geometry, blockIndex, mover, opponent, heights, pieces, moverCount) {
  const block = geometry.blocks[blockIndex];
  const rows = block.rows;
  const columns = block.columns;
  const stride = rows + 1;
  const edges = [];

  const dropMover = pieces - moverCount;
  const dropPair = pairOf(pieces + 1, dropMover);
  for (let column = 0; column < columns; column += 1) {
    const height = heights[column];
    if (height >= rows) continue;
    const grown = mover | (1n << BigInt(column * stride + height));
    if (maskHasLine(grown, rows, geometry.connect)) {
      edges.push({ terminal: WIN, sameLayer: false });
      continue;
    }
    if (pieces + 1 === geometry.cellCount) {
      edges.push({ terminal: DRAW, sameLayer: false });
      continue;
    }
    const childHeights = heights.slice();
    childHeights[column] += 1;
    edges.push({
      terminal: NOT_TERMINAL, sameLayer: false, blockIndex,
      pieces: pieces + 1, pairId: dropPair,
      slot: canonicalPairSlot(geometry, blockIndex, opponent, childHeights, pieces + 1, dropPair),
    });
  }

  const settleTransform = (nextMover, nextOpponent, nextBlock, nextHeights) => {
    const nextRows = geometry.blocks[nextBlock].rows;
    const moverLine = maskHasLine(nextMover, nextRows, geometry.connect);
    const opponentLine = maskHasLine(nextOpponent, nextRows, geometry.connect);
    if (moverLine || opponentLine) {
      edges.push({ terminal: moverLine && opponentLine ? LOSS : (moverLine ? WIN : LOSS), sameLayer: true });
      return;
    }
    edges.push({
      terminal: NOT_TERMINAL, sameLayer: true, blockIndex: nextBlock,
      pieces, pairId: pairOf(pieces, moverCount),
      slot: canonicalPairSlot(geometry, nextBlock, nextOpponent, nextHeights, pieces, pairOf(pieces, moverCount)),
    });
  };

  {
    let flippedMover = 0n;
    let flippedOpponent = 0n;
    for (let column = 0; column < columns; column += 1) {
      const height = heights[column];
      const base = BigInt(column * stride);
      const occupied = (1n << BigInt(height)) - 1n;
      const segment = (mover >> base) & occupied;
      const reversed = reverseSegment(segment, height);
      flippedMover |= reversed << base;
      flippedOpponent |= (occupied ^ reversed) << base;
    }
    settleTransform(flippedMover, flippedOpponent, blockIndex, heights);
  }

  const transposedBlock = geometry.blocks.length === 1 ? 0 : 1 - blockIndex;
  const targetStride = columns + 1;
  for (const clockwise of [true, false]) {
    let rotatedMover = 0n;
    let rotatedOpponent = 0n;
    const rotatedHeights = new Array(rows).fill(0);
    for (let targetColumn = 0; targetColumn < rows; targetColumn += 1) {
      let height = 0;
      if (clockwise) {
        for (let sourceColumn = columns - 1; sourceColumn >= 0; sourceColumn -= 1) {
          if (heights[sourceColumn] <= targetColumn) continue;
          const bit = 1n << BigInt(targetColumn * targetStride + height);
          if ((mover >> BigInt(sourceColumn * stride + targetColumn)) & 1n) rotatedMover |= bit;
          else rotatedOpponent |= bit;
          height += 1;
        }
      } else {
        const sourceRow = rows - 1 - targetColumn;
        for (let sourceColumn = 0; sourceColumn < columns; sourceColumn += 1) {
          if (heights[sourceColumn] <= sourceRow) continue;
          const bit = 1n << BigInt(targetColumn * targetStride + height);
          if ((mover >> BigInt(sourceColumn * stride + sourceRow)) & 1n) rotatedMover |= bit;
          else rotatedOpponent |= bit;
          height += 1;
        }
      }
      rotatedHeights[targetColumn] = height;
    }
    settleTransform(rotatedMover, rotatedOpponent, transposedBlock, rotatedHeights);
  }

  return edges;
}

// ---------------------------------------------------------------------------
// Range-backed lookup
// ---------------------------------------------------------------------------

const HEADER_BYTES = 24;

function readU64LE(bytes, offset) {
  let value = 0n;
  for (let index = 7; index >= 0; index -= 1) value = (value << 8n) | BigInt(bytes[offset + index]);
  return Number(value);
}

function popcount64(bytes, offset) {
  let count = 0;
  for (let index = 0; index < 8; index += 1) {
    let byte = bytes[offset + index];
    while (byte) { byte &= byte - 1; count += 1; }
  }
  return count;
}

// source: async fetchRange(fileName, offset, length) -> Uint8Array.
export async function lookupSlot(source, pieces, pairId, slot) {
  const bitsName = `pair-${pieces}-${pairId}.bits`;
  const wordIndex = Math.floor(slot / 64);
  const group = Math.floor(wordIndex / GROUP_WORDS);
  const rankBytes = await source.fetchRange(`pair-${pieces}-${pairId}.ranks`, group * 8, 8);
  let rank = readU64LE(rankBytes, 0);

  const firstWord = group * GROUP_WORDS;
  const words = await source.fetchRange(
    bitsName, HEADER_BYTES + firstWord * 8, (wordIndex - firstWord + 1) * 8,
  );
  for (let word = 0; word < wordIndex - firstWord; word += 1) rank += popcount64(words, word * 8);
  const bit = slot % 64;
  const last = words.subarray((wordIndex - firstWord) * 8, (wordIndex - firstWord) * 8 + 8);
  let below = 0;
  for (let index = 0; index < 8; index += 1) {
    const lowBits = Math.max(0, Math.min(8, bit - index * 8));
    if (lowBits === 0) break;
    let byte = last[index] & ((lowBits >= 8 ? 0xff : (1 << lowBits) - 1));
    while (byte) { byte &= byte - 1; below += 1; }
  }
  if (((last[bit >> 3] >> (bit & 7)) & 1) === 0) {
    throw new Error(`slot ${slot} of pair ${pieces}-${pairId} is not a reachable state`);
  }
  rank += below;

  const value = await source.fetchRange(`pair-${pieces}-${pairId}.values`, HEADER_BYTES + rank, 1);
  return value[0] - 1;
}

// Value of a child edge as seen by the mover of the parent state.
export async function edgeValueForMover(source, edge) {
  if (edge.terminal !== NOT_TERMINAL) return edge.terminal;
  const fromChild = await lookupSlot(source, edge.pieces, edge.pairId, edge.slot);
  return fromChild === DRAW ? DRAW : -fromChild;
}

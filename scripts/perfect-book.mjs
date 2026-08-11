#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import process from 'node:process';

const WIDTH = 7;
const HEIGHT = 6;
const STRIDE = HEIGHT + 1;
const COLUMN_BITS = (1n << BigInt(HEIGHT)) - 1n;
const COLUMN_WITH_SENTINEL = (1n << BigInt(STRIDE)) - 1n;
const BOTTOM_MASKS = Array.from(
  { length: WIDTH },
  (_, column) => 1n << BigInt(column * STRIDE),
);
const COLUMN_MASKS = BOTTOM_MASKS.map((bottom) => bottom * COLUMN_BITS);
const BOTTOM_MASK = BOTTOM_MASKS.reduce((mask, bit) => mask | bit, 0n);
const BOARD_MASK = BOTTOM_MASK * COLUMN_BITS;
const COLUMN_ORDER = Object.freeze([3, 2, 4, 1, 5, 0, 6]);
const HEADER_SIZE = 12;
const ENTRY_SIZE = 10;
const UINT64_MASK = (1n << 64n) - 1n;

function fail(message) {
  console.error(message);
  process.exitCode = 1;
}

function parseOptions(values) {
  const options = new Map();
  for (let index = 0; index < values.length; index += 1) {
    const token = values[index];
    if (!token.startsWith('--')) throw new Error(`Unexpected argument: ${token}`);
    const key = token.slice(2);
    const next = values[index + 1];
    if (next === undefined || next.startsWith('--')) options.set(key, true);
    else {
      options.set(key, next);
      index += 1;
    }
  }
  return options;
}

function integerOption(options, name, fallback, minimum = 0) {
  const raw = options.get(name);
  if (raw === undefined) return fallback;
  const value = Number.parseInt(String(raw), 10);
  if (!Number.isInteger(value) || value < minimum) {
    throw new Error(`--${name} must be an integer of at least ${minimum}.`);
  }
  return value;
}

function mix64(key) {
  let value = key & UINT64_MASK;
  value ^= value >> 33n;
  value = (value * 0xff51afd7ed558ccdn) & UINT64_MASK;
  value ^= value >> 33n;
  value = (value * 0xc4ceb9fe1a85ec53n) & UINT64_MASK;
  value ^= value >> 33n;
  return value;
}

function shardForKey(key, shardCount) {
  return Number(mix64(key) % BigInt(shardCount));
}

function possibleMoves(mask) {
  return (mask + BOTTOM_MASK) & BOARD_MASK;
}

function moveForColumn(mask, column) {
  return (mask + BOTTOM_MASKS[column]) & COLUMN_MASKS[column];
}

function play(position, move) {
  return {
    current: position.current ^ position.mask,
    mask: position.mask | move,
    moves: position.moves + 1,
  };
}

function hasAlignment(bits) {
  for (const direction of [1, HEIGHT, STRIDE, HEIGHT + 2]) {
    const shift = BigInt(direction);
    const pair = bits & (bits >> shift);
    if ((pair & (pair >> (2n * shift))) !== 0n) return true;
  }
  return false;
}

function mirrorBits(bits) {
  let mirrored = 0n;
  for (let column = 0; column < WIDTH; column += 1) {
    const columnBits = (bits >> BigInt(column * STRIDE)) & COLUMN_WITH_SENTINEL;
    mirrored |= columnBits << BigInt((WIDTH - 1 - column) * STRIDE);
  }
  return mirrored;
}

function mirrorMoveMask(mask) {
  let mirrored = 0;
  for (let column = 0; column < WIDTH; column += 1) {
    if ((mask & (1 << column)) !== 0) mirrored |= 1 << (WIDTH - 1 - column);
  }
  return mirrored;
}

function canonicalPosition(position) {
  const normal = position.current + position.mask;
  const mirrored = mirrorBits(position.current) + mirrorBits(position.mask);
  return normal <= mirrored
    ? { key: normal, mirrored: false }
    : { key: mirrored, mirrored: true };
}

function positionFromSequence(sequence) {
  let position = { current: 0n, mask: 0n, moves: 0 };

  for (let index = 0; index < sequence.length; index += 1) {
    const column = Number(sequence[index]) - 1;
    if (!Number.isInteger(column) || column < 0 || column >= WIDTH) {
      throw new Error(`Invalid column "${sequence[index]}" in sequence "${sequence}".`);
    }
    const move = moveForColumn(position.mask, column);
    if (move === 0n) throw new Error(`Full column in sequence "${sequence}".`);
    if (hasAlignment(position.current | move) && index + 1 !== sequence.length) {
      throw new Error(`Sequence continues after a win: "${sequence}".`);
    }
    position = play(position, move);
  }

  return position;
}

async function enumerate(options) {
  const depth = integerOption(options, 'depth', 6, 0);
  const shardCount = integerOption(options, 'shard-count', 1, 1);
  const shardIndex = integerOption(options, 'shard-index', 0, 0);
  if (shardIndex >= shardCount) throw new Error('--shard-index must be below --shard-count.');

  const visited = new Set();
  const lines = [];

  function explore(position, sequence) {
    const canonical = canonicalPosition(position);
    if (visited.has(canonical.key)) return;
    visited.add(canonical.key);

    if (shardForKey(canonical.key, shardCount) === shardIndex) lines.push(sequence);
    if (position.moves >= depth) return;

    const possible = possibleMoves(position.mask);
    for (const column of COLUMN_ORDER) {
      const move = moveForColumn(position.mask, column);
      if ((possible & move) === 0n) continue;
      if (hasAlignment(position.current | move)) continue;
      explore(play(position, move), `${sequence}${column + 1}`);
    }
  }

  explore({ current: 0n, mask: 0n, moves: 0 }, '');
  process.stdout.write(`${lines.join('\n')}\n`);
  console.error(
    `Enumerated ${visited.size} canonical positions; emitted ${lines.length} for shard `
      + `${shardIndex + 1}/${shardCount} through ply ${depth}.`,
  );
}

function parseScoredLine(line, lineNumber) {
  const tokens = line.trim().split(/\s+/).filter(Boolean);
  let sequence;
  let scoreTokens;

  if (tokens.length === WIDTH) {
    sequence = '';
    scoreTokens = tokens;
  } else if (tokens.length === WIDTH + 1 && /^[1-7]*$/.test(tokens[0])) {
    [sequence] = tokens;
    scoreTokens = tokens.slice(1);
  } else {
    throw new Error(`Invalid scored line ${lineNumber}: ${line}`);
  }

  const scores = scoreTokens.map((token) => Number.parseInt(token, 10));
  if (scores.some((score) => !Number.isInteger(score))) {
    throw new Error(`Non-integer score on line ${lineNumber}.`);
  }
  return { sequence, scores };
}

function encode(entries, maxPly) {
  const bytes = Buffer.alloc(HEADER_SIZE + entries.length * ENTRY_SIZE);
  bytes.write('C4PB', 0, 4, 'ascii');
  bytes.writeUInt8(1, 4);
  bytes.writeUInt8(maxPly, 5);
  bytes.writeUInt8(ENTRY_SIZE, 6);
  bytes.writeUInt8(0, 7);
  bytes.writeUInt32LE(entries.length, 8);

  entries.forEach((entry, index) => {
    const offset = HEADER_SIZE + index * ENTRY_SIZE;
    bytes.writeBigUInt64LE(entry.key, offset);
    bytes.writeUInt8(entry.moveMask, offset + 8);
    bytes.writeInt8(entry.score, offset + 9);
  });
  return bytes;
}

async function pack(options) {
  const inputPath = options.get('input');
  const outputPath = options.get('output') ?? 'assets/perfect-book.bin';
  const manifestPath = options.get('manifest') ?? 'data/perfect-book.manifest.json';
  const maxPly = integerOption(options, 'max-ply', 6, 0);
  const source = String(options.get('source') ?? 'exact solver output');

  if (!inputPath || inputPath === true) throw new Error('--input is required.');

  const text = await readFile(inputPath, 'utf8');
  const entriesByKey = new Map();

  text.split(/\r?\n/).forEach((line, index) => {
    if (!line.trim()) return;
    const { sequence, scores } = parseScoredLine(line, index + 1);
    if (sequence.length > maxPly) return;

    const position = positionFromSequence(sequence);
    const possible = possibleMoves(position.mask);
    let bestScore = -Infinity;
    let moveMask = 0;

    for (let column = 0; column < WIDTH; column += 1) {
      const move = moveForColumn(position.mask, column);
      if ((possible & move) === 0n) continue;
      const score = scores[column];
      if (score > bestScore) {
        bestScore = score;
        moveMask = 1 << column;
      } else if (score === bestScore) {
        moveMask |= 1 << column;
      }
    }

    if (!Number.isFinite(bestScore) || bestScore < -127 || bestScore > 127 || moveMask === 0) {
      throw new Error(`Invalid best move on scored line ${index + 1}.`);
    }

    const canonical = canonicalPosition(position);
    if (canonical.mirrored) moveMask = mirrorMoveMask(moveMask);
    const existing = entriesByKey.get(canonical.key);
    if (existing) {
      if (existing.strongScore !== bestScore) {
        throw new Error(`Conflicting scores for canonical key ${canonical.key}.`);
      }
      existing.moveMask |= moveMask;
    } else {
      entriesByKey.set(canonical.key, {
        key: canonical.key,
        moveMask,
        score: Math.sign(bestScore),
        strongScore: bestScore,
        ply: sequence.length,
      });
    }
  });

  const entries = [...entriesByKey.values()].sort((first, second) => (
    first.key < second.key ? -1 : first.key > second.key ? 1 : 0
  ));

  for (let index = 1; index < entries.length; index += 1) {
    if (entries[index - 1].key >= entries[index].key) {
      throw new Error('Perfect-book keys are not strictly increasing.');
    }
  }

  const bytes = encode(entries, maxPly);
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  const metadata = {
    format: 1,
    maxPly,
    entryCount: entries.length,
    source,
    scoreSemantics: 'game-theoretic outcome; move mask contains strong-optimal moves',
    generatedAt: null,
    sha256,
  };
  const manifest = {
    ...metadata,
    byteLength: bytes.length,
    optimalMoveEntries: entries.reduce((count, entry) => (
      count + ((entry.moveMask & (entry.moveMask - 1)) !== 0 ? 1 : 0)
    ), 0),
  };

  await writeFile(outputPath, bytes);
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  console.error(
    `Packed ${entries.length} exact positions through ply ${maxPly} `
      + `(${bytes.length} bytes, sha256 ${sha256}).`,
  );
}

async function main() {
  const [command, ...rest] = process.argv.slice(2);
  const options = parseOptions(rest);

  if (command === 'enumerate') await enumerate(options);
  else if (command === 'pack') await pack(options);
  else {
    throw new Error(
      'Usage:\n'
        + '  node scripts/perfect-book.mjs enumerate --depth 6\n'
        + '  node scripts/perfect-book.mjs pack --input scored.txt --output assets/perfect-book.bin --max-ply 6',
    );
  }
}

main().catch((error) => fail(error instanceof Error ? error.message : String(error)));

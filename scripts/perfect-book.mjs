#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import process from 'node:process';

import { fail, integerOption, parseArguments } from './cli-options.mjs';
import {
  COLUMN_ORDER, WIDTH, hasAlignment, mirrorBits, moveForColumn, play, possibleMoves,
} from './standard-board.mjs';

const HEADER_SIZE = 12;
const ENTRY_SIZE = 10;
const UINT64_MASK = (1n << 64n) - 1n;
// The committed book (data/perfect-book.manifest.json) is eight plies deep.
const BOOK_PLIES = 8;

// The options each command reads; anything else is refused. A misspelt
// --ouput overwrote the committed book, and --max-pyl packed six plies.
const COMMAND_OPTIONS = Object.freeze({
  enumerate: ['depth', 'shard_count', 'shard_index'],
  pack: ['input', 'output', 'manifest', 'max_ply', 'source'],
});

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
  const depth = integerOption(options.depth, BOOK_PLIES, '--depth');
  const shardCount = integerOption(options.shard_count, 1, '--shard-count', 1);
  const shardIndex = integerOption(options.shard_index, 0, '--shard-index');
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
  const inputPath = options.input;
  const outputPath = options.output ?? 'assets/perfect-book.bin';
  const manifestPath = options.manifest ?? 'data/perfect-book.manifest.json';
  const maxPly = integerOption(options.max_ply, BOOK_PLIES, '--max-ply');
  const source = String(options.source ?? 'exact solver output');

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
  const options = parseArguments(process.argv.slice(2), COMMAND_OPTIONS, {
    usage: 'Usage:\n'
      + '  node scripts/perfect-book.mjs enumerate [--depth 8] [--shard-count N --shard-index I]\n'
      + '  node scripts/perfect-book.mjs pack --input scored.txt [--output assets/perfect-book.bin] [--max-ply 8]',
  });
  if (options.command === 'enumerate') await enumerate(options);
  else await pack(options);
}

main().catch(fail);

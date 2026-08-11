#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import process from 'node:process';

const WIDTH = 7;
const HEIGHT = 6;
const STRIDE = HEIGHT + 1;
const CELL_COUNT = WIDTH * HEIGHT;
const COLUMN_BITS = (1n << BigInt(HEIGHT)) - 1n;
const COLUMN_WITH_SENTINEL = (1n << BigInt(STRIDE)) - 1n;
const BOTTOM_MASKS = Array.from({ length: WIDTH }, (_, column) => 1n << BigInt(column * STRIDE));
const COLUMN_MASKS = BOTTOM_MASKS.map((bottom) => bottom * COLUMN_BITS);
const BOTTOM_MASK = BOTTOM_MASKS.reduce((mask, bit) => mask | bit, 0n);
const BOARD_MASK = BOTTOM_MASK * COLUMN_BITS;
const COLUMN_ORDER = Object.freeze([3, 2, 4, 1, 5, 0, 6]);
const HEADER_SIZE = 12;
const ENTRY_SIZE = 10;
const MAGIC = 'C4PS';
const FORMAT_VERSION = 1;
const ROLE_FIRST = 1;
const ROLE_SECOND = 2;

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

function integerOption(options, name, fallback, minimum = 0, maximum = Infinity) {
  const raw = options.get(name);
  if (raw === undefined) return fallback;
  const value = Number.parseInt(String(raw), 10);
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new Error(`--${name} must be an integer from ${minimum} through ${maximum}.`);
  }
  return value;
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

function canonicalPosition(position) {
  const normal = position.current + position.mask;
  const mirroredCurrent = mirrorBits(position.current);
  const mirroredMask = mirrorBits(position.mask);
  const mirrored = mirroredCurrent + mirroredMask;
  return normal <= mirrored
    ? { key: normal, mirrored: false, position }
    : {
      key: mirrored,
      mirrored: true,
      position: { current: mirroredCurrent, mask: mirroredMask, moves: position.moves },
    };
}

function legalColumns(position) {
  const possible = possibleMoves(position.mask);
  return COLUMN_ORDER.filter((column) => (possible & moveForColumn(position.mask, column)) !== 0n);
}

function continuationCount(position, column) {
  const move = moveForColumn(position.mask, column);
  if (move === 0n) return Infinity;
  if (hasAlignment(position.current | move)) return 0;
  const afterMove = play(position, move);
  const possible = possibleMoves(afterMove.mask);
  if (possible === 0n) return 0;

  const replies = new Set();
  for (const opponentColumn of COLUMN_ORDER) {
    const opponentMove = moveForColumn(afterMove.mask, opponentColumn);
    if ((possible & opponentMove) === 0n) continue;
    if (hasAlignment(afterMove.current | opponentMove)) continue;
    const afterReply = play(afterMove, opponentMove);
    if (possibleMoves(afterReply.mask) === 0n) continue;
    replies.add(canonicalPosition(afterReply).key);
  }
  return replies.size;
}

function compressedChoice(position, columns) {
  let chosen = columns[0];
  let smallestFrontier = continuationCount(position, chosen);
  for (const column of columns.slice(1)) {
    const frontier = continuationCount(position, column);
    if (frontier < smallestFrontier) {
      chosen = column;
      smallestFrontier = frontier;
    }
  }
  return chosen;
}

function chooseCanonicalBest(position, scores) {
  if (!Array.isArray(scores) || scores.length !== WIDTH) {
    throw new Error('An oracle score array must contain seven values.');
  }
  const canonical = canonicalPosition(position);
  const canonicalScores = canonical.mirrored ? [...scores].reverse() : scores;
  const columns = legalColumns(canonical.position);
  if (columns.length === 0) throw new Error('Cannot choose a strategy move on a full board.');

  let bestScore = -Infinity;
  for (const column of columns) bestScore = Math.max(bestScore, canonicalScores[column]);
  if (!Number.isInteger(bestScore) || bestScore < -127 || bestScore > 127) {
    throw new Error(`Invalid exact score ${bestScore}.`);
  }
  const optimalColumns = columns.filter((column) => canonicalScores[column] === bestScore);
  const canonicalColumn = compressedChoice(canonical.position, optimalColumns);
  const actualColumn = canonical.mirrored ? WIDTH - 1 - canonicalColumn : canonicalColumn;
  return {
    key: canonical.key,
    canonicalColumn,
    actualColumn,
    moveMask: 1 << canonicalColumn,
    score: bestScore,
    outcome: Math.sign(bestScore),
  };
}

function chooseKnownExact(position, stored) {
  if (!stored || !Number.isInteger(stored.moveMask) || stored.moveMask <= 0) return null;
  const canonical = canonicalPosition(position);
  const exactColumns = COLUMN_ORDER.filter((column) => (
    (stored.moveMask & (1 << column)) !== 0
      && (possibleMoves(canonical.position.mask)
        & moveForColumn(canonical.position.mask, column)) !== 0n
  ));
  const canonicalColumn = exactColumns.length > 0
    ? compressedChoice(canonical.position, exactColumns)
    : undefined;
  if (canonicalColumn === undefined) {
    throw new Error(`Known exact entry ${canonical.key} has no legal move.`);
  }
  return {
    key: canonical.key,
    canonicalColumn,
    actualColumn: canonical.mirrored ? WIDTH - 1 - canonicalColumn : canonicalColumn,
    moveMask: 1 << canonicalColumn,
    outcome: stored.outcome,
  };
}

function parseScoredLine(line, lineNumber) {
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
    throw new Error(`Invalid oracle output line ${lineNumber}: ${line}`);
  }
  const scores = scoreTokens.map((token) => Number.parseInt(token, 10));
  if (scores.some((score) => !Number.isInteger(score))) {
    throw new Error(`Non-integer oracle score on line ${lineNumber}.`);
  }
  return { sequence, scores };
}

function partition(values, count) {
  const chunks = Array.from({ length: Math.min(count, Math.max(1, values.length)) }, () => []);
  values.forEach((value, index) => chunks[index % chunks.length].push(value));
  return chunks.filter((chunk) => chunk.length > 0);
}

function runOracleChunk(sequences, oraclePath, oracleBookPath) {
  return new Promise((resolve, reject) => {
    const child = spawn(oraclePath, ['-a', '-w', '-b', oracleBookPath], {
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Exact oracle exited with ${code}: ${stderr.trim()}`));
        return;
      }
      try {
        const parsed = new Map();
        stdout.split(/\r?\n/).forEach((line, index) => {
          if (!line.trim()) return;
          const result = parseScoredLine(line, index + 1);
          if (parsed.has(result.sequence)) {
            throw new Error(`Duplicate oracle result for sequence "${result.sequence}".`);
          }
          parsed.set(result.sequence, result.scores);
        });
        for (const sequence of sequences) {
          if (!parsed.has(sequence)) {
            throw new Error(`The exact oracle omitted sequence "${sequence}".`);
          }
        }
        resolve(parsed);
      } catch (error) {
        reject(error);
      }
    });
    child.stdin.end(`${sequences.join('\n')}\n`);
  });
}

async function scoreWithOracle(sequences, options) {
  const oraclePath = String(options.oraclePath);
  const oracleBookPath = String(options.oracleBookPath);
  const workers = Math.min(options.workers, Math.max(1, sequences.length));
  const maps = await Promise.all(
    partition(sequences, workers)
      .map((chunk) => runOracleChunk(chunk, oraclePath, oracleBookPath)),
  );
  const merged = new Map();
  for (const map of maps) {
    for (const [sequence, scores] of map) {
      if (merged.has(sequence)) throw new Error(`Duplicate merged oracle result for "${sequence}".`);
      merged.set(sequence, scores);
    }
  }
  return merged;
}

function addRepresentative(target, position, sequence) {
  const key = canonicalPosition(position).key;
  if (!target.has(key)) target.set(key, { position, sequence });
}

function startingFrontier(aiStarts) {
  const root = { current: 0n, mask: 0n, moves: 0 };
  const frontier = new Map();
  if (aiStarts) {
    addRepresentative(frontier, root, '');
    return frontier;
  }
  for (const column of legalColumns(root)) {
    const move = moveForColumn(root.mask, column);
    addRepresentative(frontier, play(root, move), String(column + 1));
  }
  return frontier;
}

function mergeEntry(entries, entry, roleFlag) {
  const existing = entries.get(entry.key);
  if (!existing) {
    entries.set(entry.key, { ...entry, roles: roleFlag });
    return;
  }
  if (existing.moveMask !== entry.moveMask || existing.outcome !== entry.outcome) {
    throw new Error(`Conflicting exact strategy for canonical key ${entry.key}.`);
  }
  existing.roles |= roleFlag;
}

function terminalAfterMove(position, move) {
  if (hasAlignment(position.current | move)) return 'mover-win';
  const next = play(position, move);
  if (possibleMoves(next.mask) === 0n) return 'draw';
  return null;
}

async function buildRole({
  aiStarts,
  handoffRemaining,
  entries,
  scoreBatch,
  exactLookup,
}) {
  const role = aiStarts ? 'first' : 'second';
  const roleFlag = aiStarts ? ROLE_FIRST : ROLE_SECOND;
  const seen = new Set();
  let frontier = startingFrontier(aiStarts);
  const stats = {
    role,
    decisionEntries: 0,
    handoffPositions: 0,
    maximumDecisionPly: -1,
    aiWins: 0,
    opponentWins: 0,
    draws: 0,
    exactTableHits: 0,
    oracleDecisions: 0,
    layerSizes: {},
  };

  while (frontier.size > 0) {
    const representatives = [...frontier.values()];
    const ply = representatives[0].position.moves;
    if (representatives.some(({ position }) => position.moves !== ply)) {
      throw new Error(`Mixed strategy frontier plies for the ${role}-player role.`);
    }
    stats.layerSizes[ply] = representatives.length;

    const actionable = [];
    for (const representative of representatives) {
      const canonical = canonicalPosition(representative.position);
      if (seen.has(canonical.key)) continue;
      seen.add(canonical.key);
      if (CELL_COUNT - representative.position.moves <= handoffRemaining) {
        stats.handoffPositions += 1;
      } else {
        actionable.push(representative);
      }
    }
    if (actionable.length === 0) break;

    const choices = new Map();
    const unresolved = [];
    for (const representative of actionable) {
      const canonical = canonicalPosition(representative.position);
      const known = typeof exactLookup === 'function' ? exactLookup(canonical.key) : null;
      if (known) {
        choices.set(representative.sequence, chooseKnownExact(representative.position, known));
        stats.exactTableHits += 1;
      } else {
        unresolved.push(representative);
      }
    }
    const scoresBySequence = unresolved.length > 0
      ? await scoreBatch(unresolved.map(({ sequence }) => sequence))
      : new Map();
    stats.oracleDecisions += unresolved.length;
    for (const { position, sequence } of unresolved) {
      const scores = scoresBySequence.get(sequence);
      if (!scores) throw new Error(`No exact scores for sequence "${sequence}".`);
      choices.set(sequence, chooseCanonicalBest(position, scores));
    }

    const next = new Map();
    for (const { position, sequence } of actionable) {
      const choice = choices.get(sequence);
      if (!choice) throw new Error(`No exact choice for sequence "${sequence}".`);
      mergeEntry(entries, {
        key: choice.key,
        moveMask: choice.moveMask,
        outcome: choice.outcome,
      }, roleFlag);
      stats.decisionEntries += 1;
      stats.maximumDecisionPly = Math.max(stats.maximumDecisionPly, position.moves);

      const aiMove = moveForColumn(position.mask, choice.actualColumn);
      const aiTerminal = terminalAfterMove(position, aiMove);
      if (aiTerminal === 'mover-win') {
        stats.aiWins += 1;
        continue;
      }
      if (aiTerminal === 'draw') {
        stats.draws += 1;
        continue;
      }

      const afterAi = play(position, aiMove);
      for (const opponentColumn of legalColumns(afterAi)) {
        const opponentMove = moveForColumn(afterAi.mask, opponentColumn);
        const opponentTerminal = terminalAfterMove(afterAi, opponentMove);
        if (opponentTerminal === 'mover-win') {
          stats.opponentWins += 1;
          continue;
        }
        if (opponentTerminal === 'draw') {
          stats.draws += 1;
          continue;
        }
        addRepresentative(
          next,
          play(afterAi, opponentMove),
          `${sequence}${choice.actualColumn + 1}${opponentColumn + 1}`,
        );
      }
    }

    console.error(
      `${role}: resolved ${actionable.length} exact AI decisions at ply ${ply} `
        + `(${unresolved.length} oracle); ${next.size} canonical replies continue.`,
    );
    frontier = next;
  }

  return stats;
}

function encodeStrategy(entries, handoffRemaining, roleFlags) {
  const sorted = [...entries.values()].sort((first, second) => (
    first.key < second.key ? -1 : first.key > second.key ? 1 : 0
  ));
  const bytes = Buffer.alloc(HEADER_SIZE + sorted.length * ENTRY_SIZE);
  bytes.write(MAGIC, 0, 4, 'ascii');
  bytes.writeUInt8(FORMAT_VERSION, 4);
  bytes.writeUInt8(handoffRemaining, 5);
  bytes.writeUInt8(ENTRY_SIZE, 6);
  bytes.writeUInt8(roleFlags, 7);
  bytes.writeUInt32LE(sorted.length, 8);
  sorted.forEach((entry, index) => {
    const offset = HEADER_SIZE + index * ENTRY_SIZE;
    bytes.writeBigUInt64LE(entry.key, offset);
    bytes.writeUInt8(entry.moveMask, offset + 8);
    bytes.writeInt8(entry.outcome, offset + 9);
  });
  return { bytes, sorted };
}

export function decodeStrategy(input) {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (bytes.length < HEADER_SIZE) throw new Error('Strategy data is truncated.');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = String.fromCharCode(...bytes.subarray(0, 4));
  if (magic !== MAGIC) throw new Error('Strategy magic is invalid.');
  const version = view.getUint8(4);
  const handoffRemaining = view.getUint8(5);
  const entrySize = view.getUint8(6);
  const roleFlags = view.getUint8(7);
  const entryCount = view.getUint32(8, true);
  if (version !== FORMAT_VERSION || entrySize !== ENTRY_SIZE) {
    throw new Error('Unsupported strategy format.');
  }
  if ((roleFlags & ~(ROLE_FIRST | ROLE_SECOND)) !== 0 || roleFlags === 0) {
    throw new Error('Strategy role flags are invalid.');
  }
  if (bytes.length !== HEADER_SIZE + entryCount * ENTRY_SIZE) {
    throw new Error('Strategy length is invalid.');
  }
  const entries = [];
  let previous = -1n;
  for (let index = 0; index < entryCount; index += 1) {
    const offset = HEADER_SIZE + index * ENTRY_SIZE;
    const key = view.getBigUint64(offset, true);
    const moveMask = view.getUint8(offset + 8);
    const outcome = view.getInt8(offset + 9);
    if (key <= previous) throw new Error('Strategy keys are not strictly increasing.');
    if (moveMask === 0 || (moveMask & (moveMask - 1)) !== 0 || (moveMask & 0x80) !== 0) {
      throw new Error('Strategy entries must contain exactly one legal-column bit.');
    }
    if (outcome < -1 || outcome > 1) throw new Error('Strategy outcome is invalid.');
    entries.push({ key, moveMask, outcome });
    previous = key;
  }
  return { version, handoffRemaining, roleFlags, entries };
}

function strategyLookup(entries) {
  return new Map(entries.map((entry) => [entry.key, entry]));
}

function initialProofFrontier(aiStarts) {
  return startingFrontier(aiStarts);
}

export function verifyClosure(decoded) {
  const lookup = strategyLookup(decoded.entries);
  const usedKeys = new Set();
  const roleResults = {};
  for (const aiStarts of [true, false]) {
    const flag = aiStarts ? ROLE_FIRST : ROLE_SECOND;
    if ((decoded.roleFlags & flag) === 0) continue;
    const role = aiStarts ? 'first' : 'second';
    const seen = new Set();
    let frontier = initialProofFrontier(aiStarts);
    let decisions = 0;
    let handoffs = 0;
    let terminals = 0;

    while (frontier.size > 0) {
      const next = new Map();
      for (const { position, sequence } of frontier.values()) {
        const canonical = canonicalPosition(position);
        if (seen.has(canonical.key)) continue;
        seen.add(canonical.key);
        if (CELL_COUNT - position.moves <= decoded.handoffRemaining) {
          handoffs += 1;
          continue;
        }
        const entry = lookup.get(canonical.key);
        if (!entry) {
          throw new Error(`Missing ${role}-role strategy entry at sequence "${sequence}".`);
        }
        usedKeys.add(canonical.key);
        const canonicalColumn = Math.log2(entry.moveMask);
        const actualColumn = canonical.mirrored ? WIDTH - 1 - canonicalColumn : canonicalColumn;
        const aiMove = moveForColumn(position.mask, actualColumn);
        if (aiMove === 0n) throw new Error(`Illegal strategy move at sequence "${sequence}".`);
        decisions += 1;
        const aiTerminal = terminalAfterMove(position, aiMove);
        if (aiTerminal) {
          terminals += 1;
          continue;
        }
        const afterAi = play(position, aiMove);
        for (const opponentColumn of legalColumns(afterAi)) {
          const opponentMove = moveForColumn(afterAi.mask, opponentColumn);
          const opponentTerminal = terminalAfterMove(afterAi, opponentMove);
          if (opponentTerminal) {
            terminals += 1;
            continue;
          }
          addRepresentative(
            next,
            play(afterAi, opponentMove),
            `${sequence}${actualColumn + 1}${opponentColumn + 1}`,
          );
        }
      }
      frontier = next;
    }
    roleResults[role] = { decisions, handoffs, terminals, visited: seen.size };
  }
  if (usedKeys.size !== decoded.entries.length) {
    throw new Error(
      `Strategy contains ${decoded.entries.length - usedKeys.size} unreachable entries.`,
    );
  }
  return roleResults;
}

export async function buildStrategy({
  handoffRemaining = 24,
  roles = 'both',
  source = 'exact oracle',
  scoreBatch,
  exactLookup = null,
}) {
  if (!Number.isInteger(handoffRemaining) || handoffRemaining < 0 || handoffRemaining > CELL_COUNT) {
    throw new Error('handoffRemaining must be between 0 and 42.');
  }
  if (typeof scoreBatch !== 'function') throw new TypeError('scoreBatch is required.');
  const entries = new Map();
  const roleStats = {};
  let roleFlags = 0;
  if (roles === 'both' || roles === 'first') {
    roleFlags |= ROLE_FIRST;
    roleStats.first = await buildRole({
      aiStarts: true,
      handoffRemaining,
      entries,
      scoreBatch,
      exactLookup,
    });
  }
  if (roles === 'both' || roles === 'second') {
    roleFlags |= ROLE_SECOND;
    roleStats.second = await buildRole({
      aiStarts: false,
      handoffRemaining,
      entries,
      scoreBatch,
      exactLookup,
    });
  }
  if (roleFlags === 0) throw new Error('roles must be first, second, or both.');
  const { bytes, sorted } = encodeStrategy(entries, handoffRemaining, roleFlags);
  const decoded = decodeStrategy(bytes);
  const closure = verifyClosure(decoded);
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  return {
    bytes,
    manifest: {
      format: FORMAT_VERSION,
      handoffRemaining,
      roleFlags,
      entryCount: sorted.length,
      byteLength: bytes.length,
      sha256,
      source,
      policy: 'one exact game-theoretic move minimizing the next canonical opponent frontier, then centre-first',
      roleStats,
      closure,
      generatedAt: null,
    },
  };
}

async function buildCommand(options) {
  const oraclePath = options.get('oracle');
  const oracleBookPath = options.get('oracle-book');
  if (!oraclePath || oraclePath === true) throw new Error('--oracle is required.');
  if (!oracleBookPath || oracleBookPath === true) throw new Error('--oracle-book is required.');
  const outputPath = String(options.get('output') ?? 'assets/perfect-strategy.bin');
  const manifestPath = String(options.get('manifest') ?? 'data/perfect-strategy.manifest.json');
  const exactTablePath = options.get('exact-table');
  let exactTable = null;
  if (exactTablePath && exactTablePath !== true) {
    const { decodePerfectBook } = await import('../src/perfect-book.js');
    exactTable = decodePerfectBook(await readFile(String(exactTablePath)));
  }
  const handoffRemaining = integerOption(options, 'handoff-remaining', 24, 0, CELL_COUNT);
  const workers = integerOption(options, 'workers', 8, 1, 32);
  const roles = String(options.get('roles') ?? 'both');
  const source = String(options.get('source') ?? 'exact oracle');
  const result = await buildStrategy({
    handoffRemaining,
    roles,
    source,
    scoreBatch: (sequences) => scoreWithOracle(sequences, {
      oraclePath,
      oracleBookPath,
      workers,
    }),
    exactLookup: exactTable ? (key) => exactTable.lookup(key) : null,
  });
  await writeFile(outputPath, result.bytes);
  await writeFile(manifestPath, `${JSON.stringify(result.manifest, null, 2)}\n`);
  console.error(
    `Packed ${result.manifest.entryCount} exact strategy decisions `
      + `(${result.manifest.byteLength} bytes, sha256 ${result.manifest.sha256}).`,
  );
}

async function verifyCommand(options) {
  const inputPath = options.get('input') ?? 'assets/perfect-strategy.bin';
  const bytes = await readFile(inputPath);
  const decoded = decodeStrategy(bytes);
  const result = verifyClosure(decoded);
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

async function main() {
  const [command, ...rest] = process.argv.slice(2);
  const options = parseOptions(rest);
  if (command === 'build') await buildCommand(options);
  else if (command === 'verify') await verifyCommand(options);
  else throw new Error(
    'Usage:\n'
      + '  node scripts/perfect-strategy.mjs build --oracle ./c4solver --oracle-book ./7x6.book --exact-table assets/perfect-book.bin\n'
      + '  node scripts/perfect-strategy.mjs verify --input assets/perfect-strategy.bin',
  );
}

const invokedAsScript = process.argv[1]
  && import.meta.url === pathToFileURL(process.argv[1]).href;
if (invokedAsScript) main().catch((error) => fail(error instanceof Error ? error.message : String(error)));

export const STRATEGY_CONSTANTS = Object.freeze({
  WIDTH,
  HEIGHT,
  CELL_COUNT,
  ROLE_FIRST,
  ROLE_SECOND,
  COLUMN_ORDER,
});

#!/usr/bin/env node
// The release gate's independent replay of classic policies, and its
// verify-reference command. The gate fingerprints this file and everything it
// imports, so generation, the small native references and merge-manifests
// live in scripts/perfect-classic-policy-generator.mjs, where an edit costs
// no replay, and policies are decoded by src/perfect-classic-format.js
// rather than the page's src/perfect-classic-policy.js.
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';

import { integerOption, parseArguments as parseCommand } from './cli-options.mjs';
import { isEntryPoint } from './entry-point.mjs';
import {
  PERFECT_CLASSIC_ROLE_FIRST,
  decodePerfectClassicRecords,
} from '../src/perfect-classic-format.js';

// A catalog names its policies as plain files beside the manifest; anything
// else could make the replay read bytes from outside the hashed catalog.
const POLICY_FILE = /^\.\/[A-Za-z0-9][A-Za-z0-9._-]*\.bin$/;
const AI_TURN_BIT = 1n << 63n;
const WIN = 1;
const DRAW = 0;
const LOSS = -1;

// The options verify-reference reads. Anything else is refused rather than
// ignored: `verify-reference --refrence candidate.json` used to verify some
// other --reference and exit 0.
const COMMAND_OPTIONS = Object.freeze({
  'verify-reference': ['reference', 'verify_table_bits', 'maximum_verify_nodes'],
});

export function parseArguments(argv) {
  return parseCommand(argv, COMMAND_OPTIONS, {
    usage: 'Usage: node scripts/perfect-classic-policy.mjs verify-reference --reference <manifest.json> '
      + '[--verify-table-bits N] [--maximum-verify-nodes N]\n'
      + 'generate, verify and merge-manifests are in scripts/perfect-classic-policy-generator.mjs.',
  });
}

function createGeometry(rows, columns, connect) {
  const stride = rows + 1;
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
  return {
    rows,
    columns,
    connect,
    stride,
    cellCount: rows * columns,
    columnWithSentinel,
    bottomMasks,
    columnMasks,
    bottomMask,
    boardMask,
    columnOrder,
    directions: [1, stride - 1, stride, stride + 1],
  };
}

function possibleMoves(geometry, mask) {
  return (mask + geometry.bottomMask) & geometry.boardMask;
}

function moveForColumn(geometry, mask, column) {
  if (!Number.isInteger(column) || column < 0 || column >= geometry.columns) return 0n;
  return (mask + geometry.bottomMasks[column]) & geometry.columnMasks[column];
}

function play(position, move) {
  return {
    current: position.current ^ position.mask,
    mask: position.mask | move,
    moves: position.moves + 1,
  };
}

function hasAlignment(geometry, bits) {
  for (const direction of geometry.directions) {
    const shift = BigInt(direction);
    let run = bits;
    for (let offset = 1; offset < geometry.connect && run !== 0n; offset += 1) {
      run &= bits >> (BigInt(offset) * shift);
    }
    if (run !== 0n) return true;
  }
  return false;
}

function mirrorBits(geometry, bits) {
  let mirrored = 0n;
  for (let column = 0; column < geometry.columns; column += 1) {
    const group = (bits >> BigInt(column * geometry.stride)) & geometry.columnWithSentinel;
    mirrored |= group << BigInt((geometry.columns - 1 - column) * geometry.stride);
  }
  return mirrored;
}

function canonicalize(geometry, position) {
  const normal = position.current + position.mask;
  const mirroredCurrent = mirrorBits(geometry, position.current);
  const mirroredMask = mirrorBits(geometry, position.mask);
  const mirrored = mirroredCurrent + mirroredMask;
  return normal <= mirrored
    ? { position, key: normal }
    : {
      position: { current: mirroredCurrent, mask: mirroredMask, moves: position.moves },
      key: mirrored,
    };
}

function immediateWinningMoves(geometry, position, pieces = position.current) {
  const possible = possibleMoves(geometry, position.mask);
  let winning = 0n;
  for (const column of geometry.columnOrder) {
    const move = moveForColumn(geometry, position.mask, column);
    if ((possible & move) !== 0n && hasAlignment(geometry, pieces | move)) winning |= move;
  }
  return winning;
}

function nonLosingMoves(geometry, position) {
  let possible = possibleMoves(geometry, position.mask);
  const opponent = position.current ^ position.mask;
  const opponentWins = immediateWinningMoves(geometry, position, opponent);
  if (opponentWins !== 0n) {
    if ((opponentWins & (opponentWins - 1n)) !== 0n) return 0n;
    possible = opponentWins;
  }

  let safe = 0n;
  for (const column of geometry.columnOrder) {
    const move = moveForColumn(geometry, position.mask, column);
    if ((possible & move) === 0n) continue;
    if (hasAlignment(geometry, position.current | move)) {
      safe |= move;
      continue;
    }
    const child = play(position, move);
    if (immediateWinningMoves(geometry, child) === 0n) safe |= move;
  }
  return safe;
}

/**
 * Independently re-solves policy handoff states with a fixed-size direct-mapped
 * table. Replacement collisions can cost work but cannot produce false hits,
 * so memory use is deterministic and an unfinished search fails closed.
 */
class IndependentExactSolver {
  constructor(geometry, options = {}) {
    this.geometry = geometry;
    this.maximumNodes = options.maximumNodes ?? Infinity;
    this.tableBits = options.tableBits ?? 22;
    if (!Number.isInteger(this.tableBits) || this.tableBits < 8 || this.tableBits > 25) {
      throw new RangeError('Independent verification table bits must be from 8 through 25.');
    }
    this.size = 2 ** this.tableBits;
    this.indexMask = BigInt(this.size - 1);
    this.keys = new BigUint64Array(this.size);
    this.lowerBounds = new Int8Array(this.size);
    this.upperBounds = new Int8Array(this.size);
    this.flags = new Uint8Array(this.size);
    this.nodes = 0;
    this.hits = 0;
    this.stores = 0;
    this.collisions = 0;
  }

  index(key) {
    return Number((key ^ (key >> 23n) ^ (key >> 41n)) & this.indexMask);
  }

  probe(key) {
    const index = this.index(key);
    if (this.keys[index] !== key + 1n) return null;
    this.hits += 1;
    return {
      lower: (this.flags[index] & 1) === 0 ? -2 : this.lowerBounds[index],
      upper: (this.flags[index] & 2) === 0 ? 2 : this.upperBounds[index],
    };
  }

  prepare(key) {
    const index = this.index(key);
    const stored = this.keys[index];
    if (stored !== 0n && stored !== key + 1n) this.collisions += 1;
    if (stored !== key + 1n) {
      this.keys[index] = key + 1n;
      this.flags[index] = 0;
    }
    return index;
  }

  storeLower(key, score) {
    const index = this.prepare(key);
    if ((this.flags[index] & 1) !== 0 && score <= this.lowerBounds[index]) return;
    this.lowerBounds[index] = score;
    this.flags[index] |= 1;
    this.stores += 1;
  }

  storeUpper(key, score) {
    const index = this.prepare(key);
    if ((this.flags[index] & 2) !== 0 && score >= this.upperBounds[index]) return;
    this.upperBounds[index] = score;
    this.flags[index] |= 2;
    this.stores += 1;
  }

  visit() {
    this.nodes += 1;
    if (this.nodes > this.maximumNodes) {
      const error = new RangeError('Independent classic policy replay exceeded its node limit.');
      error.code = 'CLASSIC_POLICY_VERIFY_NODE_LIMIT';
      error.nodes = this.nodes;
      throw error;
    }
  }

  search(rawPosition, alpha, beta) {
    this.visit();
    const canonical = canonicalize(this.geometry, rawPosition);
    const position = canonical.position;
    const possible = possibleMoves(this.geometry, position.mask);
    if (possible === 0n) return DRAW;
    if (immediateWinningMoves(this.geometry, position) !== 0n) return WIN;

    const safe = nonLosingMoves(this.geometry, position);
    if (safe === 0n) return LOSS;
    if (position.moves >= this.geometry.cellCount - 2) return DRAW;

    const cached = this.probe(canonical.key);
    if (cached) {
      if (cached.lower >= beta) return cached.lower;
      if (cached.upper <= alpha) return cached.upper;
      alpha = Math.max(alpha, cached.lower);
      beta = Math.min(beta, cached.upper);
      if (alpha >= beta) return alpha;
    }

    for (const column of this.geometry.columnOrder) {
      const move = moveForColumn(this.geometry, position.mask, column);
      if ((safe & move) === 0n) continue;
      const childValue = this.search(play(position, move), -beta, -alpha);
      const value = childValue === DRAW ? DRAW : -childValue;
      if (value >= beta) {
        this.storeLower(canonical.key, value);
        return value;
      }
      if (value > alpha) alpha = value;
    }
    this.storeUpper(canonical.key, alpha);
    return alpha;
  }

  solve(position) {
    let minimum = LOSS;
    let maximum = WIN;
    while (minimum < maximum) {
      const middle = minimum + Math.floor((maximum - minimum) / 2);
      const value = this.search(position, middle, middle + 1);
      if (value <= middle) maximum = value;
      else minimum = value;
    }
    return minimum;
  }
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

export function replayPerfectClassicPolicy(policy, options = {}) {
  const geometry = createGeometry(policy.rows, policy.columns, policy.connect);
  const exact = new IndependentExactSolver(geometry, {
    maximumNodes: options.maximumExactNodes ?? Infinity,
    tableBits: options.exactTableBits ?? 22,
  });
  const usedPolicy = new Set();
  const values = new Map();
  const visiting = new Set();
  let closureStates = 0;
  let handoffStates = 0;
  let terminalAiWins = 0;
  let terminalAiLosses = 0;
  let terminalDraws = 0;

  const evaluate = (rawPosition, aiTurn) => {
    const canonical = canonicalize(geometry, rawPosition);
    const stateKey = canonical.key | (aiTurn ? AI_TURN_BIT : 0n);
    const cached = values.get(stateKey);
    if (cached !== undefined) return cached;
    if (visiting.has(stateKey)) throw new Error('Classic policy closure unexpectedly contains a cycle.');
    visiting.add(stateKey);
    closureStates += 1;
    const position = canonical.position;
    const remaining = geometry.cellCount - position.moves;
    let value;

    if (aiTurn && remaining <= policy.handoffRemaining) {
      handoffStates += 1;
      value = exact.solve(position);
    } else if (aiTurn) {
      const record = policy.lookupKey(canonical.key);
      if (!record) {
        throw new Error(`Perfect classic policy is missing reachable key ${canonical.key}.`);
      }
      usedPolicy.add(canonical.key);
      const column = moveMaskColumn(record.moveMask, geometry.columns);
      const move = moveForColumn(geometry, position.mask, column);
      if (move === 0n) throw new Error('Perfect classic policy selects an illegal move.');
      if (hasAlignment(geometry, position.current | move)) {
        terminalAiWins += 1;
        value = WIN;
      } else {
        const child = play(position, move);
        if (possibleMoves(geometry, child.mask) === 0n) {
          terminalDraws += 1;
          value = DRAW;
        } else value = evaluate(child, false);
      }
      if (record.outcome !== value) {
        throw new Error(
          `Perfect classic policy outcome mismatch at key ${canonical.key}: `
          + `${record.outcome} instead of ${value}.`,
        );
      }
    } else {
      const possible = possibleMoves(geometry, position.mask);
      if (possible === 0n) {
        terminalDraws += 1;
        value = DRAW;
      } else {
        value = WIN;
        for (const column of geometry.columnOrder) {
          const move = moveForColumn(geometry, position.mask, column);
          if ((possible & move) === 0n) continue;
          let candidate;
          if (hasAlignment(geometry, position.current | move)) {
            terminalAiLosses += 1;
            candidate = LOSS;
          } else {
            const child = play(position, move);
            if (possibleMoves(geometry, child.mask) === 0n) {
              terminalDraws += 1;
              candidate = DRAW;
            } else candidate = evaluate(child, true);
          }
          if (candidate < value) value = candidate;
          // Continue through every legal opponent action so the replay \
          // validates the complete generated closure even after a loss is found.
        }
      }
    }

    visiting.delete(stateKey);
    values.set(stateKey, value);
    return value;
  };

  const rootValue = evaluate(
    { current: 0n, mask: 0n, moves: 0 },
    policy.role === PERFECT_CLASSIC_ROLE_FIRST,
  );
  if (rootValue !== policy.rootValue) {
    throw new Error(
      `Perfect classic policy root value mismatch: ${rootValue} instead of ${policy.rootValue}.`,
    );
  }
  if (usedPolicy.size !== policy.entryCount) {
    throw new Error(
      `Perfect classic policy has ${policy.entryCount - usedPolicy.size} unreachable record(s).`,
    );
  }
  if (closureStates !== policy.closureStates) {
    throw new Error(
      `Perfect classic policy closure mismatch: ${closureStates} instead of ${policy.closureStates}.`,
    );
  }
  return {
    format: 'connect4-perfect-classic-policy-replay-v1',
    rows: policy.rows,
    columns: policy.columns,
    connect: policy.connect,
    role: policy.role,
    handoffRemaining: policy.handoffRemaining,
    rootValue,
    entryCount: policy.entryCount,
    closureStates,
    handoffStates,
    terminalAiWins,
    terminalAiLosses,
    terminalDraws,
    exactNodes: exact.nodes,
    exactTableHits: exact.hits,
    exactTableStores: exact.stores,
    exactTableCollisions: exact.collisions,
    exactTableBits: exact.tableBits,
  };
}

async function verifyPolicyManifest(path, options = {}) {
  const manifestPath = resolve(path);
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  if (manifest?.format !== 'connect4-perfect-classic-manifest-v1'
      || !Array.isArray(manifest.policies)) {
    throw new Error('Perfect classic policy manifest format is invalid.');
  }
  // An empty catalog would pass without replaying anything.
  if (manifest.policies.length === 0) {
    throw new Error('Perfect classic policy manifest lists no policies.');
  }
  const directory = dirname(manifestPath);
  const maximumExactNodes = options.maximum_verify_nodes === undefined
    ? Infinity
    : integerOption(
      options.maximum_verify_nodes,
      0,
      'maximum-verify-nodes',
      1,
      Number.MAX_SAFE_INTEGER,
    );
  const verifyTableBits = integerOption(
    options.verify_table_bits,
    22,
    'verify-table-bits',
    8,
    25,
  );
  const replay = [];
  const identities = new Set();
  for (const entry of manifest.policies) {
    const identity = `${entry?.rows}x${entry?.columns}:c${entry?.connect}:r${entry?.role}`;
    if (identities.has(identity)) throw new Error(`Duplicate perfect classic policy ${identity}.`);
    identities.add(identity);
    if (typeof entry.file !== 'string' || !POLICY_FILE.test(entry.file)) {
      throw new Error(`Perfect classic policy ${identity} must name a ./<name>.bin file beside its manifest.`);
    }
    // One read serves both the digest and the decoder, so the bytes replayed
    // are exactly the bytes hashed even if the file changes underneath.
    const bytes = await readFile(join(directory, entry.file));
    const sha256 = createHash('sha256').update(bytes).digest('hex');
    if (bytes.length !== entry.bytes || sha256 !== entry.sha256) {
      throw new Error(`Perfect classic policy hash mismatch for ${entry.file}.`);
    }
    const policy = decodePerfectClassicRecords(bytes, entry);
    if (policy.handoffRemaining !== entry.handoffRemaining
        || policy.rootValue !== entry.rootValue
        || policy.entryCount !== entry.entryCount
        || policy.closureStates !== entry.closureStates) {
      throw new Error(`Perfect classic policy metadata mismatch for ${entry.file}.`);
    }
    replay.push(replayPerfectClassicPolicy(policy, {
      maximumExactNodes,
      exactTableBits: verifyTableBits,
    }));
  }
  return { manifestPath, replay };
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (!options.reference || options.reference === true) throw new RangeError('--reference is required.');
  const verified = await verifyPolicyManifest(options.reference, options);
  process.stdout.write(`${JSON.stringify(verified, null, 2)}\n`);
}

// Tests and the generator import the replay; only a direct run verifies.
if (isEntryPoint(import.meta.url)) await main();

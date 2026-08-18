#!/usr/bin/env node

// Independently replays the committed complete Chaos Mode certificates.
//
// Every transition comes from src/engine.js — applyAction and
// resolveActionOutcome — rather than from the solver that produced the
// certificate, so the rules used to check a policy are the rules the game plays
// by. Only the record layout is shared with the generator.
//
// What a passing replay establishes, over the complete adversarial closure from
// the empty board:
//
//   * every reachable AI position has exactly one stored action, and it is legal;
//   * the outcome the policy forces from each AI position equals the value stored
//     in its record, where a repetition cycle counts as a draw, so a "win" that
//     only shuffles pieces forever fails;
//   * the replayed root value matches the header and the manifest;
//   * no record is unreachable, and the closure size matches the header.

import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { constants as fsConstants } from 'node:fs';
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  ACTION_DROP,
  ACTION_FLIP,
  ACTION_ROTATE_CCW,
  ACTION_ROTATE_CW,
  EMPTY,
  RED,
  YELLOW,
  applyAction,
  createBoard,
  isBoardFull,
  legalActions,
  resolveActionOutcome,
} from '../src/engine.js';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const DEFAULT_REFERENCE = resolve(ROOT, 'data/perfect-chaos-complete/manifest.json');
const SOURCE = resolve(ROOT, 'native/perfect-chaos-complete.cpp');
const MANIFEST_FORMAT = 'connect4-perfect-chaos-complete-manifest-v1';
const ACTION_TYPES = Object.freeze([
  ACTION_DROP,
  ACTION_FLIP,
  ACTION_ROTATE_CW,
  ACTION_ROTATE_CCW,
]);
const MAGIC = 'C4CFUL1\0';
const HEADER_SIZE = 24;
const RECORD_SIZE = 24;
const WIN = 1;
const DRAW = 0;
const LOSS = -1;

function parseArguments(argv) {
  const options = { command: argv[0] ?? 'verify-reference' };
  for (let index = 1; index < argv.length; index += 1) {
    const argument = argv[index];
    if (!argument.startsWith('--')) throw new RangeError(`Unexpected argument: ${argument}`);
    const name = argument.slice(2).replaceAll('-', '_');
    const value = argv[index + 1];
    if (name === 'input') {
      if (value === undefined || value.startsWith('--')) throw new RangeError('--input requires a path.');
      options.inputs ??= [];
      options.inputs.push(value);
      index += 1;
    } else if (value === undefined || value.startsWith('--')) options[name] = true;
    else {
      options[name] = value;
      index += 1;
    }
  }
  return options;
}

function decode(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let magic = '';
  for (let index = 0; index < 8; index += 1) magic += String.fromCharCode(bytes[index]);
  if (magic !== MAGIC) throw new Error('Perfect Chaos policy magic is invalid.');
  if (view.getUint8(8) !== 1) throw new Error('Unsupported Perfect Chaos policy version.');

  const header = {
    rows: view.getUint8(9),
    columns: view.getUint8(10),
    connect: view.getUint8(11),
    role: view.getUint8(12),
    rootValue: view.getInt8(13),
    entryCount: view.getUint32(16, true),
    closureStates: view.getUint32(20, true),
  };
  if (view.getUint8(14) !== RECORD_SIZE) {
    throw new Error('Unsupported Perfect Chaos policy record size.');
  }
  if (header.role !== 1 && header.role !== 2) {
    throw new Error('Perfect Chaos policy role must be first or second.');
  }
  if (header.rootValue < -1 || header.rootValue > 1) {
    throw new Error('Perfect Chaos policy root value must be -1, 0, or 1.');
  }
  if (bytes.byteLength !== HEADER_SIZE + header.entryCount * RECORD_SIZE) {
    throw new Error('Perfect Chaos policy length mismatch.');
  }

  const records = new Map();
  for (let index = 0; index < header.entryCount; index += 1) {
    const offset = HEADER_SIZE + index * RECORD_SIZE;
    const mover = view.getBigUint64(offset, true);
    const opponent = view.getBigUint64(offset + 8, true);
    const rows = view.getUint8(offset + 16);
    const columns = view.getUint8(offset + 17);
    const action = view.getUint8(offset + 18);
    const column = view.getUint8(offset + 19);
    const value = view.getInt8(offset + 20);
    if ((mover & opponent) !== 0n) throw new Error('Perfect Chaos record has overlapping pieces.');
    // A rotation transposes the board, so both orientations belong in one file.
    const orientationMatches = (rows === header.rows && columns === header.columns)
      || (rows === header.columns && columns === header.rows);
    if (!orientationMatches) {
      throw new Error('Perfect Chaos record has unexpected dimensions.');
    }
    if (!ACTION_TYPES[action]) throw new Error('Perfect Chaos record has an invalid action.');
    if (action === 0 ? column >= columns : column !== 0) {
      throw new Error('Perfect Chaos record has an invalid action column.');
    }
    if (value < -1 || value > 1) throw new Error('Perfect Chaos record value is out of range.');
    const key = `${rows}x${columns}:${mover}:${opponent}`;
    if (records.has(key)) throw new Error('Perfect Chaos policy contains a duplicate record.');
    records.set(key, { action, column, value });
  }
  return { ...header, records };
}

function packedKey(board, mover) {
  const rows = board.length;
  const columns = board[0].length;
  const stride = rows + 1;
  let moverBits = 0n;
  let opponentBits = 0n;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const cell = board[row][column];
      if (cell === EMPTY) continue;
      const bit = 1n << BigInt(column * stride + (rows - 1 - row));
      if (cell === mover) moverBits |= bit;
      else opponentBits |= bit;
    }
  }
  return `${rows}x${columns}:${moverBits}:${opponentBits}`;
}

const mirrorBoard = (board) => board.map((row) => [...row].reverse());

function mirrorAction(action, columns) {
  if (action.type === ACTION_DROP) {
    return { type: ACTION_DROP, column: columns - 1 - action.column };
  }
  if (action.type === ACTION_ROTATE_CW) return { type: ACTION_ROTATE_CCW };
  if (action.type === ACTION_ROTATE_CCW) return { type: ACTION_ROTATE_CW };
  return { type: ACTION_FLIP };
}

export function replayPerfectChaosCompletePolicy(policy) {
  const { rows, columns, connect, role, rootValue, entryCount, closureStates, records } = policy;
  const aiPlayer = YELLOW;
  const startingPlayer = role === 1 ? aiPlayer : RED;
  const other = (player) => (player === RED ? YELLOW : RED);

  // Both a position and its mirror are probed, so the replay need not know which
  // representative the generator stored.
  const lookup = (board, mover) => {
    const directKey = packedKey(board, mover);
    const direct = records.get(directKey);
    if (direct) {
      return {
        key: directKey,
        value: direct.value,
        action: direct.action === 0
          ? { type: ACTION_DROP, column: direct.column }
          : { type: ACTION_TYPES[direct.action] },
      };
    }
    const flippedKey = packedKey(mirrorBoard(board), mover);
    const flipped = records.get(flippedKey);
    if (!flipped) return null;
    const stored = flipped.action === 0
      ? { type: ACTION_DROP, column: flipped.column }
      : { type: ACTION_TYPES[flipped.action] };
    // Mirror back across the board's current width: a rotated round is narrower
    // or wider than the shape the certificate is named for.
    return { key: flippedKey, value: flipped.value, action: mirrorAction(stored, board[0].length) };
  };

  const closureKey = (board, mover, aiTurn) => {
    const direct = packedKey(board, mover);
    const flipped = packedKey(mirrorBoard(board), mover);
    return `${direct < flipped ? direct : flipped}|${aiTurn ? 1 : 0}`;
  };

  const settle = (board, action, mover) => {
    const applied = applyAction(board, action, mover);
    if (!applied) return null;
    return {
      board: applied.board,
      outcome: resolveActionOutcome(
        applied.board,
        connect,
        mover,
        action.type,
        action.type === ACTION_DROP ? { row: applied.row, column: applied.column } : null,
      ),
    };
  };

  // One iterative enumeration of the closure, then two linear retrograde
  // passes over it. The previous recursive game walk re-derived every
  // cycle-tainted region on each path that touched it - 72.5 million
  // evaluations for the 1.5 million stored values of 4x6 role 2 - and could
  // not compare stored values inside those regions at all. The graph form
  // checks every stored value exactly once: a position is a forced win only
  // if the policy reaches a terminal AI win through finitely many replies, a
  // forced loss only if the opponent can compel a terminal AI loss, and a
  // draw otherwise, which is what the threefold-repetition rule makes of a
  // line that shuffles pieces forever.
  const INTERIOR = 2;
  const NO_TERMINAL = 2;
  const usedRecords = new Set();
  const counts = {
    aiStates: 0,
    opponentStates: 0,
    terminalAiWins: 0,
    terminalAiLosses: 0,
    terminalDraws: 0,
    storedValuesChecked: 0,
  };

  const idOf = new Map();
  const queue = [];
  const aiTurnOf = [];
  const storedOf = [];
  const recordKeyOf = [];
  const termOf = [];
  const oppTermBest = [];
  const succ = [];
  const succStart = [];

  const stateOf = (board, mover, aiTurn) => {
    const key = closureKey(board, mover, aiTurn);
    let id = idOf.get(key);
    if (id === undefined) {
      id = idOf.size;
      idOf.set(key, id);
      queue.push([board, mover, aiTurn]);
    }
    return id;
  };

  const rootBoard = createBoard(rows, columns);
  if (isBoardFull(rootBoard)) throw new Error('Perfect Chaos replay cannot start from a full board.');
  stateOf(rootBoard, startingPlayer, role === 1);

  for (let head = 0; head < queue.length; head += 1) {
    const [board, mover, aiTurn] = queue[head];
    queue[head] = null;
    const id = head;
    succStart[id] = succ.length;
    aiTurnOf[id] = aiTurn ? 1 : 0;
    if (aiTurn) {
      counts.aiStates += 1;
      counts.storedValuesChecked += 1;
      const chosen = lookup(board, mover);
      if (!chosen) throw new Error('Perfect Chaos policy is missing a reachable position.');
      usedRecords.add(chosen.key);
      storedOf[id] = chosen.value;
      recordKeyOf[id] = chosen.key;
      const applied = settle(board, chosen.action, mover);
      if (!applied) throw new Error('Perfect Chaos policy selects an illegal action.');
      if (applied.outcome.status === 'won') {
        if (applied.outcome.winner === mover) {
          counts.terminalAiWins += 1;
          termOf[id] = WIN;
        } else {
          counts.terminalAiLosses += 1;
          termOf[id] = LOSS;
        }
      } else if (applied.outcome.status === 'draw') {
        counts.terminalDraws += 1;
        termOf[id] = DRAW;
      } else {
        termOf[id] = INTERIOR;
        succ.push(stateOf(applied.board, other(mover), false));
      }
    } else {
      counts.opponentStates += 1;
      let bestTerminal = NO_TERMINAL;
      for (const action of legalActions(board, true)) {
        const applied = settle(board, action, mover);
        if (!applied) continue;
        if (applied.outcome.status === 'won') {
          if (applied.outcome.winner === mover) {
            counts.terminalAiLosses += 1;
            if (LOSS < bestTerminal) bestTerminal = LOSS;
          } else {
            counts.terminalAiWins += 1;
            if (WIN < bestTerminal) bestTerminal = WIN;
          }
        } else if (applied.outcome.status === 'draw') {
          counts.terminalDraws += 1;
          if (DRAW < bestTerminal) bestTerminal = DRAW;
        } else {
          succ.push(stateOf(applied.board, other(mover), true));
        }
      }
      oppTermBest[id] = bestTerminal;
    }
  }
  const stateCount = idOf.size;
  succStart[stateCount] = succ.length;

  // Reverse edges, compressed. predStart holds the prefix sums and stays
  // untouched; a working copy tracks the fill positions.
  const successors = Int32Array.from(succ);
  succ.length = 0;
  const predStart = new Int32Array(stateCount + 1);
  for (let index = 0; index < successors.length; index += 1) predStart[successors[index] + 1] += 1;
  for (let id = 0; id < stateCount; id += 1) predStart[id + 1] += predStart[id];
  const predecessors = new Int32Array(successors.length);
  const fill = predStart.slice(0, stateCount);
  for (let id = 0; id < stateCount; id += 1) {
    for (let edge = succStart[id]; edge < succStart[id + 1]; edge += 1) {
      predecessors[fill[successors[edge]]] = id;
      fill[successors[edge]] += 1;
    }
  }

  // Forced wins, retrograde. An AI state wins when its single action wins or
  // its successor does; an opponent state wins only when every reply does, so
  // it carries a countdown of interior replies and is disqualified outright by
  // any terminal reply that is not an AI win.
  const forcedWin = new Uint8Array(stateCount);
  const needWin = new Int32Array(stateCount);
  const winQueue = new Int32Array(stateCount);
  let winTail = 0;
  for (let id = 0; id < stateCount; id += 1) {
    const interiorEdges = succStart[id + 1] - succStart[id];
    if (aiTurnOf[id]) {
      if (termOf[id] === WIN) {
        forcedWin[id] = 1;
        winQueue[winTail] = id;
        winTail += 1;
      }
    } else if (oppTermBest[id] === NO_TERMINAL || oppTermBest[id] === WIN) {
      needWin[id] = interiorEdges;
      if (interiorEdges === 0) {
        forcedWin[id] = 1;
        winQueue[winTail] = id;
        winTail += 1;
      }
    } else {
      needWin[id] = -1;
    }
  }
  for (let winHead = 0; winHead < winTail; winHead += 1) {
    const id = winQueue[winHead];
    for (let edge = predStart[id]; edge < predStart[id + 1]; edge += 1) {
      const pred = predecessors[edge];
      if (forcedWin[pred]) continue;
      if (aiTurnOf[pred]) {
        forcedWin[pred] = 1;
        winQueue[winTail] = pred;
        winTail += 1;
      } else if (needWin[pred] > 0) {
        needWin[pred] -= 1;
        if (needWin[pred] === 0) {
          forcedWin[pred] = 1;
          winQueue[winTail] = pred;
          winTail += 1;
        }
      }
    }
  }

  // Forced losses, retrograde. Here one bad edge suffices for the opponent,
  // and the AI state's single action leaves it no escape either.
  const forcedLoss = new Uint8Array(stateCount);
  const lossQueue = new Int32Array(stateCount);
  let lossTail = 0;
  for (let id = 0; id < stateCount; id += 1) {
    if (aiTurnOf[id] ? termOf[id] === LOSS : oppTermBest[id] === LOSS) {
      forcedLoss[id] = 1;
      lossQueue[lossTail] = id;
      lossTail += 1;
    }
  }
  for (let lossHead = 0; lossHead < lossTail; lossHead += 1) {
    const id = lossQueue[lossHead];
    for (let edge = predStart[id]; edge < predStart[id + 1]; edge += 1) {
      const pred = predecessors[edge];
      if (!forcedLoss[pred]) {
        forcedLoss[pred] = 1;
        lossQueue[lossTail] = pred;
        lossTail += 1;
      }
    }
  }

  const failures = [];
  let suppressed = 0;
  const fail = (message) => {
    if (failures.length < 50) failures.push(message);
    else suppressed += 1;
  };

  let rootOutcome = DRAW;
  for (let id = 0; id < stateCount; id += 1) {
    if (forcedWin[id] && forcedLoss[id]) {
      fail(`state ${id} is classified as both a forced win and a forced loss`);
    }
    const outcome = forcedWin[id] ? WIN : forcedLoss[id] ? LOSS : DRAW;
    if (id === 0) rootOutcome = outcome;
    if (!aiTurnOf[id] || storedOf[id] === outcome) continue;
    if (storedOf[id] === WIN && outcome === DRAW) {
      fail(`a win is claimed at ${recordKeyOf[id]} but the line can repeat forever`);
    } else {
      fail(`stored value ${storedOf[id]} but the policy forces ${outcome} at ${recordKeyOf[id]}`);
    }
  }

  if (rootOutcome !== rootValue) {
    fail(`replayed root value ${rootOutcome} but the header claims ${rootValue}`);
  }
  if (rootValue >= 0 && counts.terminalAiLosses > 0) {
    fail(`${counts.terminalAiLosses} terminal AI losses in a non-losing certificate`);
  }
  if (usedRecords.size !== entryCount) {
    fail(`${entryCount - usedRecords.size} unreachable record(s)`);
  }
  if (stateCount !== closureStates) {
    fail(`closure size ${stateCount} does not match the header's ${closureStates}`);
  }
  if (suppressed > 0) failures.push(`${suppressed} further failure(s) suppressed`);
  if (failures.length > 0) {
    throw new Error(`Perfect Chaos replay failed: ${failures.join('; ')}`);
  }

  return {
    format: 'connect4-perfect-chaos-complete-replay-v1',
    rows,
    columns,
    connect,
    role,
    rootValue,
    replayedRootValue: rootOutcome,
    entryCount,
    closureStates: stateCount,
    ...counts,
  };
}

export async function verifyPerfectChaosCompleteReference(path) {
  const manifestPath = resolve(String(path));
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  if (manifest?.format !== 'connect4-perfect-chaos-complete-manifest-v1'
      || !Array.isArray(manifest.policies) || manifest.policies.length === 0) {
    throw new Error('Perfect Chaos manifest format is invalid or empty.');
  }

  const directory = dirname(manifestPath);
  const identities = new Set();
  const replay = [];
  for (const entry of manifest.policies) {
    const identity = `${entry.rows}x${entry.columns}:c${entry.connect}:r${entry.role}`;
    if (identities.has(identity)) throw new Error(`Duplicate Perfect Chaos policy ${identity}.`);
    identities.add(identity);

    const file = resolve(directory, entry.file);
    const bytes = new Uint8Array(await readFile(file));
    if (bytes.byteLength !== entry.bytes) {
      throw new Error(`Perfect Chaos policy byte length mismatch for ${entry.file}.`);
    }
    const digest = createHash('sha256').update(bytes).digest('hex');
    if (digest !== entry.sha256) {
      throw new Error(`Perfect Chaos policy hash mismatch for ${entry.file}.`);
    }

    const policy = decode(bytes);
    if (policy.rows !== entry.rows || policy.columns !== entry.columns
        || policy.connect !== entry.connect || policy.role !== entry.role
        || policy.rootValue !== entry.rootValue
        || policy.entryCount !== entry.entryCount
        || policy.closureStates !== entry.closureStates) {
      throw new Error(`Perfect Chaos policy metadata mismatch for ${entry.file}.`);
    }

    const record = replayPerfectChaosCompletePolicy(policy);
    // A certificate is only as good as the generator summary it claims to match.
    for (const field of ['rootValue', 'closureStates', 'terminalAiWins', 'terminalAiLosses', 'terminalDraws']) {
      if (entry.generator && entry.generator[field] !== record[field]) {
        throw new Error(`Generator summary disagrees with the replay on ${field} for ${entry.file}.`);
      }
    }
    replay.push({ file: entry.file, ...record });
  }

  return { manifestPath, policyCount: replay.length, replay };
}


async function executable(path) {
  if (!path) return false;
  try {
    await access(path, fsConstants.X_OK);
    return true;
  } catch {
    return false;
  }
}

function run(command, args, options = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'], ...options });
    const stdout = [];
    const stderr = [];
    child.stdout?.on('data', (chunk) => stdout.push(chunk));
    child.stderr?.on('data', (chunk) => stderr.push(chunk));
    child.once('error', reject);
    child.once('close', (code, signal) => resolvePromise({
      code,
      signal,
      stdout: Buffer.concat(stdout).toString('utf8'),
      stderr: Buffer.concat(stderr).toString('utf8'),
    }));
  });
}

async function findCompiler() {
  if (process.env.CXX) return process.env.CXX;
  for (const candidate of ['/usr/bin/g++', '/usr/bin/clang++']) {
    if (await executable(candidate)) return candidate;
  }
  // Fall back to whatever the PATH offers, so a toolchain installed anywhere
  // other than /usr/bin still works without setting CXX by hand.
  for (const candidate of ['g++', 'clang++']) {
    const probe = await run(candidate, ['--version']).catch(() => null);
    if (probe && probe.code === 0) return candidate;
  }
  throw new Error('A C++20 compiler is required (set CXX, or install g++/clang++).');
}

async function compile(directory) {
  const compiler = await findCompiler();
  const binary = join(directory, 'perfect-chaos-complete');
  // No -march=native: it has miscompiled this solver on at least one Zen 4
  // toolchain, and the portable build is fast enough.
  const result = await run(compiler, ['-std=c++20', '-O3', '-Wall', '-Wextra', SOURCE, '-o', binary]);
  if (result.code !== 0) {
    throw new Error(`Perfect Chaos solver compiler failed.\n${result.stderr || result.stdout}`);
  }
  return { compiler, binary, warnings: result.stderr.trim() };
}

function integerOption(value, fallback, label, minimum, maximum) {
  const selected = value === undefined ? fallback : Number.parseInt(String(value), 10);
  if (!Number.isInteger(selected) || selected < minimum || selected > maximum) {
    throw new RangeError(`${label} must be an integer from ${minimum} through ${maximum}.`);
  }
  return selected;
}

/**
 * Compiles the native solver, solves one board completely, emits both role
 * certificates, replays each through engine.js, and writes a manifest whose
 * entries carry the generator summary and the independent replay side by side.
 * An entry is written only when the two agree.
 */
export async function generatePerfectChaosComplete(options) {
  const rows = integerOption(options.rows, 4, 'rows', 1, 7);
  const columns = integerOption(options.columns, 4, 'columns', 1, 7);
  const connect = integerOption(options.connect, 4, 'connect', 1, Math.max(rows, columns));
  const output = resolve(options.output ?? join(ROOT, 'generated', `perfect-chaos-complete-${rows}x${columns}-c${connect}`));
  await mkdir(output, { recursive: true });

  const temporary = await mkdtemp(join(tmpdir(), 'connect4-chaos-complete-'));
  try {
    const compiled = await compile(temporary);
    if (compiled.warnings) process.stderr.write(`${compiled.warnings}\n`);
    const prefix = join(output, `${rows}x${columns}-c${connect}`);
    // The solver checkpoints its discovery bitset and each finished rank round
    // beside the outputs, so a killed multi-hour solve resumes instead of
    // restarting; it deletes the checkpoint files itself on success.
    const solverThreads = integerOption(options.solver_threads, 1, 'solver-threads', 1, 16);
    const result = await run(compiled.binary, [
      '--rows', String(rows), '--columns', String(columns), '--connect', String(connect),
      '--checkpoint', join(output, 'solver-checkpoint'),
      '--threads', String(solverThreads),
      '--emit-policy', prefix,
    ]);
    if (result.code !== 0) {
      throw new Error(`Perfect Chaos solver failed.\n${result.stderr || result.stdout}`);
    }
    const lines = result.stdout.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
    const solution = lines.find((line) => line.format === 'connect4-chaos-exact-solution-v1');
    if (!solution) throw new Error('The solver returned no solution summary.');

    const policies = [];
    for (const role of [1, 2]) {
      const summary = lines.find((line) => line.format === 'connect4-chaos-closure-v1' && line.role === role);
      if (!summary) throw new Error(`The solver returned no closure summary for role ${role}.`);
      const file = `${rows}x${columns}-c${connect}-role${role}.bin`;
      const bytes = new Uint8Array(await readFile(join(output, file)));
      const policy = decode(bytes);
      const replay = replayPerfectChaosCompletePolicy(policy);
      for (const field of ['rows', 'columns', 'connect', 'role', 'rootValue', 'closureStates',
        'aiStates', 'opponentStates', 'terminalAiWins', 'terminalAiLosses', 'terminalDraws']) {
        if (summary[field] !== replay[field]) {
          throw new Error(`Generator and replay disagree on ${field} for role ${role}.`);
        }
      }
      if (summary.aiStates !== policy.entryCount) {
        throw new Error(`Role ${role} entry count is not its AI-state count.`);
      }
      policies.push({
        rows, columns, connect, role,
        rootValue: policy.rootValue,
        entryCount: policy.entryCount,
        closureStates: policy.closureStates,
        file: `./${file}`,
        bytes: bytes.byteLength,
        sha256: createHash('sha256').update(bytes).digest('hex'),
        generator: summary,
        replay,
      });
    }

    const manifest = {
      format: MANIFEST_FORMAT,
      generatedAt: new Date().toISOString(),
      sourceSha256: createHash('sha256').update(await readFile(SOURCE)).digest('hex'),
      compiler: compiled.compiler,
      solution,
      policies,
    };
    await writeFile(join(output, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
    return { output, manifest };
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

/**
 * Merges per-board manifests into the runtime catalog. Duplicate
 * (board, connect, role) identities are rejected rather than overwritten.
 */
export async function mergePerfectChaosCompleteManifests(inputs, outputPath) {
  if (!Array.isArray(inputs) || inputs.length === 0) {
    throw new RangeError('At least one --input manifest is required.');
  }
  const target = resolve(outputPath);
  const outputDirectory = dirname(target);
  await mkdir(outputDirectory, { recursive: true });
  const policies = [];
  const identities = new Set();
  for (const input of inputs) {
    const path = resolve(input);
    const manifest = JSON.parse(await readFile(path, 'utf8'));
    if (manifest?.format !== MANIFEST_FORMAT || !Array.isArray(manifest.policies)) {
      throw new Error(`Invalid Perfect Chaos manifest: ${path}`);
    }
    for (const entry of manifest.policies) {
      const identity = `${entry.rows}x${entry.columns}:c${entry.connect}:r${entry.role}`;
      if (identities.has(identity)) throw new Error(`Duplicate Perfect Chaos policy ${identity}.`);
      identities.add(identity);
      const filename = `${entry.rows}x${entry.columns}-c${entry.connect}-role${entry.role}.bin`;
      await writeFile(join(outputDirectory, filename), await readFile(resolve(dirname(path), entry.file)));
      policies.push({ ...entry, file: `./${filename}` });
    }
  }
  policies.sort((first, second) => (
    first.rows - second.rows || first.columns - second.columns
    || first.connect - second.connect || first.role - second.role
  ));
  const boards = [...new Set(policies.map((entry) => `${entry.rows}x${entry.columns}-c${entry.connect}`))];
  const manifest = {
    format: MANIFEST_FORMAT,
    generatedAt: new Date().toISOString(),
    note: 'Complete Chaos Mode solutions. Every position reachable from the empty board under the selected policy is covered, so there is no piece-count frontier and no handoff to bounded search. A rotation transposes the board, so each certificate spans both orientations of its orbit. Repetition cycles are draws under the threefold rule.',
    policies,
    coverage: { boards, boardCount: boards.length, roleCount: policies.length },
  };
  await writeFile(target, `${JSON.stringify(manifest, null, 2)}\n`);
  return manifest;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.command === 'generate') {
    const generated = await generatePerfectChaosComplete(options);
    process.stdout.write(`${JSON.stringify(generated, null, 2)}\n`);
    return;
  }
  if (options.command === 'merge-manifests') {
    if (!options.output || options.output === true) throw new RangeError('--output is required.');
    const manifest = await mergePerfectChaosCompleteManifests(options.inputs, options.output);
    process.stdout.write(`${JSON.stringify(manifest.coverage, null, 2)}\n`);
    return;
  }
  if (options.command !== 'verify-reference') {
    throw new RangeError(`Unknown command: ${options.command}`);
  }
  const reference = options.reference && options.reference !== true
    ? options.reference
    : DEFAULT_REFERENCE;
  const verified = await verifyPerfectChaosCompleteReference(reference);
  process.stdout.write(`${JSON.stringify(verified, null, 2)}\n`);
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  await main();
}

import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  buildStrategy,
  decodeStrategy,
  scoreWithOracle,
  verifyClosure,
  STRATEGY_CONSTANTS,
} from '../scripts/perfect-strategy.mjs';

function neutralScores(sequences) {
  return new Map(sequences.map((sequence) => [sequence, Array(7).fill(0)]));
}

test('the deterministic strategy format covers both starting roles to its handoff', async () => {
  const { bytes, manifest } = await buildStrategy({
    handoffRemaining: 40,
    roles: 'both',
    source: 'deterministic test oracle',
    scoreBatch: neutralScores,
  });
  const decoded = decodeStrategy(bytes);

  assert.equal(decoded.version, 1);
  assert.equal(decoded.handoffRemaining, 40);
  assert.equal(decoded.roleFlags, STRATEGY_CONSTANTS.ROLE_FIRST | STRATEGY_CONSTANTS.ROLE_SECOND);
  assert.equal(decoded.entries.length, 5);
  assert.equal(manifest.entryCount, 5);
  assert.deepEqual(verifyClosure(decoded), manifest.closure);
});

// An oracle that knows only immediate wins: 1 for a column that wins at once
// for the side to move, `otherwise` for every other column.
function immediateWinScores(otherwise) {
  const winsAtOnce = (sequence, column) => {
    const heights = Array(7).fill(0);
    const owner = new Map();
    [...sequence].forEach((digit, ply) => {
      const played = Number(digit) - 1;
      owner.set(`${played},${heights[played]}`, ply % 2);
      heights[played] += 1;
    });
    if (heights[column] >= 6) return false;
    const row = heights[column];
    const mine = (c, r) => (c === column && r === row) || owner.get(`${c},${r}`) === sequence.length % 2;
    return [[1, 0], [0, 1], [1, 1], [1, -1]].some(([dc, dr]) => {
      let count = 1;
      for (const sign of [1, -1]) {
        for (let step = 1; step < 4 && mine(column + sign * dc * step, row + sign * dr * step); step += 1) {
          count += 1;
        }
      }
      return count >= 4;
    });
  };
  return (sequences) => new Map(sequences.map((sequence) => [
    sequence,
    Array.from({ length: 7 }, (_, column) => (winsAtOnce(sequence, column) ? 1 : otherwise)),
  ]));
}

test('the closure refuses a stored draw whose move lets the opponent win at once', async () => {
  // Claiming a draw everywhere, the frontier-minimising choice happily picks
  // moves that hand the opponent an immediate win; those entries lie.
  await assert.rejects(
    buildStrategy({
      handoffRemaining: 33, roles: 'first', source: 'draw-claiming test oracle',
      scoreBatch: immediateWinScores(0),
    }),
    /stores outcome 0, but its move lets the opponent win with column \d/,
  );
  // The same moves stored as losses are consistent, and are counted apart
  // from the non-losing terminals.
  const { manifest } = await buildStrategy({
    handoffRemaining: 33, roles: 'first', source: 'loss-claiming test oracle',
    scoreBatch: immediateWinScores(-1),
  });
  assert.ok(manifest.closure.first.aiLosses > 0);
});

test('the closure refuses a stored outcome that the move itself contradicts', async () => {
  // Neutral scores store 0 even for a move that wins on the spot.
  await assert.rejects(
    buildStrategy({
      handoffRemaining: 35, roles: 'first', source: 'deterministic test oracle',
      scoreBatch: neutralScores,
    }),
    /stores outcome 0, but its move ends the game with 1/,
  );
});

test('strategy decoding rejects truncation and multi-move entries', async () => {
  const { bytes } = await buildStrategy({
    handoffRemaining: 41,
    roles: 'first',
    source: 'deterministic test oracle',
    scoreBatch: neutralScores,
  });

  assert.throws(() => decodeStrategy(bytes.subarray(0, bytes.length - 1)), /length/);
  const invalid = bytes.slice();
  invalid[20] = 3;
  assert.throws(() => decodeStrategy(invalid), /exactly one/);
});

// A stand-in for the exact oracle: the module `lines` make, run by Node in the
// oracle's place and with its arguments, the last of which is the book.
async function standInOracle(context, lines) {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-strategy-oracle-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const script = join(directory, 'oracle.mjs');
  await writeFile(script, lines.join('\n'));
  return { directory, start: (command, args, options) => spawn(process.execPath, [script, ...args], options) };
}

test('the oracle pool shares the sequences out and merges the answers', async (context) => {
  // Each oracle scores a sequence by how many it was given.
  const { directory, start } = await standInOracle(context, [
    "import { createInterface } from 'node:readline';",
    'const sequences = [];',
    'for await (const sequence of createInterface({ input: process.stdin })) sequences.push(sequence);',
    'for (const sequence of sequences) console.log(`${sequence} ${Array(7).fill(sequences.length).join(" ")}`);',
  ]);
  const scores = await scoreWithOracle(['1', '2', '3'],
    { oraclePath: 'c4solver', oracleBookPath: directory, workers: 2 }, start);
  assert.deepEqual(scores, new Map([['1', Array(7).fill(2)], ['2', Array(7).fill(1)], ['3', Array(7).fill(2)]]));
});

// Signal 0 only asks whether the process still runs.
function stillRunning(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error.code !== 'ESRCH';
  }
}

// A busy machine can take tens of seconds to start the two oracles below, so
// this only ends a run that hangs.
const STOP_TIMEOUT = 180_000;

// One oracle's failure fails its layer, but the others ran on: the build
// reported the failure and stayed until each had finished its chunk, hours in
// the deepest layers. Of the two stand-ins here, the first to start would
// sleep for a minute, and the other fails once the first is running. Neither
// reads its input, which is more than a pipe holds: the write to an oracle
// that has exited, or been stopped, fails, and with nothing listening that
// error ended the build before the failure was reported. The pool runs in
// this process, which stays the children's parent, so Windows, which ends a
// build's children when it exits, cannot hide one left running.
test('a failed oracle stops the others before the build gives up', { timeout: STOP_TIMEOUT }, async (context) => {
  // On Linux and macOS the sleeper takes half a second to exit after SIGTERM,
  // so a pool that gave up without waiting for it to close would leave it
  // running. Windows ends it outright, but there too its close comes after
  // the kill, and the closes are counted. Left alone, it writes sleeper.slept
  // when its minute is up, so a pool that waits for it instead of stopping it
  // fails on that file rather than on the timeout.
  const { directory, start } = await standInOracle(context, [
    "import { existsSync, openSync, renameSync, writeFileSync } from 'node:fs';",
    "import { join } from 'node:path';",
    'const state = process.argv.at(-1);',
    "const started = join(state, 'sleeper.pid');",
    'let sleeper = true;',
    "try { openSync(join(state, 'sleeper.claim'), 'wx'); } catch { sleeper = false; }",
    'if (sleeper) {',
    "  process.on('SIGTERM', () => setTimeout(() => process.exit(143), 500));",
    "  writeFileSync(join(state, 'sleeper.part'), String(process.pid));",
    "  renameSync(join(state, 'sleeper.part'), started);",
    "  setTimeout(() => writeFileSync(join(state, 'sleeper.slept'), ''), 60_000);",
    '} else {',
    `  const deadline = Date.now() + ${STOP_TIMEOUT};`,
    '  const poll = setInterval(() => {',
    '    if (!existsSync(started) && Date.now() < deadline) return;',
    '    clearInterval(poll);',
    "    console.error('stand-in refusal');",
    '    process.exitCode = 1;',
    '  }, 10);',
    '}',
  ]);
  // Each oracle's close is counted, and reaches the pool a moment after its
  // exit, as it does while something still holds the oracle's output open: a
  // pool that settled for the exit would give up before the second close.
  let closed = 0;
  const counted = (...oracle) => {
    const child = start(...oracle);
    const emit = child.emit;
    child.emit = function emitLater(event, ...values) {
      if (event !== 'close') return emit.call(this, event, ...values);
      setTimeout(() => {
        closed += 1;
        emit.call(this, event, ...values);
      }, 250);
      return true;
    };
    return child;
  };
  const sequences = Array.from({ length: 80_000 }, (_, index) => String(1_000_000 + index));
  await assert.rejects(
    scoreWithOracle(sequences, { oraclePath: 'c4solver', oracleBookPath: directory, workers: 2 }, counted),
    { message: 'Exact oracle exited with 1: stand-in refusal' },
  );
  const closedByThen = closed;
  const pid = Number(await readFile(join(directory, 'sleeper.pid'), 'utf8'));
  const leftRunning = stillRunning(pid);
  // An oracle the pool left behind is killed here, so it does not sleep out its minute.
  if (leftRunning) process.kill(pid, 'SIGKILL');
  assert.equal(leftRunning, false, 'an oracle was still running when the build gave up');
  assert.equal(closedByThen, 2, 'the build gave up before both oracles had closed');
  assert.equal(existsSync(join(directory, 'sleeper.slept')), false,
    'the build waited for the other oracle to finish instead of stopping it');
});

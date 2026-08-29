import assert from 'node:assert/strict';
import { mkdtemp, open, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { pythonCommand } from '../scripts/python-command.mjs';
import {
  DRAW, LOSS, WIN,
  canonicalPairSlot, edgeValueForMover, lookupSlot, makeGeometry, maskHasLine,
  pairOf, successors,
} from '../scripts/perfect-chaos-remote-lookup.mjs';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const SOURCE = join(ROOT, 'native', 'perfect-chaos-paired.cpp');
const SIDECARS = join(ROOT, 'scripts', 'build-pair-rank-sidecars.py');

function findCompiler() {
  if (process.env.CXX) return process.env.CXX;
  for (const candidate of ['g++', 'clang++']) {
    const probe = spawnSync(candidate, ['--version'], { encoding: 'utf8' });
    if (probe.status === 0) return candidate;
  }
  return null;
}

function run(command, args) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] });
    const stdout = [];
    const stderr = [];
    child.stdout.on('data', (chunk) => stdout.push(chunk));
    child.stderr.on('data', (chunk) => stderr.push(chunk));
    child.once('error', reject);
    child.once('close', (code) => resolvePromise({
      code,
      stdout: Buffer.concat(stdout).toString('utf8'),
      stderr: Buffer.concat(stderr).toString('utf8'),
    }));
  });
}

// File-backed range source with the access pattern the browser tier uses.
function fileSource(directory) {
  const handles = new Map();
  return {
    async fetchRange(name, offset, length) {
      if (!handles.has(name)) handles.set(name, await open(join(directory, name), 'r'));
      const buffer = Buffer.alloc(length);
      const { bytesRead } = await handles.get(name).read(buffer, 0, length, offset);
      assert.equal(bytesRead, length, `short read on ${name}`);
      return buffer;
    },
    async close() {
      for (const handle of handles.values()) await handle.close();
    },
  };
}

// Deterministic PRNG so a failure reproduces.
function mulberry32(seed) {
  let state = seed;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

test('remote lookup agrees with the solved 4x4 c4 tables everywhere sampled', async (context) => {
  const compiler = findCompiler();
  let python = null;
  try { python = pythonCommand(); } catch { /* stay null */ }
  if (!compiler || !python) {
    context.skip('a C++ compiler and Python are required');
    return;
  }
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-remote-'));
  const source = fileSource(join(directory, 'out'));
  try {
    const binary = join(directory, process.platform === 'win32' ? 'paired.exe' : 'paired');
    const compiled = await run(compiler, ['-O2', '-std=c++20', '-static', '-o', binary, SOURCE]);
    assert.equal(compiled.code, 0, `compile failed: ${compiled.stderr.slice(0, 2000)}`);
    const solved = await run(binary, [
      '--rows', '4', '--columns', '4', '--connect', '4',
      '--threads', '2', '--output', join(directory, 'out'),
    ]);
    assert.equal(solved.code, 0, `solve failed: ${solved.stderr.slice(0, 2000)}`);
    const sidecars = await run(python.command, [...python.args, SIDECARS, join(directory, 'out')]);
    assert.equal(sidecars.code, 0, `sidecars failed: ${sidecars.stderr.slice(0, 2000)}`);

    const geometry = makeGeometry(4, 4, 4);

    // The empty board is slot 0 of pair (0, 0) and the recorded root draw.
    assert.equal(await lookupSlot(source, 0, 0, 0), DRAW, 'root value');

    // Random playouts; every visited state must satisfy the game's value
    // equation through the remote reader: a win iff some child wins for the
    // mover, a loss iff every child loses for the mover, a draw otherwise.
    // A single mis-ranked slot would surface as garbage values instantly.
    const random = mulberry32(20260829);
    let checked = 0;
    for (let playout = 0; playout < 40; playout += 1) {
      let state = {
        blockIndex: 0, mover: 0n, opponent: 0n,
        heights: [0, 0, 0, 0], pieces: 0, moverCount: 0,
      };
      for (let ply = 0; ply < 24; ply += 1) {
        const edges = successors(
          geometry, state.blockIndex, state.mover, state.opponent,
          state.heights, state.pieces, state.moverCount,
        );
        const pairId = pairOf(state.pieces, state.moverCount);
        const slot = canonicalPairSlot(
          geometry, state.blockIndex, state.mover, state.heights, state.pieces, pairId,
        );
        const value = await lookupSlot(source, state.pieces, pairId, slot);
        let anyWin = false;
        let allLoss = true;
        for (const edge of edges) {
          const forMover = await edgeValueForMover(source, edge);
          if (forMover === WIN) anyWin = true;
          if (forMover !== LOSS) allLoss = false;
        }
        const expected = anyWin ? WIN : (allLoss ? LOSS : DRAW);
        assert.equal(value, expected,
          `value equation failed at pieces=${state.pieces} pair=${pairId} slot=${slot}`);
        checked += 1;

        // Walk one random non-terminal drop; that keeps the playout inside
        // states whose full child sets the equation above already verified.
        const open = [];
        for (let c = 0; c < 4; c += 1) {
          if (state.heights[c] >= 4) continue;
          const grown = state.mover | (1n << BigInt(c * 5 + state.heights[c]));
          const wins = maskHasLine(grown, 4, 4);
          if (!wins && state.pieces + 1 < geometry.cellCount) open.push({ c, grown });
        }
        if (open.length === 0) break;
        const pick = open[Math.floor(random() * open.length)];
        const childHeights = state.heights.slice();
        childHeights[pick.c] += 1;
        state = {
          blockIndex: state.blockIndex,
          mover: state.opponent, opponent: pick.grown,
          heights: childHeights,
          pieces: state.pieces + 1,
          moverCount: state.pieces - state.moverCount,
        };
      }
    }
    assert.ok(checked >= 150, `only ${checked} states checked`);
  } finally {
    await source.close();
    await rm(directory, { recursive: true, force: true });
  }
});

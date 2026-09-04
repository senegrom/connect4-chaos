import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import {
  ACTION_DROP,
  RED,
  YELLOW,
  applyAction,
  createBoard,
  legalActions,
  normalizeConfig,
  resolveActionOutcome,
  supportsPerfectChaosConfig,
  supportsPerfectChaosStart,
  supportsPerfectConfig,
} from '../src/engine.js';
import {
  decodePerfectChaosCompletePolicy,
  findPerfectChaosCompletePolicy,
  loadPerfectChaosCompleteManifest,
  loadVerifiedPerfectChaosCompletePolicy,
  perfectChaosCompleteRole,
} from '../src/perfect-chaos-complete.js';
import {
  choosePerfectChaosMove,
  isPerfectChaosVariant,
} from '../src/perfect-chaos-runtime.js';

const MANIFEST_URL = new URL('../data/perfect-chaos-complete/manifest.json', import.meta.url);

test('solved Chaos configurations are advertised and unsolved ones are not', () => {
  assert.equal(supportsPerfectChaosConfig(4, 4, 4, true), true);
  assert.equal(supportsPerfectChaosConfig(4, 4, 3, true), true);
  assert.equal(supportsPerfectChaosConfig(4, 5, 4, true), true);
  assert.equal(supportsPerfectChaosConfig(4, 5, 3, true), true);
  // A rotation transposes the board, so the orbit's other orientation counts.
  assert.equal(supportsPerfectChaosConfig(5, 4, 4, true), true);
  assert.equal(supportsPerfectChaosConfig(5, 4, 3, true), true);
  assert.equal(supportsPerfectChaosConfig(4, 5, 5, true), true);
  assert.equal(supportsPerfectChaosConfig(5, 5, 4, true), true);
  assert.equal(supportsPerfectChaosConfig(4, 6, 4, true), true);
  assert.equal(supportsPerfectChaosConfig(6, 4, 4, true), true);
  assert.equal(supportsPerfectChaosConfig(5, 5, 3, true), true);
  assert.equal(supportsPerfectChaosConfig(4, 6, 3, true), true);
  assert.equal(supportsPerfectChaosConfig(4, 7, 3, true), true);
  assert.equal(supportsPerfectChaosConfig(7, 4, 3, true), true);
  assert.equal(supportsPerfectChaosConfig(5, 6, 3, true), true);
  assert.equal(supportsPerfectChaosConfig(6, 5, 3, true), true);
  // Solved but unpublished: the 5x5 c5, 4x6 c5 and 4x6 c6 certificates exceed
  // the repository's file-size budget, so Perfect must not be advertised.
  assert.equal(supportsPerfectChaosConfig(5, 5, 5, true), false);
  assert.equal(supportsPerfectChaosConfig(4, 6, 5, true), false);
  assert.equal(supportsPerfectChaosConfig(4, 6, 6, true), false);
  assert.equal(supportsPerfectChaosConfig(5, 6, 4, true), false);
  assert.equal(supportsPerfectChaosConfig(6, 7, 4, true), false);
  // Chaos support must not leak into the classic predicate or vice versa.
  assert.equal(supportsPerfectChaosConfig(4, 4, 4, false), false);
  assert.equal(supportsPerfectConfig(4, 4, 4, true), true);
  assert.equal(supportsPerfectConfig(6, 7, 4, true), false);
  assert.equal(supportsPerfectConfig(6, 7, 4, false), true);

  // A round may start only in a certificate's own orientation: the empty
  // transposed board is a position the certificate never reached.
  assert.equal(supportsPerfectChaosStart(4, 5, 4, true), true);
  assert.equal(supportsPerfectChaosStart(5, 4, 4, true), false);
  assert.equal(supportsPerfectChaosStart(4, 4, 4, true), true);
  assert.equal(supportsPerfectChaosStart(6, 4, 4, true), false);
  assert.equal(supportsPerfectConfig(5, 4, 4, true), false);
  assert.equal(supportsPerfectConfig(4, 5, 4, true), true);
});

test('Perfect survives normalisation on a solved Chaos board and not otherwise', () => {
  assert.equal(normalizeConfig({
    rows: 4, cols: 4, connect: 4, chaosMode: true, opponent: 'perfect',
  }).opponent, 'perfect');
  assert.equal(normalizeConfig({
    rows: 6, cols: 7, connect: 4, chaosMode: true, opponent: 'perfect',
  }).opponent, 'brutal');
});

test('the committed Chaos catalog matches its own artifacts', async () => {
  const manifest = await loadPerfectChaosCompleteManifest(MANIFEST_URL);
  assert.equal(manifest.format, 'connect4-perfect-chaos-complete-manifest-v1');
  assert.equal(manifest.policies.length, 22);

  const { createHash } = await import('node:crypto');
  for (const entry of manifest.policies) {
    const bytes = new Uint8Array(await readFile(
      new URL(entry.file, MANIFEST_URL),
    ));
    assert.equal(bytes.byteLength, entry.bytes, `${entry.file} byte length`);
    assert.equal(
      createHash('sha256').update(bytes).digest('hex'),
      entry.sha256,
      `${entry.file} digest`,
    );
    const policy = decodePerfectChaosCompletePolicy(bytes, {
      rows: entry.rows,
      columns: entry.columns,
      connect: entry.connect,
      role: entry.role,
    });
    assert.equal(policy.entryCount, entry.entryCount);
    assert.equal(policy.closureStates, entry.closureStates);
    assert.equal(policy.rootValue, entry.rootValue);
    // The independent replay recorded in the manifest must agree with the
    // generator it is meant to check. The verifier throws rather than reporting
    // a flag, so a present record already means it passed.
    assert.equal(
      entry.replay.format,
      'connect4-perfect-chaos-complete-replay-v1',
      `${entry.file} replay`,
    );
    assert.equal(entry.replay.replayedRootValue, entry.rootValue);
    assert.equal(entry.replay.entryCount, entry.entryCount);
    assert.equal(entry.replay.closureStates, entry.closureStates);
    assert.equal(entry.generator.terminalAiLosses, entry.replay.terminalAiLosses);
    assert.equal(entry.generator.terminalAiWins, entry.replay.terminalAiWins);
    if (entry.rootValue >= 0) {
      assert.equal(entry.replay.terminalAiLosses, 0, `${entry.file} must never lose`);
    }
  }
});

test('published Chaos root values are recorded for both starting roles', async () => {
  const manifest = await loadPerfectChaosCompleteManifest(MANIFEST_URL);
  const value = (connect, role) => findPerfectChaosCompletePolicy(
    manifest, 4, 4, connect, role,
  ).rootValue;
  // 4x4 Chaos Connect Four is a draw; Connect Three is a first-player win, so
  // the second-player certificate is a lost game played optimally.
  assert.equal(value(4, 1), 0);
  assert.equal(value(4, 2), 0);
  assert.equal(value(3, 1), 1);
  assert.equal(value(3, 2), -1);
});

test('a decoded policy rejects a mismatched configuration', async () => {
  const bytes = new Uint8Array(await readFile(
    new URL('../data/perfect-chaos-complete/4x4-c4-role1.bin', import.meta.url),
  ));
  assert.throws(
    () => decodePerfectChaosCompletePolicy(bytes, { role: 2 }),
    /does not match the requested configuration/,
  );
  assert.throws(
    () => decodePerfectChaosCompletePolicy(bytes.slice(0, 40)),
    /length mismatch/,
  );
});

test('the runtime plays the certificate and refuses to guess without it', async () => {
  const policy = await loadVerifiedPerfectChaosCompletePolicy(4, 4, 4, 1, {
    manifestUrl: MANIFEST_URL,
  });
  assert.ok(policy);
  const position = {
    board: createBoard(4, 4),
    currentPlayer: YELLOW,
    startingPlayer: YELLOW,
    connect: 4,
    chaosMode: true,
  };
  assert.equal(isPerfectChaosVariant(position), true);

  const result = choosePerfectChaosMove(position, {
    difficulty: 'perfect',
    aiPlayer: YELLOW,
    perfectChaosCompletePolicy: policy,
  });
  assert.equal(result.solved, true);
  assert.equal(result.solver, 'perfect-chaos-complete');
  assert.equal(result.value, 0);
  assert.ok(result.action);

  // Missing certificate data must fail closed, never fall back to search.
  assert.throws(
    () => choosePerfectChaosMove(position, { difficulty: 'perfect', aiPlayer: YELLOW }),
    /could not be loaded/,
  );
  // A policy for the wrong role must not be accepted for this round.
  const secondRole = await loadVerifiedPerfectChaosCompletePolicy(4, 4, 4, 2, {
    manifestUrl: MANIFEST_URL,
  });
  assert.throws(
    () => choosePerfectChaosMove(position, {
      difficulty: 'perfect',
      aiPlayer: YELLOW,
      perfectChaosCompletePolicy: secondRole,
    }),
    /does not match the current round/,
  );
});

test('the certificate never loses a 4x4 Chaos game from either role', async () => {
  const manifest = await loadPerfectChaosCompleteManifest(MANIFEST_URL);
  let random = 20260817;
  const next = () => {
    random ^= random << 13;
    random >>>= 0;
    random ^= random >> 17;
    random ^= random << 5;
    random >>>= 0;
    return random / 0x100000000;
  };

  for (const role of [1, 2]) {
    const policy = await loadVerifiedPerfectChaosCompletePolicy(4, 4, 4, role, {
      manifestUrl: MANIFEST_URL,
    });
    const aiPlayer = YELLOW;
    const startingPlayer = role === 1 ? YELLOW : RED;
    let losses = 0;

    for (let game = 0; game < 25; game += 1) {
      let board = createBoard(4, 4);
      let currentPlayer = startingPlayer;
      let status = 'playing';
      let plies = 0;

      while (status === 'playing' && plies < 200) {
        let action;
        if (currentPlayer === aiPlayer) {
          const chosen = choosePerfectChaosMove(
            { board, currentPlayer, startingPlayer, connect: 4, chaosMode: true },
            { difficulty: 'perfect', aiPlayer, perfectChaosCompletePolicy: policy },
          );
          action = chosen.action;
        } else {
          const legal = legalActions(board, true);
          action = legal[Math.floor(next() * legal.length)];
        }
        const applied = applyAction(board, action, currentPlayer);
        assert.ok(applied, 'the chosen action must be legal');
        const outcome = resolveActionOutcome(
          applied.board,
          4,
          currentPlayer,
          action.type,
          action.type === ACTION_DROP ? { row: applied.row, column: applied.column } : null,
        );
        board = applied.board;
        status = outcome.status;
        if (status === 'won' && outcome.winner !== aiPlayer) losses += 1;
        currentPlayer = currentPlayer === RED ? YELLOW : RED;
        plies += 1;
      }
    }
    // 4x4 Chaos Connect Four is a draw, so a certified policy cannot be beaten
    // from either starting role.
    assert.equal(losses, 0, `role ${role} lost a game it should never lose`);
  }
});

test('a non-square certificate is found from either orientation', async () => {
  const manifest = await loadPerfectChaosCompleteManifest(MANIFEST_URL);
  // A 4x5 round becomes 5x4 the moment either player rotates, so the same
  // certificate has to answer for both shapes.
  const upright = findPerfectChaosCompletePolicy(manifest, 4, 5, 4, 1);
  const rotated = findPerfectChaosCompletePolicy(manifest, 5, 4, 4, 1);
  assert.ok(upright);
  assert.equal(rotated, upright);

  const policy = await loadVerifiedPerfectChaosCompletePolicy(5, 4, 4, 1, {
    manifestUrl: MANIFEST_URL,
  });
  assert.ok(policy, 'the rotated shape must resolve to the committed certificate');
  assert.equal(policy.rows, 4);
  assert.equal(policy.columns, 5);
});

test('the certificate keeps playing after a rotation changes the board shape', async () => {
  const policy = await loadVerifiedPerfectChaosCompletePolicy(4, 5, 4, 1, {
    manifestUrl: MANIFEST_URL,
  });
  const aiPlayer = YELLOW;
  let board = createBoard(4, 5);
  let currentPlayer = aiPlayer;
  let rotatedShapeSeen = false;

  // Force the opponent to rotate whenever it can, so play spends most of its
  // time in the transposed orientation.
  for (let ply = 0; ply < 24; ply += 1) {
    if (currentPlayer === aiPlayer) {
      const chosen = choosePerfectChaosMove(
        { board, currentPlayer, startingPlayer: aiPlayer, connect: 4, chaosMode: true },
        { difficulty: 'perfect', aiPlayer, perfectChaosCompletePolicy: policy },
      );
      const applied = applyAction(board, chosen.action, currentPlayer);
      assert.ok(applied);
      board = applied.board;
      const outcome = resolveActionOutcome(
        board,
        4,
        currentPlayer,
        chosen.action.type,
        chosen.action.type === ACTION_DROP
          ? { row: applied.row, column: applied.column }
          : null,
      );
      if (outcome.status !== 'playing') break;
    } else {
      const applied = applyAction(board, { type: 'rotateCW' }, currentPlayer);
      assert.ok(applied);
      board = applied.board;
      const outcome = resolveActionOutcome(board, 4, currentPlayer, 'rotateCW', null);
      if (outcome.status !== 'playing') break;
    }
    if (board.length === 5 && board[0].length === 4) rotatedShapeSeen = true;
    currentPlayer = currentPlayer === RED ? YELLOW : RED;
  }
  assert.equal(rotatedShapeSeen, true, 'the test must actually exercise the rotated shape');
});

test('the role helper mirrors the classic convention', () => {
  assert.equal(perfectChaosCompleteRole(YELLOW, YELLOW), 1);
  assert.equal(perfectChaosCompleteRole(RED, YELLOW), 2);
  assert.equal(perfectChaosCompleteRole(0, YELLOW), null);
});

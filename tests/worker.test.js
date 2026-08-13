import test from 'node:test';
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';

import { RED, YELLOW, createBoard, positionKey } from '../src/engine.js';

function chaosRepetitionHistory(board, currentPlayer, startingPlayer = RED) {
  const entries = new Map();
  const initial = createBoard(6, 7);
  entries.set(positionKey(initial, startingPlayer, 4, true), 1);
  const currentKey = positionKey(board, currentPlayer, 4, true);
  if (!entries.has(currentKey)) entries.set(currentKey, 1);
  return [...entries];
}

function createBrowserWorkerShim() {
  const workerModuleUrl = new URL('../src/ai-worker.js', import.meta.url).href;
  const source = `
    import { parentPort } from 'node:worker_threads';
    globalThis.self = {
      addEventListener(type, listener) {
        if (type === 'message') {
          parentPort.on('message', (data) => listener({ data }));
        }
      },
      postMessage(message) {
        parentPort.postMessage(message);
      },
    };
    await import(${JSON.stringify(workerModuleUrl)});
  `;

  return new Worker(new URL(`data:text/javascript,${encodeURIComponent(source)}`), {
    type: 'module',
  });
}

test('the browser worker returns a legal AI action with the matching request id', async (context) => {
  const worker = createBrowserWorkerShim();
  context.after(() => worker.terminate());

  const board = createBoard(6, 7);
  board[5][3] = RED;
  const requestId = 42;
  const currentPlayer = YELLOW;

  const messages = await new Promise((resolve, reject) => {
    const received = [];
    const timeout = setTimeout(() => reject(new Error('AI worker response timed out.')), 2_000);
    worker.once('error', reject);
    worker.on('message', (message) => {
      received.push(message);
      if (message.kind === 'result' || message.kind === 'error') {
        clearTimeout(timeout);
        resolve(received);
      }
    });
    worker.postMessage({
      requestId,
      position: {
        board,
        currentPlayer,
        connect: 4,
        chaosMode: false,
        repetitionCounts: [[positionKey(board, currentPlayer, 4, false), 1]],
      },
      options: {
        difficulty: 'medium',
        aiPlayer: YELLOW,
        timeBudgetMs: 40,
        maximumDepth: 4,
      },
    });
  });

  const progress = messages.filter((message) => message.kind === 'progress');
  const response = messages.at(-1);

  assert.equal(response.requestId, requestId);
  assert.equal(response.kind, 'result');
  assert.equal(response.error, undefined);
  assert.ok(progress.length >= 1);
  assert.ok(progress.every((message) => message.progress.depth >= 1));
  assert.equal(response.result.action.type, 'drop');
  assert.ok(response.result.action.column >= 0 && response.result.action.column < 7);
});

test('the browser worker avoids the bounded-search opening rotation regression', async (context) => {
  const worker = createBrowserWorkerShim();
  context.after(() => worker.terminate());

  const board = createBoard(6, 7);
  board[5] = [YELLOW, RED, 0, RED, RED, 0, YELLOW];
  const requestId = 77;

  const response = await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Chaos worker response timed out.')), 5_000);
    worker.once('error', reject);
    worker.on('message', (message) => {
      if (message.requestId !== requestId || message.kind === 'progress') return;
      clearTimeout(timeout);
      if (message.kind === 'error') reject(new Error(message.error));
      else resolve(message);
    });
    worker.postMessage({
      requestId,
      position: {
        board,
        currentPlayer: YELLOW,
        connect: 4,
        chaosMode: true,
        repetitionCounts: [[positionKey(board, YELLOW, 4, true), 1]],
      },
      options: {
        difficulty: 'brutal',
        aiPlayer: YELLOW,
        maximumDepth: 3,
        quiescenceDepth: 2,
        chaosExactEmptyThreshold: 0,
      },
    });
  });

  assert.equal(response.kind, 'result');
  assert.deepEqual(response.result.action, { type: 'drop', column: 2 });
  assert.equal(response.result.depth, 3);
});

test('one browser worker handles consecutive AI requests', async (context) => {
  const worker = createBrowserWorkerShim();
  context.after(() => worker.terminate());

  const ask = (requestId, column) => new Promise((resolve, reject) => {
    const board = createBoard(6, 7);
    board[5][column] = RED;
    const currentPlayer = YELLOW;
    const timeout = setTimeout(() => reject(new Error(`AI worker request ${requestId} timed out.`)), 2_000);
    const onError = (error) => {
      clearTimeout(timeout);
      worker.off('message', onMessage);
      reject(error);
    };
    const onMessage = (message) => {
      if (message.requestId !== requestId || message.kind === 'progress') return;
      clearTimeout(timeout);
      worker.off('error', onError);
      worker.off('message', onMessage);
      if (message.kind === 'error') reject(new Error(message.error));
      else resolve(message.result);
    };
    worker.once('error', onError);
    worker.on('message', onMessage);
    worker.postMessage({
      requestId,
      position: {
        board,
        currentPlayer,
        connect: 4,
        chaosMode: false,
        repetitionCounts: [[positionKey(board, currentPlayer, 4, false), 1]],
      },
      options: { difficulty: 'medium', aiPlayer: YELLOW, maximumDepth: 3 },
    });
  });

  const first = await ask(101, 3);
  const second = await ask(102, 2);
  for (const result of [first, second]) {
    assert.equal(result.action.type, 'drop');
    assert.ok(result.action.column >= 0 && result.action.column < 7);
  }
});

test('the browser worker uses the certified Chaos policy for Brutal standard play', async (context) => {
  const worker = createBrowserWorkerShim();
  context.after(() => worker.terminate());

  const board = createBoard(6, 7);
  board[5][2] = RED;
  const requestId = 202;
  const currentPlayer = YELLOW;

  const messages = await new Promise((resolve, reject) => {
    const received = [];
    const timeout = setTimeout(() => reject(new Error('Certified Chaos policy worker response timed out.')), 2_000);
    worker.once('error', reject);
    worker.on('message', (message) => {
      if (message.requestId !== requestId) return;
      received.push(message);
      if (message.kind === 'result' || message.kind === 'error') {
        clearTimeout(timeout);
        resolve(received);
      }
    });
    worker.postMessage({
      requestId,
      position: {
        board,
        currentPlayer,
        connect: 4,
        chaosMode: true,
        startingPlayer: RED,
        repetitionCounts: chaosRepetitionHistory(board, currentPlayer),
      },
      options: { difficulty: 'brutal', aiPlayer: YELLOW },
    });
  });

  const response = messages.at(-1);
  assert.equal(response.kind, 'result');
  assert.equal(response.result.solver, 'chaos-certified-prefix');
  assert.equal(response.result.nodes, 0);
  assert.equal(response.result.depth, 0);
  assert.equal(response.result.certifiedThroughPieces, 8);
  assert.deepEqual(response.result.action, { type: 'drop', column: 3 });
  assert.ok(messages.some((message) => (
    message.kind === 'progress' && message.progress.solver === 'chaos-certified-prefix'
  )));

  const secondBoard = createBoard(6, 7);
  secondBoard[5][2] = RED;
  secondBoard[5][3] = YELLOW;
  secondBoard[4][3] = RED;
  const secondRequestId = 203;
  const secondResponse = await new Promise((resolve, reject) => {
    const timeout = setTimeout(
      () => reject(new Error('Second certified Chaos worker response timed out.')),
      2_000,
    );
    const onMessage = (message) => {
      if (message.requestId !== secondRequestId || message.kind === 'progress') return;
      clearTimeout(timeout);
      worker.off('message', onMessage);
      resolve(message);
    };
    worker.on('message', onMessage);
    worker.postMessage({
      requestId: secondRequestId,
      position: {
        board: secondBoard,
        currentPlayer: YELLOW,
        connect: 4,
        chaosMode: true,
        startingPlayer: RED,
        repetitionCounts: chaosRepetitionHistory(secondBoard, YELLOW),
      },
      options: { difficulty: 'brutal', aiPlayer: YELLOW },
    });
  });

  assert.equal(secondResponse.kind, 'result');
  assert.equal(secondResponse.result.solver, 'chaos-certified-prefix');
  assert.deepEqual(secondResponse.result.action, { type: 'drop', column: 3 });
});

test('the certified Chaos policy preserves the starting role after transform-only turns', async (context) => {
  const worker = createBrowserWorkerShim();
  context.after(() => worker.terminate());

  // This equal-count state is covered only by the second-player certificate. Piece counts
  // cannot reveal the starting role in Chaos because either player may transform instead
  // of dropping, so the worker must use the explicit round configuration.
  const board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [YELLOW, 0, 0, 0, 0, 0, 0],
    [YELLOW, RED, 0, 0, 0, 0, 0],
    [RED, YELLOW, RED, 0, 0, 0, 0],
  ];
  const requestId = 204;

  const response = await new Promise((resolve, reject) => {
    const timeout = setTimeout(
      () => reject(new Error('Transform-history Chaos policy response timed out.')),
      2_000,
    );
    worker.once('error', reject);
    worker.on('message', (message) => {
      if (message.requestId !== requestId || message.kind === 'progress') return;
      clearTimeout(timeout);
      resolve(message);
    });
    worker.postMessage({
      requestId,
      position: {
        board,
        currentPlayer: YELLOW,
        connect: 4,
        chaosMode: true,
        repetitionCounts: chaosRepetitionHistory(board, YELLOW),
      },
      options: { difficulty: 'brutal', aiPlayer: YELLOW },
    });
  });

  assert.equal(response.kind, 'result');
  assert.equal(response.result.solver, 'chaos-certified-prefix');
  assert.deepEqual(response.result.action, { type: 'drop', column: 3 });
});

test('the browser worker lazy-loads the certified 8→10 Chaos policy layer', async (context) => {
  const worker = createBrowserWorkerShim();
  context.after(() => worker.terminate());

  const board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [YELLOW, 0, 0, 0, 0, 0, 0],
    [YELLOW, RED, 0, 0, 0, 0, 0],
    [RED, RED, 0, 0, 0, 0, 0],
    [YELLOW, YELLOW, RED, 0, 0, 0, 0],
  ];
  const requestId = 205;

  const response = await new Promise((resolve, reject) => {
    const timeout = setTimeout(
      () => reject(new Error('Second certified Chaos policy layer response timed out.')),
      2_000,
    );
    worker.once('error', reject);
    worker.on('message', (message) => {
      if (message.requestId !== requestId || message.kind === 'progress') return;
      clearTimeout(timeout);
      resolve(message);
    });
    worker.postMessage({
      requestId,
      position: {
        board,
        currentPlayer: YELLOW,
        connect: 4,
        chaosMode: true,
        startingPlayer: RED,
        repetitionCounts: chaosRepetitionHistory(board, YELLOW),
      },
      options: { difficulty: 'brutal', aiPlayer: YELLOW },
    });
  });

  assert.equal(response.kind, 'result');
  assert.equal(response.result.solver, 'chaos-certified-prefix');
  assert.equal(response.result.certifiedFromPieces, 8);
  assert.equal(response.result.certifiedThroughPieces, 10);
  assert.deepEqual(response.result.action, { type: 'drop', column: 3 });
});

test('the browser worker returns a proved Perfect Chaos endgame move', async (context) => {
  const worker = createBrowserWorkerShim();
  context.after(() => worker.terminate());

  const board = [
    [RED, RED, RED, YELLOW, RED, 0, 0],
    [YELLOW, YELLOW, YELLOW, RED, YELLOW, 0, 0],
    [YELLOW, RED, YELLOW, RED, YELLOW, RED, 0],
    [YELLOW, RED, RED, RED, YELLOW, YELLOW, 0],
    [RED, YELLOW, YELLOW, YELLOW, RED, YELLOW, YELLOW],
    [RED, RED, YELLOW, YELLOW, RED, RED, RED],
  ];
  const requestId = 303;

  const response = await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Perfect Chaos worker response timed out.')), 5_000);
    worker.once('error', reject);
    worker.on('message', (message) => {
      if (message.requestId !== requestId || message.kind === 'progress') return;
      clearTimeout(timeout);
      if (message.kind === 'error') reject(new Error(message.error));
      else resolve(message);
    });
    worker.postMessage({
      requestId,
      position: {
        board,
        currentPlayer: RED,
        connect: 4,
        chaosMode: true,
        repetitionCounts: [[positionKey(board, RED, 4, true), 1]],
      },
      options: { difficulty: 'perfect', aiPlayer: RED },
    });
  });

  assert.equal(response.kind, 'result');
  assert.equal(response.result.solver, 'chaos-exact-graph');
  assert.equal(response.result.solved, true);
  assert.deepEqual(response.result.action, { type: 'rotateCW' });
});

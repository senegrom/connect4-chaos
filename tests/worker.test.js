import test from 'node:test';
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';

import { RED, YELLOW, createBoard, positionKey } from '../src/engine.js';

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

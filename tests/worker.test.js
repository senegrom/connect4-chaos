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

  const response = await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('AI worker response timed out.')), 2_000);
    worker.once('error', reject);
    worker.once('message', (message) => {
      clearTimeout(timeout);
      resolve(message);
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

  assert.equal(response.requestId, requestId);
  assert.equal(response.error, undefined);
  assert.equal(response.result.action.type, 'drop');
  assert.ok(response.result.action.column >= 0 && response.result.action.column < 7);
});

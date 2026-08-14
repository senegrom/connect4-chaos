import test from 'node:test';
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';

import { requireCertifiedChaosPolicy } from '../src/ai-worker.js';
import { RED, YELLOW, createBoard, positionKey } from '../src/engine.js';

test('the certified Brutal policy fails closed when a covered position is missing', () => {
  const policy = requireCertifiedChaosPolicy({
    fromBoundary: 0,
    boundary: 8,
    entryCount: 1,
    lookup() {
      return null;
    },
  });

  assert.throws(
    () => policy.lookup(createBoard(6, 7), YELLOW, YELLOW),
    /does not cover this reachable position/,
  );
});

function createPolicyLoadFailureWorker() {
  const workerModuleUrl = new URL('../src/ai-worker.js', import.meta.url).href;
  const source = `
    import { parentPort } from 'node:worker_threads';
    Object.defineProperty(globalThis, 'process', {
      value: undefined,
      configurable: true,
    });
    globalThis.fetch = async () => {
      throw new Error('forced certificate load failure');
    };
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

test('Brutal reports a certified-policy load failure instead of using heuristic search', async (context) => {
  const worker = createPolicyLoadFailureWorker();
  context.after(() => worker.terminate());

  const board = createBoard(6, 7);
  board[5][3] = RED;
  const initialBoard = createBoard(6, 7);
  const requestId = 901;

  const response = await new Promise((resolve, reject) => {
    const timeout = setTimeout(
      () => reject(new Error('Brutal certificate failure response timed out.')),
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
        repetitionCounts: [
          [positionKey(initialBoard, RED, 4, true), 1],
          [positionKey(board, YELLOW, 4, true), 1],
        ],
      },
      options: { difficulty: 'brutal', aiPlayer: YELLOW },
    });
  });

  assert.equal(response.kind, 'error');
  assert.match(response.error, /Could not load the certified Brutal Chaos policy/);
  assert.match(response.error, /forced certificate load failure/);
});

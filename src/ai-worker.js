import { chooseMove } from './ai.js';

let perfectBookPromise = null;
let perfectStrategyPromise = null;

function canUseStandardExactData(position) {
  return Boolean(
    position
      && !position.chaosMode
      && position.connect === 4
      && Array.isArray(position.board)
      && position.board.length === 6
      && position.board.every((row) => Array.isArray(row) && row.length === 7),
  );
}

async function exactDataFor(position, options) {
  if (!canUseStandardExactData(position)) {
    return { perfectBook: null, perfectStrategy: null };
  }

  if (options?.difficulty === 'perfect') {
    if (!perfectStrategyPromise) {
      perfectStrategyPromise = import('./perfect-strategy.js')
        .then((module) => module.loadPerfectStrategy())
        .catch(() => null);
    }
    return { perfectBook: null, perfectStrategy: await perfectStrategyPromise };
  }

  if (!perfectBookPromise) {
    perfectBookPromise = import('./perfect-book.js')
      .then((module) => module.loadPerfectBook())
      .catch(() => null);
  }
  return { perfectBook: await perfectBookPromise, perfectStrategy: null };
}

self.addEventListener('message', async (event) => {
  const { requestId, position, options } = event.data ?? {};
  try {
    const exactData = await exactDataFor(position, options);
    const result = chooseMove(position, {
      ...options,
      ...exactData,
      onIteration(progress) {
        self.postMessage({ requestId, kind: 'progress', progress });
      },
    });
    self.postMessage({ requestId, kind: 'result', result });
  } catch (error) {
    self.postMessage({
      requestId,
      kind: 'error',
      error: error instanceof Error ? error.message : String(error),
    });
  }
});

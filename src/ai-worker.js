import { chooseMove } from './ai.js';

let perfectBookPromise = null;

function canUsePerfectBook(position) {
  return Boolean(
    position
      && !position.chaosMode
      && position.connect === 4
      && Array.isArray(position.board)
      && position.board.length === 6
      && position.board.every((row) => Array.isArray(row) && row.length === 7),
  );
}

async function perfectBookFor(position) {
  if (!canUsePerfectBook(position)) return null;
  if (!perfectBookPromise) {
    perfectBookPromise = import('./perfect-book.js')
      .then((module) => module.loadPerfectBook())
      .catch(() => null);
  }
  return perfectBookPromise;
}

self.addEventListener('message', async (event) => {
  const { requestId, position, options } = event.data ?? {};
  try {
    const perfectBook = await perfectBookFor(position);
    const result = chooseMove(position, {
      ...options,
      perfectBook,
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

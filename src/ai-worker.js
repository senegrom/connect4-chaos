import { chooseMove } from './ai.js';
import { isBitboardPosition } from './bitboard.js';

const BOOK_DIFFICULTIES = new Set(['medium', 'hard', 'brutal']);

async function loadPerfectStrategy() {
  const { loadPerfectStrategy: load } = await import('./perfect-strategy.js');
  return load();
}

async function loadPerfectBook() {
  try {
    const { loadPerfectBook: load } = await import('./perfect-book.js');
    return await load();
  } catch {
    return null;
  }
}

async function exactDataFor(position, options) {
  if (!isBitboardPosition(position)) {
    return { perfectBook: null, perfectStrategy: null };
  }
  const difficulty = options?.difficulty ?? position?.difficulty ?? 'medium';
  if (difficulty === 'perfect') {
    return { perfectBook: null, perfectStrategy: await loadPerfectStrategy() };
  }
  const useBook = BOOK_DIFFICULTIES.has(difficulty)
    && options?.maximumDepth === undefined
    && options?.useBook !== false;
  return {
    perfectBook: useBook ? await loadPerfectBook() : null,
    perfectStrategy: null,
  };
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

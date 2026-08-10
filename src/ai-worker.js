import { chooseMove } from './ai.js';

self.addEventListener('message', (event) => {
  const { requestId, position, options } = event.data ?? {};
  try {
    const result = chooseMove(position, {
      ...options,
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

import { chooseMove } from './ai.js';

self.addEventListener('message', (event) => {
  const { requestId, position, options } = event.data ?? {};
  try {
    const result = chooseMove(position, options);
    self.postMessage({ requestId, result });
  } catch (error) {
    self.postMessage({
      requestId,
      error: error instanceof Error ? error.message : String(error),
    });
  }
});

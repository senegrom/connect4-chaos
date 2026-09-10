// Neural request orchestration is separate from board rendering and rules.
import { DOWNLOAD_BYTES, cancelNeuralLoad, loadNeuralNetwork, neuralLoadState,
  recordSearch, simulationsFor, invalidateNeuralNetwork } from './neural-client.js';
import { bestAction, searchPosition } from './neural-search.js';
import { requestDownload, showDownloadProgress } from './download-gate.js';
import { waitFor } from './async-control.js';

export async function runNeuralRequest(request, {
  isCurrent, onSearch, onFraction, shouldStop, finish, fail,
}) {
  const signal = request.controller.signal;
  const stale = () => signal.aborted || !isCurrent();
  let panel = null;
  let network = null;
  const abortNetwork = () => { if (network) invalidateNeuralNetwork(network); };
  signal.addEventListener('abort', abortNetwork, { once: true });
  try {
    if (stale()) return;
    if (neuralLoadState() !== 'ready') {
      if (neuralLoadState() === 'idle') {
        const agreed = await requestDownload({
          id: 'neural-opponent', title: 'Neural opponent', signal,
          description: 'The neural opponent is a trained network plus a search. Playing it needs a one-time download of the network and its runtime.',
          bytes: DOWNLOAD_BYTES.model + DOWNLOAD_BYTES.runtime,
        });
        if (stale()) return;
        if (!agreed) {
          fail('The neural opponent needs a one-time download. Choose Download when asked, or pick another opponent.');
          return;
        }
      }
      panel = showDownloadProgress({
        title: 'Neural opponent', signal,
        note: 'Downloading the network and its runtime.',
        onCancel: () => {
          if (stale()) return;
          // Invalidate the game request now; native startup can finish later,
          // but it must never play a move for this cancelled request.
          cancelNeuralLoad();
          fail('The neural download or startup was cancelled. Retry, or pick another opponent.');
        },
      });
    }
    network = await waitFor(loadNeuralNetwork({
      signal,
      onProgress(progress) {
        if (stale()) return;
        if (progress.stage === 'session') panel?.note(`Starting the network on ${progress.backend}.`);
        else panel?.update(progress.loaded ?? 0, progress.total ?? 0, 'Downloaded');
        onSearch({ solver: 'neural-loading', note: progress.stage === 'session'
          ? `Starting the network on ${progress.backend}` : 'Downloading the network (once)' });
      },
    }), { signal });
    panel?.close();
    panel = null;
    if (stale()) return;
    const simulations = simulationsFor(network);
    onSearch({ solver: 'neural-searching', note: `Neural search · up to ${simulations} simulations on ${network.backend}` });
    const started = performance.now();
    const result = await searchPosition(request.position,
      (...args) => waitFor(network.evaluate(...args), { signal, timeoutMs: 45_000, label: 'Network evaluation' }), {
        simulations, signal, shouldStop: () => stale() || shouldStop(),
        // One call per batch of leaves: the GPU is nearly idle on a single
        // position, so this is most of the search budget. A backend without
        // it - an older worker, a test stub - still plays, one leaf at a time.
        evaluateMany: typeof network.evaluateMany === 'function'
          ? (items) => waitFor(network.evaluateMany(items),
            { signal, timeoutMs: 45_000, label: 'Network evaluation' })
          : null,
        onProgress: (done, total) => { if (!stale()) onFraction(done / total); },
      });
    if (stale()) return;
    const elapsedMs = performance.now() - started;
    // Cached terminal edges require no network evaluation. Calibrate using
    // actual evaluations, not the requested (possibly interrupted) budget.
    recordSearch(network, elapsedMs, result.evaluations);
    const action = bestAction(result);
    if (!action) { fail('The neural opponent found no legal move.'); return; }
    finish({ action, score: result.value, depth: 0, nodes: result.completedSimulations,
      evaluations: result.evaluations, elapsedMs, solver: 'neural', solved: false, backend: network.backend });
  } catch (error) {
    if (network) invalidateNeuralNetwork(network);
    if (!stale()) fail(`The neural opponent failed: ${error.message}. Retry to restart it.`);
  } finally {
    signal.removeEventListener('abort', abortNetwork);
    panel?.close();
  }
}

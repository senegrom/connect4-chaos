import test from 'node:test';
import assert from 'node:assert/strict';

import {
  isRetriableBrowserLaunchFailure,
  parseAttemptCount,
  runBrowserSmoke,
} from '../scripts/browser-smoke-retry.mjs';

function result(code, diagnostics = '') {
  return { code, signal: null, diagnostics };
}

test('only pre-DevTools Chromium startup failures are retried', () => {
  assert.equal(
    isRetriableBrowserLaunchFailure('Error: Timed out waiting for DevToolsActivePort.'),
    true,
  );
  assert.equal(
    isRetriableBrowserLaunchFailure('Browser exited before exposing DevTools.\ncrash'),
    true,
  );
  assert.equal(
    isRetriableBrowserLaunchFailure('Error: Timed out navigating to http://127.0.0.1/.'),
    false,
  );
  assert.equal(
    isRetriableBrowserLaunchFailure('Brutal Chaos returned an illegal move.'),
    false,
  );
});

test('attempt-count configuration is deterministic and fail-closed', () => {
  assert.equal(parseAttemptCount(undefined), 3);
  assert.equal(parseAttemptCount(''), 3);
  assert.equal(parseAttemptCount('1'), 1);
  assert.equal(parseAttemptCount('5'), 5);
  assert.throws(() => parseAttemptCount('0'), /between 1 and 5/);
  assert.throws(() => parseAttemptCount('6'), /between 1 and 5/);
  assert.throws(() => parseAttemptCount('1.5'), /integer/);
  assert.throws(() => parseAttemptCount('nope'), /integer/);
});

test('a clean second launch succeeds after one exact startup failure', async () => {
  const calls = [];
  const waits = [];
  const logs = [];
  const sequence = [
    result(1, 'Timed out waiting for DevToolsActivePort.'),
    result(0, '{"ok":true}'),
  ];
  const outcome = await runBrowserSmoke({
    attempts: 3,
    retryDelayMs: 7,
    runAttempt: async () => {
      calls.push(calls.length + 1);
      return sequence.shift();
    },
    wait: async (milliseconds) => waits.push(milliseconds),
    log: (message) => logs.push(message),
  });

  assert.equal(outcome.code, 0);
  assert.equal(outcome.attemptsUsed, 2);
  assert.deepEqual(calls, [1, 2]);
  assert.deepEqual(waits, [7]);
  assert.equal(logs.length, 2);
  assert.match(logs[0], /fresh profile/);
  assert.match(logs[1], /passed on clean launch attempt 2\/3/);
});

test('semantic browser failures are never retried', async () => {
  let calls = 0;
  const outcome = await runBrowserSmoke({
    attempts: 5,
    retryDelayMs: 0,
    runAttempt: async () => {
      calls += 1;
      return result(1, 'Perfect policy returned an illegal action.');
    },
    log: () => {},
  });
  assert.equal(calls, 1);
  assert.equal(outcome.code, 1);
  assert.equal(outcome.attemptsUsed, 1);
  assert.equal(outcome.retriable, false);
});

test('startup retries remain bounded and preserve the final failure', async () => {
  let calls = 0;
  const waits = [];
  const outcome = await runBrowserSmoke({
    attempts: 3,
    retryDelayMs: 2,
    runAttempt: async () => {
      calls += 1;
      return result(17, 'Browser exited before exposing DevTools.');
    },
    wait: async (milliseconds) => waits.push(milliseconds),
    log: () => {},
  });
  assert.equal(calls, 3);
  assert.deepEqual(waits, [2, 4]);
  assert.equal(outcome.code, 17);
  assert.equal(outcome.attemptsUsed, 3);
  assert.equal(outcome.retriable, true);
});

test('programmatic retry limits are also bounded', async () => {
  await assert.rejects(runBrowserSmoke({ attempts: 0 }), /between 1 and 5/);
  await assert.rejects(runBrowserSmoke({ attempts: 6 }), /between 1 and 5/);
  await assert.rejects(
    runBrowserSmoke({ attempts: 1, retryDelayMs: 10_001 }),
    /between 0 and 10000/,
  );
});

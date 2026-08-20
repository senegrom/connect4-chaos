#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const DEFAULT_ATTEMPTS = 3;
const MAXIMUM_ATTEMPTS = 5;
const DEFAULT_RETRY_DELAY_MS = 1_000;
const MAXIMUM_RETRY_DELAY_MS = 10_000;
const MAXIMUM_DIAGNOSTIC_BYTES = 256 * 1024;
const BROWSER_SMOKE = fileURLToPath(new URL('./browser-smoke.mjs', import.meta.url));

function delay(milliseconds) {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
}

function appendDiagnosticTail(current, chunk) {
  const next = current + chunk;
  return next.length <= MAXIMUM_DIAGNOSTIC_BYTES
    ? next
    : next.slice(-MAXIMUM_DIAGNOSTIC_BYTES);
}

export function parseAttemptCount(value = process.env.BROWSER_SMOKE_ATTEMPTS) {
  if (value === undefined || value === '') return DEFAULT_ATTEMPTS;
  if (!/^[0-9]+$/.test(String(value))) {
    throw new RangeError('BROWSER_SMOKE_ATTEMPTS must be an integer.');
  }
  const attempts = Number(value);
  if (!Number.isSafeInteger(attempts) || attempts < 1 || attempts > MAXIMUM_ATTEMPTS) {
    throw new RangeError(
      `BROWSER_SMOKE_ATTEMPTS must be between 1 and ${MAXIMUM_ATTEMPTS}.`,
    );
  }
  return attempts;
}

export function isRetriableBrowserLaunchFailure(output) {
  return output.includes('Timed out waiting for DevToolsActivePort.')
    || output.includes('Browser exited before exposing DevTools.');
}

export function runBrowserSmokeAttempt({
  scriptPath = BROWSER_SMOKE,
  spawnImplementation = spawn,
  environment = process.env,
  stdout = process.stdout,
  stderr = process.stderr,
} = {}) {
  return new Promise((resolveAttempt, rejectAttempt) => {
    const child = spawnImplementation(process.execPath, [scriptPath], {
      env: environment,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let diagnostics = '';

    const collect = (stream, destination) => {
      stream?.setEncoding?.('utf8');
      stream?.on?.('data', (chunk) => {
        const text = String(chunk);
        diagnostics = appendDiagnosticTail(diagnostics, text);
        destination?.write?.(text);
      });
    };
    collect(child.stdout, stdout);
    collect(child.stderr, stderr);

    child.once('error', rejectAttempt);
    child.once('close', (code, signal) => {
      resolveAttempt({
        code: Number.isInteger(code) ? code : 1,
        signal,
        diagnostics,
      });
    });
  });
}

export async function runBrowserSmoke({
  attempts = parseAttemptCount(),
  retryDelayMs = DEFAULT_RETRY_DELAY_MS,
  runAttempt = runBrowserSmokeAttempt,
  wait = delay,
  log = (message) => process.stderr.write(`${message}\n`),
} = {}) {
  if (!Number.isSafeInteger(attempts) || attempts < 1 || attempts > MAXIMUM_ATTEMPTS) {
    throw new RangeError(`Browser smoke attempts must be between 1 and ${MAXIMUM_ATTEMPTS}.`);
  }
  if (!Number.isFinite(retryDelayMs)
      || retryDelayMs < 0
      || retryDelayMs > MAXIMUM_RETRY_DELAY_MS) {
    throw new RangeError(
      `Browser smoke retry delay must be between 0 and ${MAXIMUM_RETRY_DELAY_MS} milliseconds.`,
    );
  }

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    const result = await runAttempt();
    if (result.code === 0) {
      if (attempt > 1) log(`Browser smoke passed on clean launch attempt ${attempt}/${attempts}.`);
      return { ...result, attemptsUsed: attempt };
    }

    const retriable = isRetriableBrowserLaunchFailure(result.diagnostics);
    if (!retriable || attempt === attempts) {
      return { ...result, attemptsUsed: attempt, retriable };
    }

    log(
      `Chromium did not expose DevTools; retrying browser smoke with a fresh profile `
      + `(attempt ${attempt + 1}/${attempts}).`,
    );
    if (retryDelayMs > 0) await wait(retryDelayMs * attempt);
  }

  throw new Error('Browser smoke retry loop ended without a result.');
}

const invokedAsScript = process.argv[1]
  && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (invokedAsScript) {
  runBrowserSmoke().then((result) => {
    if (result.code !== 0) process.exitCode = result.code || 1;
  }).catch((error) => {
    console.error(error instanceof Error ? error.stack : String(error));
    process.exitCode = 1;
  });
}

/** Compiler discovery and cached builds of the native solvers, shared by the
 * proof scripts that compile them and by the native tests.
 *
 * CXX is honoured as given - a name on the PATH or a path - so CXX=clang++ on
 * the Darwin job selects clang instead of whatever /usr/bin happens to hold.
 * A build is cached in the OS temp directory under a key over the source
 * bytes, the headers it includes, the compiler, its version banner and the
 * flags, so every test and script in a run shares one binary per source and
 * flag set, and an edit to any of them builds afresh. A lock file keeps test
 * files running in parallel processes from compiling the same binary at once;
 * since the name is content-keyed, a build that loses a race anyway publishes
 * nothing and uses the copy that won.
 */
import { spawn, spawnSync } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import { mkdir, open, readFile, rename, rm, stat, utimes } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

import { nativeLinkFlags } from './native-toolchain.mjs';

// The flags the proof scripts have always built with; the host link flags are
// appended by buildNative.
const PROOF_FLAGS = Object.freeze(['-std=c++20', '-O3', '-Wall', '-Wextra', '-Wpedantic']);

const CACHE = join(tmpdir(), 'connect4-native-builds');
// A lock's owner touches it every HEARTBEAT_MS while it builds; one left
// untouched for STALE_LOCK_MS is abandoned, whatever process its PID now names.
const HEARTBEAT_MS = 5_000;
const STALE_LOCK_MS = 60_000;
const builds = new Map();

/** The compiler to use, or null when none is installed. */
export function findCompiler() {
  if (process.env.CXX) return process.env.CXX;
  // Probe the PATH first: under Git Bash on Windows /usr/bin/g++ is the MSYS
  // compiler, whose executables crash silently, while the PATH carries the
  // real toolchain. On Linux the PATH g++ is /usr/bin/g++ anyway.
  for (const candidate of ['g++', 'clang++', '/usr/bin/g++', '/usr/bin/clang++']) {
    if (spawnSync(candidate, ['--version'], { encoding: 'utf8' }).status === 0) return candidate;
  }
  return null;
}

/** Runs a process to completion; resolves with its code, signal and output. */
export function runProcess(command, args, options = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'], ...options });
    const stdout = [];
    const stderr = [];
    child.stdout?.on('data', (chunk) => stdout.push(chunk));
    child.stderr?.on('data', (chunk) => stderr.push(chunk));
    child.once('error', reject);
    child.once('close', (code, signal) => resolvePromise({
      code,
      signal,
      stdout: Buffer.concat(stdout).toString('utf8'),
      stderr: Buffer.concat(stderr).toString('utf8'),
    }));
  });
}

/** The records a native tool printed, one JSON object per line. */
export function parseJsonLines(output) {
  return output.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

// Whether the process a lock names still runs. Signal 0 only asks: ESRCH is
// gone, and EPERM is alive under another user. A lock with no readable
// owner yet - it is written just after the file is created - counts as live.
function ownerAlive(owner) {
  const pid = Number.parseInt(owner, 10);
  if (!Number.isSafeInteger(pid) || pid <= 0) return true;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error?.code !== 'ESRCH';
  }
}

/** Holds an exclusive lock file, which names this process, while build()
 * runs; done() says whether another holder already built the target. A lock
 * whose owner has exited - a test run killed mid-compile - is broken at once.
 * The owner refreshes the lock's mtime as it builds, and a lock not refreshed
 * for `staleMs` is broken too: Windows reuses PIDs quickly, and a dead
 * owner's PID taken by another process used to hold every later build for
 * fifteen minutes. */
export async function underLock(lock, done, build, { heartbeatMs = HEARTBEAT_MS, staleMs = STALE_LOCK_MS } = {}) {
  for (;;) {
    let handle;
    try {
      handle = await open(lock, 'wx');
    } catch (error) {
      if (error?.code !== 'EEXIST') throw error;
      if (await done()) return;
      const [owner, age] = await Promise.all([
        readFile(lock, 'utf8').catch(() => ''),
        stat(lock).then((info) => Date.now() - info.mtimeMs, () => 0),
      ]);
      if (age > staleMs || !ownerAlive(owner)) await rm(lock, { force: true });
      else await sleep(200);
      continue;
    }
    const heartbeat = setInterval(() => {
      const now = new Date();
      utimes(lock, now, now).catch(() => {});
    }, heartbeatMs);
    try {
      await handle.writeFile(String(process.pid));
      await build();
      return;
    } finally {
      clearInterval(heartbeat);
      await handle.close();
      await rm(lock, { force: true });
    }
  }
}

/** Moves a finished compile to its cache name. Two builds can still race -
 * waiters that each judged the same lock abandoned - and on Windows renaming
 * onto a binary another process is running fails with EPERM. The name is
 * content-keyed, so a copy that is already there is as good as this one. */
export async function publishBinary(partial, binary, move = rename) {
  try {
    await move(partial, binary);
  } catch (error) {
    if (!['EPERM', 'EEXIST', 'EACCES'].includes(error?.code) || !(await exists(binary))) throw error;
  }
}

/**
 * The digest of every header a source includes with quotes, and of the
 * headers those include, by file name. The cache used to be keyed on the
 * .cpp alone: after an edit to checkpoint-io.hpp the local tests ran the
 * binary built before it, while CI, building fresh, tested the new code.
 */
export async function includedHeaders(source, found = new Map()) {
  const text = await readFile(source, 'utf8');
  for (const [, header] of text.matchAll(/^[ \t]*#[ \t]*include[ \t]*"([^"]+)"/gm)) {
    const path = resolve(dirname(source), header);
    if (found.has(path)) continue;
    found.set(path, createHash('sha256').update(await readFile(path)).digest('hex'));
    await includedHeaders(path, found);
  }
  return Object.fromEntries([...found].map(([path, digest]) => [basename(path), digest]).sort());
}

/**
 * Compiles source once per identity and returns the cached binary with the
 * identity it was built under: the source digest, its headers' digests, the
 * compiler, its version banner and the complete argument list.
 */
export async function buildNative(source, options = {}) {
  const compiler = options.compiler ?? findCompiler();
  if (!compiler) throw new Error('A C++20 compiler is required (set CXX, or install g++/clang++).');
  const flags = [...(options.flags ?? PROOF_FLAGS), ...nativeLinkFlags()];
  const version = spawnSync(compiler, ['--version'], { encoding: 'utf8' });
  if (version.error || version.status !== 0) {
    throw new Error(`${compiler} --version failed; is CXX a working C++ compiler?`);
  }
  const sourceSha256 = createHash('sha256').update(await readFile(source)).digest('hex');
  const identity = {
    sourceSha256,
    headersSha256: await includedHeaders(source),
    compiler,
    compilerVersion: version.stdout.trim(),
    flags,
  };
  const key = createHash('sha256').update(JSON.stringify(identity)).digest('hex');
  const name = options.name ?? basename(source, '.cpp');
  const extension = process.platform === 'win32' ? '.exe' : '';
  const binary = join(CACHE, `${name}-${key.slice(0, 16)}${extension}`);

  if (!builds.has(binary)) {
    builds.set(binary, (async () => {
      await mkdir(CACHE, { recursive: true });
      if (await exists(binary)) return '';
      let warnings = '';
      await underLock(`${binary}.lock`, () => exists(binary), async () => {
        if (await exists(binary)) return;
        // Compile beside the target and rename, so no process ever runs a
        // half-written binary. The suffix keeps .exe last for MinGW.
        const partial = `${binary}.${process.pid}-${randomBytes(4).toString('hex')}.partial${extension}`;
        try {
          const result = await runProcess(compiler, [...flags, source, '-o', partial]);
          if (result.code !== 0) {
            throw new Error(`${name} failed to compile.\n${result.stderr || result.stdout}`);
          }
          warnings = result.stderr.trim();
          await publishBinary(partial, binary);
        } finally {
          await rm(partial, { force: true });
        }
      });
      return warnings;
    })().catch((error) => {
      builds.delete(binary);
      throw error;
    }));
  }
  const warnings = await builds.get(binary);
  return { ...identity, binary, warnings };
}

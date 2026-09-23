/** Compiler discovery and cached builds of the native solvers, shared by the
 * proof scripts that compile them and by the native tests.
 *
 * CXX is honoured as given - a name on the PATH or a path - so CXX=clang++ on
 * the Darwin job selects clang instead of whatever /usr/bin happens to hold.
 * A build is cached in the OS temp directory under a key over the source
 * bytes, the compiler, its version banner and the flags, so every test and
 * script in a run shares one binary per source and flag set, and an edit to
 * any of them builds afresh. A lock file keeps test files running in parallel
 * processes from compiling the same binary at once.
 */
import { spawn, spawnSync } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import { mkdir, open, readFile, rename, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

import { nativeLinkFlags } from './native-toolchain.mjs';

// The flags the proof scripts have always built with; the host link flags are
// appended by buildNative.
export const PROOF_FLAGS = Object.freeze(['-std=c++20', '-O3', '-Wall', '-Wextra', '-Wpedantic']);

const CACHE = join(tmpdir(), 'connect4-native-builds');
const STALE_LOCK_MS = 15 * 60_000;
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

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

// Holds an exclusive lock file while build() runs. A lock left behind by a
// killed process is broken once it is older than any compile could take.
async function underLock(lock, done, build) {
  for (;;) {
    let handle;
    try {
      handle = await open(lock, 'wx');
    } catch (error) {
      if (error?.code !== 'EEXIST') throw error;
      if (await done()) return;
      const age = await stat(lock).then((info) => Date.now() - info.mtimeMs, () => 0);
      if (age > STALE_LOCK_MS) await rm(lock, { force: true });
      else await sleep(200);
      continue;
    }
    try {
      await build();
      return;
    } finally {
      await handle.close();
      await rm(lock, { force: true });
    }
  }
}

/**
 * Compiles source once per identity and returns the cached binary with the
 * identity it was built under: the source digest, the compiler, its version
 * banner and the complete argument list.
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
          await rename(partial, binary);
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

/** Compiler discovery and cached builds of the native solvers, shared by the
 * proof scripts that compile them and by the native tests.
 *
 * CXX is honoured as given - a name on the PATH or a path - so CXX=clang++ on
 * the Darwin job selects clang instead of whatever /usr/bin happens to hold.
 * A build is cached in the OS temp directory under a key over the source
 * bytes, the headers it includes, the compiler, its version banner and the
 * flags, so every test and script in a run shares one binary per source, and
 * an edit to any of them builds afresh. A lock file keeps test files running
 * in parallel processes from compiling the same binary at once; since the
 * name is content-keyed, a build that loses a race anyway publishes nothing
 * and uses the copy that won. The cache runs whatever it finds under that
 * name, so on POSIX it is refused unless only this user can write to it.
 */
import { spawn, spawnSync } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import { lstat, mkdir, mkdtemp, open, readFile, rename, rm, stat, utimes } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

// The flags the proof scripts have always built with; buildNative appends
// -pthread for a source that starts threads, and the host link flags.
const PROOF_FLAGS = Object.freeze(['-std=c++20', '-O3', '-Wall', '-Wextra', '-Wpedantic']);
// The standard headers a thread is started through: std::thread, std::async.
const THREAD_HEADER = /^[ \t]*#[ \t]*include[ \t]*<(?:thread|future)>/m;

// One cache per user on POSIX, where os.tmpdir() is usually the /tmp that
// every user shares: the first user's directory left every other user unable
// to take a build lock in it. The name is no protection - user ids are public
// - so privateCache checks the directory it gets.
const CACHE = join(tmpdir(),
  process.getuid ? `connect4-native-builds-${process.getuid()}` : 'connect4-native-builds');
// A lock's owner touches it every HEARTBEAT_MS while it builds; one left
// untouched for STALE_LOCK_MS is abandoned, whatever process its PID now names.
const HEARTBEAT_MS = 5_000;
const STALE_LOCK_MS = 60_000;
const builds = new Map();

/** The host link flags every native build appends. MinGW must not pick up an
 * unrelated libstdc++ DLL from Git for Windows, so Windows links statically;
 * Darwin needs its dynamic system runtime, and Linux needs no workaround.
 * The builds are for the host, with GCC-style compiler drivers. */
export function nativeLinkFlags(platform = process.platform) {
  if (typeof platform !== 'string' || !platform) throw new TypeError('A host platform is required.');
  return platform === 'win32' ? ['-static'] : [];
}

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

/** Why a directory with these lstat results is not `uid`'s alone, or ''. */
function exposure(info, uid) {
  if (!info.isDirectory()) return 'is a link or not a directory';
  if (info.uid !== uid) return 'belongs to another user';
  if ((info.mode & 0o022) !== 0) return 'is writable by other users';
  return '';
}

/** Creates the cache directory and refuses it unless it is a real directory
 * that this user owns and nobody else can write to. A binary found in the
 * cache under its expected name is run as whoever asked for it, and that name
 * is built from public inputs only, so on a shared host another user who made
 * the directory first could plant one there. Windows has no POSIX owner to
 * check (process.getuid is POSIX-only) and gives each user a temp directory
 * of their own.
 *
 * Some filesystems do not keep the owner and mode a directory is made with,
 * and report the same ones for whatever is made on them: WSL's drvfs without
 * the metadata option, exFAT, ntfs-3g, NFS with root_squash for root. A cache
 * there is refused like any other, since nothing shows who else can write to
 * it, but "Delete it" cannot help: the directory made in its place reports
 * the same. So before a cache is refused, a directory is made beside it for
 * this user alone, and when that one is no more private the refusal names
 * the filesystem and the remedy that works. `uid` and `status` stand in for
 * the process's user and lstat in the tests. */
export async function privateCache(cache = CACHE, uid = process.getuid?.(), status = lstat) {
  await mkdir(cache, { recursive: true, mode: 0o700 });
  if (uid === undefined) return;
  const problem = exposure(await status(cache), uid);
  if (!problem) return;
  const planted = `The native build cache ${cache} ${problem}, so a binary in it could have been planted.`;
  let unkept = '';
  try {
    const probe = await mkdtemp(`${cache}-probe-`);
    try {
      unkept = exposure(await status(probe), uid);
    } finally {
      await rm(probe, { recursive: true, force: true });
    }
  } catch {
    // No probe could be made or read: the cache is refused as it stands.
  }
  if (unkept) {
    throw new Error(`${planted} Deleting it cannot help: a directory just made beside it for this user alone`
      + ` ${unkept}, so this filesystem does not keep the owner and mode a directory is made with. Set TMPDIR to a`
      + ' directory on a filesystem that does.');
  }
  throw new Error(`${planted} Delete it, or set TMPDIR to a directory of your own.`);
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

/** Flushes a finished compile to disk and moves it to its cache name. A
 * rename can reach the disk before the data it names, so after a power loss
 * an unflushed binary could read back truncated or as zeros under its final
 * name, and every later run would trust it because it exists; the solvers'
 * checkpoints are flushed before their rename for the same reason. Two builds
 * can still race - waiters that each judged the same lock abandoned - and on
 * Windows renaming onto a binary another process is running fails with
 * EPERM. The name is content-keyed, so a copy that is already there is as
 * good as this one. */
export async function publishBinary(partial, binary, move = rename) {
  const handle = await open(partial, 'r+');
  try {
    await handle.sync();
  } finally {
    await handle.close();
  }
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
 * -pthread for a source that starts threads: one that includes <thread> or
 * <future>, itself or through a header of its own. Where pthread_create is
 * in a library of its own, as in glibc before 2.34, a build without the flag
 * does not link, or links and then refuses to start a thread, and the
 * solvers were built without it here: only the documented recipes passed it.
 * A source that starts no thread keeps the flags it has always had. They key
 * its cached binary and, for the prefix solver, every entry of a
 * generation's journal, which a flag it has no use for would orphan.
 */
export async function nativeThreadFlags(source) {
  const headers = new Map();
  await includedHeaders(source, headers);
  for (const path of [source, ...headers.keys()]) {
    if (THREAD_HEADER.test(await readFile(path, 'utf8'))) return ['-pthread'];
  }
  return [];
}

/**
 * Compiles source once per identity and returns the cached binary with the
 * identity it was built under: the source digest, its headers' digests, the
 * compiler, its version banner and the complete argument list.
 */
export async function buildNative(source) {
  const compiler = findCompiler();
  if (!compiler) throw new Error('A C++20 compiler is required (set CXX, or install g++/clang++).');
  // One list for the cache key and for the compiler, so a binary is always
  // found under the flags it was built with.
  const flags = [...PROOF_FLAGS, ...(await nativeThreadFlags(source)), ...nativeLinkFlags()];
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
  const name = basename(source, '.cpp');
  const extension = process.platform === 'win32' ? '.exe' : '';
  const binary = join(CACHE, `${name}-${key.slice(0, 16)}${extension}`);

  if (!builds.has(binary)) {
    builds.set(binary, (async () => {
      await privateCache();
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

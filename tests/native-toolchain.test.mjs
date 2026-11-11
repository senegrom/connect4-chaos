import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { buildNative, findCompiler, nativeLinkFlags, nativeThreadFlags } from '../scripts/native-build.mjs';

const run = promisify(execFile);
const read = (path) => readFile(new URL(`../${path}`, import.meta.url), 'utf8');
// What every solver is built with, before the flags its source or host adds.
const PROOF_FLAGS = ['-std=c++20', '-O3', '-Wall', '-Wextra', '-Wpedantic'];
const builders = [
  'scripts/perfect-classic.mjs', 'scripts/perfect-classic-policy-generator.mjs',
  'scripts/perfect-chaos-prefix.mjs', 'scripts/perfect-chaos-complete.mjs',
  'tests/perfect-chaos-layered.test.js', 'tests/perfect-chaos-paired.test.js',
  'tests/perfect-chaos-complete-checkpoint.test.js',
  'tests/perfect-chaos-prefix-tooling.test.js', 'tests/native-writers.test.mjs',
];

test('only Windows native builds use the MinGW static-link workaround', () => {
  assert.deepEqual(nativeLinkFlags('win32'), ['-static']);
  for (const platform of ['darwin', 'linux', 'freebsd']) assert.deepEqual(nativeLinkFlags(platform), []);
  assert.deepEqual(nativeLinkFlags(), nativeLinkFlags(process.platform));
  const flags = nativeLinkFlags('win32');
  flags.push('-unexpected');
  assert.deepEqual(nativeLinkFlags('win32'), ['-static']);
  for (const invalid of [null, false, 0, '']) assert.throws(() => nativeLinkFlags(invalid), TypeError);
});

test('every solver build entry point uses the shared host flags', async () => {
  // A build either passes the host flags itself or goes through
  // scripts/native-build.mjs, which appends them to every compile.
  assert.match(await read('scripts/native-build.mjs'), /\.\.\.nativeLinkFlags\(\)/);
  for (const path of builders) {
    const source = await read(path);
    if (!/import \{[^}]*\bbuildNative\b[^}]*\} from '(?:\.|\.\.\/scripts)\/native-build\.mjs'/.test(source)) {
      assert.match(source, /import \{[^}]*\bnativeLinkFlags\b[^}]*\} from /, path);
      assert.match(source, /\.\.\.nativeLinkFlags\(\)/, path);
    }
    assert.doesNotMatch(source, /['"]-static['"]/, path);
  }
});

// buildNative compiled the complete, layered and paired solvers, which start
// threads, without -pthread; only the documented recipes passed it. Where
// pthread_create is in a library of its own, as in glibc before 2.34, such a
// build does not link, or links and then refuses --threads above 1.
test('a solver that starts threads is built with -pthread, and no other is', async (t) => {
  const solver = (name) => fileURLToPath(new URL(`../native/${name}.cpp`, import.meta.url));
  for (const name of ['perfect-chaos-complete', 'perfect-chaos-layered', 'perfect-chaos-paired']) {
    assert.deepEqual(await nativeThreadFlags(solver(name)), ['-pthread'], name);
  }
  // These start none, so their flags stay as they were, and with them their
  // cached binaries and every entry of a prefix generation's journal.
  for (const name of ['perfect-chaos-prefix', 'perfect-classic', 'perfect-classic-policy']) {
    assert.deepEqual(await nativeThreadFlags(solver(name)), [], name);
  }
  // <thread> counts however many of the source's own headers away it is, as
  // it is in checkpoint-io.hpp for the three solvers; a comment does not.
  const directory = await mkdtemp(join(tmpdir(), 'connect4-native-threads-'));
  t.after(() => rm(directory, { recursive: true, force: true, maxRetries: 5 }));
  const source = join(directory, 'solver.cpp');
  await writeFile(source, '#include <vector>\n#include "pool.hpp"\nint main() {}\n');
  await writeFile(join(directory, 'pool.hpp'), '#pragma once\n// #include <thread>\n#include "worker.hpp"\n');
  await writeFile(join(directory, 'worker.hpp'), '#pragma once\n#include <atomic>\n');
  assert.deepEqual(await nativeThreadFlags(source), []);
  await writeFile(join(directory, 'worker.hpp'), '#pragma once\n  #  include <thread>\n');
  assert.deepEqual(await nativeThreadFlags(source), ['-pthread']);
  // The compiler is handed the list the cache key covers, and nothing else.
  assert.match(await read('scripts/native-build.mjs'),
    /runProcess\(compiler, \[\.\.\.flags, source, '-o', partial\]\)/);
});

// The flag has to reach the build itself: its identity, which names the
// cached binary, and the binary, which then starts a thread.
test('a source that starts a thread builds with -pthread and runs', async (t) => {
  if (!findCompiler()) {
    if (process.env.CI) throw new Error('No C++ compiler: the Darwin CI job supplies clang++.');
    t.skip('No local C++ compiler; the required Darwin CI job supplies clang++.');
    return;
  }
  const directory = await mkdtemp(join(tmpdir(), 'connect4-native-thread-build-'));
  t.after(() => rm(directory, { recursive: true, force: true, maxRetries: 5 }));
  const source = join(directory, 'threaded.cpp');
  await writeFile(source, '#include <thread>\nint main() { int ran = 0; std::thread worker([&ran] { ran = 1; });'
    + ' worker.join(); return ran == 1 ? 0 : 1; }\n');
  const threaded = await buildNative(source);
  assert.deepEqual(threaded.flags, [...PROOF_FLAGS, '-pthread', ...nativeLinkFlags()]);
  await run(threaded.binary, [], { timeout: 10_000 });
  const plain = join(directory, 'plain.cpp');
  await writeFile(plain, 'int main() { return 0; }\n');
  assert.deepEqual((await buildNative(plain)).flags, [...PROOF_FLAGS, ...nativeLinkFlags()]);
});

test('host compiler builds and runs an iostream file-writing fixture', async (t) => {
  const compiler = process.env.CXX || (process.platform === 'win32' ? 'g++' : 'c++');
  try { await run(compiler, ['--version'], { timeout: 10_000 }); }
  catch (error) {
    if (error.code !== 'ENOENT' || process.env.CI) throw error;
    t.skip('No local C++ compiler; the required Darwin CI job supplies clang++.');
    return;
  }
  const directory = await mkdtemp(join(tmpdir(), 'connect4-native-flags-'));
  try {
    const source = join(directory, 'io.cpp');
    const binary = join(directory, process.platform === 'win32' ? 'io.exe' : 'io');
    const output = join(directory, 'output.txt');
    await writeFile(source, '#include <fstream>\n#include <iostream>\n#include <cstdint>\n#include "atomic-load.hpp"\nint main(int argc, char** argv) { if (argc != 2) return 2; std::uint64_t word = 17; std::uint8_t byte = 3; const auto& view = word; if (connect4::atomicLoad(view, std::memory_order_acquire) != 17 || connect4::atomicLoad(byte, std::memory_order_relaxed) != 3) return 4; std::ofstream out(argv[1]); out << "native runtime works"; out.close(); if (!out) return 3; std::cout << "ok"; }\n');
    await run(compiler, ['-std=c++20', ...nativeLinkFlags(), '-I', fileURLToPath(new URL('../native/', import.meta.url)), source, '-o', binary], { timeout: 60_000 });
    assert.equal((await run(binary, [output], { timeout: 10_000 })).stdout, 'ok');
    assert.equal(await readFile(output, 'utf8'), 'native runtime works');
  } finally { await rm(directory, { recursive: true, force: true, maxRetries: 5 }); }
});


test('complete native solver compiles and emits a tiny certified policy', async (t) => {
  if (!findCompiler()) {
    if (process.env.CI) throw new Error('No C++ compiler: the Darwin CI job supplies clang++.');
    t.skip('No local C++ compiler; the required Darwin CI job supplies clang++.');
    return;
  }
  // The cached build the checkpoint test and the proof script use, under CXX
  // as they are; this used to compile the solver a third time, at -O2. It is
  // built with the -pthread that the two threads it is run with below need.
  const source = fileURLToPath(new URL('../native/perfect-chaos-complete.cpp', import.meta.url));
  const { binary, flags } = await buildNative(source);
  assert.deepEqual(flags, [...PROOF_FLAGS, '-pthread', ...nativeLinkFlags()]);
  const directory = await mkdtemp(join(tmpdir(), 'connect4-complete-portable-'));
  try {
    const prefix = join(directory, 'policy');
    const result = await run(binary, ['--rows', '2', '--columns', '2', '--connect', '2',
      '--threads', '2', '--max-states', '100000', '--emit-policy', prefix], { timeout: 30_000 });
    const records = result.stdout.trim().split(/\r?\n/).map((line) => JSON.parse(line));
    assert.equal(records[0].rootValue, 1);
    assert.equal(records.length, 3);
    for (const role of [1, 2]) assert.ok((await readFile(`${prefix}-role${role}.bin`)).length > 0);
  } finally { await rm(directory, { recursive: true, force: true, maxRetries: 5 }); }
});

import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import test from 'node:test';
import { nativeLinkFlags } from '../scripts/native-toolchain.mjs';

const run = promisify(execFile);
const read = (path) => readFile(new URL(`../${path}`, import.meta.url), 'utf8');
const builders = [
  'scripts/perfect-classic.mjs', 'scripts/perfect-classic-policy.mjs',
  'scripts/perfect-classic-shards.mjs', 'scripts/perfect-chaos-native.mjs',
  'scripts/perfect-chaos-prefix.mjs', 'scripts/perfect-chaos-complete.mjs',
  'tests/perfect-chaos-classification.test.js', 'tests/perfect-chaos-incremental-repair.test.js',
  'tests/perfect-chaos-policy-partition.test.js', 'tests/perfect-chaos-policy-slice.test.js',
  'tests/perfect-chaos-layered.test.js', 'tests/perfect-chaos-paired.test.js',
  'tests/perfect-chaos-remote-lookup.test.js',
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
  for (const path of builders) {
    const source = await read(path);
    assert.match(source, /import \{ nativeLinkFlags \} from /, path);
    assert.match(source, /\.\.\.nativeLinkFlags\(\)/, path);
    assert.doesNotMatch(source, /['"]-static['"]/, path);
  }
});

test('Darwin builds are required by Pages and linker changes invalidate replay receipts', async () => {
  const ci = await read('.github/workflows/ci.yml');
  const needs = ci.match(/  pages:[\s\S]*?needs: \[([^\]]+)\]/)?.[1];
  assert.ok(needs?.split(',').map((s) => s.trim()).includes('native-portability'));
  assert.match(ci, /uses: \.\/\.github\/workflows\/native-portability\.yml/);
  const native = await read('.github/workflows/native-portability.yml');
  assert.match(native, /runs-on: macos-latest/);
  assert.match(native, /CXX: clang\+\+/);
  assert.match(native, /npm run classic:verify/);
  assert.doesNotMatch(native, /continue-on-error:|\|\| true/);
  const replay = await read('.github/workflows/verify-perfect-classic-policies.yml');
  assert.match(replay, /git ls-tree -r HEAD --[\s\S]*?scripts\/native-toolchain\.mjs/);
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
    await writeFile(source, '#include <fstream>\n#include <iostream>\nint main(int argc, char** argv) { if (argc != 2) return 2; std::ofstream out(argv[1]); out << "native runtime works"; out.close(); if (!out) return 3; std::cout << "ok"; }\n');
    await run(compiler, ['-std=c++20', ...nativeLinkFlags(), source, '-o', binary], { timeout: 60_000 });
    assert.equal((await run(binary, [output], { timeout: 10_000 })).stdout, 'ok');
    assert.equal(await readFile(output, 'utf8'), 'native runtime works');
  } finally { await rm(directory, { recursive: true, force: true, maxRetries: 5 }); }
});

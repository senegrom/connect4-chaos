import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { mkdir, mkdtemp, rm, rmdir, symlink, unlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { isEntryPoint } from '../scripts/entry-point.mjs';

const HELPER = new URL('../scripts/entry-point.mjs', import.meta.url);

function node(script) {
  const result = spawnSync(process.execPath, [script], { encoding: 'utf8', timeout: 60_000 });
  assert.ifError(result.error);
  assert.equal(result.status, 0, result.stderr);
  return result.stdout.trim();
}

test('a script reached through a junction or symlink still recognises itself', async (context) => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-entry-point-'));
  const real = join(directory, 'real');
  const link = join(directory, 'link');
  await mkdir(real);
  // A directory junction needs no privilege on Windows; elsewhere the type is
  // ignored and this is an ordinary symlink.
  await symlink(real, link, 'junction');
  context.after(async () => {
    // Remove the link itself first, so nothing ever recurses through it.
    await (process.platform === 'win32' ? rmdir(link) : unlink(link));
    await rm(directory, { recursive: true, force: true });
  });
  await writeFile(join(real, 'probe.mjs'), [
    "import { resolve } from 'node:path';",
    "import { fileURLToPath } from 'node:url';",
    `import { isEntryPoint } from ${JSON.stringify(HELPER.href)};`,
    'const naive = resolve(process.argv[1]) === fileURLToPath(import.meta.url);',
    'console.log(JSON.stringify({ entry: isEntryPoint(import.meta.url), naive }));',
  ].join('\n'));
  await writeFile(join(real, 'importer.mjs'),
    `import ${JSON.stringify(pathToFileURL(join(real, 'probe.mjs')).href)};\n`);

  assert.deepEqual(JSON.parse(node(join(real, 'probe.mjs'))), { entry: true, naive: true });
  // The comparison the helper replaces skips main() through the link.
  assert.deepEqual(JSON.parse(node(join(link, 'probe.mjs'))), { entry: true, naive: false });
  // Imported by another script, the module is not the entry point.
  assert.deepEqual(JSON.parse(node(join(real, 'importer.mjs'))), { entry: false, naive: false });
});

test('the helper answers false without a usable entry path', () => {
  assert.equal(isEntryPoint(HELPER.href, ['node']), false);
  assert.equal(isEntryPoint(HELPER.href, ['node', '']), false);
  assert.equal(isEntryPoint(HELPER.href, ['node', join(tmpdir(), 'no-such-entry-point.mjs')]), false);
  assert.equal(isEntryPoint(HELPER.href, ['node', fileURLToPath(HELPER)]), true);
});

test('every proof script guards main() with the shared helper', () => {
  for (const script of ['perfect-chaos-complete.mjs', 'verify-perfect-classic-parallel.mjs',
    'perfect-chaos-bridge.mjs', 'perfect-classic-shards.mjs', 'perfect-strategy.mjs',
    'perfect-classic-policy.mjs', 'perfect-chaos-native.mjs', 'perfect-chaos-prefix.mjs']) {
    const source = readFileSync(new URL(`../scripts/${script}`, import.meta.url), 'utf8');
    assert.match(source, /import \{ isEntryPoint \} from '\.\/entry-point\.mjs';/, script);
    assert.match(source, /if \(isEntryPoint\(import\.meta\.url\)\)/, script);
    assert.doesNotMatch(source, /process\.argv\[1\]/, script);
    assert.doesNotMatch(source, /^await main\(\);$/m, script);
  }
});

test('importing the classic policy verifier runs nothing', async (context) => {
  // It used to end in a bare `await main()`, so every importer - the shard
  // tests among them - compiled and ran the C++ generator's self-test.
  const directory = await mkdtemp(join(tmpdir(), 'connect4-entry-import-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const importer = join(directory, 'importer.mjs');
  const verifier = new URL('../scripts/perfect-classic-policy.mjs', import.meta.url);
  await writeFile(importer, [
    `const module = await import(${JSON.stringify(verifier.href)});`,
    "console.log(typeof module.replayPerfectClassicPolicy);",
  ].join('\n'));
  assert.equal(node(importer), 'function');
});

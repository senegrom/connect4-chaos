import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const workflow = (name) => readFileSync(new URL(`../.github/workflows/${name}`, import.meta.url), 'utf8');

function job(source, name) {
  const found = source.match(new RegExp(`^  ${name}:\\n([\\s\\S]*?)(?=^  [\\w-]+:|$(?![\\s\\S]))`, 'm'));
  assert.ok(found, `missing ${name} job`);
  return found[0];
}

test('Darwin builds remain an unconditional same-commit Pages dependency', () => {
  const ci = workflow('ci.yml');
  const needs = job(ci, 'pages').match(/^    needs: \[([^\]]+)\]/m)?.[1].split(',').map((s) => s.trim());
  assert.ok(needs?.includes('native-portability'));
  assert.doesNotMatch(job(ci, 'pages'), /always\(|!cancelled\(|continue-on-error:/);
  const call = job(ci, 'native-portability');
  assert.match(call, /uses: \.\/\.github\/workflows\/native-portability\.yml/);
  assert.doesNotMatch(call, /^    (if|continue-on-error):/m);
  const portability = workflow('native-portability.yml');
  assert.match(portability, /  workflow_call:/);
  const darwin = job(portability, 'darwin');
  assert.match(darwin, /runs-on: macos-/);
  assert.doesNotMatch(darwin, /^\s+(if|continue-on-error):|\|\| true/m);
  for (const command of ['tests/native-toolchain.test.mjs', 'tests/perfect-chaos-layered.test.js',
    'tests/perfect-chaos-paired.test.js', 'npm run classic:verify', 'npm run classic:policy:verify',
    'npm run chaos:prefix:verify', 'node scripts/perfect-chaos-native.mjs']) {
    assert.ok(darwin.includes(command), `Darwin gate must exercise ${command}`);
  }
});



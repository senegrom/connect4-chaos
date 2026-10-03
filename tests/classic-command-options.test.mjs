import assert from 'node:assert/strict';
import test from 'node:test';

import { parseArguments as parseVerifier } from '../scripts/perfect-classic-policy.mjs';
import { parseArguments as parseGenerator } from '../scripts/perfect-classic-policy-generator.mjs';
import { parseArguments as parseParallel } from '../scripts/verify-perfect-classic-parallel.mjs';

test('the release verifier has verify-reference alone, and refuses options it does not read', () => {
  assert.deepEqual(
    parseVerifier(['verify-reference', '--reference', 'a.json', '--verify-table-bits', '14']),
    { command: 'verify-reference', reference: 'a.json', verify_table_bits: '14' },
  );
  // A misspelling used to verify the --reference the npm script had supplied.
  assert.throws(() => parseVerifier(['verify-reference', '--refrence', 'b.json']),
    /verify-reference has no option --refrence\./);
  // Generation moved out of the fingerprinted file; the refusal says where.
  for (const argv of [[], ['verify'], ['generate', '--rows', '4'], ['merge-manifests']]) {
    assert.throws(() => parseVerifier(argv), /perfect-classic-policy-generator\.mjs/, argv.join(' '));
  }
});

test('each classic policy generator command refuses options it does not read', () => {
  assert.deepEqual(parseGenerator([]), { command: 'verify' });
  assert.deepEqual(parseGenerator(['generate', '--rows', '4', '--handoff-remaining', '8']),
    { command: 'generate', rows: '4', handoff_remaining: '8' });
  assert.throws(() => parseGenerator(['verify', '--reference', 'a.json']), /verify has no option --reference\./);
  assert.throws(() => parseGenerator(['generate', '--input', 'a.json']), /generate has no option --input\./);
  assert.throws(() => parseGenerator(['verify-reference']), /Unknown command: verify-reference/);
  assert.deepEqual(
    parseGenerator(['merge-manifests', '--input', 'a.json', '--input', 'b.json', '--output', 'c.json']),
    { command: 'merge-manifests', inputs: ['a.json', 'b.json'], output: 'c.json' },
  );
  assert.throws(() => parseGenerator(['merge-manifests', '--input', '--output', 'c.json']),
    /--input requires a value\./);
});

test('the parallel catalog replay refuses options it does not read', () => {
  assert.deepEqual(
    parseParallel(['--reference', 'data/perfect-classic/manifest.json', '--workers', '4']),
    { reference: 'data/perfect-classic/manifest.json', workers: '4' },
  );
  assert.throws(() => parseParallel(['--worker', '4']), /verify-perfect-classic-parallel has no option --worker\./);
  // Nothing passed --output, and the summary goes to stdout anyway.
  assert.throws(() => parseParallel(['--output', 'summary.json']), /has no option --output\./);
  assert.throws(() => parseParallel(['verify-reference']), /Unexpected argument: verify-reference/);
});

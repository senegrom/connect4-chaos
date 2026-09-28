import assert from 'node:assert/strict';
import test from 'node:test';

import { parseArguments as parsePolicy } from '../scripts/perfect-classic-policy.mjs';
import { parseArguments as parseParallel } from '../scripts/verify-perfect-classic-parallel.mjs';

test('each classic policy command refuses options it does not read', () => {
  assert.deepEqual(parsePolicy([]), { command: 'verify' });
  assert.deepEqual(
    parsePolicy(['verify-reference', '--reference', 'a.json', '--verify-table-bits', '14']),
    { command: 'verify-reference', reference: 'a.json', verify_table_bits: '14' },
  );
  // A misspelling used to verify the --reference the npm script had supplied.
  assert.throws(() => parsePolicy(['verify-reference', '--refrence', 'b.json']),
    /verify-reference has no option --refrence\./);
  assert.throws(() => parsePolicy(['verify', '--reference', 'a.json']), /verify has no option --reference\./);
  assert.throws(() => parsePolicy(['generate', '--input', 'a.json']), /generate has no option --input\./);
  assert.throws(() => parsePolicy(['verfy']), /Unknown command: verfy/);
  assert.deepEqual(
    parsePolicy(['merge-manifests', '--input', 'a.json', '--input', 'b.json', '--output', 'c.json']),
    { command: 'merge-manifests', inputs: ['a.json', 'b.json'], output: 'c.json' },
  );
  assert.throws(() => parsePolicy(['merge-manifests', '--input', '--output', 'c.json']),
    /--input requires a manifest path\./);
});

test('the parallel catalog replay refuses options it does not read', () => {
  assert.deepEqual(
    parseParallel(['--reference', 'data/perfect-classic/manifest.json', '--workers', '4']),
    { reference: 'data/perfect-classic/manifest.json', workers: '4' },
  );
  assert.throws(() => parseParallel(['--worker', '4']), /Unknown option --worker\./);
  assert.throws(() => parseParallel(['verify-reference']), /Unexpected argument: verify-reference/);
});

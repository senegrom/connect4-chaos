import assert from 'node:assert/strict';
import test from 'node:test';

import { integerOption } from '../scripts/cli-options.mjs';
import { parseArguments as parseVerifier } from '../scripts/perfect-classic-policy.mjs';
import { parseArguments as parseGenerator, roleSelection } from '../scripts/perfect-classic-policy-generator.mjs';
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
  assert.deepEqual(parseGenerator(['generate', '--rows', '4', '--columns', '5', '--handoff-remaining', '8']),
    { command: 'generate', rows: '4', columns: '5', handoff_remaining: '8' });
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

test('classic policy generation names its board', () => {
  // With no flags it used to generate standard 6x7 - the longest run there
  // is, for a catalog entry the release gate refuses.
  for (const argv of [['generate'], ['generate', '--rows', '6'], ['generate', '--columns', '7', '--connect', '4']]) {
    assert.throws(() => parseGenerator(argv), /generate requires --rows and --columns\./, argv.join(' '));
  }
  // Asked for by name, 6x7 still generates.
  assert.deepEqual(parseGenerator(['generate', '--rows', '6', '--columns', '7']),
    { command: 'generate', rows: '6', columns: '7' });
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

test('an integer option is a plain decimal integer, not whatever starts with digits', () => {
  assert.equal(integerOption('14', 22, 'verify-table-bits', 8, 25), 14);
  assert.equal(integerOption(undefined, 22, 'verify-table-bits', 8, 25), 22);
  assert.equal(integerOption('-1', 0, 'expected-root', -1, 1), -1);
  // The generator's own small references pass numbers.
  assert.equal(integerOption(10_000_000, 0, 'maximum-nodes'), 10_000_000);
  // parseInt read the leading digits: `pack --max-ply 1O` packed a one-ply
  // book over the committed one and exited 0.
  for (const value of ['1O', '6e1', '8x', '1.9', '0x10', '+5', ' 5', '', true, 2.5]) {
    assert.throws(() => integerOption(value, 0, '--max-ply'), /--max-ply must be an integer of at least 0\./,
      JSON.stringify(value));
  }
  assert.throws(() => integerOption('16.5', 22, 'verify-table-bits', 8, 25),
    /verify-table-bits must be an integer from 8 through 25\./);
});

test('the generator takes --role 1, 2 or both, written exactly', () => {
  assert.deepEqual(roleSelection(undefined), [1, 2]);
  assert.deepEqual(roleSelection('both'), [1, 2]);
  assert.deepEqual(roleSelection('1'), [1]);
  assert.deepEqual(roleSelection('2'), [2]);
  // `--role 2x` generated role 2, and a bare --role is true.
  for (const value of ['2x', '1.0', '3', 'first', true]) {
    assert.throws(() => roleSelection(value), /role must be 1, 2, or both\./, String(value));
  }
});

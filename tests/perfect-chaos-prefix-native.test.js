import assert from 'node:assert/strict';
import { mkdtemp, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { buildNative, findCompiler, runProcess } from '../scripts/native-build.mjs';

const SOURCE = fileURLToPath(new URL('../native/perfect-chaos-prefix.cpp', import.meta.url));

// std::stoul skipped a leading space, took a '+' and negated a '-': where
// unsigned long is 32 bits, as on Windows, `--maximum-states -1` was
// 4294967295, passed the range check and set no limit. A value with no digits
// said only "stoul".
test('the prefix solver refuses a number that is not all digits, and names its option', async (context) => {
  if (!findCompiler()) {
    context.skip('no C++ compiler available');
    return;
  }
  const { binary } = await buildNative(SOURCE);
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-prefix-native-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const file = (name) => join(directory, name);
  const run = (command, options) => runProcess(binary, [command, ...Object.entries(options).flat()]);

  // A generation to two pieces and its extension to four, a fraction of a
  // second each. As given both succeed, so each refusal below is of the one
  // value changed.
  const commands = {
    generate: { '--role': 'red', '--frontier-pieces': '2', '--maximum-states': '100000',
      '--policy': file('0-2.policy.bin'), '--frontier': file('0-2.frontier.bin') },
    extend: { '--input-frontier': file('0-2.frontier.bin'), '--frontier-pieces': '4', '--maximum-states': '100000',
      '--policy': file('2-4.policy.bin'), '--frontier': file('2-4.frontier.bin') },
  };
  for (const [command, options] of Object.entries(commands)) {
    const accepted = await run(command, options);
    assert.equal(accepted.code, 0, `${command}: ${accepted.stderr}`);
  }
  const written = (await readdir(directory)).sort();
  assert.deepEqual(written, ['0-2.frontier.bin', '0-2.policy.bin', '2-4.frontier.bin', '2-4.policy.bin']);

  const refused = { '--policy': file('refused.policy.bin'), '--frontier': file('refused.frontier.bin') };
  for (const [command, options] of Object.entries(commands)) {
    for (const option of ['--frontier-pieces', '--maximum-states']) {
      const digits = options[option];
      // A leading space, a '+', a negative number, trailing characters, no
      // value at all, and 2^32, one more than a 32-bit count can hold.
      for (const value of [` ${digits}`, `+${digits}`, '-1', `${digits}x`, '', '4294967296']) {
        const result = await run(command, { ...options, ...refused, [option]: value });
        assert.equal(result.code, 1, `${command} ${option} '${value}'`);
        assert.equal(result.stderr.trim(), `${option.slice(2)} is invalid.`, `${command} ${option} '${value}'`);
      }
    }
  }
  assert.deepEqual((await readdir(directory)).sort(), written, 'a refused run wrote nothing');
});

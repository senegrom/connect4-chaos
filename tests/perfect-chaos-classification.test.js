import assert from 'node:assert/strict';
import { chmod, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawn } from 'node:child_process';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { buildNative, findCompiler } from '../scripts/native-build.mjs';
import { pythonCommand } from '../scripts/python-command.mjs';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const NATIVE_SOURCE = join(ROOT, 'native', 'perfect-chaos-prefix.cpp');
const CLASSIFIER = join(ROOT, 'scripts', 'perfect-chaos-classify.py');
const MERGER = join(ROOT, 'scripts', 'perfect-chaos-merge-classification.py');

function run(command, args, options = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, {
      cwd: ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
      ...options,
    });
    const stdout = [];
    const stderr = [];
    child.stdout.on('data', (chunk) => stdout.push(chunk));
    child.stderr.on('data', (chunk) => stderr.push(chunk));
    child.once('error', reject);
    child.once('close', (code, signal) => {
      const output = Buffer.concat(stdout).toString('utf8');
      const errors = Buffer.concat(stderr).toString('utf8');
      if (code === 0) {
        resolvePromise(output);
        return;
      }
      reject(new Error(
        `${command} exited with ${code ?? signal}.\n${errors || output}`,
      ));
    });
  });
}

async function python() {
  try {
    return pythonCommand();
  } catch {
    return null;
  }
}

test('distributed Perfect Chaos classification matches the direct native segment', async (context) => {
  const pythonCommand = await python();
  if (!findCompiler() || !pythonCommand) {
    context.skip('A C++20 compiler and Python are required for proof-tool integration.');
    return;
  }

  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-classification-'));
  try {
    const { binary: solver } = await buildNative(NATIVE_SOURCE, { name: 'perfect-chaos-prefix' });

    const rootPolicy = join(directory, '0-4.policy.bin');
    const rootFrontier = join(directory, '0-4.frontier.bin');
    await run(solver, [
      'generate', '--role', 'red', '--frontier-pieces', '4',
      '--maximum-states', '1000000', '--policy', rootPolicy, '--frontier', rootFrontier,
    ]);

    const directPolicy = join(directory, 'direct.policy.bin');
    const directFrontier = join(directory, 'direct.frontier.bin');
    await run(solver, [
      'extend', '--input-frontier', rootFrontier, '--frontier-pieces', '6',
      '--maximum-states', '2000000', '--policy', directPolicy, '--frontier', directFrontier,
      '--rejected', join(directory, 'direct.rejected.bin'),
    ]);

    const shardDirectory = join(directory, 'shards');
    await mkdir(shardDirectory, { recursive: true });
    for (let shard = 0; shard < 2; shard += 1) {
      const output = await run(pythonCommand.command, [...pythonCommand.args, 
        CLASSIFIER,
        '--solver', solver,
        '--input', rootFrontier,
        '--role', 'red',
        '--target-pieces', '6',
        '--shard-index', String(shard),
        '--shard-count', '2',
        '--maximum-states', '10000',
        '--rejected', join(shardDirectory, `rejected-${shard}.bin`),
        '--policy', join(shardDirectory, `policy-${shard}.bin`),
        '--frontier', join(shardDirectory, `frontier-${shard}.bin`),
        '--summary', join(shardDirectory, `summary-${shard}.json`),
      ]);
      const summary = JSON.parse(output);
      assert.equal(summary.classificationComplete, true);
      assert.equal(summary.rejectedRoots, 0);
      assert.ok(summary.splitEvents > 0);
    }

    const mergedRejected = join(directory, 'merged.rejected.bin');
    const mergedPolicy = join(directory, 'merged.policy.bin');
    const mergedFrontier = join(directory, 'merged.frontier.bin');
    const mergedOutput = await run(pythonCommand.command, [...pythonCommand.args, 
      MERGER,
      '--directory', shardDirectory,
      '--input', rootFrontier,
      '--role', 'red',
      '--target-pieces', '6',
      '--shard-count', '2',
      '--rejected', mergedRejected,
      '--policy', mergedPolicy,
      '--frontier', mergedFrontier,
      '--summary', join(directory, 'merged-summary.json'),
    ]);
    const merged = JSON.parse(mergedOutput);
    assert.equal(merged.classificationComplete, true);
    assert.equal(merged.inputRoots, 59);
    assert.equal(merged.safeInputRoots, 59);
    assert.equal(merged.rejectedRoots, 0);
    assert.equal(merged.policyConflicts, 0);
    assert.deepEqual(await readFile(mergedPolicy), await readFile(directPolicy));
    assert.deepEqual(await readFile(mergedFrontier), await readFile(directFrontier));
    assert.equal((await readFile(mergedRejected)).length, 16);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});


test('classification merger fails closed on actions that conflict across shards', async (context) => {
  const pythonCommand = await python();
  if (!pythonCommand) {
    context.skip('Python is required for proof-table validation.');
    return;
  }

  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-merge-conflict-'));
  try {
    const shardDirectory = join(directory, 'shards');
    await mkdir(shardDirectory, { recursive: true });
    const source = join(directory, 'source.frontier.bin');
    const setup = `
import json
import struct
from pathlib import Path
from perfect_chaos_tables import (
    FRONTIER_MAGIC, FRONTIER_RECORD_SIZE, POLICY_MAGIC, POLICY_RECORD_SIZE,
    file_summary, write_table,
)

root = Path(${JSON.stringify(directory)})
shards = root / 'shards'

def frontier(rows, columns):
    value = bytearray(FRONTIER_RECORD_SIZE)
    struct.pack_into('<QQ', value, 0, 0, 0)
    value[16] = rows
    value[17] = columns
    value[18] = 1
    return bytes(value)

def policy(action):
    value = bytearray(POLICY_RECORD_SIZE)
    struct.pack_into('<QQ', value, 0, 0, 0)
    value[16] = 6
    value[17] = 7
    value[18] = action
    value[19] = 0
    return bytes(value)

write_table(Path(${JSON.stringify(source)}), FRONTIER_MAGIC, 1, 0, FRONTIER_RECORD_SIZE, [
    frontier(6, 7), frontier(7, 6),
])
for shard, action in enumerate((1, 2)):
    rejected = shards / f'rejected-{shard}.bin'
    policy_path = shards / f'policy-{shard}.bin'
    frontier_path = shards / f'frontier-{shard}.bin'
    write_table(rejected, FRONTIER_MAGIC, 1, 0, FRONTIER_RECORD_SIZE, [])
    write_table(policy_path, POLICY_MAGIC, 1, 2, POLICY_RECORD_SIZE, [policy(action)])
    write_table(frontier_path, FRONTIER_MAGIC, 1, 2, FRONTIER_RECORD_SIZE, [])
    summary = {
        'format': 'connect4-chaos-frontier-classification-shard-v1',
        'role': 'red',
        'fromPieces': 0,
        'targetPieces': 2,
        'shardIndex': shard,
        'shardCount': 2,
        'inputRoots': 1,
        'rejectedRoots': 0,
        'safeInputRoots': 1,
        'classificationComplete': True,
        'safePolicyEntries': 1,
        'safeFrontierStates': 0,
        'policyConflicts': 0,
        'attempts': 1,
        'splitEvents': 0,
        'maximumSplitDepth': 0,
        'safeLeaves': 1,
        'rejectedLeaves': 0,
        'maximumStatesPerLeaf': 10000,
        'targetRejectSha256': None,
        'artifacts': {
            'rejected': file_summary(rejected),
            'policy': file_summary(policy_path),
            'frontier': file_summary(frontier_path),
        },
    }
    (shards / f'summary-{shard}.json').write_text(json.dumps(summary) + '\\n')
`;
    await run(pythonCommand.command, [...pythonCommand.args, '-c', setup], {
      env: { ...process.env, PYTHONPATH: join(ROOT, 'scripts') },
    });

    await assert.rejects(
      run(pythonCommand.command, [...pythonCommand.args, 
        MERGER,
        '--directory', shardDirectory,
        '--input', source,
        '--role', 'red',
        '--target-pieces', '2',
        '--shard-count', '2',
        '--rejected', join(directory, 'merged.rejected.bin'),
        '--policy', join(directory, 'merged.policy.bin'),
        '--frontier', join(directory, 'merged.frontier.bin'),
        '--summary', join(directory, 'merged-summary.json'),
      ]),
      /Conflicting Perfect Chaos policy actions across classification shards: 1/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});


test('adaptive classifier fails closed on actions that conflict across leaves', async (context) => {
  const pythonCommand = await python();
  if (!pythonCommand) {
    context.skip('Python is required for proof-table validation.');
    return;
  }

  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-classifier-conflict-'));
  try {
    const source = join(directory, 'source.frontier.bin');
    const solver = join(directory, 'conflicting-solver.py');
    const setup = `
import struct
from pathlib import Path
from perfect_chaos_tables import FRONTIER_MAGIC, FRONTIER_RECORD_SIZE, write_table

def frontier(rows, columns):
    value = bytearray(FRONTIER_RECORD_SIZE)
    struct.pack_into('<QQ', value, 0, 0, 0)
    value[16] = rows
    value[17] = columns
    value[18] = 1
    return bytes(value)

write_table(Path(${JSON.stringify(source)}), FRONTIER_MAGIC, 1, 0, FRONTIER_RECORD_SIZE, [
    frontier(6, 7), frontier(7, 6),
])
`;
    await run(pythonCommand.command, [...pythonCommand.args, '-c', setup], {
      env: { ...process.env, PYTHONPATH: join(ROOT, 'scripts') },
    });
    await writeFile(solver, `#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from perfect_chaos_tables import (
    FRONTIER_MAGIC, FRONTIER_RECORD_SIZE, POLICY_MAGIC, POLICY_RECORD_SIZE,
    read_table, write_table,
)

arguments = sys.argv[2:]
options = {arguments[index]: arguments[index + 1] for index in range(0, len(arguments), 2)}
input_path = Path(options['--input-frontier'])
source = read_table(input_path, FRONTIER_MAGIC, FRONTIER_RECORD_SIZE)
if len(source.records) > 1:
    print('Prefix graph exceeded its state limit.', file=sys.stderr)
    raise SystemExit(1)

action = 1 if '-0' in input_path.name else 2
record = bytearray(POLICY_RECORD_SIZE)
record[16] = 6
record[17] = 7
record[18] = action
record[19] = 0
write_table(
    Path(options['--policy']), POLICY_MAGIC, source.role,
    int(options['--frontier-pieces']), POLICY_RECORD_SIZE, [bytes(record)],
)
write_table(
    Path(options['--frontier']), FRONTIER_MAGIC, source.role,
    int(options['--frontier-pieces']), FRONTIER_RECORD_SIZE, [],
)
print(json.dumps({'ok': True}))
`);
    await chmod(solver, 0o755);

    await assert.rejects(
      run(pythonCommand.command, [...pythonCommand.args, 
        CLASSIFIER,
        '--solver', solver,
        '--input', source,
        '--role', 'red',
        '--target-pieces', '2',
        '--shard-index', '0',
        '--shard-count', '1',
        '--maximum-states', '10000',
        '--rejected', join(directory, 'rejected.bin'),
        '--policy', join(directory, 'policy.bin'),
        '--frontier', join(directory, 'frontier.bin'),
        '--summary', join(directory, 'summary.json'),
      ], {
        env: { ...process.env, PYTHONPATH: join(ROOT, 'scripts') },
      }),
      /Conflicting Perfect Chaos policy actions across classifier leaves: 1/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

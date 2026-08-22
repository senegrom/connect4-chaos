import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
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
      reject(new Error(`${command} exited with ${code ?? signal}.\n${errors || output}`));
    });
  });
}

test('classification merger rejects a preserved shard that reports internal conflicts', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-summary-conflict-'));
  try {
    const shardDirectory = join(directory, 'shards');
    await mkdir(shardDirectory);
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

def policy():
    value = bytearray(POLICY_RECORD_SIZE)
    struct.pack_into('<QQ', value, 0, 0, 0)
    value[16] = 6
    value[17] = 7
    value[18] = 1
    value[19] = 0
    return bytes(value)

write_table(Path(${JSON.stringify(source)}), FRONTIER_MAGIC, 1, 0, FRONTIER_RECORD_SIZE, [
    frontier(6, 7), frontier(7, 6),
])
for shard in range(2):
    rejected = shards / f'rejected-{shard}.bin'
    policy_path = shards / f'policy-{shard}.bin'
    frontier_path = shards / f'frontier-{shard}.bin'
    write_table(rejected, FRONTIER_MAGIC, 1, 0, FRONTIER_RECORD_SIZE, [])
    write_table(policy_path, POLICY_MAGIC, 1, 2, POLICY_RECORD_SIZE, [policy()])
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
        'policyConflicts': 1 if shard == 0 else 0,
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
    await run('python3', ['-c', setup], {
      env: { ...process.env, PYTHONPATH: join(ROOT, 'scripts') },
    });

    await assert.rejects(
      run('python3', [
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
      /Shard 0 contains conflicting policy actions: 1/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

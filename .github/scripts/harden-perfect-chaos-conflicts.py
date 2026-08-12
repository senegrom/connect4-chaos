from pathlib import Path

root = Path('.')

# Harden the JavaScript prefix writer/merger and add a permanent self-test.
path = root / 'scripts/perfect-chaos-prefix.mjs'
text = path.read_text()
old = '''function compareAction(first, second) {
  if (first.type !== second.type) return first.type - second.type;
  return first.column - second.column;
}

function sameAction(first, second) {
'''
new = '''function sameAction(first, second) {
'''
if text.count(old) != 1:
    raise SystemExit('compareAction anchor mismatch')
text = text.replace(old, new)
old = '''    const key = stateKey(record.state);
    const existing = selected.get(key);
    if (!existing || compareAction(record.action, existing.action) < 0) {
      selected.set(key, record);
    }
'''
new = '''    const key = stateKey(record.state);
    const existing = selected.get(key);
    if (existing && !sameAction(existing.action, record.action)) {
      throw new Error(`Conflicting Perfect Chaos policy actions for ${key}.`);
    }
    if (!existing) selected.set(key, record);
'''
if text.count(old) != 1:
    raise SystemExit('encodePolicy anchor mismatch')
text = text.replace(old, new)
old = '''  const records = new Map();
  let conflicts = 0;
  for (const path of paths) {
'''
new = '''  const records = new Map();
  for (const path of paths) {
'''
if text.count(old) != 1:
    raise SystemExit('mergePolicies declaration anchor mismatch')
text = text.replace(old, new)
old = '''      const key = stateKey(record.state);
      const existing = records.get(key);
      if (existing && !sameAction(existing.action, record.action)) conflicts += 1;
      if (!existing || compareAction(record.action, existing.action) < 0) {
        records.set(key, record);
      }
'''
new = '''      const key = stateKey(record.state);
      const existing = records.get(key);
      if (existing && !sameAction(existing.action, record.action)) {
        throw new Error(`Conflicting Perfect Chaos policy actions for ${key}.`);
      }
      if (!existing) records.set(key, record);
'''
if text.count(old) != 1:
    raise SystemExit('mergePolicies action anchor mismatch')
text = text.replace(old, new)
old = '''  await writeFile(target, encodePolicy(role, boundary, [...records.values()]));
  return { count: (await readPolicy(target)).count, conflicts };
}

async function mergeFrontiers'''
new = '''  await writeFile(target, encodePolicy(role, boundary, [...records.values()]));
  return { count: (await readPolicy(target)).count, conflicts: 0 };
}

async function mergeFrontiers'''
if text.count(old) != 1:
    raise SystemExit('mergePolicies return anchor mismatch')
text = text.replace(old, new)
anchor = '''async function verifyShardedSmall(binary, temporary) {
'''
insert = '''async function verifyPolicyConflicts(temporary) {
  const directory = join(temporary, 'policy-conflict');
  await mkdir(directory, { recursive: true });
  const state = {
    mover: 0n,
    opponent: 0n,
    rows: 6,
    columns: 7,
    aiTurn: true,
  };
  const flip = { state, action: { type: ACTION_FLIP, column: 0 } };
  const rotate = { state, action: { type: ACTION_CW, column: 0 } };

  let encodingRejected = false;
  try {
    encodePolicy(ROLE_CODES.red, 2, [flip, rotate]);
  } catch (error) {
    if (!/Conflicting Perfect Chaos policy actions/.test(String(error))) throw error;
    encodingRejected = true;
  }
  if (!encodingRejected) {
    throw new Error('The Perfect Chaos policy encoder silently selected a conflicting action.');
  }

  const first = join(directory, 'first.policy.bin');
  const second = join(directory, 'second.policy.bin');
  const merged = join(directory, 'merged.policy.bin');
  await writeFile(first, encodePolicy(ROLE_CODES.red, 2, [flip]));
  await writeFile(second, encodePolicy(ROLE_CODES.red, 2, [rotate]));
  let mergeRejected = false;
  try {
    await mergePolicies(merged, [first, second]);
  } catch (error) {
    if (!/Conflicting Perfect Chaos policy actions/.test(String(error))) throw error;
    mergeRejected = true;
  }
  if (!mergeRejected) {
    throw new Error('The Perfect Chaos policy merger silently selected a conflicting action.');
  }
  if (await exists(merged)) {
    throw new Error('The Perfect Chaos policy merger wrote output after a conflict.');
  }
  return { encodingRejected, mergeRejected };
}

async function verifyShardedSmall(binary, temporary) {
'''
if text.count(anchor) != 1:
    raise SystemExit('verifyShardedSmall anchor mismatch')
text = text.replace(anchor, insert)
old = '''  const sharding = await verifyShardedSmall(binary, temporary);
  const generated = join(temporary, 'small-reference');
'''
new = '''  const sharding = await verifyShardedSmall(binary, temporary);
  const policyConflicts = await verifyPolicyConflicts(temporary);
  const generated = join(temporary, 'small-reference');
'''
if text.count(old) != 1:
    raise SystemExit('verifySmall setup anchor mismatch')
text = text.replace(old, new)
old = '''  return { native: native.records, sharding, replay: manifest.roles };
}
'''
new = '''  return {
    native: native.records,
    sharding,
    policyConflicts,
    replay: manifest.roles,
  };
}
'''
if text.count(old) != 1:
    raise SystemExit('verifySmall return anchor mismatch')
text = text.replace(old, new)
path.write_text(text)

# Fail closed inside one classifier shard when adaptive leaves disagree.
path = root / 'scripts/perfect-chaos-classify.py'
text = path.read_text()
old = '''        classify(selected, f"{args.shard_index:03d}", 0)

    write_table(
'''
new = '''        classify(selected, f"{args.shard_index:03d}", 0)

    if policy_conflicts:
        raise RuntimeError(
            f"Conflicting Perfect Chaos policy actions across classifier leaves: "
            f"{policy_conflicts}."
        )

    write_table(
'''
if text.count(old) != 1:
    raise SystemExit('classifier output anchor mismatch')
path.write_text(text.replace(old, new))

# Fail closed before writing any merged classification output.
path = root / 'scripts/perfect-chaos-merge-classification.py'
text = path.read_text()
old = '''    merged_frontier, _ = merge_records(
        frontier_records,
        FRONTIER_RECORD_SIZE,
    )
    write_table(
'''
new = '''    merged_frontier, _ = merge_records(
        frontier_records,
        FRONTIER_RECORD_SIZE,
    )
    if policy_conflicts:
        raise RuntimeError(
            f"Conflicting Perfect Chaos policy actions across classification shards: "
            f"{policy_conflicts}."
        )
    write_table(
'''
if text.count(old) != 1:
    raise SystemExit('merger output anchor mismatch')
path.write_text(text.replace(old, new))

# Extend integration tests to prove both conflict boundaries fail closed.
path = root / 'tests/perfect-chaos-classification.test.js'
text = path.read_text()
old = "import { access, mkdtemp, readFile, rm } from 'node:fs/promises';"
new = "import { access, chmod, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';"
if text.count(old) != 1:
    raise SystemExit('test import anchor mismatch')
text = text.replace(old, new)
append = r'''


test('classification merger fails closed on actions that conflict across shards', async (context) => {
  const pythonCommand = await python();
  if (!pythonCommand) {
    context.skip('Python is required for proof-table validation.');
    return;
  }

  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-merge-conflict-'));
  try {
    const shardDirectory = join(directory, 'shards');
    await run('mkdir', ['-p', shardDirectory]);
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
    await run(pythonCommand, ['-c', setup], {
      env: { ...process.env, PYTHONPATH: join(ROOT, 'scripts') },
    });

    await assert.rejects(
      run(pythonCommand, [
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
    await run(pythonCommand, ['-c', setup], {
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
      run(pythonCommand, [
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
'''
if 'classification merger fails closed on actions that conflict across shards' in text:
    raise SystemExit('tests already appended')
path.write_text(text + append)

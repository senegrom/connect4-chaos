import test from 'node:test';
import assert from 'node:assert/strict';
import {
  mkdtempSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const REPOSITORY_ROOT = fileURLToPath(new URL('..', import.meta.url));
const SCRIPT = join(REPOSITORY_ROOT, 'scripts', 'perfect-classic-download-fragments.py');
const RUN_ID = 123456;
const RUN_SHA = 'a'.repeat(40);
const PREFIX = 'c4cert-7x7-r1-prefix1-';

function sha256(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

function makeZip(path, entries) {
  const code = String.raw`
import json, os, stat, zipfile
from pathlib import Path
path = Path(os.environ['ZIP_PATH'])
entries = json.loads(os.environ['ZIP_ENTRIES'])
with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for entry in entries:
        info = zipfile.ZipInfo(entry['name'])
        kind = entry.get('kind', 'file')
        if kind == 'symlink':
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
        elif kind == 'directory':
            if not info.filename.endswith('/'):
                info.filename += '/'
            info.create_system = 3
            info.external_attr = (stat.S_IFDIR | 0o755) << 16
        else:
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
        archive.writestr(info, entry.get('content', '').encode())
`;
  const result = spawnSync('python3', ['-c', code], {
    env: {
      ...process.env,
      ZIP_PATH: path,
      ZIP_ENTRIES: JSON.stringify(entries),
    },
    encoding: 'utf8',
  });
  assert.equal(result.status, 0, result.stderr);
}

function artifact(root, {
  id,
  name = `${PREFIX}${String(id).padStart(3, '0')}`,
  entries = [
    { name: 'fragment.json', content: JSON.stringify({ id }) },
    { name: 'policy.bin', content: `policy-${id}` },
  ],
  runId = RUN_ID,
  runSha = RUN_SHA,
  expired = false,
  digestOverride,
  archiveName = `${id}.zip`,
} = {}) {
  const archive = join(root, archiveName);
  makeZip(archive, entries);
  return {
    id,
    name,
    size_in_bytes: statSync(archive).size,
    digest: `sha256:${digestOverride ?? sha256(archive)}`,
    expired,
    created_at: '2026-08-18T00:00:00Z',
    updated_at: '2026-08-18T00:00:01Z',
    workflow_run: { id: runId, head_sha: runSha },
    archive_path: basename(archive),
  };
}

function execute(root, artifacts, ...extraArguments) {
  const index = join(root, 'artifacts.json');
  const output = join(root, 'downloaded');
  const metadata = join(root, 'metadata.json');
  writeFileSync(index, `${JSON.stringify(artifacts, null, 2)}\n`);
  const result = spawnSync('python3', [
    SCRIPT,
    '--repository', 'senegrom/connect4-chaos',
    '--run-id', String(RUN_ID),
    '--run-sha', RUN_SHA,
    '--artifact-prefix', PREFIX,
    '--output', output,
    '--metadata', metadata,
    '--offline-index', index,
    ...extraArguments,
  ], {
    cwd: REPOSITORY_ROOT,
    encoding: 'utf8',
  });
  return { result, output, metadata };
}

function parsed(execution) {
  assert.equal(execution.result.status, 0, execution.result.stderr);
  return JSON.parse(readFileSync(execution.metadata, 'utf8'));
}

function expectFailure(execution, pattern) {
  assert.notEqual(execution.result.status, 0, 'downloader unexpectedly succeeded');
  assert.match(`${execution.result.stdout}\n${execution.result.stderr}`, pattern);
}

function withRoot(callback) {
  const root = mkdtempSync(join(tmpdir(), 'c4-classic-fragments-'));
  try {
    callback(root);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

test('one exact artifact is digest-checked and extracted under its name', () => {
  withRoot((root) => {
    const item = artifact(root, { id: 1 });
    const execution = execute(root, [item]);
    const manifest = parsed(execution);
    assert.equal(manifest.format, 'connect4-perfect-classic-fragment-download-v1');
    assert.equal(manifest.artifactNames, 1);
    assert.equal(manifest.archiveCandidates, 1);
    assert.deepEqual(readdirSync(execution.output), [item.name]);
    assert.deepEqual(
      JSON.parse(readFileSync(join(execution.output, item.name, 'fragment.json'))),
      { id: 1 },
    );
    assert.equal(manifest.artifacts[0].selectedArtifactId, 1);
    assert.equal(manifest.artifacts[0].equivalentArchives.length, 1);
  });
});

test('multiple artifact names are extracted in deterministic lexical order', () => {
  withRoot((root) => {
    const later = artifact(root, { id: 20, name: `${PREFIX}020` });
    const earlier = artifact(root, { id: 10, name: `${PREFIX}010` });
    const manifest = parsed(execute(root, [later, earlier]));
    assert.deepEqual(manifest.artifacts.map((item) => item.name), [earlier.name, later.name]);
  });
});

test('same-name rerun artifacts are accepted only when their extracted bytes agree', () => {
  withRoot((root) => {
    const name = `${PREFIX}007`;
    const entries = [
      { name: 'fragment.json', content: '{"fragment":7}' },
      { name: 'policy.bin', content: 'same-policy' },
    ];
    const first = artifact(root, { id: 7, name, entries, archiveName: 'first.zip' });
    const second = artifact(root, { id: 8, name, entries, archiveName: 'second.zip' });
    const manifest = parsed(execute(root, [second, first]));
    assert.equal(manifest.artifactNames, 1);
    assert.equal(manifest.archiveCandidates, 2);
    assert.equal(manifest.artifacts[0].selectedArtifactId, 8);
    assert.deepEqual(
      manifest.artifacts[0].equivalentArchives.map((item) => item.id),
      [7, 8],
    );
  });
});

test('same-name artifacts with different extracted proof bytes fail closed', () => {
  withRoot((root) => {
    const name = `${PREFIX}009`;
    const first = artifact(root, {
      id: 9,
      name,
      archiveName: 'first.zip',
      entries: [{ name: 'fragment.json', content: '{"value":1}' }],
    });
    const second = artifact(root, {
      id: 10,
      name,
      archiveName: 'second.zip',
      entries: [{ name: 'fragment.json', content: '{"value":2}' }],
    });
    expectFailure(execute(root, [first, second]), /conflicting contents/i);
  });
});

test('archive digest, size, run and commit mismatches fail closed', () => {
  withRoot((root) => {
    expectFailure(
      execute(root, [artifact(root, { id: 11, digestOverride: '0'.repeat(64) })]),
      /archive digest mismatch/i,
    );
  });
  withRoot((root) => {
    const item = artifact(root, { id: 12 });
    item.size_in_bytes += 1;
    expectFailure(execute(root, [item]), /archive size mismatch/i);
  });
  withRoot((root) => {
    expectFailure(
      execute(root, [artifact(root, { id: 13, runId: RUN_ID + 1 })]),
      /belongs to run/i,
    );
  });
  withRoot((root) => {
    expectFailure(
      execute(root, [artifact(root, { id: 14, runSha: 'b'.repeat(40) })]),
      /belongs to commit/i,
    );
  });
});

test('expired required artifacts and duplicate artifact ids fail closed', () => {
  withRoot((root) => {
    expectFailure(
      execute(root, [artifact(root, { id: 15, expired: true })]),
      /is expired/i,
    );
  });
  withRoot((root) => {
    const first = artifact(root, { id: 16, archiveName: 'one.zip' });
    const duplicate = { ...first, archive_path: 'one.zip' };
    expectFailure(execute(root, [first, duplicate]), /appears more than once/i);
  });
});

test('unrelated artifacts are ignored but an empty matching set fails closed', () => {
  withRoot((root) => {
    const unrelated = artifact(root, { id: 17, name: 'ordinary-build-output' });
    expectFailure(execute(root, [unrelated]), /No unexpired artifacts beginning/i);
  });
});

test('path traversal, absolute paths, backslashes and symbolic links are rejected', () => {
  const cases = [
    [{ name: '../fragment.json', content: '{}' }, /Unsafe ZIP entry path/i],
    [{ name: '/fragment.json', content: '{}' }, /Unsafe ZIP entry path/i],
    [{ name: 'nested\\fragment.json', content: '{}' }, /Unsafe ZIP entry name/i],
    [{ name: 'fragment.json', content: 'target', kind: 'symlink' }, /symbolic link/i],
  ];
  for (const [entry, pattern] of cases) {
    withRoot((root) => {
      expectFailure(
        execute(root, [artifact(root, { id: 18, entries: [entry] })]),
        pattern,
      );
    });
  }
});

test('duplicate and case-colliding ZIP paths fail closed', () => {
  withRoot((root) => {
    const entries = [
      { name: 'fragment.json', content: '{}' },
      { name: 'fragment.json', content: '{}' },
    ];
    expectFailure(
      execute(root, [artifact(root, { id: 19, entries })]),
      /duplicate path/i,
    );
  });
  withRoot((root) => {
    const entries = [
      { name: 'fragment.json', content: '{}' },
      { name: 'FRAGMENT.JSON', content: '{}' },
    ];
    expectFailure(
      execute(root, [artifact(root, { id: 20, entries })]),
      /case-colliding path/i,
    );
  });
});

test('each artifact must contain exactly one fragment manifest', () => {
  withRoot((root) => {
    expectFailure(
      execute(root, [artifact(root, {
        id: 21,
        entries: [{ name: 'policy.bin', content: 'missing manifest' }],
      })]),
      /exactly one fragment\.json/i,
    );
  });
  withRoot((root) => {
    expectFailure(
      execute(root, [artifact(root, {
        id: 22,
        entries: [
          { name: 'fragment.json', content: '{}' },
          { name: 'nested/fragment.json', content: '{}' },
        ],
      })]),
      /exactly one fragment\.json/i,
    );
  });
});

test('entry-count and uncompressed-size bounds fail closed', () => {
  withRoot((root) => {
    expectFailure(
      execute(
        root,
        [artifact(root, {
          id: 23,
          entries: [
            { name: 'fragment.json', content: '{}' },
            { name: 'extra.bin', content: 'x' },
          ],
        })],
        '--maximum-entries', '1',
      ),
      /exceeding the limit/i,
    );
  });
  withRoot((root) => {
    expectFailure(
      execute(
        root,
        [artifact(root, {
          id: 24,
          entries: [{ name: 'fragment.json', content: '1234567890' }],
        })],
        '--maximum-uncompressed-bytes', '5',
      ),
      /uncompressed size exceeds/i,
    );
  });
});

test('a nonempty output directory is refused unless replacement is explicit', () => {
  withRoot((root) => {
    const item = artifact(root, { id: 25 });
    const first = execute(root, [item]);
    parsed(first);
    const second = execute(root, [item]);
    expectFailure(second, /Output directory is not empty/i);
    const replacement = execute(root, [item], '--replace');
    parsed(replacement);
  });
});

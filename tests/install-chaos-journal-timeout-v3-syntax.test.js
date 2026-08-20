import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const PATH = '.github/workflows/install-chaos-journal-timeout-v3.yml';

test('embedded current-main patcher is syntactically valid Python', (context) => {
  if (!existsSync(new URL(`../${PATH}`, import.meta.url))) {
    context.skip('self-cleaning installer has already published');
    return;
  }
  const source = readFileSync(new URL(`../${PATH}`, import.meta.url), 'utf8');
  const match = source.match(
    /cat > \/tmp\/install-chaos-journal-timeout-v3\.py <<'PY'\n([\s\S]*?)\n\s+PY\n/,
  );
  assert.ok(match, 'embedded Python patcher is missing');
  const lines = match[1].split('\n');
  const nonempty = lines.filter((line) => line.trim());
  const indent = Math.min(...nonempty.map((line) => line.match(/^\s*/)[0].length));
  const script = lines.map((line) => line.slice(Math.min(indent, line.length))).join('\n');
  const parsed = spawnSync(
    'python3',
    ['-c', 'import ast,sys; ast.parse(sys.stdin.read())'],
    { cwd: ROOT, input: script, encoding: 'utf8' },
  );
  assert.equal(parsed.status, 0, parsed.stderr);
});

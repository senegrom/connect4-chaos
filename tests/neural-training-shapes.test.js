import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { normalizeConfig } from '../src/engine.js';
import { pythonCommand } from '../scripts/python-command.mjs';

// Evaluate the actual pure enumerator without importing torch into the JS job.
// The training/arena modules share this function through parse_shapes('all').
const script = `
import ast, json
from pathlib import Path
path = Path('neural/gpu_selfplay.py')
nodes = [n for n in ast.parse(path.read_text()).body
         if isinstance(n, ast.FunctionDef) and n.name in ('all_shapes', 'parse_shapes')]
namespace = {}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
print(json.dumps(namespace['parse_shapes']('all')))
`;

test('default self-play and arena shapes cover every normalized UI configuration', () => {
  const python = pythonCommand();
  const result = spawnSync(python.command, [...python.args, '-c', script], {
    cwd: fileURLToPath(new URL('..', import.meta.url)), encoding: 'utf8', timeout: 30_000,
  });
  assert.equal(result.status, 0, result.stderr || String(result.error));
  const shapes = JSON.parse(result.stdout);
  const keys = new Set(shapes.map((shape) => JSON.stringify(shape)));
  assert.equal(keys.size, shapes.length, 'enumerator must not duplicate shapes');
  let connectSix = 0;
  const configurations = new Map();
  for (let rows = 4; rows <= 10; rows += 1) {
    for (let cols = 4; cols <= 10; cols += 1) {
      for (let connect = 3; connect <= 10; connect += 1) {
        for (const chaosMode of [false, true]) {
          const config = normalizeConfig({ rows, cols, connect, chaosMode, opponent: 'neural' });
          const key = JSON.stringify([config.rows, config.cols, config.connect, config.chaosMode]);
          configurations.set(key, config);
        }
      }
    }
  }
  for (const [key, config] of configurations) {
    assert.ok(keys.has(key), `Missing supported neural configuration: ${key}`);
    if (config.connect === 6) connectSix += 1;
  }
  assert.ok(connectSix > 0, 'coverage must exercise Connect-6');
  for (const [rows, cols, connect] of shapes) {
    assert.ok(connect <= Math.max(rows, cols), 'connect length must fit the board');
  }
});

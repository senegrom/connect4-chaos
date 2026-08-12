import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const root = new URL('../', import.meta.url);

async function json(path) {
  return JSON.parse(await readFile(new URL(path, root), 'utf8'));
}

test('Perfect Chaos foundation manifest records the verified boundary without claiming full play', async () => {
  const [manifest, packageJson] = await Promise.all([
    json('data/perfect-chaos-foundation.manifest.json'),
    json('package.json'),
  ]);

  assert.equal(manifest.version, packageJson.version);
  assert.equal(manifest.format, 'connect4-chaos-perfect-chaos-foundation-v1');
  assert.equal(manifest.status.fullRootProved, false);
  assert.equal(manifest.status.uiPerfectEnabled, false);
  assert.equal(manifest.status.exactEmptyThreshold, 6);
  assert.equal(manifest.status.runtimeMaximumStates, 250_000);
  assert.deepEqual(manifest.referenceCases.map(({ name, value, states, rank }) => ({
    name,
    value,
    states,
    rank,
  })), [
    { name: '2x2-connect2', value: 1, states: 6, rank: 3 },
    { name: '3x3-connect3', value: 0, states: 628, rank: 0 },
    { name: '6x7-endgame-fixture', value: 1, states: 2_585, rank: 1 },
  ]);
  assert.deepEqual(manifest.rootEnumeration.at(-1), {
    depth: 8,
    frontier: 152_581,
    cumulative: 212_379,
  });
});

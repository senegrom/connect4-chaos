import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { vendorRuntime } from '../scripts/vendor-onnx-runtime.mjs';

async function fixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'c4-vendor-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const put = async (name, data) => {
    await mkdir(path.dirname(path.join(root, name)), { recursive: true });
    await writeFile(path.join(root, name), typeof data === 'object' && !Buffer.isBuffer(data) ? JSON.stringify(data) : data);
  };
  await put('package.json', { devDependencies: { 'onnxruntime-web': '1.31.1' } });
  await put('package-lock.json', { packages: {
    '': { devDependencies: { 'onnxruntime-web': '1.31.1' } },
    'node_modules/onnxruntime-web': { version: '1.31.1' },
  } });
  await put('node_modules/onnxruntime-web/package.json', { version: '1.31.1' });
  await put('node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs', '// ONNX Runtime Web v1.31.1\nconst loader="ort-wasm-simd-threaded.asyncify.mjs";');
  await put('node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs', '// loader');
  await put('node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm', Buffer.from([0,97,115,109,1,0,0,0]));
  const source = "export const RUNTIME_RELEASE = '1.30.0';\nexport const DOWNLOAD_BYTES = { model: 106_433_918, runtime: 26_781_914 };\n";
  await put('src/neural-runtime.js', source);
  await put('assets/neural/ort-1.30.0/old.wasm', 'old');
  await put('assets/neural/model.json', 'preserve me');
  return { root, put, source };
}

test('vendoring couples real bytes, release, and size without changing model metadata', async (t) => {
  const { root } = await fixture(t);
  const result = await vendorRuntime(root);
  assert.equal(result.version, '1.31.1');
  assert.equal(result.wasmBytes, 8);
  assert.deepEqual((await readdir(path.join(root, 'assets/neural'))).sort(), ['model.json', 'ort-1.31.1']);
  assert.equal(await readFile(path.join(root, 'assets/neural/model.json'), 'utf8'), 'preserve me');
  const source = await readFile(path.join(root, 'src/neural-runtime.js'), 'utf8');
  assert.match(source, /RUNTIME_RELEASE = '1\.31\.1'/);
  assert.match(source, /model: 106_433_918, runtime: 8/);
  for (const name of result.files) {
    assert.deepEqual(await readFile(path.join(root, 'assets/neural/ort-1.31.1', name)),
      await readFile(path.join(root, 'node_modules/onnxruntime-web/dist', name)));
  }
  await vendorRuntime(root); // Idempotent; no package or model changes.
});

for (const [name, file, data] of [
  ['installed release mismatch', 'node_modules/onnxruntime-web/package.json', { version: '1.30.0' }],
  ['new loader layout', 'node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs', '// ONNX Runtime Web v1.31.1\nconst loader="ort-wasm-simd-threaded.jsep.mjs";'],
  ['invalid wasm', 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm', 'not wasm'],
  ['unexpected runtime directory', 'assets/neural/ort-unreviewed/keep.txt', 'keep'],
]) {
  test(`vendoring rejects ${name} before changing the existing runtime`, async (t) => {
    const { root, put, source } = await fixture(t);
    await put(file, data);
    await assert.rejects(vendorRuntime(root));
    assert.equal(await readFile(path.join(root, 'src/neural-runtime.js'), 'utf8'), source);
    assert.equal(await readFile(path.join(root, 'assets/neural/ort-1.30.0/old.wasm'), 'utf8'), 'old');
  });
}

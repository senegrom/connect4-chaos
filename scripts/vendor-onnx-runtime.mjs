#!/usr/bin/env node
// Run after npm ci. Keep the npm release and the browser's three vendored files together.
import assert from 'node:assert/strict';
import { mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { isEntryPoint } from './entry-point.mjs';

const FILES = ['ort.webgpu.min.mjs', 'ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm'];
const EXACT = /^\d+\.\d+\.\d+$/;
const RELEASE = /export const RUNTIME_RELEASE = '(\d+\.\d+\.\d+)';/g;
const SIZES = /export const DOWNLOAD_BYTES = \{ model: ([\d_]+), runtime: ([\d_]+) \};/g;

export async function vendorRuntime(root = fileURLToPath(new URL('../', import.meta.url))) {
  const json = async (name) => JSON.parse(await readFile(path.join(root, name), 'utf8'));
  const pkg = await json('package.json');
  const lock = await json('package-lock.json');
  const installed = await json('node_modules/onnxruntime-web/package.json');
  const version = pkg.devDependencies['onnxruntime-web'];
  assert.match(version, EXACT, 'Pin onnxruntime-web to an exact release before vendoring');
  assert.equal(lock.packages[''].devDependencies['onnxruntime-web'], version, 'Root lockfile pin differs');
  assert.equal(lock.packages['node_modules/onnxruntime-web'].version, version, 'Lockfile release differs');
  assert.equal(installed.version, version, 'Run npm ci: installed release differs');

  const dist = path.join(root, 'node_modules/onnxruntime-web/dist');
  const bytes = await Promise.all(FILES.map((name) => readFile(path.join(dist, name))));
  const bundle = bytes[0].toString('utf8');
  assert.equal(bundle.match(/ONNX Runtime Web v(\d+\.\d+\.\d+)/)?.[1], version, 'Runtime banner differs');
  assert.deepEqual([...new Set(bundle.match(/ort-wasm-simd-threaded[\w.]*\.mjs/g) ?? [])].sort(), [FILES[1]],
    'The runtime loader layout changed; review the browser URLs instead of guessing');
  assert.ok(bytes[1].length > 0, 'Runtime loader is empty');
  assert.ok(bytes[2].length >= 8 && bytes[2].subarray(0, 8).equals(Buffer.from([0, 97, 115, 109, 1, 0, 0, 0])),
    'Runtime is not a WebAssembly v1 binary');

  const sourcePath = path.join(root, 'src/neural-runtime.js');
  const source = await readFile(sourcePath, 'utf8');
  assert.equal([...source.matchAll(RELEASE)].length, 1, 'Expected one runtime-release declaration');
  assert.equal([...source.matchAll(SIZES)].length, 1, 'Expected one download-size declaration');
  const size = String(bytes[2].length).replace(/\B(?=(\d{3})+(?!\d))/g, '_');
  const updated = source.replace(RELEASE, `export const RUNTIME_RELEASE = '${version}';`)
    .replace(SIZES, (_match, model) => `export const DOWNLOAD_BYTES = { model: ${model}, runtime: ${size} };`);
  const assets = path.join(root, 'assets/neural');
  const runtimes = (await readdir(assets)).filter((name) => name.startsWith('ort'));
  assert.ok(runtimes.every((name) => /^ort-\d+\.\d+\.\d+$/.test(name)),
    'Unexpected runtime path: review it before removing generated files');

  // All inputs are validated before changing any tracked output. Never touch model files.
  const target = path.join(assets, `ort-${version}`);
  if (runtimes.includes(`ort-${version}`)) {
    assert.ok((await readdir(target)).every((name) => FILES.includes(name)), 'Unexpected file in target runtime');
  }
  await mkdir(target, { recursive: true });
  for (let i = 0; i < FILES.length; i += 1) await writeFile(path.join(target, FILES[i]), bytes[i]);
  assert.deepEqual((await readdir(target)).sort(), [...FILES].sort(), 'Unexpected file in target runtime');
  await writeFile(sourcePath, updated);
  for (const name of runtimes) if (name !== `ort-${version}`) await rm(path.join(assets, name), { recursive: true });
  return { version, wasmBytes: bytes[2].length, files: FILES };
}

if (isEntryPoint(import.meta.url)) {
  const result = await vendorRuntime();
  console.log(`Vendored ONNX Runtime ${result.version} (${result.wasmBytes} WASM bytes); run the full regression suite before committing.`);
}

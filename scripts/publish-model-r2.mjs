// Publishes the exported network to R2, where the browser fetches it from.
//
// The model used to ship inside this repository, which cost 106 MB of git
// history at every generation swap and spent GitHub Pages' whole monthly
// bandwidth allowance on roughly 950 visitors. It now lives in a private R2
// bucket, read through workers/model-cdn, and the repository carries only
// the manifest that names it.
//
// One-time setup (after `npx wrangler login`):
//   npx wrangler r2 bucket create connect4-models
//   npx wrangler deploy --config workers/model-cdn/wrangler.jsonc
//
// Then, for each generation:
//   node scripts/publish-model-r2.mjs [model.onnx]
//
// With no argument it reassembles the parts named by assets/neural/model.json,
// so the currently shipped network can be published as it stands. The object
// is stored gzipped: R2 returns exactly the bytes it holds and compresses
// nothing on the fly, and uncompressed the model would be 7 MB larger over
// the wire than GitHub Pages managed.
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { gzip } from 'node:zlib';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const BUCKET = process.env.R2_BUCKET ?? 'connect4-models';
const compress = promisify(gzip);
const run = promisify(execFile);

const manifestPath = join(ROOT, 'assets/neural/model.json');
const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));

// The checkpoint's own name is the version: keys are never overwritten, so a
// response can be immutable, and rolling back is a one-line manifest edit.
const version = String(manifest.source ?? '').replace(/\.pt$/, '');
if (!/^[A-Za-z0-9._-]{1,64}$/.test(version)) {
  throw new Error(`assets/neural/model.json has no usable "source": ${manifest.source}`);
}

async function modelBytes() {
  const [given] = process.argv.slice(2);
  if (given) return readFile(resolve(given));
  const names = manifest.parts ?? ['model.onnx'];
  const pieces = await Promise.all(
    names.map((name) => readFile(join(ROOT, 'assets/neural', name))));
  return Buffer.concat(pieces);
}

const bytes = await modelBytes();
const sha256 = createHash('sha256').update(bytes).digest('hex');
console.log(`model ${(bytes.length / 1e6).toFixed(1)} MB, sha256 ${sha256.slice(0, 16)}…`);

const packed = await compress(bytes, { level: 6 });
console.log(`stored gzipped: ${(packed.length / 1e6).toFixed(1)} MB `
  + `(${(100 * packed.length / bytes.length).toFixed(1)}% of the model)`);

const scratch = await mkdtemp(join(process.env.TEMP ?? '/tmp', 'model-r2-'));
const staged = join(scratch, 'model.onnx.gz');
await writeFile(staged, packed);

const key = `models/${version}/model.onnx`;
console.log(`uploading to ${BUCKET}/${key} …`);
try {
  const { stdout, stderr } = await run('npx', ['--yes', 'wrangler', 'r2', 'object', 'put',
    `${BUCKET}/${key}`,
    '--file', staged,
    '--content-type', 'application/octet-stream',
    '--content-encoding', 'gzip',
    '--remote',
  ], { cwd: ROOT, shell: process.platform === 'win32', maxBuffer: 1 << 24 });
  process.stdout.write(stdout || stderr);
} finally {
  await rm(scratch, { recursive: true, force: true, maxRetries: 5 }).catch(() => {});
}

console.log('\nPut these in assets/neural/model.json:');
console.log(JSON.stringify({
  source: manifest.source,
  object: key,
  bytes: bytes.length,
  storedBytes: packed.length,
  sha256,
}, null, 2));
console.log('\nThen check it end to end:');
console.log(`  curl -s -o /dev/null -w '%{http_code} %{size_download}\\n' \\`);
console.log(`    -H 'Origin: https://senegrom.github.io' <worker-url>/${key}`);

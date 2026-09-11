// Publish a verified export under a content-addressed key. Old generation-only
// URLs are never written by this tool, so their immutable responses stay valid.
// Usage: node scripts/publish-model-r2.mjs [model.onnx] [--manifest model.json]
// By default the committed assets/neural/model.json describes the input. For
// a new export, pass its JSON sidecar explicitly. No upload precedes validation.
import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { gzip } from 'node:zlib';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { promisify } from 'node:util';
import { modelIdentity, verifyModelBytes, ModelIntegrityError } from '../src/model-integrity.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const MANIFEST = join(ROOT, 'assets/neural/model.json');
const compress = promisify(gzip);
const run = promisify(execFile);

async function uploadR2(bucket, key, staged) {
  const { stdout, stderr } = await run('npx', ['--yes', 'wrangler', 'r2', 'object', 'put',
    `${bucket}/${key}`, '--file', staged, '--content-type', 'application/octet-stream',
    '--content-encoding', 'gzip', '--remote',
  ], { cwd: ROOT, shell: process.platform === 'win32', maxBuffer: 1 << 24 });
  process.stdout.write(stdout || stderr);
}

export async function publishModel({ modelPath, manifestPath = MANIFEST,
  bucket = process.env.R2_BUCKET ?? 'connect4-models', upload = uploadR2 } = {}) {
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$/.test(bucket)) throw new Error('Invalid R2 bucket name.');
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  const identity = modelIdentity(manifest);
  const version = String(manifest.source ?? '').replace(/\.pt$/, '');
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(version)) throw new Error('Model manifest has no usable source.');
  const path = modelPath ? resolve(modelPath) : join(dirname(resolve(manifestPath)), 'model.onnx');
  const info = await stat(path);
  if (!info.isFile() || info.size !== identity.bytes) {
    throw new ModelIntegrityError(`Model length mismatch: found ${info.size}, expected ${identity.bytes}.`);
  }
  const bytes = await readFile(path);
  await verifyModelBytes(bytes, identity);
  // The full digest is the filename (no extension is needed for octet-stream).
  // Different exports of the same checkpoint cannot share an immutable URL.
  // Re-uploading this key can only send identical, verified model content.
  const key = `models/${version}/${identity.sha256}`;
  const packed = await compress(bytes, { level: 6 });
  const scratch = await mkdtemp(join(tmpdir(), 'model-r2-'));
  try {
    const staged = join(scratch, 'model.onnx.gz');
    await writeFile(staged, packed, { flag: 'wx' });
    await upload(bucket, key, staged);
  } finally {
    await rm(scratch, { recursive: true, force: true, maxRetries: 5 });
  }
  return { source: manifest.source, object: key, bytes: bytes.length,
    storedBytes: packed.length, sha256: identity.sha256,
    ...(manifest.origin ? { origin: manifest.origin } : {}) };
}

export async function main(argv = process.argv.slice(2)) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === '--manifest') {
      if (options.manifestPath || !argv[index + 1] || argv[index + 1].startsWith('--')) {
        throw new Error('--manifest requires one JSON sidecar path.');
      }
      options.manifestPath = resolve(argv[++index]);
    } else if (argv[index].startsWith('--') || options.modelPath) {
      throw new Error('Usage: publish-model-r2.mjs [model.onnx] [--manifest model.json]');
    } else options.modelPath = argv[index];
  }
  const release = await publishModel(options);
  console.log('\nPublished verified model. Update assets/neural/model.json with:');
  console.log(JSON.stringify(release, null, 2));
  console.log('Also update MODEL_OBJECT, MODEL_SHA256 and DOWNLOAD_BYTES.model in src/neural-runtime.js.');
  console.log('Run the model identity tests before committing. Do not overwrite legacy model.onnx objects.');
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await main();
}

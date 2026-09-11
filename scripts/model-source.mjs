// Where Node-side tools and tests get the network from.
//
// The browser fetches the model from R2 and keeps it in Cache Storage, but a
// test runner has neither. It looks in three places, cheapest first: an
// explicit path in NEURAL_MODEL, the working copy under assets/neural if one
// is present, and failing those the published object, downloaded once into a
// gitignored cache. Returning null rather than throwing lets a caller skip
// instead of fail, which is what an offline checkout wants.
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const CACHE = join(ROOT, '.model-cache');

// A file that exists but is empty or the wrong length is worse than none: it
// reaches the runtime as "No graph was found in the protobuf", which says
// nothing about where the bytes came from. An interrupted download leaves
// exactly that, and so does a shell redirect whose command then failed.
async function readable(path, expected = 0) {
  try {
    const info = await stat(path);
    if (!info.isFile() || info.size === 0) return false;
    return expected ? info.size === expected : true;
  } catch {
    return false;
  }
}

/** The manifest that names the published object and its size. */
export async function modelManifest() {
  return JSON.parse(await readFile(join(ROOT, 'assets/neural/model.json'), 'utf8'));
}

/**
 * The exported network as bytes, or null when it cannot be had.
 *
 * `allowDownload` is false by default so that no test reaches the network
 * without being asked to: a suite that silently downloads 99 MB is a suite
 * that fails differently on a train.
 */
export async function readModelBytes({ allowDownload = false } = {}) {
  const override = process.env.NEURAL_MODEL;
  if (override && await readable(override)) return readFile(override);

  const manifest = await modelManifest();
  const local = join(ROOT, 'assets/neural');
  // Whatever the manifest says ships locally: one file, or parts that
  // concatenate back into it byte for byte.
  const names = manifest.parts ?? ['model.onnx'];
  const paths = names.map((name) => join(local, name));
  // Only a single file can be size-checked against the manifest; parts carry
  // their own sizes, and the concatenation is checked below either way.
  const each = names.length === 1 ? manifest.bytes : 0;
  if ((await Promise.all(paths.map((path) => readable(path, each)))).every(Boolean)) {
    const joined = Buffer.concat(await Promise.all(paths.map((path) => readFile(path))));
    if (!manifest.bytes || joined.length === manifest.bytes) return joined;
  }

  const version = String(manifest.source ?? '').replace(/\.pt$/, '');
  const cached = join(CACHE, `${version}.onnx`);
  if (await readable(cached, manifest.bytes ?? 0)) return readFile(cached);

  if (!allowDownload || !manifest.origin || !manifest.object) return null;
  const url = `${manifest.origin.replace(/\/$/, '')}/${manifest.object}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (manifest.bytes && bytes.length !== manifest.bytes) {
    throw new Error(`${url} gave ${bytes.length} bytes, expected ${manifest.bytes}`);
  }
  await mkdir(CACHE, { recursive: true });
  await writeFile(cached, bytes);
  return bytes;
}

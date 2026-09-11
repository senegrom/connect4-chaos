// Verified model bytes for Node tools. Downloads are opt-in; disk caching is
// optional and content-addressed. NEURAL_MODEL is also checked, so an override
// cannot quietly substitute different weights for a named release.
import { mkdir, readFile, rename, stat, unlink, writeFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { modelIdentity, verifyModelBytes, ModelIntegrityError } from '../src/model-integrity.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const MANIFEST = join(ROOT, 'assets/neural/model.json');
const CACHE = join(ROOT, '.model-cache');

export async function modelManifest(path = MANIFEST) {
  return JSON.parse(await readFile(path, 'utf8'));
}

async function candidate(path, identity, { removeInvalid = false } = {}) {
  try {
    const info = await stat(path);
    if (!info.isFile()) return null;
    if (info.size !== identity.bytes) throw new ModelIntegrityError('Cached model length mismatch.');
    const bytes = await readFile(path);
    await verifyModelBytes(bytes, identity); // Recheck the bytes, not just a racy stat.
    return bytes;
  } catch (error) {
    if (error instanceof ModelIntegrityError && removeInvalid) await unlink(path).catch(() => {});
    if (error instanceof ModelIntegrityError || ['ENOENT', 'ENOTDIR', 'EISDIR', 'EACCES', 'EPERM'].includes(error.code)) return null;
    throw error;
  }
}

/** An explicit manifestPath permits tools to evaluate a different exported
 * model together with its own identity. There is no unverified override. */
export async function readModelBytes({ allowDownload = false, manifestPath = MANIFEST,
  cacheDirectory = CACHE } = {}) {
  const manifest = await modelManifest(manifestPath);
  const identity = modelIdentity(manifest);
  const override = process.env.NEURAL_MODEL;
  if (override) {
    const bytes = await readFile(override);
    await verifyModelBytes(bytes, identity);
    return bytes;
  }

  const local = dirname(resolve(manifestPath));
  const names = manifest.parts ?? ['model.onnx'];
  if (!Array.isArray(names) || !names.length || names.some((name) =>
    typeof name !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(name))) {
    throw new ModelIntegrityError('Invalid local model filenames.');
  }
  if (names.length === 1) {
    const bytes = await candidate(join(local, names[0]), identity);
    if (bytes) return bytes;
  } else {
    // Legacy exports may still be assembled from parts, but only the verified
    // concatenation can escape this resolver.
    try {
      const bytes = Buffer.concat(await Promise.all(names.map((name) => readFile(join(local, name)))));
      await verifyModelBytes(bytes, identity);
      return bytes;
    } catch (error) {
      if (!(error instanceof ModelIntegrityError) && !['ENOENT', 'ENOTDIR', 'EISDIR', 'EACCES', 'EPERM'].includes(error.code)) throw error;
    }
  }

  const cached = join(cacheDirectory, `${identity.sha256}.onnx`);
  const kept = await candidate(cached, identity, { removeInvalid: true });
  if (kept) return kept;
  // Preserve offline use of an older generation-named cache, but verify it
  // before returning and evict it if it contains wrong or interrupted bytes.
  const version = String(manifest.source ?? '').replace(/\.pt$/, '');
  if (/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(version)) {
    const legacy = await candidate(join(cacheDirectory, `${version}.onnx`), identity, { removeInvalid: true });
    if (legacy) return legacy;
  }
  if (!allowDownload || !manifest.origin || !manifest.object) return null;
  const url = `${manifest.origin.replace(/\/$/, '')}/${manifest.object}`;
  const response = await fetch(url, { signal: AbortSignal.timeout(600_000) });
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  await verifyModelBytes(bytes, identity);

  // Each writer owns a unique temporary file. Interrupted/concurrent writers
  // never expose partial bytes at the cache key, and disk failure is optional.
  const temporary = `${cached}.${randomUUID()}.tmp`;
  try {
    await mkdir(cacheDirectory, { recursive: true });
    await writeFile(temporary, bytes, { flag: 'wx' });
    await rename(temporary, cached);
  } catch {
    // The verified model remains usable without a writable disk cache.
  } finally {
    await unlink(temporary).catch(() => {});
  }
  return bytes;
}

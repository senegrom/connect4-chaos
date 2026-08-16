import {
  decodePerfectClassicPolicy,
  findPerfectClassicPolicy,
  loadPerfectClassicManifest,
} from './perfect-classic-policy.js';

const DEFAULT_MANIFEST_URL = new URL('../data/perfect-classic/manifest.json', import.meta.url);
const LOADS = new Map();

async function readBytes(url) {
  if (url.protocol === 'file:' && typeof process !== 'undefined' && process.versions?.node) {
    const { readFile } = await import('node:fs/promises');
    return new Uint8Array(await readFile(url));
  }
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load perfect classic policy (${response.status}).`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

function hex(bytes) {
  return [...bytes].map((value) => value.toString(16).padStart(2, '0')).join('');
}

async function sha256(bytes) {
  if (globalThis.crypto?.subtle) {
    return hex(new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', bytes)));
  }
  if (typeof process !== 'undefined' && process.versions?.node) {
    const { createHash } = await import('node:crypto');
    return createHash('sha256').update(bytes).digest('hex');
  }
  throw new Error('SHA-256 support is unavailable for Perfect classic verification.');
}

function validateArtifactMetadata(entry) {
  if (!Number.isInteger(entry.bytes) || entry.bytes < 24
      || typeof entry.sha256 !== 'string'
      || !/^[0-9a-f]{64}$/i.test(entry.sha256)) {
    throw new Error('Perfect classic manifest has invalid artifact metadata.');
  }
}

export async function loadVerifiedPerfectClassicPolicy(
  rows,
  columns,
  connect,
  role,
  options = {},
) {
  const manifestUrl = options.manifestUrl instanceof URL
    ? options.manifestUrl
    : options.manifestUrl
      ? new URL(String(options.manifestUrl), import.meta.url)
      : DEFAULT_MANIFEST_URL;
  const manifest = options.manifest ?? await loadPerfectClassicManifest(manifestUrl);
  const entry = findPerfectClassicPolicy(manifest, rows, columns, connect, role);
  if (!entry) return null;
  validateArtifactMetadata(entry);

  const policyUrl = options.url instanceof URL
    ? options.url
    : options.url
      ? new URL(String(options.url), import.meta.url)
      : new URL(entry.file, manifestUrl);
  const cacheKey = [policyUrl.href, entry.bytes, entry.sha256].join('|');
  let load = LOADS.get(cacheKey);
  if (!load) {
    load = (async () => {
      const bytes = await readBytes(policyUrl);
      if (bytes.byteLength !== entry.bytes) {
        throw new Error(
          `Perfect classic policy length mismatch: expected ${entry.bytes}, `
          + `found ${bytes.byteLength}.`,
        );
      }
      const actualHash = await sha256(bytes);
      if (actualHash.toLowerCase() !== entry.sha256.toLowerCase()) {
        throw new Error('Perfect classic policy SHA-256 does not match its manifest.');
      }
      const policy = decodePerfectClassicPolicy(bytes, { rows, columns, connect, role });
      if (policy.handoffRemaining !== entry.handoffRemaining
          || policy.rootValue !== entry.rootValue
          || policy.entryCount !== entry.entryCount
          || policy.closureStates !== entry.closureStates) {
        throw new Error('Perfect classic policy metadata does not match its manifest.');
      }
      return policy;
    })();
    LOADS.set(cacheKey, load);
    load.catch(() => {
      if (LOADS.get(cacheKey) === load) LOADS.delete(cacheKey);
    });
  }
  return load;
}

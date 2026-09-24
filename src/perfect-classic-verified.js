import { readData, cachedDataLoad } from './data-loader.js';
import {
  decodePerfectClassicPolicy,
  findPerfectClassicPolicy,
  loadPerfectClassicManifest,
} from './perfect-classic-policy.js';
import { sha256Hex } from './sha256.js';

const DEFAULT_MANIFEST_URL = new URL('../data/perfect-classic/manifest.json', import.meta.url);
const LOADS = new Map();


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
  const manifest = options.manifest ?? await loadPerfectClassicManifest(manifestUrl, options);
  const entry = findPerfectClassicPolicy(manifest, rows, columns, connect, role);
  if (!entry) return null;
  validateArtifactMetadata(entry);

  const policyUrl = options.url instanceof URL
    ? options.url
    : options.url
      ? new URL(String(options.url), import.meta.url)
      : new URL(entry.file, manifestUrl);
  const cacheKey = [policyUrl.href, entry.bytes, entry.sha256].join('|');
  return cachedDataLoad(LOADS, cacheKey, async () => {
      const bytes = await readData(policyUrl, 'Perfect classic policy', options);
      if (bytes.byteLength !== entry.bytes) {
        throw new Error(
          `Perfect classic policy length mismatch: expected ${entry.bytes}, `
          + `found ${bytes.byteLength}.`,
        );
      }
      const actualHash = await sha256Hex(bytes, 'Perfect classic verification');
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
  }, options);
}

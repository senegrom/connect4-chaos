import { loadManifestPolicy } from './data-loader.js';
import {
  PERFECT_CLASSIC_MANIFEST_URL,
  decodePerfectClassicPolicy,
  findPerfectClassicPolicy,
  loadPerfectClassicManifest,
} from './perfect-classic-policy.js';

const LOADS = new Map();

export function loadVerifiedPerfectClassicPolicy(rows, columns, connect, role, options = {}) {
  return loadManifestPolicy(LOADS, {
    label: 'Perfect classic',
    defaultManifestUrl: PERFECT_CLASSIC_MANIFEST_URL,
    loadManifest: loadPerfectClassicManifest,
    find: (manifest) => findPerfectClassicPolicy(manifest, rows, columns, connect, role),
    decode: (bytes) => decodePerfectClassicPolicy(bytes, { rows, columns, connect, role }),
    fields: ['handoffRemaining', 'rootValue', 'entryCount', 'closureStates'],
  }, options);
}

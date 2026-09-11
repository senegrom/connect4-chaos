import {
  STANDARD_POSITION_KEY_LIMIT,
  createExactTableLoader,
  decodeExactTable,
} from './exact-table.js';

const DEFAULT_URL = new URL('../assets/perfect-strategy.bin', import.meta.url);

// Trust is pinned to the independently replayed release, not to a manifest
// fetched beside potentially stale/corrupt bytes. The regression suite checks
// these fields against data/perfect-strategy.manifest.json and the real asset.
export const PERFECT_STRATEGY_CERTIFICATE = Object.freeze({
  version: 1,
  handoffRemaining: 24,
  roleFlags: 3,
  entryCount: 470494,
  byteLength: 4704952,
  sha256: '91de1bd2a5bef3805c19a018b9dcb3a11d240e0569086a03da2872b981363f7a',
});

export const PERFECT_ROLE_FIRST = 1;
export const PERFECT_ROLE_SECOND = 2;
export const PERFECT_ROLE_BOTH = PERFECT_ROLE_FIRST | PERFECT_ROLE_SECOND;

export function decodePerfectStrategy(input) {
  const table = decodeExactTable(input, {
    magic: 'C4PS',
    label: 'Perfect-strategy',
    readMetadata(view) {
      const handoffRemaining = view.getUint8(5);
      const roleFlags = view.getUint8(7);
      if (handoffRemaining > 42) {
        throw new Error('Perfect-strategy handoff is outside the standard board.');
      }
      if (roleFlags === 0 || (roleFlags & ~PERFECT_ROLE_BOTH) !== 0) {
        throw new Error('Perfect-strategy role flags are invalid.');
      }
      return { handoffRemaining, roleFlags };
    },
    validKey: (key) => key < STANDARD_POSITION_KEY_LIMIT,
    validMoveMask: (mask) => mask !== 0 && (mask & (mask - 1)) === 0 && (mask & 0x80) === 0,
    moveMaskError: 'Perfect-strategy entries must contain exactly one of seven columns.',
  });

  return Object.freeze({
    ...table,
    coversRole(role) {
      return (role === PERFECT_ROLE_FIRST || role === PERFECT_ROLE_SECOND)
        && (table.roleFlags & role) !== 0;
    },
  });
}

async function decodeVerifiedStrategy(bytes) {
  const expected = PERFECT_STRATEGY_CERTIFICATE;
  if (bytes.byteLength !== expected.byteLength) {
    throw new Error('Perfect strategy length does not match its certificate.');
  }
  let actualHash;
  if (globalThis.crypto?.subtle) {
    const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
    actualHash = [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, '0')).join('');
  } else if (typeof process !== 'undefined' && process.versions?.node) {
    const { createHash } = await import('node:crypto');
    actualHash = createHash('sha256').update(bytes).digest('hex');
  } else {
    throw new Error('SHA-256 support is unavailable for Perfect strategy verification.');
  }
  if (actualHash !== expected.sha256) {
    throw new Error('Perfect strategy SHA-256 does not match its certificate.');
  }
  // Raw decoding remains available to generators; runtime loading must never
  // cache or return a table until both integrity and metadata are verified.
  const table = decodePerfectStrategy(bytes);
  for (const field of ['version', 'handoffRemaining', 'roleFlags', 'entryCount', 'byteLength']) {
    if (table[field] !== expected[field]) {
      throw new Error(`Perfect strategy ${field} does not match its certificate.`);
    }
  }
  return table;
}

const loadStrategy = createExactTableLoader(decodeVerifiedStrategy, 'Perfect strategy');

export function loadPerfectStrategy(url = DEFAULT_URL, options = {}) {
  return loadStrategy(url instanceof URL ? url : new URL(String(url), import.meta.url), options);
}

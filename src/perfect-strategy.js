import { decodeExactTable } from './exact-table.js';
import { createVerifiedTableLoader } from './verified-table.js';

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

const PERFECT_ROLE_FIRST = 1;
const PERFECT_ROLE_SECOND = 2;
export const PERFECT_ROLE_BOTH = PERFECT_ROLE_FIRST | PERFECT_ROLE_SECOND;

export function decodePerfectStrategy(input) {
  return decodeExactTable(input, {
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
    validMoveMask: (mask) => mask !== 0 && (mask & (mask - 1)) === 0 && (mask & 0x80) === 0,
    moveMaskError: 'Perfect-strategy entries must contain exactly one of seven columns.',
  });
}

export const loadPerfectStrategy = createVerifiedTableLoader(
  PERFECT_STRATEGY_CERTIFICATE,
  decodePerfectStrategy,
  'Perfect strategy',
  new URL('../assets/perfect-strategy.bin', import.meta.url),
);

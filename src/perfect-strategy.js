import {
  STANDARD_POSITION_KEY_LIMIT,
  createExactTableLoader,
  decodeExactTable,
} from './exact-table.js';

const DEFAULT_URL = new URL('../assets/perfect-strategy.bin', import.meta.url);

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

const loadStrategy = createExactTableLoader(decodePerfectStrategy, 'Perfect strategy');

export function loadPerfectStrategy(url = DEFAULT_URL) {
  return loadStrategy(url instanceof URL ? url : new URL(String(url), import.meta.url));
}

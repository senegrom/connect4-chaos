import {
  STANDARD_POSITION_KEY_LIMIT,
  createExactTableLoader,
  decodeExactTable,
} from './exact-table.js';

const DEFAULT_URL = new URL('../assets/perfect-book.bin', import.meta.url);

export function decodePerfectBook(input) {
  return decodeExactTable(input, {
    magic: 'C4PB',
    label: 'Perfect-book',
    readMetadata(view) {
      const maxPly = view.getUint8(5);
      if (maxPly > 42) throw new Error('Perfect-book maximum ply is outside the standard board.');
      if (view.getUint8(7) !== 0) throw new Error('Perfect-book reserved header byte must be zero.');
      return { maxPly };
    },
    validKey: (key) => key < STANDARD_POSITION_KEY_LIMIT,
    validMoveMask: (mask) => mask !== 0 && (mask & 0x80) === 0,
    moveMaskError: 'Perfect-book move masks must contain at least one of seven columns.',
  });
}

const loadBook = createExactTableLoader(decodePerfectBook, 'Perfect-play book');

export function loadPerfectBook(url = DEFAULT_URL) {
  return loadBook(url instanceof URL ? url : new URL(String(url), import.meta.url));
}

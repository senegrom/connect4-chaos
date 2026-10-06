import { decodeExactTable } from './exact-table.js';
import { createVerifiedTableLoader } from './verified-table.js';

// Pin runtime trust to the released book, just like the standard strategy.
// Tests bind these fields to the committed manifest and binary. Raw decoding
// stays available for generators; alternate download URLs do not change trust.
export const PERFECT_BOOK_CERTIFICATE = Object.freeze({
  version: 1,
  maxPly: 8,
  entryCount: 129498,
  byteLength: 1294992,
  sha256: '7d9e4e39f469083b1297671c015309ede049515716b7d0cbae0a8ddb5e8ced13',
});

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
    validMoveMask: (mask) => mask !== 0 && (mask & 0x80) === 0,
    moveMaskError: 'Perfect-book move masks must contain at least one of seven columns.',
  });
}

export const loadPerfectBook = createVerifiedTableLoader(
  PERFECT_BOOK_CERTIFICATE,
  decodePerfectBook,
  'Perfect-play book',
  new URL('../assets/perfect-book.bin', import.meta.url),
);

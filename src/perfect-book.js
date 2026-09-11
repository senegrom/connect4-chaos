import {
  STANDARD_POSITION_KEY_LIMIT,
  createExactTableLoader,
  decodeExactTable,
} from './exact-table.js';

const DEFAULT_URL = new URL('../assets/perfect-book.bin', import.meta.url);

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
    validKey: (key) => key < STANDARD_POSITION_KEY_LIMIT,
    validMoveMask: (mask) => mask !== 0 && (mask & 0x80) === 0,
    moveMaskError: 'Perfect-book move masks must contain at least one of seven columns.',
  });
}

async function decodeVerifiedBook(bytes) {
  const expected = PERFECT_BOOK_CERTIFICATE;
  if (bytes.byteLength !== expected.byteLength) {
    throw new Error('Perfect-play book length does not match its certificate.');
  }
  let actualHash;
  if (globalThis.crypto?.subtle) {
    const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
    actualHash = [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, '0')).join('');
  } else if (typeof process !== 'undefined' && process.versions?.node) {
    const { createHash } = await import('node:crypto');
    actualHash = createHash('sha256').update(bytes).digest('hex');
  } else {
    throw new Error('SHA-256 support is unavailable for Perfect-play book verification.');
  }
  if (actualHash !== expected.sha256) {
    throw new Error('Perfect-play book SHA-256 does not match its certificate.');
  }
  const table = decodePerfectBook(bytes);
  for (const field of ['version', 'maxPly', 'entryCount', 'byteLength']) {
    if (table[field] !== expected[field]) {
      throw new Error(`Perfect-play book ${field} does not match its certificate.`);
    }
  }
  return table;
}

const loadBook = createExactTableLoader(decodeVerifiedBook, 'Perfect-play book');

export function loadPerfectBook(url = DEFAULT_URL, options = {}) {
  return loadBook(url instanceof URL ? url : new URL(String(url), import.meta.url), options);
}

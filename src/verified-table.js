import { createExactTableLoader } from './exact-table.js';
import { sha256Hex } from './sha256.js';

/** The loader of a table whose trust is pinned to its release certificate,
 * whatever URL it is fetched from. The free size check runs before any
 * hashing, and a table is neither cached nor returned until its SHA-256 and
 * every other field the certificate pins match. The certificate's size is
 * also the download's progress total: a compressed response's
 * Content-Length counts compressed bytes.
 *
 * It has a module of its own because exact-table.js may come from the cache:
 * Pages keeps modules for ten minutes, and a page reloaded soon after a
 * deploy can load a new perfect-strategy.js beside an exact-table.js that
 * perfect-book.js or perfect-chaos-prefix.js fetched before it. */
export function createVerifiedTableLoader(certificate, decode, label, defaultUrl) {
  const load = createExactTableLoader(async (bytes) => {
    if (bytes.byteLength !== certificate.byteLength) {
      throw new Error(`${label} length does not match its certificate.`);
    }
    if (await sha256Hex(bytes, `${label} verification`) !== certificate.sha256) {
      throw new Error(`${label} SHA-256 does not match its certificate.`);
    }
    const table = decode(bytes);
    for (const field of Object.keys(certificate)) {
      if (field !== 'sha256' && table[field] !== certificate[field]) {
        throw new Error(`${label} ${field} does not match its certificate.`);
      }
    }
    return table;
  }, label);
  return (url = defaultUrl, options = {}) => load(url, { ...options, expectedBytes: certificate.byteLength });
}

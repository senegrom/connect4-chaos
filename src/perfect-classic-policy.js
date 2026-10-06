// Classic policies as the page uses them: the decoded format with a board
// lookup, and the catalog loader. The release gate replays the catalog
// through perfect-classic-format.js alone, so an edit here, or to the rules
// and transport this module uses, costs no replay.
import {
  boardToClassicBitboard, canonicalClassicPosition, moveForColumn,
} from './classic-geometry.js';
import { readData, cachedDataLoad, CATALOG_LOAD_TIMEOUT_MS } from './data-loader.js';
import { ACTION_DROP, RED, YELLOW } from './engine.js';
import {
  PERFECT_CLASSIC_ROLE_FIRST,
  PERFECT_CLASSIC_ROLE_SECOND,
  decodePerfectClassicRecords,
  moveMaskColumn,
  policyGeometry,
  validateRole,
} from './perfect-classic-format.js';

export { PERFECT_CLASSIC_ROLE_FIRST, PERFECT_CLASSIC_ROLE_SECOND };

const DEFAULT_MANIFEST_URL = new URL('../data/perfect-classic/manifest.json', import.meta.url);
const MANIFEST_PROMISES = new Map();

export function perfectClassicRole(startingPlayer, aiPlayer) {
  if ((startingPlayer !== RED && startingPlayer !== YELLOW)
      || (aiPlayer !== RED && aiPlayer !== YELLOW)) return null;
  return startingPlayer === aiPlayer
    ? PERFECT_CLASSIC_ROLE_FIRST
    : PERFECT_CLASSIC_ROLE_SECOND;
}

/** The decoded policy, with lookup(board, currentPlayer, aiPlayer,
 * startingPlayer): the move it stores for the side to move, or null where it
 * does not cover the position. */
export function decodePerfectClassicPolicy(input, expectations = {}) {
  const records = decodePerfectClassicRecords(input, expectations);
  const { rows, columns, connect, role, handoffRemaining } = records;
  return Object.freeze({
    ...records,
    lookup(board, currentPlayer, aiPlayer, startingPlayer) {
      if (currentPlayer !== aiPlayer) return null;
      if (perfectClassicRole(startingPlayer, aiPlayer) !== role) return null;
      const encoded = boardToClassicBitboard(board, currentPlayer, connect);
      if (!encoded || encoded.geometry.rows !== rows || encoded.geometry.columns !== columns) return null;
      if (encoded.geometry.cellCount - encoded.moves <= handoffRemaining) return null;

      const canonical = canonicalClassicPosition(encoded.geometry, encoded);
      const record = records.lookupKey(canonical.key);
      if (!record) return null;
      let column = moveMaskColumn(record.moveMask, columns);
      if (canonical.mirrored) column = columns - 1 - column;
      if (moveForColumn(encoded.geometry, encoded.mask, column) === 0n) {
        throw new Error('Perfect classic policy returned an illegal move.');
      }
      return Object.freeze({
        action: Object.freeze({ type: ACTION_DROP, column }),
        outcome: record.outcome,
        mirrored: canonical.mirrored,
      });
    },
  });
}

export function loadPerfectClassicManifest(url = DEFAULT_MANIFEST_URL, options = {}) {
  const target = url instanceof URL ? url : new URL(String(url), import.meta.url);
  return cachedDataLoad(MANIFEST_PROMISES, target.href, async () => {
    const manifest = await readData(target, "Perfect classic manifest", { timeoutMs: CATALOG_LOAD_TIMEOUT_MS, ...options, json: true });
    if (manifest?.format !== 'connect4-perfect-classic-manifest-v1'
        || !Array.isArray(manifest.policies)) {
      throw new Error('Perfect classic manifest format is invalid.');
    }
    const policies = manifest.policies.map((entry) => {
      policyGeometry(entry.rows, entry.columns, entry.connect);
      validateRole(entry.role);
      if (typeof entry.file !== 'string' || entry.file.length === 0
          || !Number.isInteger(entry.handoffRemaining)
          || !Number.isInteger(entry.entryCount)
          || !Number.isInteger(entry.rootValue)) {
        throw new Error('Perfect classic manifest contains an invalid policy entry.');
      }
      return Object.freeze({ ...entry });
    });
    return Object.freeze({ ...manifest, policies: Object.freeze(policies) });
  }, options);
}

export function findPerfectClassicPolicy(manifest, rows, columns, connect, role) {
  return manifest?.policies?.find((entry) => (
    entry.rows === rows
    && entry.columns === columns
    && entry.connect === connect
    && entry.role === role
  )) ?? null;
}

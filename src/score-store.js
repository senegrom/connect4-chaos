import { normalizeScores } from './round-storage.js';

export const SCORE_CHANGE_KEY = 'connect4-chaos.scores.changed.v2';
export const SCORE_DATABASE = 'connect4-chaos.results.v2';
export function resultId() {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}
export function newLedger(legacy = {}) {
  return { version: 2, epoch: resultId(), revision: 0, base: normalizeScores(legacy), results: {} };
}

function validLedger(ledger) {
  return ledger?.version === 2 && typeof ledger.epoch === 'string'
    && Number.isSafeInteger(ledger.revision) && Boolean(ledger.results) && Boolean(ledger.base);
}

/** Pure transition, executed inside a single IndexedDB readwrite transaction. */
export function scoreTransition(ledger, operation = { type: 'read' }) {
  if (!validLedger(ledger)) {
    throw new Error('Saved score data is damaged. Reset the score to start it again.');
  }
  let changed = false;
  let receipt = null;
  if (operation.type === 'record') {
    const { id, winner, supersedes = [] } = operation;
    if (typeof id !== 'string' || !/^[a-zA-Z0-9._:-]{1,128}$/.test(id)
        || !['1', '2', 'draw'].includes(String(winner))) throw new Error('Invalid round result.');
    if (Object.hasOwn(ledger.results, id) && ledger.results[id] !== String(winner)) {
      throw new Error('Another tab already recorded a different result for this round. Start a new round.');
    }
    // Results of this round whose write failed. One that outlived its
    // deadline can have landed after all; with the round now ending
    // differently it would count the round twice, and no Undo holds it.
    for (const stale of Array.isArray(supersedes) ? supersedes : []) {
      if (stale !== id && typeof stale === 'string' && Object.hasOwn(ledger.results, stale)) {
        delete ledger.results[stale];
        changed = true;
      }
    }
    if (!Object.hasOwn(ledger.results, id)) {
      Object.defineProperty(ledger.results, id, { value: String(winner), enumerable: true, writable: true, configurable: true });
      changed = true;
    }
    receipt = { id, epoch: ledger.epoch };
  } else if (operation.type === 'undo') {
    for (const result of operation.receipts ?? []) {
      // Reset rotates the epoch. A pre-reset Undo can never remove new wins.
      if (result?.epoch === ledger.epoch && Object.hasOwn(ledger.results, result.id)) {
        delete ledger.results[result.id];
        changed = true;
      }
    }
  } else if (operation.type === 'reset') {
    ledger.epoch = resultId();
    ledger.base = normalizeScores({});
    ledger.results = {};
    changed = true;
  } else if (operation.type !== 'read') throw new Error('Unknown score operation.');
  if (changed) ledger.revision += 1;
  const scores = normalizeScores(ledger.base);
  for (const winner of Object.values(ledger.results)) {
    if (!Object.hasOwn(scores, winner)) throw new Error('Invalid saved winner.');
    scores[winner] += 1;
  }
  return { changed, receipt, scores, revision: ledger.revision };
}

export function createScoreStore({
  indexedDB = globalThis.indexedDB, legacyScores = () => ({}), retireLegacy = () => {}, onWarning = () => {},
  timeoutMs = 10_000,
} = {}) {
  let retired = false;
  let connection;
  let activeConnection;
  let memory;
  let committed;
  let failedWrites = 0;
  const forget = (db) => {
    // A delayed close/versionchange from an old handle must not evict a new one.
    if (activeConnection === db) {
      activeConnection = null;
      connection = null;
    }
  };
  // Fallback is deliberately tab-local. Never pretend unlocked localStorage
  // read/modify/write is a transaction when persistent storage is unavailable.
  const local = () => {
    if (!memory) {
      memory = committed ?? newLedger(legacyScores());
      onWarning('Scores are available in this tab only because browser storage is unavailable.');
    }
    return null;
  };
  const open = () => {
    if (!connection) connection = new Promise((resolve, reject) => {
      if (!indexedDB) { resolve(local()); return; }
      let expired = false;
      const request = indexedDB.open(SCORE_DATABASE, 1);
      const timer = setTimeout(() => { expired = true; reject(new Error('Score database did not open.')); }, timeoutMs);
      request.onupgradeneeded = () => {
        if (expired) { request.transaction.abort(); return; }
        if (!request.result.objectStoreNames.contains('scores')) request.result.createObjectStore('scores');
      };
      request.onerror = () => { clearTimeout(timer); reject(request.error); };
      request.onsuccess = () => {
        clearTimeout(timer);
        if (expired) { request.result.close(); return; }
        const db = request.result;
        activeConnection = db;
        db.onversionchange = () => { db.close(); forget(db); };
        db.onclose = () => forget(db);
        resolve(db);
      };
    }).catch(local);
    return connection;
  };
  // Storage that keeps refusing writes moves the scores to this tab, from the
  // last committed ledger, so a round can still end: a full quota or a
  // browser's generic storage failure at once, anything else the second time
  // in a row. A single passing failure only puts the move back to retry. Every
  // later operation stays in the tab - a late close of the old handle must not
  // reopen storage beneath the tab's ledger.
  const storageGivesUp = (error) => {
    failedWrites += 1;
    if (!['QuotaExceededError', 'UnknownError'].includes(error?.name) && failedWrites < 2) return false;
    activeConnection = null;
    connection = Promise.resolve(local());
    return true;
  };
  const transact = async (operation, retryConnection = true) => {
    const db = await open();
    // `persistent` says whether the result reached the database: a caller
    // must not report a tab-only result as saved.
    if (!db) return { ...scoreTransition(memory, operation), persistent: false };
    let tx;
    try { tx = db.transaction('scores', 'readwrite'); }
    catch (error) {
      // Closing may precede the close event. No transaction was started here,
      // so opening a fresh handle and retrying once cannot duplicate a write.
      if (error?.name === 'InvalidStateError') {
        forget(db);
        if (retryConnection) return transact(operation, false);
      }
      throw error;
    }
    return new Promise((resolve, reject) => {
      const store = tx.objectStore('scores');
      const request = store.get('ledger');
      let ledger;
      let result;
      let failure;
      let refused = false;     // the ledger refused the operation; storage did not fail
      let settled = false;
      const failed = (error) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (refused || !storageGivesUp(error)) { reject(error); return; }
        try { resolve({ ...scoreTransition(memory, operation), persistent: false }); } catch (refusal) { reject(refusal); }
      };
      const timer = setTimeout(() => {
        failure = new Error('Score update did not finish. Your board is unchanged by this storage failure.');
        try { tx.abort(); } catch { /* already completed */ }
        failed(failure);
      }, timeoutMs);
      request.onsuccess = () => {
        try {
          const stored = request.result;
          // A damaged ledger cannot be read, but Reset replaces it: the one
          // way back that does not mean clearing the site's storage by hand.
          ledger = stored === undefined ? newLedger(legacyScores())
            : operation.type === 'reset' && !validLedger(stored) ? newLedger() : stored;
          result = scoreTransition(ledger, operation);
          if (stored === undefined || result.changed) store.put(ledger, 'ledger');
        } catch (error) { failure = error; refused = true; tx.abort(); }
      };
      tx.oncomplete = () => {
        // Keep receipts, epoch and revision as well as totals. Publish the
        // snapshot only after commit, so aborted writes never reach fallback.
        committed = ledger;
        // A committed ledger holds the v1 totals it was seeded from, so the
        // old copy goes: it only showed pre-reset totals on every load, and
        // seeded a tab-only ledger with them when the database failed.
        if (!retired) {
          retired = true;
          try { retireLegacy(); } catch { /* storage refused; the ledger still wins */ }
        }
        if (settled) return;       // a commit after its deadline: that caller has its answer
        settled = true;
        failedWrites = 0;
        clearTimeout(timer);
        resolve({ ...result, persistent: true });
      };
      // A failed request's error event reaches the transaction before the
      // abort that sets transaction.error, so its own error is the one that
      // names a full quota (WebKit and Gecko fail the put itself).
      tx.onabort = tx.onerror = (event) => failed(failure ?? event?.target?.error ?? tx.error
        ?? new Error('Could not save score.'));
    });
  };
  return {
    read: () => transact({ type: 'read' }),
    record: (id, winner, supersedes = []) => transact({ type: 'record', id, winner, supersedes }),
    undo: (receipts) => transact({ type: 'undo', receipts }),
    reset: () => transact({ type: 'reset' }),
  };
}

import { normalizeScores } from './round-storage.js';

export const SCORE_CHANGE_KEY = 'connect4-chaos.scores.changed.v2';
export const SCORE_DATABASE = 'connect4-chaos.results.v2';
export function resultId() {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}
export function newLedger(legacy = {}) {
  return { version: 2, epoch: resultId(), revision: 0, base: normalizeScores(legacy), results: {} };
}

/** Pure transition, executed inside a single IndexedDB readwrite transaction. */
export function scoreTransition(ledger, operation = { type: 'read' }) {
  if (ledger?.version !== 2 || typeof ledger.epoch !== 'string'
      || !Number.isSafeInteger(ledger.revision) || !ledger.results || !ledger.base) {
    throw new Error('Saved score data is invalid. Reset browser storage to recover it.');
  }
  let changed = false;
  let receipt = null;
  if (operation.type === 'record') {
    const { id, winner } = operation;
    if (typeof id !== 'string' || !/^[a-zA-Z0-9._:-]{1,128}$/.test(id)
        || !['1', '2', 'draw'].includes(String(winner))) throw new Error('Invalid round result.');
    if (Object.hasOwn(ledger.results, id) && ledger.results[id] !== String(winner)) {
      throw new Error('Another tab already recorded a different result for this round. Start a new round.');
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

export function createScoreStore({ indexedDB = globalThis.indexedDB, legacyScores = () => ({}), onWarning = () => {}, timeoutMs = 10_000 } = {}) {
  let connection;
  let activeConnection;
  let memory;
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
      memory = newLedger(legacyScores());
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
  const transact = async (operation, retryConnection = true) => {
    const db = await open();
    if (!db) return scoreTransition(memory, operation);
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
      let result;
      let failure;
      const timer = setTimeout(() => {
        failure = new Error('Score update did not finish. Your board is unchanged by this storage failure.');
        try { tx.abort(); } catch { /* already completed */ }
        reject(failure);
      }, timeoutMs);
      request.onsuccess = () => {
        try {
          const ledger = request.result ?? newLedger(legacyScores());
          result = scoreTransition(ledger, operation);
          if (request.result === undefined || result.changed) store.put(ledger, 'ledger');
        } catch (error) { failure = error; tx.abort(); }
      };
      tx.oncomplete = () => { clearTimeout(timer); resolve(result); };
      tx.onabort = tx.onerror = () => { clearTimeout(timer); reject(failure ?? tx.error ?? new Error('Could not save score.')); };
    });
  };
  return {
    read: () => transact({ type: 'read' }),
    record: (id, winner) => transact({ type: 'record', id, winner }),
    undo: (receipts) => transact({ type: 'undo', receipts }),
    reset: () => transact({ type: 'reset' }),
  };
}

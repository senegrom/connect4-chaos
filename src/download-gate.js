// An explicit gate in front of large downloads, with a progress bar.
//
// The neural opponent is a 73 MB fetch and the biggest exact tables are
// tens of megabytes. Nobody should start that by accident from a select
// box, so the page asks first, remembers a yes, and shows how far along
// the download is instead of a spinner.

const CONSENT_PREFIX = 'connect4-chaos.download.';

// Looked up on first use rather than at import, so the module loads in a
// worker or under Node, where there is no document and no dialog.
let elements = null;

function ui() {
  if (elements) return elements;
  if (typeof document === 'undefined') return null;
  const dialog = document.querySelector('#downloadDialog');
  if (!dialog || typeof dialog.showModal !== 'function') return null;
  elements = {
    dialog,
    title: document.querySelector('#downloadTitle'),
    message: document.querySelector('#downloadMessage'),
    progress: document.querySelector('#downloadProgress'),
    detail: document.querySelector('#downloadDetail'),
    confirm: document.querySelector('#downloadConfirmButton'),
    cancel: document.querySelector('#downloadCancelButton'),
  };
  return elements;
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '';
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${Math.round(bytes / 1e6)} MB`;
  return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
}

function remembered(id) {
  try {
    return localStorage.getItem(CONSENT_PREFIX + id) === 'yes';
  } catch {
    return false;
  }
}

function remember(id) {
  try {
    localStorage.setItem(CONSENT_PREFIX + id, 'yes');
  } catch {
    // Storage may be unavailable; the user simply gets asked next time.
  }
}

// A superseded request must never update or close a newer request's dialog.
let activeDialog = null;

/** Consent is tied to its request and disappears immediately on abort. */
export function requestDownload({ id, title, description, bytes, remember: keep = true, signal }) {
  if (signal?.aborted) return Promise.resolve(false);
  const elements = ui();
  if (!elements || (keep && remembered(id))) return Promise.resolve(true);
  activeDialog?.();
  elements.title.textContent = title;
  elements.message.textContent = description;
  elements.progress.hidden = true;
  elements.detail.textContent = bytes ? `${formatBytes(bytes)}, downloaded once and kept by your browser.` : '';
  elements.confirm.hidden = false;
  elements.confirm.disabled = false;
  elements.cancel.hidden = false;
  elements.cancel.textContent = 'Not now';
  return new Promise((resolve) => {
    let finished = false;
    const finish = (accepted) => {
      if (finished) return;
      finished = true;
      elements.confirm.removeEventListener('click', onConfirm);
      elements.cancel.removeEventListener('click', onCancel);
      elements.dialog.removeEventListener('cancel', onCancel);
      signal?.removeEventListener('abort', close);
      if (activeDialog === close) {
        activeDialog = null;
        if (elements.dialog.open) elements.dialog.close();
      }
      resolve(accepted);
    };
    const close = () => finish(false);
    const onConfirm = () => { if (keep) remember(id); finish(true); };
    const onCancel = (event) => { event?.preventDefault?.(); finish(false); };
    activeDialog = close;
    elements.confirm.addEventListener('click', onConfirm);
    elements.cancel.addEventListener('click', onCancel);
    elements.dialog.addEventListener('cancel', onCancel);
    signal?.addEventListener('abort', close, { once: true });
    if (!elements.dialog.open) elements.dialog.showModal();
    elements.confirm.focus();
  });
}

/** Cancel and Escape dismiss immediately, even during native startup. */
export function showDownloadProgress({ title, note, onCancel = null, signal }) {
  const elements = ui();
  if (!elements || signal?.aborted) return { update() {}, note() {}, close() {} };
  activeDialog?.();
  elements.title.textContent = title;
  elements.message.textContent = note ?? '';
  elements.confirm.hidden = true;
  elements.cancel.hidden = !onCancel;
  elements.cancel.textContent = 'Cancel';
  elements.progress.hidden = false;
  elements.progress.removeAttribute('value');
  elements.detail.textContent = '';
  let closed = false;
  const ownsDialog = () => !closed && activeDialog === close;
  const close = () => {
    if (closed) return;
    closed = true;
    elements.cancel.removeEventListener('click', cancel);
    elements.dialog.removeEventListener('cancel', cancel);
    signal?.removeEventListener('abort', close);
    if (activeDialog === close) {
      activeDialog = null;
      if (elements.dialog.open) elements.dialog.close();
    }
  };
  const cancel = (event) => {
    event?.preventDefault?.();
    if (!ownsDialog()) return;
    close();
    onCancel?.();
  };
  activeDialog = close;
  elements.cancel.addEventListener('click', cancel);
  elements.dialog.addEventListener('cancel', cancel);
  signal?.addEventListener('abort', close, { once: true });
  if (!elements.dialog.open) elements.dialog.showModal();
  if (onCancel) elements.cancel.focus();
  return {
    update(loaded, total, label) {
      if (!ownsDialog()) return;
      if (Number.isFinite(total) && total > 0) {
        elements.progress.max = total;
        elements.progress.value = Math.min(loaded, total);
        const percent = Math.min(100, Math.round((100 * loaded) / total));
        elements.detail.textContent = `${label ?? ''} ${formatBytes(loaded)} of ${formatBytes(total)} (${percent}%)`.trim();
      } else {
        elements.progress.removeAttribute('value');
        elements.detail.textContent = `${label ?? ''} ${formatBytes(loaded)}`.trim();
      }
    },
    note(text) { if (ownsDialog()) elements.message.textContent = text; },
    close,
  };
}

/**
 * Fetches a URL while reporting bytes received. `signal` aborts the
 * transfer. The total is Content-Length unless the transfer is compressed,
 * where Content-Length is the compressed size while the stream yields
 * decompressed bytes; then `expectedBytes`, the file's known size, counts.
 */
export async function fetchWithProgress(url, onProgress, {
  signal = undefined, expectedBytes = 0, retain = true,
} = {}) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`${url.split('/').pop()} returned ${response.status}`);
  const encoded = Boolean(response.headers.get('content-encoding'));
  const length = Number(response.headers.get('content-length')) || 0;
  const total = encoded ? expectedBytes : (length || expectedBytes);
  if (!response.body || typeof response.body.getReader !== 'function') {
    const buffer = await response.arrayBuffer();
    onProgress?.(buffer.byteLength, total || buffer.byteLength);
    return retain ? buffer : null;
  }
  const reader = response.body.getReader();
  // The model's known size lets us fill one allocation instead of retaining
  // every chunk plus a second model-sized buffer at the end. Unexpected
  // lengths still work: retain the filled prefix and use the general path.
  let buffer = retain && Number.isSafeInteger(expectedBytes) && expectedBytes > 0
    ? new Uint8Array(expectedBytes) : null;
  const chunks = [];
  let loaded = 0;
  for (;;) {
    // eslint-disable-next-line no-await-in-loop
    const { done, value } = await reader.read();
    if (done) break;
    if (retain) {
      if (buffer && loaded + value.byteLength <= buffer.length) buffer.set(value, loaded);
      else {
        if (buffer) { chunks.push(buffer.subarray(0, loaded)); buffer = null; }
        chunks.push(value);
      }
    }
    loaded += value.byteLength;
    onProgress?.(loaded, total);
  }
  // A cache warm-up only needs progress; do not retain/concatenate another
  // entire WASM binary while the model is being loaded alongside it.
  if (!retain) {
    onProgress?.(loaded, total || loaded);
    return null;
  }
  if (buffer) {
    onProgress?.(loaded, total || loaded);
    return loaded === buffer.length ? buffer.buffer : buffer.slice(0, loaded).buffer;
  }
  buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }
  onProgress?.(loaded, total || loaded);
  return buffer.buffer;
}

// An explicit gate in front of large downloads, with a progress bar.
//
// The neural opponent is a 75 MB fetch and the biggest exact tables are
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

/**
 * Asks before a large download. Resolves true when the user agrees, or at
 * once when they agreed before. `bytes` is shown so the size is a fact,
 * not a surprise.
 */
export function requestDownload({ id, title, description, bytes, remember: keep = true }) {
  const elements = ui();
  if (!elements) return Promise.resolve(true);
  if (keep && remembered(id)) return Promise.resolve(true);

  elements.title.textContent = title;
  elements.message.textContent = description;
  elements.progress.hidden = true;
  elements.progress.value = 0;
  elements.detail.textContent = bytes ? `${formatBytes(bytes)}, downloaded once and kept by your browser.` : '';
  elements.confirm.hidden = false;
  elements.cancel.hidden = false;
  elements.cancel.textContent = 'Not now';
  elements.confirm.disabled = false;

  return new Promise((resolve) => {
    const finish = (accepted) => {
      elements.confirm.removeEventListener('click', onConfirm);
      elements.cancel.removeEventListener('click', onCancel);
      elements.dialog.removeEventListener('cancel', onCancel);
      if (!accepted && elements.dialog.open) elements.dialog.close();
      resolve(accepted);
    };
    const onConfirm = () => {
      if (keep) remember(id);
      // The dialog stays open to show progress; the caller closes it.
      elements.confirm.hidden = true;
      elements.cancel.hidden = true;
      finish(true);
    };
    const onCancel = (event) => {
      event?.preventDefault?.();
      finish(false);
    };
    elements.confirm.addEventListener('click', onConfirm);
    elements.cancel.addEventListener('click', onCancel);
    elements.dialog.addEventListener('cancel', onCancel);
    if (!elements.dialog.open) elements.dialog.showModal();
    elements.confirm.focus();
  });
}

/**
 * Shows progress in the dialog, opening it if the prompt was skipped. With
 * `onCancel` the dialog offers a Cancel button, and Escape calls it too;
 * without one the modal cannot be dismissed, so callers should pass it.
 */
export function showDownloadProgress({ title, note, onCancel = null }) {
  const elements = ui();
  if (!elements) {
    return { update() {}, note() {}, close() {} };
  }
  elements.title.textContent = title;
  elements.message.textContent = note ?? '';
  elements.confirm.hidden = true;
  elements.cancel.hidden = !onCancel;
  elements.cancel.textContent = 'Cancel';
  elements.progress.hidden = false;
  elements.progress.removeAttribute('value');          // indeterminate until sized
  elements.detail.textContent = '';
  let closed = false;
  const cancel = (event) => {
    event?.preventDefault?.();
    if (!closed) onCancel?.();
  };
  elements.cancel.addEventListener('click', cancel);
  elements.dialog.addEventListener('cancel', cancel);
  if (!elements.dialog.open) elements.dialog.showModal();
  if (onCancel) elements.cancel.focus();
  return {
    update(loaded, total, label) {
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
    note(text) {
      elements.message.textContent = text;
    },
    close() {
      closed = true;
      elements.cancel.removeEventListener('click', cancel);
      elements.dialog.removeEventListener('cancel', cancel);
      if (elements.dialog.open) elements.dialog.close();
    },
  };
}

/**
 * Fetches a URL while reporting bytes received. `signal` aborts the
 * transfer. The total is Content-Length unless the transfer is compressed,
 * where Content-Length is the compressed size while the stream yields
 * decompressed bytes; then `expectedBytes`, the file's known size, counts.
 */
export async function fetchWithProgress(url, onProgress, { signal = undefined, expectedBytes = 0 } = {}) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`${url.split('/').pop()} returned ${response.status}`);
  const encoded = Boolean(response.headers.get('content-encoding'));
  const length = Number(response.headers.get('content-length')) || 0;
  const total = encoded ? expectedBytes : (length || expectedBytes);
  if (!response.body || typeof response.body.getReader !== 'function') {
    const buffer = await response.arrayBuffer();
    onProgress?.(buffer.byteLength, total || buffer.byteLength);
    return buffer;
  }
  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    // eslint-disable-next-line no-await-in-loop
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, total);
  }
  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }
  onProgress?.(loaded, total || loaded);
  return buffer.buffer;
}

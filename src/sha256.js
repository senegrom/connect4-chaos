// SHA-256 as lowercase hex, for checking downloaded tables and the model
// against their release hashes. Web Crypto exists in every secure context and
// in Node 22 and later; a page served over plain HTTP from another host has
// none, and cannot verify anything, so it is told so. `digest` is looked up
// on every call, which lets tests stand in for it.
export async function sha256Hex(bytes, purpose) {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error(`SHA-256 support is unavailable for ${purpose}.`);
  const digest = new Uint8Array(await subtle.digest('SHA-256', bytes));
  return [...digest].map((value) => value.toString(16).padStart(2, '0')).join('');
}

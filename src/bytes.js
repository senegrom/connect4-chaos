// Byte helpers for the binary tables and policies. They live apart from
// exact-table.js because the classic policy decoder uses them, and the
// release gate replays the classic catalog through that decoder: an edit
// here costs a replay, an edit to exact-table.js does not.

/** A Uint8Array over `input`, an ArrayBuffer or any typed array. */
export function bytesFrom(input, label) {
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  }
  throw new TypeError(`${label} data must be an ArrayBuffer or typed array.`);
}

/** `length` bytes from `offset` read as ASCII, for a format's magic. */
export function ascii(bytes, offset, length) {
  let value = '';
  for (let index = 0; index < length; index += 1) {
    value += String.fromCharCode(bytes[offset + index]);
  }
  return value;
}

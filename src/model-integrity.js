import { sha256Hex } from './sha256.js';

/** Model identity is checked before bytes reach a cache, publisher or runtime. */
export class ModelIntegrityError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ModelIntegrityError';
    this.code = 'MODEL_INTEGRITY';
  }
}

export function modelIdentity(manifest) {
  if (!Number.isSafeInteger(manifest?.bytes) || manifest.bytes <= 0
      || typeof manifest.sha256 !== 'string' || !/^[a-f0-9]{64}$/i.test(manifest.sha256)) {
    throw new ModelIntegrityError('The model requires a positive byte count and a SHA-256 digest.');
  }
  return Object.freeze({ bytes: manifest.bytes, sha256: manifest.sha256.toLowerCase() });
}

export async function verifyModelBytes(input, manifest) {
  const expected = modelIdentity(manifest);
  const bytes = input instanceof ArrayBuffer ? new Uint8Array(input)
    : ArrayBuffer.isView(input) ? new Uint8Array(input.buffer, input.byteOffset, input.byteLength)
      : null;
  if (!bytes) throw new TypeError('Model bytes must be an ArrayBuffer or typed array.');
  if (bytes.byteLength !== expected.bytes) {
    throw new ModelIntegrityError(`Model length mismatch: found ${bytes.byteLength}, expected ${expected.bytes}.`);
  }
  const digest = await sha256Hex(bytes, 'model verification');
  if (digest !== expected.sha256) throw new ModelIntegrityError('Model SHA-256 does not match its release.');
  return bytes;
}

import test from 'node:test';
import assert from 'node:assert/strict';
import { fetchWithProgress } from '../src/download-gate.js';

// The model is one object, downloaded straight into a buffer of its known
// size (neural-model-cache.js), so a response of the wrong length must never
// be taken for it or written past it.

function responseFor(bytes, { streamed, headers = {}, onCancel = () => {} }) {
  if (!streamed) {
    return { ok: true, headers: new Headers(headers), body: null,
      arrayBuffer: async () => Uint8Array.from(bytes).buffer };
  }
  let offset = 0;
  return new Response(new ReadableStream({
    pull(controller) {
      if (offset === bytes.length) { controller.close(); return; }
      const next = Math.min(offset + 2, bytes.length);
      controller.enqueue(Uint8Array.from(bytes.slice(offset, next)));
      offset = next;
    },
    cancel: onCancel,
  }), { headers });
}

for (const streamed of [true, false]) {
  const mode = streamed ? 'streamed' : 'buffered';

  test(`${mode} model fills its buffer, with compressed or absent length headers`, async (t) => {
    // Content-Length is the compressed size when the object travels gzipped;
    // progress counts against the model's real length either way.
    for (const headers of [{}, { 'content-encoding': 'gzip', 'content-length': '2' }]) {
      t.mock.method(globalThis, 'fetch', async () => responseFor([1, 2, 3, 4], { streamed, headers }));
      const into = new Uint8Array(4);
      const progress = [];
      const written = await fetchWithProgress('https://example.test/model.onnx',
        (loaded, total) => progress.push([loaded, total]), { expectedBytes: 4, into });
      assert.equal(written, 4);
      assert.deepEqual([...into], [1, 2, 3, 4]);
      assert.deepEqual(progress.at(-1), [4, 4]);
      t.mock.restoreAll();
    }
  });

  test(`${mode} oversized model is refused before anything is written past its size`, async (t) => {
    let cancelled = false;
    t.mock.method(globalThis, 'fetch', async () => responseFor(
      [1, 2, 3, 4, 5, 6, 7, 8, 9], {
        streamed, headers: { 'content-length': '9' }, onCancel: () => { cancelled = true; },
      }));
    // A buffer with room to spare: the expected size, not the buffer, is the limit.
    const into = new Uint8Array(12).fill(99);
    const progress = [];
    await assert.rejects(fetchWithProgress('https://example.test/model.onnx',
      (loaded) => progress.push(loaded), { expectedBytes: 4, into }),
    /model\.onnx is larger than expected/);
    assert.ok(into.subarray(4).every((byte) => byte === 99), 'nothing past the expected size is touched');
    assert.ok(progress.every((loaded) => loaded <= 4));
    if (streamed) assert.equal(cancelled, true, 'stop reading an oversized response');
  });

  test(`${mode} truncated model is refused even when Content-Length matches the response`, async (t) => {
    for (const bytes of [[], [1, 2, 3]]) {
      t.mock.method(globalThis, 'fetch', async () => responseFor(bytes,
        { streamed, headers: { 'content-length': String(bytes.length) } }));
      await assert.rejects(fetchWithProgress('https://example.test/model.onnx', null,
        { expectedBytes: 4, into: new Uint8Array(4) }),
      new RegExp(`model\\.onnx downloaded ${bytes.length} bytes, expected 4`));
      t.mock.restoreAll();
    }
  });
}

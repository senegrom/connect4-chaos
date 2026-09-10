import test from 'node:test';
import assert from 'node:assert/strict';
import { fetchWithProgress } from '../src/download-gate.js';

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

  test(`${mode} model parts fill their regions with compressed or absent length headers`, async (t) => {
    for (const headers of [{}, { 'content-encoding': 'gzip', 'content-length': '2' }]) {
      t.mock.method(globalThis, 'fetch', async (url) => responseFor(
        url.endsWith('part1') ? [1, 2, 3, 4] : [5, 6, 7, 8], { streamed, headers }));
      const into = new Uint8Array(10).fill(99);
      const progress = [[], []];
      const written = await Promise.all([0, 1].map((part) => fetchWithProgress(
        `https://example.test/model.part${part + 1}`,
        (loaded, total) => progress[part].push([loaded, total]),
        { expectedBytes: 4, into, offset: 1 + part * 4 },
      )));
      assert.deepEqual(written, [4, 4]);
      assert.deepEqual([...into], [99, 1, 2, 3, 4, 5, 6, 7, 8, 99]);
      for (const updates of progress) assert.deepEqual(updates.at(-1), [4, 4]);
      t.mock.restoreAll();
    }
  });

  test(`${mode} oversized model parts never overwrite the following region`, async (t) => {
    let cancelled = false;
    t.mock.method(globalThis, 'fetch', async () => responseFor(
      [1, 2, 3, 4, 5, 6, 7, 8, 9], {
        streamed, headers: { 'content-length': '9' }, onCancel: () => { cancelled = true; },
      }));
    const into = new Uint8Array(12).fill(99);
    const progress = [];
    await assert.rejects(fetchWithProgress('https://example.test/model.part1',
      (loaded) => progress.push(loaded), { expectedBytes: 4, into, offset: 2 }),
    /model\.part1 is larger than expected/);
    assert.deepEqual([...into.subarray(0, 2)], [99, 99]);
    assert.ok(into.subarray(6).every((byte) => byte === 99), 'the next part must stay untouched');
    assert.ok(progress.every((loaded) => loaded <= 4));
    if (streamed) assert.equal(cancelled, true, 'stop reading an oversized response');
  });

  test(`${mode} truncated model parts reject even when Content-Length matches the response`, async (t) => {
    for (const bytes of [[], [1, 2, 3]]) {
      t.mock.method(globalThis, 'fetch', async () => responseFor(bytes,
        { streamed, headers: { 'content-length': String(bytes.length) } }));
      await assert.rejects(fetchWithProgress('https://example.test/model.part2', null,
        { expectedBytes: 4, into: new Uint8Array(8), offset: 4 }),
      new RegExp(`model\\.part2 downloaded ${bytes.length} bytes, expected 4`));
      t.mock.restoreAll();
    }
  });

  test(`${mode} mismatched part lengths cannot compensate for each other`, async (t) => {
    t.mock.method(globalThis, 'fetch', async (url) => responseFor(
      url.endsWith('part1') ? [1, 2, 3, 4, 5] : [5, 6, 7], { streamed }));
    const into = new Uint8Array(8);
    const results = await Promise.allSettled([0, 1].map((part) => fetchWithProgress(
      `https://example.test/model.part${part + 1}`, null,
      { expectedBytes: 4, into, offset: part * 4 },
    )));
    assert.deepEqual(results.map((result) => result.status), ['rejected', 'rejected']);
    assert.match(results[0].reason.message, /model\.part1 is larger than expected/);
    assert.match(results[1].reason.message, /model\.part2 downloaded 3 bytes, expected 4/);
  });
}

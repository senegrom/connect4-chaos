import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import test from 'node:test';

const root = new URL('../', import.meta.url);
const site = new URL('https://example.test/connect4-chaos/');
const html = readFileSync(new URL('index.html', root), 'utf8');
const head = html.match(/<head\b[^>]*>([\s\S]*?)<\/head>/i)?.[1] ?? '';

function attributes(tag) {
  return Object.fromEntries(
    [...tag.matchAll(/([\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g)]
      .map(([, key, doubleQuoted, singleQuoted]) => [key, doubleQuoted ?? singleQuoted]),
  );
}

const links = [...head.matchAll(/<link\b[^>]*>/gi)].map(([tag]) => attributes(tag));
const metas = [...head.matchAll(/<meta\b[^>]*>/gi)].map(([tag]) => attributes(tag));

function link(rel) {
  const matches = links.filter((entry) => entry.rel?.split(/\s+/).includes(rel));
  assert.equal(matches.length, 1, `Expected one ${rel} link in the source HTML`);
  return matches[0];
}

// GitHub Pages hosts this app under /connect4-chaos/, not at the origin root.
function assetPath(href, base = site) {
  const url = new URL(href, base);
  assert.equal(url.origin, site.origin, 'Installation resources must be same-origin');
  assert.ok(url.pathname.startsWith(site.pathname), `Resource escaped the app path: ${href}`);
  return url.pathname.slice(site.pathname.length);
}

const crcTable = Uint32Array.from({ length: 256 }, (_, value) => {
  for (let bit = 0; bit < 8; bit++) value = (value >>> 1) ^ ((value & 1) ? 0xedb88320 : 0);
  return value >>> 0;
});

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = (crc >>> 8) ^ crcTable[(crc ^ byte) & 255];
  return (crc ^ 0xffffffff) >>> 0;
}

// Inspect the complete PNG, not just its dimensions: damaged uploads can
// retain a valid IHDR while the compressed pixel data is unusable on iOS.
function pngSize(path) {
  const bytes = readFileSync(new URL(path, root));
  assert.ok(bytes.length >= 45, `Truncated PNG: ${path}`);
  assert.equal(bytes.subarray(0, 8).toString('hex'), '89504e470d0a1a0a', `Not a PNG: ${path}`);
  assert.equal(bytes.toString('ascii', 12, 16), 'IHDR', `Missing PNG header: ${path}`);
  assert.equal(bytes.readUInt32BE(8), 13, `Invalid PNG header length: ${path}`);
  const width = bytes.readUInt32BE(16);
  const height = bytes.readUInt32BE(20);
  assert.ok(width > 0 && width <= 1024 && height > 0 && height <= 1024);
  assert.deepEqual([...bytes.subarray(24, 29)], [8, 2, 0, 0, 0], `Expected non-interlaced opaque RGB PNG: ${path}`);
  let offset = 8;
  let ended = false;
  const imageData = [];
  while (offset < bytes.length) {
    assert.ok(offset + 12 <= bytes.length, `Truncated PNG chunk: ${path}`);
    const length = bytes.readUInt32BE(offset);
    const end = offset + 8 + length;
    assert.ok(end + 4 <= bytes.length, `Truncated PNG chunk payload: ${path}`);
    const type = bytes.toString('ascii', offset + 4, offset + 8);
    assert.equal(crc32(bytes.subarray(offset + 4, end)), bytes.readUInt32BE(end), `Invalid ${type} checksum: ${path}`);
    if (type === 'IDAT') imageData.push(bytes.subarray(offset + 8, end));
    offset = end + 4;
    if (type === 'IEND') {
      assert.equal(length, 0);
      assert.equal(offset, bytes.length, `Trailing data after PNG end: ${path}`);
      ended = true;
      break;
    }
  }
  assert.ok(ended && imageData.length > 0, `Incomplete PNG: ${path}`);
  const stride = width * 3 + 1;
  const raw = inflateSync(Buffer.concat(imageData), { maxOutputLength: stride * height + 1 });
  assert.equal(raw.length, stride * height, `Incomplete PNG pixels: ${path}`);
  for (let row = 0; row < height; row++) assert.ok(raw[row * stride] <= 4, `Invalid PNG row filter: ${path}`);
  return `${width}x${height}`;
}

test('source HTML explicitly declares the 180px iPhone Home Screen icon', () => {
  const icon = link('apple-touch-icon');
  assert.equal(icon.sizes, '180x180');
  assert.equal(pngSize(assetPath(icon.href)), icon.sizes);
  assert.equal(pngSize('apple-touch-icon.png'), '180x180');
});

test('web app identity, launch URL and scope stay inside the GitHub Pages project', () => {
  const manifestLink = link('manifest');
  const manifestUrl = new URL(manifestLink.href, site);
  const manifest = JSON.parse(readFileSync(new URL(assetPath(manifestLink.href), root), 'utf8'));
  assert.equal(manifest.name, 'Connect 4: Chaos Edition');
  assert.equal(manifest.short_name, 'Connect 4');
  assert.equal(manifest.display, 'standalone');
  for (const field of ['start_url', 'scope']) {
    assert.equal(new URL(manifest[field], manifestUrl).href, site.href, field);
  }
  // Unlike start_url/scope, a relative id is resolved against the start URL's
  // ORIGIN (W3C manifest #id-member). "./" would identify every project as "/".
  const start = new URL(manifest.start_url, manifestUrl);
  const id = new URL(manifest.id, `${start.origin}/`);
  id.hash = '';
  assert.equal(id.href, site.href, 'App ID must not collide with other GitHub Pages projects');
});

test('manifest supplies real 192px, 512px and maskable PNGs of the declared sizes', () => {
  const manifestLink = link('manifest');
  const manifestUrl = new URL(manifestLink.href, site);
  const manifest = JSON.parse(readFileSync(new URL(assetPath(manifestLink.href), root), 'utf8'));
  const available = new Set();
  for (const icon of manifest.icons) {
    assert.equal(icon.type, 'image/png');
    assert.equal(pngSize(assetPath(icon.src, manifestUrl)), icon.sizes);
    for (const purpose of (icon.purpose ?? 'any').split(/\s+/)) {
      available.add(`${icon.sizes}:${purpose}`);
    }
  }
  for (const required of ['192x192:any', '512x512:any', '512x512:maskable']) {
    assert.ok(available.has(required), `Missing installation icon: ${required}`);
  }
});

test('legacy Apple standalone mode and short app title are present in the source', () => {
  for (const name of ['mobile-web-app-capable', 'apple-mobile-web-app-capable']) {
    assert.equal(metas.find((meta) => meta.name === name)?.content, 'yes');
  }
  assert.equal(metas.find((meta) => meta.name === 'apple-mobile-web-app-title')?.content, 'Connect 4');
});

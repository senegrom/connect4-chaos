import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
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

function pngSize(path) {
  const bytes = readFileSync(new URL(path, root));
  assert.ok(bytes.length >= 45, `Truncated PNG: ${path}`);
  assert.equal(bytes.subarray(0, 8).toString('hex'), '89504e470d0a1a0a', `Not a PNG: ${path}`);
  assert.equal(bytes.toString('ascii', 12, 16), 'IHDR', `Missing PNG header: ${path}`);
  assert.equal(bytes.toString('ascii', bytes.length - 8, bytes.length - 4), 'IEND', `Incomplete PNG: ${path}`);
  return `${bytes.readUInt32BE(16)}x${bytes.readUInt32BE(20)}`;
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
  for (const field of ['id', 'start_url', 'scope']) {
    assert.equal(new URL(manifest[field], manifestUrl).href, site.href, field);
  }
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

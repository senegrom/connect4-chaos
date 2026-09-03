import assert from 'node:assert/strict';
import test from 'node:test';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { assetUrls } from '../src/neural-runtime.js';

// A relative specifier in a dynamic import resolves against the module, not
// the page, so './assets/...' from src/ silently looked inside src/assets
// and the opponent hung waiting for a file that was never there.
test('the runtime, model and metadata resolve to files that exist', () => {
  const urls = assetUrls();
  for (const [name, url] of Object.entries(urls)) {
    if (name === 'base') continue;
    assert.ok(url.includes('/assets/neural/'), `${name} should live in assets/neural: ${url}`);
    assert.ok(!url.includes('/src/assets/'), `${name} resolved inside src/: ${url}`);
    assert.ok(existsSync(fileURLToPath(url)), `${name} is missing on disk: ${url}`);
  }
});

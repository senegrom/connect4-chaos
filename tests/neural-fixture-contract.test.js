import test from 'node:test';
import assert from 'node:assert/strict';
import { cp, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));

// The browser suites swap one page module for a stand-in: the neural runtime,
// the neural client or the whole neural request. A name the real page imports
// from that module but the stand-in lacks stops the module graph from
// linking, so the app never starts and the suite reports only a timeout far
// from the cause. On 2026-09-23 one new import did that to every browser.
// Linking the page against each stand-in here keeps them in step with it.
const STAND_INS = [
  { suite: 'neural-worker-regressions.py', name: 'CPU_FIXTURE', replaces: 'neural-runtime.js' },
  { suite: 'failure-browser-regressions.py', name: 'NEURAL', replaces: 'neural-client.js' },
  { suite: 'browser-regressions.py', name: 'NEURAL_STUB', replaces: 'neural-client.js' },
  { suite: 'handoff-browser-regressions.py', name: 'NEURAL', replaces: 'neural-app.js' },
];
// app.js links neural-client.js statically and loads neural-app.js on demand,
// and the worker links the runtime, so these three reach every stand-in.
const ENTRIES = ['app.js', 'neural-app.js', 'neural-worker.js'];

async function standIn(suite, name) {
  const source = await readFile(join(ROOT, 'scripts', suite), 'utf8');
  const body = source.match(new RegExp(`^${name} = """([\\s\\S]*?)"""`, 'm'))?.[1];
  assert.ok(body, `${suite} should define ${name}`);
  return body;
}

for (const { suite, name, replaces } of STAND_INS) {
  test(`${suite} ${name} provides everything the page imports from ${replaces}`, async (t) => {
    const directory = await mkdtemp(join(tmpdir(), 'neural-stand-in-'));
    t.after(() => rm(directory, { recursive: true, force: true }));
    await cp(join(ROOT, 'src'), join(directory, 'src'), { recursive: true });
    await writeFile(join(directory, 'package.json'), '{"type":"module"}');
    await writeFile(join(directory, 'src', replaces), await standIn(suite, name));
    for (const entry of ENTRIES) {
      if (entry === replaces) continue;
      try {
        await import(pathToFileURL(join(directory, 'src', entry)).href);
      } catch (error) {
        // Evaluating page code under Node fails on the first DOM access; only
        // a failure to link is the stand-in's fault.
        assert.doesNotMatch(String(error?.message), /does not provide an export named/,
          `${entry} does not link against ${suite}'s ${name}`);
      }
    }
    if (replaces === 'neural-app.js') {
      // app.js takes these from its on-demand import, which cannot fail to
      // link: a missing name would only arrive as undefined.
      const app = await readFile(join(ROOT, 'src/app.js'), 'utf8');
      const wanted = [...app.matchAll(/const \{([^}]+)\} = await import\('\.\/neural-app\.js'\)/g)]
        .flatMap(([, names]) => names.split(',').map((part) => part.trim().split(/\s*:\s*/)[0]).filter(Boolean));
      assert.ok(wanted.length > 0, 'app.js should load neural-app.js on demand');
      const module = await import(pathToFileURL(join(directory, 'src', replaces)).href);
      for (const exported of wanted) assert.equal(typeof module[exported], 'function', `${name} lacks ${exported}`);
    }
  });
}

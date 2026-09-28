// A page left open across a deploy still loads code after startup: the
// modules it imports on first use, and the whole module graph of each AI
// worker it starts. Those come from the newer site. A lazy module then runs
// against this page's copies of the modules they share, and a new worker
// speaks its newer protocol to this older page. The deploy stamps its build
// into the page and into build.json; before loading more code the page
// compares the two, and asks for a reload rather than mixing builds.

export const RELOAD_MESSAGE = 'Connect 4 was updated while this page was open. Reload the page to continue; your round resumes where it is.';
const CHECK_INTERVAL_MS = 60_000;
const CHECK_TIMEOUT_MS = 5_000;

export class OutdatedBuildError extends Error {
  constructor() {
    super(RELOAD_MESSAGE);
    this.name = 'OutdatedBuildError';
  }
}

async function deployedBuild() {
  const response = await fetch(new URL('../build.json', import.meta.url), {
    cache: 'no-store', signal: AbortSignal.timeout?.(CHECK_TIMEOUT_MS),
  });
  return response.ok ? (await response.json())?.build : null;
}

export function createBuildCheck({
  build = globalThis.document?.querySelector('meta[name="connect4-build"]')?.content ?? 'dev',
  currentBuild = deployedBuild,
  now = () => Date.now(),
} = {}) {
  // The checkout served for development, or a test, carries no stamp and
  // has nothing to compare: it loads as it always has.
  const stamped = build !== 'dev';
  let outdated = false;
  let checkedAt = -Infinity;
  let checking = null;
  function isOutdated() {
    if (!stamped || outdated) return Promise.resolve(outdated);
    if (now() - checkedAt < CHECK_INTERVAL_MS) return Promise.resolve(false);
    checking ??= Promise.resolve()
      .then(currentBuild)
      .then((current) => { outdated = typeof current === 'string' && current !== build; },
        () => { /* offline, or refused: keep running the build that is loaded */ })
      .then(() => {
        checkedAt = now();
        checking = null;
        return outdated;
      });
    return checking;
  }
  return {
    stamped,
    isOutdated,
    async ensureCurrent() {
      if (await isOutdated()) throw new OutdatedBuildError();
    },
  };
}

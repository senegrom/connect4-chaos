// A page left open across a deploy still loads code after startup: the
// modules it imports on first use, and the whole module graph of each AI
// worker it starts. Those come from the newer site, whatever build their URL
// names. A lazy module then runs beside this page's older modules, and a new
// worker speaks its newer protocol to this older page. The deploy stamps its
// build into this module and into build.json; before loading more code the
// page compares the two, and asks for a reload rather than mixing builds.
//
// A page of the newer deploy cannot pick up older modules from the HTTP cache
// (Pages lets a browser keep them ten minutes): scripts/build-site.sh puts the
// build into every module URL, and no older deploy was ever asked for those.
// A worker imports its whole graph as it starts, right after the check.

export const RELOAD_MESSAGE = 'Connect 4 was updated while this page was open. Reload the page to continue; your round resumes where it is.';
const CHECK_INTERVAL_MS = 60_000;
const CHECK_TIMEOUT_MS = 5_000;
// scripts/build-site.sh writes the site's content digest here.
const BUILD = 'dev';

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
  return response.ok ? response.json() : null;
}

/** Renews the cached copies of what build.json lists - the deployed page - and
 * settles once every renewal has. A reload revalidates the page anyway, but a
 * new tab, or a relaunch of the installed app, which has no reload control,
 * can take the old page from the cache for up to ten minutes. */
function renewCache(deployed) {
  const paths = Array.isArray(deployed?.refresh) ? deployed.refresh : [];
  return Promise.allSettled(paths.filter((path) => typeof path === 'string' && !path.includes('..'))
    .map((path) => fetch(new URL(`../${path}`, import.meta.url), {
      cache: 'reload', signal: AbortSignal.timeout?.(CHECK_TIMEOUT_MS),
    })));
}

export function createBuildCheck({
  build = BUILD,
  currentBuild = deployedBuild,
  onOutdated = renewCache,
  reloadPage = () => globalThis.location.reload(),
  now = () => Date.now(),
} = {}) {
  // The checkout served for development, or a test, carries no stamp and
  // has nothing to compare: it loads as it always has.
  const stamped = build !== 'dev';
  let outdated = false;
  let checkedAt = -Infinity;
  let checking = null;
  let renewed = Promise.resolve();
  function isOutdated() {
    if (!stamped || outdated) return Promise.resolve(outdated);
    if (now() - checkedAt < CHECK_INTERVAL_MS) return Promise.resolve(false);
    checking ??= Promise.resolve()
      .then(currentBuild)
      .then((deployed) => {
        outdated = typeof deployed?.build === 'string' && deployed.build !== build;
        if (outdated) renewed = Promise.resolve(deployed).then(onOutdated).catch(() => {});
      }, () => { /* offline, or refused: keep running the build that is loaded */ })
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
    /** Reloads into the deployed build once the renewals have settled. */
    async reload() {
      await renewed;
      reloadPage();
    },
  };
}

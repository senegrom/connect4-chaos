// Asks for the cross-origin isolation that multi-threaded WebAssembly needs.
//
// The headers come from a service worker (../cross-origin-isolation-worker.js)
// because this host cannot send them. The first visit registers it without
// interrupting play; the next navigation can use several neural evaluation
// threads. Every failure here is silent and harmless:
// without isolation it simply runs on one thread, as it always has.

const ATTEMPTED = 'connect4-chaos.coi-attempted';
const WORKER_URL = new URL('../cross-origin-isolation-worker.js', import.meta.url);
const WORKER_SCOPE = new URL('../', import.meta.url);

async function unregisterIsolationWorker() {
  const registrations = await navigator.serviceWorker?.getRegistrations?.() ?? [];
  // Other applications on this origin own their registrations. Match both
  // our exact scope and our script, including a worker still installing.
  const owned = registrations.filter((registration) => (
    registration.scope === WORKER_SCOPE.href
    && [registration.active, registration.waiting, registration.installing]
      .some((worker) => worker?.scriptURL === WORKER_URL.href)
  ));
  await Promise.all(owned.map((registration) => registration.unregister()));
}

/**
 * Returns whether the page is cross-origin isolated, registering the worker
 * for the next navigation if it is not. `?coi=off` removes only this app's
 * isolation worker, which is the way out if it ever misbehaves.
 */
export async function enableCrossOriginIsolation() {
  if (typeof window === 'undefined' || !navigator.serviceWorker) return false;
  if (new URL(window.location.href).searchParams.get('coi') === 'off') {
    await unregisterIsolationWorker();
    return false;
  }
  if (window.crossOriginIsolated) return true;
  if (!window.isSecureContext) return false;
  try {
    // Registration is idempotent, but there is no reason to ask twice a tab.
    if (sessionStorage.getItem(ATTEMPTED)) return false;
    sessionStorage.setItem(ATTEMPTED, '1');
  } catch {
    return false;                     // storage unavailable; stay on one thread
  }
  try {
    // Registered, never reloaded. Isolation is decided when a page is
    // navigated to, so this visit keeps running on one thread and the next
    // one starts isolated - which is worth more than interrupting whatever
    // is on the board to gain it a few seconds earlier.
    await navigator.serviceWorker.register(WORKER_URL, { scope: WORKER_SCOPE.href });
  } catch {
    // Registration refused (private window, policy, an unsupported browser).
  }
  return false;
}

// Asks for the cross-origin isolation that multi-threaded WebAssembly needs.
//
// The headers come from a service worker (../cross-origin-isolation-worker.js)
// because this host cannot send them, so the first visit registers it and
// reloads once; from then on the page is isolated and the neural opponent
// evaluates on several threads. Every failure here is silent and harmless:
// without isolation it simply runs on one thread, as it always has.

const ATTEMPTED = 'connect4-chaos.coi-attempted';
const WORKER_URL = new URL('../cross-origin-isolation-worker.js', import.meta.url);

async function unregisterAll() {
  const registrations = await navigator.serviceWorker?.getRegistrations?.() ?? [];
  await Promise.all(registrations.map((registration) => registration.unregister()));
}

/**
 * Returns whether the page is cross-origin isolated, registering the worker
 * and reloading once if it is not. `?coi=off` removes the worker instead,
 * which is the way out if it ever misbehaves.
 */
export async function enableCrossOriginIsolation() {
  if (typeof window === 'undefined' || !navigator.serviceWorker) return false;
  if (new URL(window.location.href).searchParams.get('coi') === 'off') {
    await unregisterAll();
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
    await navigator.serviceWorker.register(WORKER_URL, { scope: './' });
  } catch {
    // Registration refused (private window, policy, an unsupported browser).
  }
  return false;
}

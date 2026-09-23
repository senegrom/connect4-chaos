/** Whether the module at moduleUrl is the script node was asked to run.
 * Proof scripts double as libraries, so each runs main() only when invoked
 * directly. Comparing path.resolve(process.argv[1]) with the module's own path
 * fails whenever the script is reached through a junction or symlink: node
 * resolves the entry module through the link, argv[1] keeps the link, main()
 * is skipped and a verifier exits 0 having verified nothing. Both sides go
 * through the native realpath instead, which also settles letter case on
 * case-insensitive file systems.
 */
import { realpathSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

function realPath(path) {
  try {
    return realpathSync.native(path);
  } catch {
    return realpathSync(path);
  }
}

export function isEntryPoint(moduleUrl, argv = process.argv) {
  if (typeof argv?.[1] !== 'string' || argv[1].length === 0) return false;
  try {
    return realPath(argv[1]) === realPath(fileURLToPath(moduleUrl));
  } catch {
    // An entry path that does not exist (node -e, a REPL) is not this module.
    return false;
  }
}

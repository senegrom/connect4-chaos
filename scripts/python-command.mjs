// Resolves a working Python 3 interpreter once per process. Windows ships
// Microsoft Store aliases named python/python3 that exit 9009 instead of
// running anything, so every candidate is probed with --version and the
// first real interpreter wins; the py launcher covers stock Windows
// installs. Set PYTHON to put an explicit interpreter first.
import { spawnSync } from 'node:child_process';

let resolved;

export function pythonCommand() {
  if (resolved !== undefined) return resolved;
  const candidates = [];
  if (process.env.PYTHON) candidates.push({ command: process.env.PYTHON, args: [] });
  candidates.push(
    { command: 'python3', args: [] },
    { command: 'python', args: [] },
    { command: 'py', args: ['-3'] },
  );
  for (const candidate of candidates) {
    const probe = spawnSync(candidate.command, [...candidate.args, '--version'], {
      encoding: 'utf8',
    });
    if (probe.status === 0) {
      resolved = Object.freeze({ command: candidate.command, args: candidate.args });
      return resolved;
    }
  }
  throw new Error(
    'No working Python 3 interpreter found (tried PYTHON, python3, python, py -3).',
  );
}

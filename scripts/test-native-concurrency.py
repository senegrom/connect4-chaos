#!/usr/bin/env python3
"""Exercise production discovery and resolution with 1 and 4 threads.

--sanitize enables ThreadSanitizer; an unavailable sanitizer is an error,
never a passing/skipped race check. No committed tables are modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(command):
    proc = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if proc.returncode:
        raise RuntimeError(f"{' '.join(map(str, command))}\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def build(source, binary, sanitize):
    flags = ['-std=c++20', '-pthread', '-g', '-O1']
    if os.name == 'nt':
        flags.append('-static')   # as scripts/native-toolchain.mjs: MinGW's DLLs can be shadowed
    if sanitize:
        flags += ['-fsanitize=thread', '-fno-omit-frame-pointer', '-fno-pie', '-no-pie']
    run(['g++', *flags, str(source), '-o', str(binary)])


def agree(label, results):
    # Explicit, not assert: `python -O` strips asserts, and this check is the
    # whole point of the job.
    if any(result != results[0] for result in results[1:]):
        raise RuntimeError(f'{label} differs across thread counts: {results}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    results = []
    with tempfile.TemporaryDirectory(prefix='connect4-concurrency-') as temp:
        directory = Path(temp)
        for solver in ('paired', 'layered'):
            binary = directory / solver
            build(ROOT / 'native' / f'perfect-chaos-{solver}.cpp', binary, args.sanitize)
            outcomes = []
            for threads in (1, 4):
                output = directory / f'{solver}-{threads}'
                text = run([str(binary), '--rows', '3', '--columns', '3', '--connect', '3',
                            '--threads', str(threads), '--output', str(output)])
                result = json.loads(text.strip().splitlines()[-1])
                outcomes.append({key: result[key] for key in ('states', 'wins', 'draws', 'losses', 'rootValue')})
                print(f"PASS {solver}: {threads} threads, sanitizer={args.sanitize}, {outcomes[-1]}", flush=True)
            agree(solver, outcomes)
            results.append(outcomes[0])
        agree('the paired and layered solvers', results)

        # The complete solver writes the shipped certificates with up to 16
        # threads: its atomic rank sweep and parallel action pass must give
        # the same solution and byte-identical certificates on any count.
        binary = directory / 'complete'
        build(ROOT / 'native' / 'perfect-chaos-complete.cpp', binary, args.sanitize)
        certificates = []
        for threads in (1, 4):
            prefix = directory / f'complete-{threads}'
            text = run([str(binary), '--rows', '4', '--columns', '4', '--connect', '3',
                        '--threads', str(threads), '--emit-policy', str(prefix)])
            lines = [json.loads(line) for line in text.strip().splitlines() if line.startswith('{')]
            files = sorted(directory.glob(f'complete-{threads}-role*.bin'))
            if len(files) != 2:
                raise RuntimeError(f'the complete solver wrote {len(files)} certificates, not 2')
            digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]
            certificates.append({'lines': [{key: value for key, value in line.items() if key != 'elapsedMs'}
                                           for line in lines], 'sha256': digests})
            print(f"PASS complete: {threads} threads, sanitizer={args.sanitize}, "
                  f"root {lines[0]['rootValue']}, certificates {[digest[:12] for digest in digests]}", flush=True)
        agree('the complete solver', certificates)
        # And they are the certificates the site ships for this board.
        manifest = json.loads((ROOT / 'data' / 'perfect-chaos-complete' / 'manifest.json').read_text(encoding='utf-8'))
        shipped = [entry['sha256'] for entry in sorted(manifest['policies'], key=lambda entry: entry['role'])
                   if (entry['rows'], entry['columns'], entry['connect']) == (4, 4, 3)]
        if certificates[0]['sha256'] != shipped:
            raise RuntimeError(f'the 4x4 Connect-3 certificates differ from the shipped ones: '
                               f'{certificates[0]["sha256"]} != {shipped}')
    print('Every solver agrees with itself across thread counts, the two layouts with each other, '
          'and the complete solver with the shipped 4x4 Connect-3 certificates.', flush=True)


if __name__ == '__main__': main()

#!/usr/bin/env python3
"""Exercise production discovery and resolution with 1 and 4 threads.

--sanitize enables ThreadSanitizer; an unavailable sanitizer is an error,
never a passing/skipped race check. No committed tables are modified.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(command):
    proc = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if proc.returncode:
        raise RuntimeError(f"{' '.join(map(str, command))}\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    results = []
    with tempfile.TemporaryDirectory(prefix='connect4-concurrency-') as temp:
        directory = Path(temp)
        for solver in ('paired', 'layered'):
            source = ROOT / 'native' / f'perfect-chaos-{solver}.cpp'
            binary = directory / solver
            flags = ['-std=c++20', '-pthread', '-g', '-O1']
            if args.sanitize:
                flags += ['-fsanitize=thread', '-fno-omit-frame-pointer', '-fno-pie', '-no-pie']
            run(['g++', *flags, str(source), '-o', str(binary)])
            baseline = None
            for threads in (1, 4):
                output = directory / f'{solver}-{threads}'
                text = run([str(binary), '--rows', '3', '--columns', '3', '--connect', '3',
                            '--threads', str(threads), '--output', str(output)])
                result = json.loads(text.strip().splitlines()[-1])
                comparable = {key: result[key] for key in ('states', 'wins', 'draws', 'losses', 'rootValue')}
                if baseline is not None:
                    assert baseline == comparable, (baseline, comparable)
                baseline = comparable
                print(f"PASS {solver}: {threads} threads, sanitizer={args.sanitize}, {comparable}", flush=True)
            results.append(baseline)
        assert results[0] == results[1], results
    print('Both independent solver layouts agree across thread counts.', flush=True)


if __name__ == '__main__': main()

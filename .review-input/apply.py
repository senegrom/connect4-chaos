"""Temporary, hash-pinned source assembly; never updates a branch."""
from pathlib import Path
import base64
import hashlib
import json
import os
import subprocess
import sys
import urllib.request

BASE = 'de0e9962e16349cfb02805574b6c1c5ea8834fc4'
BASE_TREE = 'add8bca883006a28a68de62f2f41bae9aa7ffe9c'
EXPECTED = json.loads(Path('.review-input/expected.json').read_text())


def original(path):
    base = os.environ.get('REVIEW_LOCAL_BASE', BASE)
    return subprocess.check_output(['git', 'show', f'{base}:{path}']).decode()


def write(path, data):
    Path(path).write_text(data)


def apply():
    p = 'neural/pair_tables.py'
    s = original(p)
    s = s.replace('import mmap\n', 'from collections import OrderedDict\nfrom contextlib import ExitStack\nfrom dataclasses import dataclass\nimport mmap\nimport os\n')
    s = s.replace('import random\n', 'import random\nimport re\n')
    s = s.replace('GROUP_WORDS = 2048\n', 'GROUP_WORDS = 2048\nMAX_MAPPED_BLOCKS = 32\n')
    write(p, s[:s.index('class PairTable:')] + Path('.review-input/pair-class.txt').read_text())

    p = 'neural/build_dataset.py'
    s = original(p).replace("    chaos = mode != 'classic'", "    if mode not in ('classic', 'chaos'):\n        raise ValueError(\"Dataset mode must be classic or chaos\")\n    chaos = mode == 'chaos'")
    a = s.index('    table = PairTable('); b = s.index('\n\ndef main()', a)
    old = s[a:b].splitlines()
    s = s[:a] + old[0].replace('table = PairTable(', 'with PairTable(').replace('chaos=chaos)', 'chaos=chaos) as table:') + '\n        table.validate()\n' + '\n'.join('    ' + line if line else '' for line in old[1:]) + '\n' + s[b:]
    write(p, s)

    p = 'neural/distill.py'
    s = original(p)
    a = s.index('    holdout = {tag.strip()', s.index('def load_shards')); b = s.index('    window = ', a)
    s = s[:a] + '    holdout, holdout_shapes = training_holdouts()\n' + s[b:]
    a = s.index('def load_shards(')
    helper = '''def training_holdouts(spec=None):
    """One holdout parser for both replay staging and the training loader."""
    if spec is None:
        spec = os.environ.get("DISTILL_HOLDOUT_CONFIGS", "")
    holdout = {tag.strip() for tag in spec.split(",") if tag.strip()}
    if "all" in holdout:
        raise ValueError("A training holdout must name specific configurations")
    shapes = [shape for tag in sorted(holdout) for shape in (parse_shape_spec(tag) or [])]
    return holdout, shapes


'''
    write(p, s[:a] + helper + s[a:])

    p = 'neural/modal_app.py'
    s = original(p); a = s.index('def learn('); b = s.index('\n\n@app.function', a); part = s[a:b]
    part = part.replace('    from neural.data_split import SPLIT_VERSION', '    from neural.distill import filtered_chunks, training_holdouts\n\n    if type(replay_window) is not int or replay_window < 0:\n        raise ValueError("replay_window must be a nonnegative integer")\n    holdout_spec = os.environ.get("DISTILL_HOLDOUT_CONFIGS", "")\n    _, holdout_shapes = training_holdouts(holdout_spec)')
    part = part.replace('    skipped = 0\n', '    skipped = 0\n    excluded = 0\n')
    part = part.replace('''            if payload.get("split_version") == SPLIT_VERSION and "validation" in payload:
                positions += int((~payload["validation"].bool()).sum())
            else:
                positions += len(payload["wdl"])
            staged_shards += 1''', '''            try:
                if payload.get("source") != "selfplay":
                    raise ValueError("Replay archive does not contain a self-play shard")
                # Count exactly what load_shards can consume, including whole-board
                # exclusions, legacy position hashing, newest-tail order and the cap.
                eligible = sum(len(chunk["planes"]) for chunk in filtered_chunks(
                    payload, holdout_shapes, limit=replay_window - positions, newest_first=True))
            finally:
                del payload  # release the mmap before removing an excluded shard
            if not eligible:
                excluded += 1
                out.unlink()
                continue
            positions += eligible
            staged_shards += 1''')
    part = part.replace('DISTILL_ENTROPY_BONUS=str(entropy_bonus),', 'DISTILL_ENTROPY_BONUS=str(entropy_bonus), DISTILL_HOLDOUT_CONFIGS=holdout_spec,')
    part = part.replace('"skipped_shards": skipped,', '"skipped_shards": skipped, "excluded_shards": excluded,')
    part = part.replace('key=lambda path: path.stat().st_mtime, reverse=True)', 'key=lambda path: (-path.stat().st_mtime, path.name))')
    write(p, s[:a] + part + s[b:])

    for p in ('neural/test_rereview.py', 'neural/test_optimizations.py'):
        s = original(p).replace('''            def __init__(self, *_args, **_kwargs): pass
            def sample_state''', '''            def __init__(self, *_args, **_kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *_exc): pass
            def validate(self): pass
            def sample_state''')
        write(p, s)
    p = '.github/workflows/ci.yml'
    write(p, original(p).replace('          python -m neural.test_search_settings\n', '          python -m neural.test_search_settings\n          python -m neural.test_pair_tables\n          python -m neural.test_replay_staging\n'))


def verify():
    for path, expected in EXPECTED.items():
        data = Path(path).read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if actual != expected:
            raise ValueError(f'{path}: source hash mismatch: {actual} != {expected}')
    print(f'Verified all {len(EXPECTED)} exact locally tested source files.', flush=True)


def stage():
    if os.environ['GITHUB_REPOSITORY'] != 'senegrom/connect4-chaos':
        raise ValueError('unexpected repository')
    def post(endpoint, payload):
        request = urllib.request.Request(
            'https://api.github.com/repos/senegrom/connect4-chaos/git/' + endpoint,
            data=json.dumps(payload).encode(), method='POST', headers={
                'Authorization': 'Bearer ' + os.environ['GH_TOKEN'],
                'Accept': 'application/vnd.github+json', 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    entries = []
    for path, sha in EXPECTED.items():
        result = post('blobs', {'content': base64.b64encode(Path(path).read_bytes()).decode(), 'encoding': 'base64'})
        if result['sha'] != sha:
            raise ValueError(f'{path}: uploaded blob hash mismatch')
        entries.append({'path': path, 'mode': '100644', 'type': 'blob', 'sha': sha})
    tree = post('trees', {'base_tree': BASE_TREE, 'tree': entries})
    print('FINAL_TREE=' + tree['sha'], flush=True)
    Path('validated-tree.json').write_text(json.dumps({'base': BASE, 'tree': tree['sha'], 'files': EXPECTED}, indent=2))


apply()
verify()
if '--stage' in sys.argv:
    stage()

"""Validate and apply the readable review diff; optionally create an unreferenced commit.

No branch is updated here. The reviewed commit is published separately after
all validation jobs pass. This helper and its patch files are temporary.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

EXPECTED = '7f27aa4d3cf2b224a08f076a7c18afe9482c601c83a156b5864988de46116269'
parts = [Path(f'.review-source.{i}.patch') for i in range(8)]
data = b''.join(path.read_bytes() for path in parts)
data = data.replace(b'\n diff --git ', b'\ndiff --git ')
actual = hashlib.sha256(data).hexdigest()
if actual != EXPECTED:
    raise SystemExit(f'Readable candidate checksum mismatch: {actual}')
subprocess.run(['git', 'apply', '--check', '-'], input=data, check=True)
subprocess.run(['git', 'apply', '-'], input=data, check=True)
for path in parts + [Path('.github/workflows/review-maintenance.yml'), Path(__file__)]:
    path.unlink()
print(f'Applied readable review candidate {actual}', flush=True)

if '--object' in sys.argv:
    def git(*args):
        return subprocess.check_output(['git', *args]).decode().strip()
    parent = git('rev-parse', 'HEAD')
    base = git('rev-parse', 'HEAD^{tree}')
    subprocess.run(['git', 'add', '-A'], check=True)
    paths = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z']).decode().strip('\0').split('\0')
    entries = []
    for name in paths:
        path = Path(name)
        entry = {'path': name, 'mode': '100644', 'type': 'blob'}
        if path.exists():
            entry['mode'] = git('ls-files', '-s', '--', name).split()[0]
            entry['content'] = path.read_text(encoding='utf-8')
        else:
            entry['sha'] = None
        entries.append(entry)
    api = f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}"
    def post(endpoint, payload):
        request = urllib.request.Request(api + endpoint, data=json.dumps(payload).encode(), method='POST', headers={
            'Authorization': 'Bearer ' + os.environ['GH_TOKEN'],
            'Accept': 'application/vnd.github+json', 'Content-Type': 'application/json',
            'X-GitHub-Api-Version': '2022-11-28'})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    tree = post('/git/trees', {'base_tree': base, 'tree': entries})['sha']
    commit = post('/git/commits', {'tree': tree, 'parents': [parent],
        'message': 'Fix full-code review: atomic solvers, bounded AI loading, authorised artifacts, transactional results and training safeguards'})['sha']
    metadata = {'parent': parent, 'tree': tree, 'commit': commit, 'patchSha256': actual, 'files': paths}
    Path('review-candidate.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)

"""Apply readable, checksum-pinned fixes; optionally create a tested commit object.

The branch is published separately after every validation job passes. This
helper, its source patches and the staging workflow are removed by the fix.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

parts = [Path(f'.review-source.{i}.patch') for i in range(8)]
data = b''.join(path.read_bytes() for path in parts)
data = data.replace(b'\n diff --git ', b'\ndiff --git ')
followup = Path('.review-source.8.patch')
stages = [(data, '7f27aa4d3cf2b224a08f076a7c18afe9482c601c83a156b5864988de46116269'),
          (followup.read_bytes(), 'c687a4e04c795f7aa150c5c3acbe2f6e136e259568000b40884db8d7232e462e')]
for patch, expected in stages:
    actual = hashlib.sha256(patch).hexdigest()
    if actual != expected:
        raise SystemExit(f'Readable candidate checksum mismatch: {actual}')
    subprocess.run(['git', 'apply', '--check', '-'], input=patch, check=True)
    subprocess.run(['git', 'apply', '-'], input=patch, check=True)
    print(f'Applied readable review stage {actual}', flush=True)
parts.append(followup)
for path in parts + [Path('.github/workflows/review-maintenance.yml'), Path(__file__)]:
    path.unlink()

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
    metadata = {'parent': parent, 'tree': tree, 'commit': commit, 'patchSha256': [digest for _, digest in stages], 'files': paths}
    Path('review-candidate.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)

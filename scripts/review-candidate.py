"""Apply checksum-pinned, validated fixes and prepare normal source objects.

Workflow changes are returned separately for the connected GitHub writer;
the Actions token is not used to edit workflow files or advance a branch.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import urllib.error

parts = [Path(f'.review-source.{i}.patch') for i in range(8)]
data = b''.join(path.read_bytes() for path in parts)
data = data.replace(b'\n diff --git ', b'\ndiff --git ')
stages = [(data, '7f27aa4d3cf2b224a08f076a7c18afe9482c601c83a156b5864988de46116269')]
for index, digest in [(8, '00ab65598ae47d11f0e55606a4eaac6af0405fba8c9ffde2c67f52ad21ca5f4b'),
                      (9, '2f58e0a0a547bb958e36810ddca8287432c16d193323663d09a0cd1122ad420a')]:
    path = Path(f'.review-source.{index}.patch')
    parts.append(path)
    stages.append((path.read_bytes(), digest))
for patch, expected in stages:
    actual = hashlib.sha256(patch).hexdigest()
    if actual != expected:
        raise SystemExit(f'Readable candidate checksum mismatch: {actual}')
    subprocess.run(['git', 'apply', '--check', '-'], input=patch, check=True)
    subprocess.run(['git', 'apply', '-'], input=patch, check=True)
    print(f'Applied readable review stage {actual}', flush=True)
for path in parts + [Path('.github/workflows/review-maintenance.yml'), Path(__file__)]:
    path.unlink()

if '--object' in sys.argv:
    def git(*args):
        return subprocess.check_output(['git', *args]).decode().strip()
    parent = git('rev-parse', 'HEAD')
    base = git('rev-parse', 'HEAD^{tree}')
    subprocess.run(['git', 'add', '-A'], check=True)
    paths = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z']).decode().strip('\0').split('\0')
    entries, workflows = [], []
    for name in paths:
        path = Path(name)
        entry = {'path': name, 'mode': '100644', 'type': 'blob'}
        if path.exists():
            entry['mode'] = git('ls-files', '-s', '--', name).split()[0]
            entry['content'] = path.read_text(encoding='utf-8')
        else:
            entry['sha'] = None
        (workflows if name.startswith('.github/workflows/') else entries).append(entry)
    api = f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}"
    def post(endpoint, payload):
        request = urllib.request.Request(api + endpoint, data=json.dumps(payload).encode(), method='POST', headers={
            'Authorization': 'Bearer ' + os.environ['GH_TOKEN'],
            'Accept': 'application/vnd.github+json', 'Content-Type': 'application/json',
            'X-GitHub-Api-Version': '2022-11-28'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'GitHub {endpoint}: {error.code}: {error.read().decode()[:2000]}') from error
    tree = post('/git/trees', {'base_tree': base, 'tree': entries})['sha']
    metadata = {'parent': parent, 'tree': tree, 'workflowEntries': workflows,
                'patchSha256': [digest for _, digest in stages], 'files': paths}
    Path('review-candidate.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)

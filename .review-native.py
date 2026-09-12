"""Additional Darwin conformance fix found by the real native build gate."""
from pathlib import Path
import re
import subprocess

BASE = 'b8caed9e53530c6e0e0f3abcb8f7c499b4ace0ab'
pattern = re.compile(r'std::atomic_ref<const std::uint(?:64|8)_t>\(\s*([^()]+?)\)\s*\.load\((std::memory_order_\w+)\)')
changes = {}
for name, count in [('perfect-chaos-layered.cpp', 4), ('perfect-chaos-paired.cpp', 3), ('perfect-chaos-complete.cpp', 2)]:
    path = Path('native') / name
    original = subprocess.check_output(['git', 'show', f'{BASE}:{path.as_posix()}'])
    if path.read_bytes() != original:
        raise RuntimeError(f'Refusing to overwrite changed native source: {path}')
    source, replaced = pattern.subn(lambda match: 'connect4::atomicLoad(' + match[1].strip() + ', ' + match[2] + ')', original.decode('utf8'))
    if replaced != count or source.count('#include <atomic>') != 1:
        raise RuntimeError(f'Unexpected atomic read sites in {path}: {replaced}')
    source = source.replace('#include <atomic>', '#include "atomic-load.hpp"\n\n#include <atomic>')
    changes[path] = source
for path, source in changes.items():
    path.write_text(source, encoding='utf8')
    print('Portable atomic reads:', path)

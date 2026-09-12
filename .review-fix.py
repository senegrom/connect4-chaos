from pathlib import Path
import ast
import json
import re
import subprocess

NEW_FILES = {name: Path(name).read_text(encoding="utf8") for name in ['scripts/native-toolchain.mjs', 'tests/native-toolchain.test.mjs', '.github/workflows/native-portability.yml', 'neural/test_search_quality.py', 'neural/test_search_settings.py', 'docs/REVIEW_REGRESSIONS.md']}

BASE = 'b8caed9e53530c6e0e0f3abcb8f7c499b4ace0ab'
BUILDERS = [
    'scripts/perfect-classic.mjs', 'scripts/perfect-classic-policy.mjs',
    'scripts/perfect-classic-shards.mjs', 'scripts/perfect-chaos-native.mjs',
    'scripts/perfect-chaos-prefix.mjs', 'scripts/perfect-chaos-complete.mjs',
    'tests/perfect-chaos-classification.test.js', 'tests/perfect-chaos-incremental-repair.test.js',
    'tests/perfect-chaos-policy-partition.test.js', 'tests/perfect-chaos-policy-slice.test.js',
    'tests/perfect-chaos-layered.test.js', 'tests/perfect-chaos-paired.test.js',
    'tests/perfect-chaos-remote-lookup.test.js',
]

def original(path):
    expected = subprocess.check_output(['git', 'show', f'{BASE}:{path}'])
    actual = Path(path).read_bytes()
    if actual != expected:
        raise RuntimeError(f'Refusing to overwrite changed source: {path}')
    return actual.decode('utf8')


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f'Expected exactly one replacement: {old!r}; found {source.count(old)}')
    return source.replace(old, new)


def quality_patch(source):
    helpers = '''def unique_budgets(budgets):
    """Validate and deduplicate once, including one-shot iterables."""
    values = tuple(budgets)
    if not values or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                         for value in values):
        raise ValueError("Search budgets must be nonnegative integers, with at least one budget")
    return tuple(dict.fromkeys(values))


def evaluation_budgets(sims, *, directory):
    unique_budgets((sims,))
    if sims == 0:
        return (0,)
    return unique_budgets((0, 32, sims, 2 * sims if directory else 512))


'''
    source = replace_once(source, 'def held_out_shards(shard_dir):', helpers + 'def held_out_shards(shard_dir):')
    source = replace_once(source, '    pools = {name: {budget: [0, 0] for budget in budgets}',
                          '    budgets = unique_budgets(budgets)\n    pools = {name: {budget: [0, 0] for budget in budgets}')
    source = replace_once(source, '        counted = next(iter(pool.values()))[1]\n        if not counted:',
                          '        if not any(count for _wrong, count in pool.values()):')
    source = replace_once(source,
        '        parts = [f"{label(budget)} {wrong / counted:.4f}" for budget, (wrong, _n) in pool.items()]',
        '        parts = [f"{label(budget)} {wrong / count:.4f}" if count else f"{label(budget)} n/a"\n'
        '                 for budget, (wrong, count) in pool.items()]')
    source = replace_once(source,
        '        lines.append(f"pooled {name:8s} {\'  \'.join(parts)}  ({counted} positions)")',
        '        counts = {count for _wrong, count in pool.values()}\n'
        '        count_note = (f"{next(iter(counts))} positions" if len(counts) == 1 else\n'
        '                      ", ".join(f"{label(budget)}: {count} positions"\n'
        '                                for budget, (_wrong, count) in pool.items()))\n'
        '        lines.append(f"pooled {name:8s} {\'  \'.join(parts)}  ({count_note})")')
    source = replace_once(source, '    net = load(model_path, device)\n    if Path(target).is_dir():',
                          '    directory = Path(target).is_dir()\n'
                          '    budgets = evaluation_budgets(sims, directory=directory)\n'
                          '    net = load(model_path, device)\n    if directory:')
    source = replace_once(source, '        budgets = (0,) if sims == 0 else (0, 32, sims, 2 * sims)\n', '')
    return replace_once(source, '    for budget in (0, 32, sims, 512):', '    for budget in budgets:')


def mcts_patch(source):
    source = replace_once(source, 'import os\n', 'import math\nimport os\n')
    helpers = '''def search_configuration():
    """Snapshot the Python settings whose values are baked into CUDA kernels."""
    if (isinstance(C_PUCT, bool) or not isinstance(C_PUCT, (int, float))
            or not math.isfinite(C_PUCT) or C_PUCT < 0):
        raise ValueError("C_PUCT must be finite and nonnegative")
    if not isinstance(Q_SEED, bool):
        raise ValueError("Q_SEED must be a boolean")
    return float(C_PUCT), Q_SEED


'''
    source = replace_once(source, 'def bucket(width: int) -> int:', helpers + 'def bucket(width: int) -> int:')
    source = replace_once(source,
        '    def __init__(self, games: int, sims: int, device, max_connect: int = 10, any_chaos=True):\n        capacity = sims + 2',
        '    def __init__(self, games: int, sims: int, device, max_connect: int = 10, any_chaos=True,\n'
        '                 settings=None):\n'
        '        self.c_puct, self.q_seed = search_configuration() if settings is None else settings\n'
        '        capacity = sims + 2')
    source = replace_once(source, '        u = C_PUCT * self.prior[index]', '        u = self.c_puct * self.prior[index]')
    source = replace_once(source, '        if q_logits is None or not Q_SEED:', '        if q_logits is None or not self.q_seed:')
    source = replace_once(source, '                 history_capacity: int = HISTORY_CAPACITY):',
                          '                 history_capacity: int = HISTORY_CAPACITY, *, settings=None):')
    source = replace_once(source,
        '        self.forest = Forest(games, sims, self.device, max_connect, any_chaos)',
        '        self.settings = search_configuration() if settings is None else settings\n'
        '        self.forest = Forest(games, sims, self.device, max_connect, any_chaos, self.settings)')
    source = replace_once(source,
        '    key = (id(net), id(forward), games, sims, str(torch.device(device)), max_connect, any_chaos,\n'
        '           MAX_DEPTH, history_capacity)',
        '    settings = search_configuration()\n'
        '    # Python scalars and branches are fixed at capture. A changed setting\n'
        '    # must never reuse an older graph; eager forests freeze the same values.\n'
        '    key = (id(net), id(forward), games, sims, str(torch.device(device)), max_connect, any_chaos,\n'
        '           MAX_DEPTH, history_capacity, settings, bool(USE_GRAPHS))')
    return replace_once(source,
        '        ws = Workspace(net, forward, games, sims, device, max_connect, any_chaos, history_capacity)',
        '        ws = Workspace(net, forward, games, sims, device, max_connect, any_chaos, history_capacity,\n'
        '                       settings=settings)')


def apply():
    changes = dict(NEW_FILES)
    for path in BUILDERS:
        source = original(path)
        if source.count("'-static'") < 1:
            raise RuntimeError(f'Missing expected static linker flag: {path}')
        source = source.replace("'-static'", '...nativeLinkFlags()')
        # Replace the old unqualified comments with a pointer to host selection.
        source = re.sub(r'(?m)^([ \t]*)// Statically linked: a dynamic libstdc\+\+ resolves[^\n]*\n'
                        r'\1// first on PATH[^\n]*\n\1// std::ofstream[^\n]*\n',
                        r'\1// Host-specific runtime linking is centralized in native-toolchain.mjs.\n', source)
        source = re.sub(r'(?m)^([ \t]*)// -static keeps the WinLibs[^\n]*\n\1// older libstdc\+\+[^\n]*\n',
                        r'\1// Host-specific runtime linking is centralized in native-toolchain.mjs.\n', source)
        helper = './native-toolchain.mjs' if path.startswith('scripts/') else '../scripts/native-toolchain.mjs'
        line = f"import {{ nativeLinkFlags }} from '{helper}';\n"
        if source.startswith('#!'):
            first, rest = source.split('\n', 1)
            source = first + '\n' + line + rest
        else:
            source = line + source
        changes[path] = source
    changes['neural/search_quality.py'] = quality_patch(original('neural/search_quality.py'))
    changes['neural/gpu_mcts.py'] = mcts_patch(original('neural/gpu_mcts.py'))
    ci = original('.github/workflows/ci.yml')
    ci = replace_once(ci, '          python -m neural.test_export_onnx\n',
                      '          python -m neural.test_export_onnx\n'
                      '          python -m neural.test_search_quality\n'
                      '          python -m neural.test_search_settings\n')
    ci = replace_once(ci, '  native-concurrency:\n',
                      '  native-portability:\n    uses: ./.github/workflows/native-portability.yml\n\n  native-concurrency:\n')
    ci = replace_once(ci, 'needs: [test, training-regressions, native-concurrency,',
                      'needs: [test, training-regressions, native-concurrency, native-portability,')
    changes['.github/workflows/ci.yml'] = ci
    path = '.github/workflows/verify-perfect-classic-policies.yml'
    classic = original(path)
    classic = replace_once(classic, '            scripts/perfect-classic-policy.mjs \\\n',
                           '            scripts/perfect-classic-policy.mjs \\\n            scripts/native-toolchain.mjs \\\n')
    changes[path] = classic
    path = 'tests/release-gate.test.mjs'
    gate = original(path)
    changes[path] = replace_once(gate, "'scripts/perfect-classic-policy.mjs', 'src/perfect-classic-policy.js'",
                                "'scripts/perfect-classic-policy.mjs', 'scripts/native-toolchain.mjs', 'src/perfect-classic-policy.js'")
    path = 'docs/PERFECT_CHAOS.md'
    doc = original(path)
    old = 'g++ -O3 -std=c++20 -static -o'
    if old not in doc:
        raise RuntimeError('Missing documented native compile command')
    changes[path] = doc.replace(old, 'g++ -O3 -std=c++20 -o') + (
        '\nNative build note: the direct compiler example above uses ordinary linking. '
        'With Windows MinGW, add `-static` to avoid loading an unrelated C++ runtime DLL '
        'from PATH. The Node build wrappers select this flag only on Windows through '
        '`scripts/native-toolchain.mjs`; do not use full static linking on macOS.\n')
    # Validate every replacement in memory before writing any source file.
    for path, source in changes.items():
        if path.endswith('.py'):
            ast.parse(source, filename=path)
    for path, source in changes.items():
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(source, encoding='utf8')
    print(json.dumps({'base': BASE, 'changed': sorted(changes)}, indent=2))


if __name__ == '__main__':
    apply()

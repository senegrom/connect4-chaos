"""Runs the loop driver's main() against stub Modal functions.

Usage: python -m neural.test_modal_loop

Exercises the real control flow - spawn, collect, publish, arena every
ARENA_EVERY generations, stop file - without touching Modal or the GPU, so
a missing initialisation or a bad branch fails here instead of at 22:30.
"""
import sys
import types
from pathlib import Path

import tempfile

ROOT = Path(tempfile.mkdtemp(prefix="loopcheck-"))
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / "gpu-replay").mkdir(exist_ok=True)

calls = {"actor": 0, "learner": 0, "arena": 0}


class Call:
    def __init__(self, kind, payload):
        self.kind, self.payload, self.object_id = kind, payload, f"fc-{kind}-{calls[kind]}"
        self.polls = 0

    def get(self, timeout=None):
        self.polls += 1
        if self.polls < 2:
            raise TimeoutError
        return self.payload


class Stub:
    def __init__(self, kind):
        self.kind = kind

    def spawn(self, *args, **kwargs):
        calls[self.kind] += 1
        index = calls[self.kind]
        if self.kind == "actor":
            payload = {"exit": 0, "shard": f"shard-{index}.pt.gz", "seconds": 1.0,
                       "gpu": "H100", "out": "self-play: 8192 games, 260000 positions, 1s -> x"}
        elif self.kind == "learner":
            payload = {"exit": 0, "model": f"big{100 + index}-abc.pt", "seconds": 1.0,
                       "gpu": "H100", "replay_positions": 4_000_000, "replay_shards": 30,
                       "staging_seconds": 1.0, "lines": ["train samples: 1 (exact 1, replay 0)"]}
        else:
            payload = {"exit": 0, "seconds": 1.0, "out": "arena A vs B: 55.0% over 24 games"}
        return Call(self.kind, payload)


class Volume:
    def read_file(self, name):
        return [b"stub"]

    def listdir(self, path):
        # Six earlier generations, so the arena can fire on the first
        # multiple of ARENA_EVERY after a restart.
        return [types.SimpleNamespace(path=f"models/big{n}-old.pt") for n in range(1, 7)]


modal = types.ModuleType("modal")
modal.Function = types.SimpleNamespace(
    from_name=lambda app, name: Stub({"selfplay_gpu": "actor", "learn": "learner",
                                      "arena": "arena"}[name]))
modal.Volume = types.SimpleNamespace(from_name=lambda name: Volume())
sys.modules["modal"] = modal

import os
os.environ["C4_NEURAL_ROOT"] = str(ROOT)
sys.argv = ["modal_loop.py", "big0-seed.pt", "1", "2", "8192", "10", "64", "4e-4",
            "4000000", "100000", "64", "2", "2"]      # arena every 2 generations, lag 2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural import modal_loop

# Stop after a handful of generations by planting the stop file mid-run.
original_fetch = modal_loop.fetch_shard
generations = {"count": 0}


def fetch(shard_gz):
    return ROOT / "gpu-replay" / shard_gz[:-3], 1000


def mirror(name):
    generations["count"] += 1
    if generations["count"] >= 6:
        (ROOT / "modal-loop.stop").write_text("stop")
    return ROOT / name


modal_loop.fetch_shard = fetch
modal_loop.mirror_model = mirror
(ROOT / "modal-loop.stop").unlink(missing_ok=True)
(ROOT / "modal-loop.log").unlink(missing_ok=True)
modal_loop.time.sleep = lambda seconds: None

modal_loop.main()

log = (ROOT / "modal-loop.log").read_text(encoding="utf-8")
arenas = [line for line in log.splitlines() if "arena" in line]
print(f"generations published: {generations['count']}, arena events: {len(arenas)}")
for line in arenas:
    print("  ", line.split(" ", 1)[1][:100])
assert generations["count"] >= 6, "loop did not publish generations"
assert arenas, "no arena was ever run"
assert "loop end" in log
print("LOOP FLOW OK")

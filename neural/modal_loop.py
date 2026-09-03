"""Modal training loop driver (stage 2: actors AND learner on H100s).

Keeps K `selfplay_gpu` calls in flight with the newest checkpoint in
models/ on the Volume, and one `learn` call training the next generation
from that checkpoint over the exact shards plus the newest replay window.
Finished shards are mirrored into neural/gpu-replay and checkpoints into
neural/modal-models (so local evaluation and serving keep working), and
neural/current-model.txt points at the newest mirrored checkpoint.
Stop with neural/modal-loop.stop (in-flight calls are collected first).
Log: neural/modal-loop.log.

Usage: python -m neural.modal_loop <init model name on Volume> <first gen> [K=3]
       [games=4096] [steps=6000] [batch=1024] [lr=4e-4] [window=4000000]
       [min_new_positions=2000000] [sims] [arena_every] [arena_lag] [shapes]
"""
import os
import re
import sys
import time
from pathlib import Path

import modal

# Local mirror root (logs, replay mirror, checkpoint mirror, current-model.txt).
ROOT = Path(os.environ.get("C4_NEURAL_ROOT", "E:/tmp-claude/connect4/neural"))
REPLAY = ROOT / "gpu-replay"
MODELS = ROOT / "modal-models"
LOG = ROOT / "modal-loop.log"
STOP = ROOT / "modal-loop.stop"
INIT_MODEL = sys.argv[1]
GEN = int(sys.argv[2])
K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
GAMES = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 6000
BATCH = int(sys.argv[6]) if len(sys.argv) > 6 else 1024
LR = float(sys.argv[7]) if len(sys.argv) > 7 else 4e-4
WINDOW = int(sys.argv[8]) if len(sys.argv) > 8 else 4_000_000
# Pacing: a generation starts only after MIN_NEW fresh self-play positions
# arrived since the previous one was spawned (the first is exempt), so the
# learner re-sees each position about (steps*batch*0.75)/MIN_NEW times
# instead of spinning on stale data; idle learner time is unbilled.
MIN_NEW = int(sys.argv[9]) if len(sys.argv) > 9 else 2_000_000
# Simulations per move in the actors: 0 keeps the two-ply lookahead, >0 runs
# batched PUCT search (better targets, one network evaluation per simulation).
SIMS = int(sys.argv[10]) if len(sys.argv) > 10 else 0
# Every ARENA_EVERY generations the newest model plays the one ARENA_LAG
# generations older, over boards with no exact table. That is the only
# measurement of progress where the tables cannot reach.
ARENA_EVERY = int(sys.argv[11]) if len(sys.argv) > 11 else 5
ARENA_LAG = int(sys.argv[12]) if len(sys.argv) > 12 else 5
ARENA_GAMES = 24
ARENA_SIMS = 32
# "all" is every playable board from 4x1 to 10x10 in both rule sets (412 of
# them). The network's heads are size-agnostic, so it should see the whole
# space rather than a fixed handful.
SHAPES = sys.argv[13] if len(sys.argv) > 13 else "all"
OUT_SUBDIR = "replay-gpu"

actor_fn = modal.Function.from_name("connect4-chaos", "selfplay_gpu")
learn_fn = modal.Function.from_name("connect4-chaos", "learn")
arena_fn = modal.Function.from_name("connect4-chaos", "arena")
vol = modal.Volume.from_name("connect4-tables")


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# Local shard mirror: the learner reads the Volume, so the mirror is only a
# backup. Shards stay gzipped (a twentieth of the disk) and the newest
# MIRROR_KEEP are kept; the Volume holds the full history either way.
MIRROR_KEEP = int(os.environ.get("C4_MIRROR_KEEP", "400"))


def fetch_shard(shard_gz):
    data = b"".join(vol.read_file(f"{OUT_SUBDIR}/{shard_gz}"))
    out = REPLAY / shard_gz
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(out)
    stale = sorted(REPLAY.glob("*.pt.gz"), key=lambda p: p.stat().st_mtime, reverse=True)[MIRROR_KEEP:]
    for path in stale:
        try:
            path.unlink()
        except OSError:
            pass
    return out, len(data)


def mirror_model(name):
    data = b"".join(vol.read_file(f"models/{name}"))
    MODELS.mkdir(parents=True, exist_ok=True)
    out = MODELS / name
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(out)
    (ROOT / "current-model.txt").write_text(str(out).replace("/", chr(92)) + "\n")
    return out


# A dropped connection says nothing about the job on the other side: the
# call keeps running and will finish. Only a real error means the work is
# gone, so polling distinguishes the two and keeps waiting through outages.
TRANSIENT = ("connectionerror", "getaddrinfo", "connection lost", "connection reset",
             "streamterminated", "broken pipe", "unavailable", "deadline",
             "timed out", "temporarily unavailable", "eof occurred")


def is_transient(exc):
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in TRANSIENT)


def published_history():
    """Checkpoint names already on the Volume, oldest first. Without this a
    restart forgets every earlier generation and the next arena waits for
    ARENA_LAG fresh ones."""
    try:
        names = [Path(entry.path).name for entry in vol.listdir("models")]
    except Exception as exc:
        log(f"could not list models/: {type(exc).__name__}: {str(exc)[:120]}")
        return []
    numbered = []
    for name in names:
        match = re.match(r"big(\d+)-", name)
        if match:
            numbered.append((int(match.group(1)), name))
    return [name for _generation, name in sorted(numbered)]


def main():
    REPLAY.mkdir(parents=True, exist_ok=True)
    model = INIT_MODEL
    gen = GEN
    seed_base = (int(time.time()) % 10_000_000) * 100
    actors = {}
    learner = None
    arena = None
    published = published_history()   # so a restart does not delay the next arena
    spawned = finished = 0
    new_positions = None          # None = first generation, no pacing
    waiting_logged = False
    log(f"loop start init={model} gen={gen} K={K} games={GAMES} steps={STEPS} batch={BATCH} "
        f"lr={LR} window={WINDOW} minNew={MIN_NEW} sims={SIMS} seedBase={seed_base}")
    while True:
        stopping = STOP.exists()
        if not stopping:
            ready = new_positions is None or new_positions >= MIN_NEW
            if learner is None and not ready and not waiting_logged:
                log(f"learner pacing: {new_positions} of {MIN_NEW} fresh positions since gen {gen - 1}")
                waiting_logged = True
            if learner is None and ready:
                try:
                    call = learn_fn.spawn(gen, model, STEPS, BATCH, LR, 0.75, WINDOW)
                    learner = (call, gen, model, time.time())
                    log(f"learner spawned {call.object_id} gen={gen} init={model} "
                        f"(fresh positions since last spawn: {new_positions})")
                    new_positions = 0
                    waiting_logged = False
                except Exception as exc:
                    log(f"learner spawn failed: {type(exc).__name__}: {str(exc)[:200]}; retry in 60 s")
                    time.sleep(60)
            while len(actors) < K:
                try:
                    spawned += 1
                    seed = seed_base + spawned
                    call = actor_fn.spawn(model, GAMES, SHAPES, seed, OUT_SUBDIR, SIMS)
                except Exception as exc:
                    log(f"actor spawn failed: {type(exc).__name__}: {str(exc)[:200]}; retry in 60 s")
                    time.sleep(60)
                    break
                actors[call.object_id] = (call, seed, model, time.time())
                log(f"actor spawned {call.object_id} seed={seed} model={model}")
        for cid, (call, seed, used, t0) in list(actors.items()):
            try:
                result = call.get(timeout=0)
            except TimeoutError:
                continue
            except Exception as exc:
                if is_transient(exc):
                    log(f"actor {cid}: {type(exc).__name__} while polling; still tracked")
                    continue
                log(f"actor {cid} failed: {type(exc).__name__}: {str(exc)[:200]}")
                del actors[cid]
                continue
            del actors[cid]
            if result.get("exit") == 0 and result.get("shard"):
                try:
                    out, size = fetch_shard(result["shard"])
                except Exception as exc:
                    log(f"actor {cid} fetch failed: {type(exc).__name__}: {str(exc)[:200]}")
                    continue
                finished += 1
                summary = (result.get("out") or "").strip().splitlines()
                match = re.search(r"games, (\d+) positions", result.get("out") or "")
                if match and new_positions is not None:
                    new_positions += int(match.group(1))
                log(f"actor {cid} done {result['seconds']}s on {result.get('gpu')} -> {out.name} "
                    f"({size / 1e6:.1f} MB gz) {summary[-1] if summary else ''}")
            else:
                log(f"actor {cid} exit={result.get('exit')} err={(result.get('err') or '')[-300:]!r}")
                time.sleep(30)
        if learner is not None:
            call, lgen, init, t0 = learner
            try:
                result = call.get(timeout=0)
            except TimeoutError:
                result = None
            except Exception as exc:
                if is_transient(exc):
                    log(f"learner {call.object_id}: {type(exc).__name__} while polling; still tracked")
                    result = None
                else:
                    log(f"learner {call.object_id} failed: {type(exc).__name__}: "
                        f"{str(exc)[:200]}; retry in 120 s")
                    learner = None
                    time.sleep(120)
                    result = None
            if result is not None:
                learner = None
                if result.get("exit") == 0 and result.get("model"):
                    model = result["model"]
                    gen = lgen + 1
                    try:
                        local = mirror_model(model)
                    except Exception as exc:
                        local = f"(mirror failed: {type(exc).__name__}: {str(exc)[:120]})"
                    published.append(model)
                    if (ARENA_EVERY and arena is None and len(published) > ARENA_LAG
                            and lgen % ARENA_EVERY == 0):
                        older = published[-1 - ARENA_LAG]
                        try:
                            call = arena_fn.spawn(model, older, ARENA_GAMES, ARENA_SIMS, "", 7)
                            arena = (call, model, older)
                            log(f"arena spawned {call.object_id}: {model} vs {older}")
                        except Exception as exc:
                            log(f"arena spawn failed: {type(exc).__name__}: {str(exc)[:150]}")
                    log(f"learner gen {lgen} done {result['seconds']}s on {result.get('gpu')} "
                        f"(staging {result.get('staging_seconds')}s, replay {result.get('replay_positions')} "
                        f"positions / {result.get('replay_shards')} shards) -> models/{model}; mirrored {local}")
                    for line in result.get("lines", []):
                        log(f"  gen {lgen} {line}")
                else:
                    log(f"learner gen {lgen} exit={result.get('exit')} err={(result.get('err') or '')[-400:]!r}; "
                        f"retry in 120 s")
                    time.sleep(120)
        if arena is not None:
            call, newer, older = arena
            try:
                outcome = call.get(timeout=0)
            except TimeoutError:
                outcome = None
            except Exception as exc:
                if is_transient(exc):
                    log(f"arena: {type(exc).__name__} while polling; still tracked")
                    outcome = None
                else:
                    log(f"arena failed: {type(exc).__name__}: {str(exc)[:150]}")
                    arena, outcome = None, None
            if outcome is not None:
                arena = None
                if outcome.get("exit") == 0:
                    for line in (outcome.get("out") or "").strip().splitlines():
                        log(f"  {line.strip()}")
                else:
                    log(f"arena exit={outcome.get('exit')} {(outcome.get('err') or '')[-200:]!r}")

        if stopping and not actors and learner is None and arena is None:
            break
        time.sleep(10)
    log(f"loop end: actors spawned {spawned}, finished {finished}, next gen {gen}, model {model}")


if __name__ == "__main__":
    main()

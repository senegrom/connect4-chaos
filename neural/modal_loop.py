"""Modal training loop driver (stage 2: actors AND learner on H100s).

Keeps K `selfplay_gpu` calls in flight with the newest checkpoint in
models/ on the Volume, and one `learn` call training the next generation
from that checkpoint over the exact shards plus the newest replay window.
The Volume is the record: every shard and checkpoint lives there and the
learner reads it there. Mirroring them to <root>/gpu-replay and
<root>/modal-models is opt-in (C4_MIRROR=1) - it costs about 60 GB of local
disk per 50-generation block, and an export needs one checkpoint, which
`modal volume get connect4-tables models/<name>` fetches on demand. With
mirroring on, <root>/current-model.txt points at the newest mirrored
checkpoint. <root> is C4_NEURAL_ROOT (default
E:/tmp-claude/connect4-tools/neural) and is created if missing.
Stop with <root>/modal-loop.stop (in-flight calls are collected first).
Log: <root>/modal-loop.log.
Known terminal Modal failures release their tracked slot; connection outages
keep the existing call. A stop request drains calls without submitting replacements.
So does a role (actors, learner, arena) failing C4_MAX_FAILURES times in a row
(default 3), instead of paying for replacements of work that keeps failing.
Every spawned call is journaled in <root>/modal-loop.calls.json until it is
collected; a driver that starts with calls in the journal reattaches to them
(a learner only for the generation it starts at; any other is cancelled).
The initial checkpoint must be on the Volume before anything is spawned.

Usage: python -m neural.modal_loop <init model name on Volume> <first gen> [K=3]
       [games=4096] [steps=6000] [batch=1024] [lr=4e-4] [window=4000000]
       [min_new_positions=2000000] [sims] [arena_every] [arena_lag] [shapes]
       [target_sims] [target_share] [entropy_bonus=0] [q_seed=1] [replay_fraction=0.75]
       [policy_target=visits|gumbel] [root_value_weight=0] [exact_subdir=datasets-v3]
       [until_gen=0]
"""
import json
import os
from neural.training_config import DEFAULT_SIMS, parse_shape_spec, validate_selfplay
import re
import threading
import sys
import time
from pathlib import Path

import modal

# Local root: the log and stop file always; the replay and checkpoint
# mirrors only when asked for. The Volume holds everything either way, so
# the default keeps a 50-generation block from writing 60 GB to this disk.
ROOT = Path(os.environ.get("C4_NEURAL_ROOT", "E:/tmp-claude/connect4-tools/neural"))
MIRROR = os.environ.get("C4_MIRROR", "0") == "1"
REPLAY = ROOT / "gpu-replay"
MODELS = ROOT / "modal-models"
LOG = ROOT / "modal-loop.log"
STOP = ROOT / "modal-loop.stop"
# Every call spawned and not yet collected, rewritten whenever that set
# changes: a driver that crashes or is killed leaves the IDs behind, and the
# next start reattaches to them instead of paying for a second set.
JOURNAL = ROOT / "modal-loop.calls.json"
ROOT.mkdir(parents=True, exist_ok=True)
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
# Every actor uses PUCT; the removed two-ply mode is not a valid budget.
SIMS = int(sys.argv[10]) if len(sys.argv) > 10 else DEFAULT_SIMS
# Every ARENA_EVERY generations the newest model plays the ancestor
# ARENA_LAG successful learner steps behind it, across every playable board.
# Explicit lineage, not all files in models/, defines those predecessors.
ARENA_EVERY = int(sys.argv[11]) if len(sys.argv) > 11 else 5
ARENA_LAG = int(sys.argv[12]) if len(sys.argv) > 12 else 5
ARENA_GAMES = 6            # per board
ARENA_SIMS = 32
# "all" includes every UI-supported board and the narrower training boards,
# with Connect 3 through 5 in both rule sets. The heads are size-agnostic.
SHAPES = sys.argv[13] if len(sys.argv) > 13 else "all"
# Deep targets on a share of plies (0 keeps every ply at SIMS).
TARGET_SIMS = int(sys.argv[14]) if len(sys.argv) > 14 else 0
TARGET_SHARE = float(sys.argv[15]) if len(sys.argv) > 15 else 0.25
# Weight of the learner's policy-entropy bonus (DISTILL_ENTROPY_BONUS); 0 = off.
ENTROPY_BONUS = float(sys.argv[16]) if len(sys.argv) > 16 else 0.0
# 0 = the actors' searches start unvisited children from zero instead of the
# Q head's expected value (MCTS_Q_SEED); fewer deep-search blunders measured.
# Anything but 0 or 1 is rejected before the first spawn.
Q_SEED = {"1": True, "0": False}.get(sys.argv[17] if len(sys.argv) > 17 else "1")
# Share of each learner batch drawn from replay (the rest from the exact tables).
REPLAY_FRACTION = float(sys.argv[18]) if len(sys.argv) > 18 else 0.75
# "visits" (deep plies' visit counts) or "gumbel" (improved policy on every ply).
POLICY_TARGET = sys.argv[19] if len(sys.argv) > 19 else "visits"
# Weight of the search-value term in the learner's value loss (0 = off).
ROOT_VALUE_WEIGHT = float(sys.argv[20]) if len(sys.argv) > 20 else 0.0
# The exact-table corpus on the Volume; docs/NEURAL_CHAOS.md records how
# datasets-v3 was built. The learner fails when it is missing or empty.
EXACT_SUBDIR = sys.argv[21] if len(sys.argv) > 21 else "datasets-v3"
# The last generation to train; 0 trains until the stop file appears. Once
# that generation is published nothing more is paid for: no learner, no
# self-play, and the actors still running are cancelled, since no learner
# will ever read their shards. Its arena, when one is due, still plays.
UNTIL_GEN = int(sys.argv[22]) if len(sys.argv) > 22 else 0
OUT_SUBDIR = "replay-gpu"
# Read here and passed to the remote functions as arguments: a container does
# not inherit this environment, so setting them only here used to do nothing.
GZIP_LEVEL = int(os.environ.get("C4_REPLAY_GZIP_LEVEL", "1"))
HOLDOUT_CONFIGS = os.environ.get("DISTILL_HOLDOUT_CONFIGS", "")
# Consecutive failures of one role that stop all submission (see above).
MAX_FAILURES = int(os.environ.get("C4_MAX_FAILURES", "3"))
ROLES = ("actor", "learner", "arena")

actor_fn = modal.Function.from_name("connect4-chaos", "selfplay_gpu")
learn_fn = modal.Function.from_name("connect4-chaos", "learn")
arena_fn = modal.Function.from_name("connect4-chaos", "arena")
vol = modal.Volume.from_name("connect4-tables")


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# Local shard mirror (C4_MIRROR=1 only): the learner reads the Volume, so
# the mirror is only a backup. Shards stay gzipped (a twentieth of the disk)
# and the newest MIRROR_KEEP are kept; the Volume holds the full history.
MIRROR_KEEP = int(os.environ.get("C4_MIRROR_KEEP", "400"))


def fetch_shard(shard_gz):
    data = b"".join(vol.read_file(f"{OUT_SUBDIR}/{shard_gz}"))
    REPLAY.mkdir(parents=True, exist_ok=True)
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


def read_model(name):
    """No side effects: a late deadline thread cannot publish an old model."""
    return b"".join(vol.read_file(f"models/{name}"))


def mirror_model(name):
    data = with_timeout(300, read_model, name)
    MODELS.mkdir(parents=True, exist_ok=True)
    out = MODELS / name
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(out)
    pointer = ROOT / "current-model.txt"
    temporary = pointer.with_suffix(".tmp")
    temporary.write_text(str(out.resolve()) + "\n", encoding="utf-8")
    temporary.replace(pointer)
    return out


# A dropped connection says nothing about the job on the other side: the
# call keeps running and will finish. Only a real error means the work is
# gone, so polling distinguishes the two and keeps waiting through outages.
TRANSIENT = ("connectionerror", "getaddrinfo", "connection lost", "connection reset",
             "streamterminated", "broken pipe", "unavailable", "deadline",
             "timed out", "temporarily unavailable", "eof occurred")


def is_transient(exc):
    # get(timeout=0) raises Python's TimeoutError when no output is ready.
    # These SDK exceptions instead describe a completed failure or an output
    # that cannot be retrieved. Their messages may contain transport words,
    # but polling the same call will never recover it. Retire it through the
    # normal failure path (and never replace it after shutdown is requested).
    # Look up optional classes for compatibility across Modal SDK versions.
    exceptions = getattr(modal, "exception", None)
    terminal = tuple(cls for name in (
        "FunctionTimeoutError", "OutputExpiredError", "RemoteError",
        "ExecutionError", "InternalFailure", "DeserializationError",
    ) if isinstance(cls := getattr(exceptions, name, None), type))
    if isinstance(exc, terminal):
        return False
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in TRANSIENT)


def with_timeout(seconds, work, *args):
    """Runs `work` on a helper thread and gives up after `seconds`.

    A Volume read that never returns froze the driver for three hours: it
    raised nothing, so the retry logic never saw a failure and the loop
    simply stopped. Every network call the loop depends on now has a
    deadline, and a mirror that misses it is skipped rather than fatal.
    """
    result = {}

    def run():
        try:
            result["value"] = work(*args)
        except BaseException as exc:                     # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise TimeoutError(f"{getattr(work, '__name__', 'call')} exceeded {seconds}s")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def published_history():
    """Restore only INIT_MODEL's successful ancestry from immutable sidecars.

    Legacy checkpoints are roots: missing provenance is never reconstructed
    from filenames, experiments or failed-but-retained checkpoints. A read
    failure disables old-history arenas but does not stop new training.
    Only the newest ARENA_LAG + 1 are read: no arena looks further back, and
    walking a long ancestry under one deadline could only fail it.
    """
    from neural.checkpoint_lineage import read_history

    try:
        history = with_timeout(60, read_history, vol.read_file, INIT_MODEL, ARENA_LAG + 1)
    except Exception as exc:
        log(f"could not read checkpoint lineage: {type(exc).__name__}: {str(exc)[:120]}; "
            "arena history starts at the initial checkpoint")
        return [INIT_MODEL]
    log(f"restored {len(history)} checkpoints in the initial model's lineage")
    return history


def require_initial_model():
    """Fail before the first spawn when models/<INIT_MODEL> is not on the
    Volume. Otherwise K actors and a learner each start a container, fail
    minutes later, and count towards the failure cap on the way."""
    missing = (FileNotFoundError,) + tuple(
        cls for cls in (getattr(getattr(modal, "exception", None), "NotFoundError", None),)
        if isinstance(cls, type))
    try:
        entries = with_timeout(60, vol.listdir, f"models/{INIT_MODEL}")
    except missing:
        entries = []
    if not any(Path(str(entry.path)).name == INIT_MODEL for entry in entries):
        raise FileNotFoundError(f"models/{INIT_MODEL} is not on the Volume; upload it first with "
                                f"`modal volume put connect4-tables <file> models/{INIT_MODEL}`")


def write_journal(actors, learner, arena):
    """Record every call in flight, atomically (see JOURNAL)."""
    calls = [dict(id=cid, role="actor", seed=seed, model=used, spawned=t0)
             for cid, (_call, seed, used, t0, _restored) in actors.items()]
    if learner is not None:
        _call, cid, lgen, init, t0, _restored = learner
        calls.append(dict(id=cid, role="learner", gen=lgen, init=init, spawned=t0))
    if arena is not None:
        _call, cid, newer, older, _restored = arena
        calls.append(dict(id=cid, role="arena", newer=newer, older=older))
    temporary = JOURNAL.with_suffix(".tmp")
    temporary.write_text(json.dumps({"version": 1, "calls": calls}, indent=1) + "\n", encoding="utf-8")
    temporary.replace(JOURNAL)


def restore_journal():
    """The calls a previous driver spawned and never collected, reattached.

    Actors and an arena are taken over as they are: their shards and reports
    are as good as new ones. A learner is taken over only when it trains the
    generation this run starts at, from the same checkpoint; any other would
    publish a generation this run does not expect, so it is cancelled. The
    restored calls' failures do not count towards the failure cap.
    """
    actors, learner, arena = {}, None, None
    try:
        record = json.loads(JOURNAL.read_text(encoding="utf-8"))
        calls = record["calls"] if record.get("version") == 1 else None
        entries = [(entry["id"], entry["role"], entry) for entry in calls]
    except FileNotFoundError:
        return actors, learner, arena
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"{JOURNAL} is not a readable call journal ({type(exc).__name__}); check the "
                         "Modal dashboard for calls still running, then remove it") from exc
    for cid, role, entry in entries:
        call = modal.FunctionCall.from_id(cid)
        if role == "actor":
            actors[cid] = (call, entry["seed"], entry["model"], entry["spawned"], True)
            log(f"actor {cid} reattached from {JOURNAL.name} (seed={entry['seed']} model={entry['model']})")
            continue
        if role == "learner" and learner is None and (entry["gen"], entry["init"]) == (GEN, INIT_MODEL):
            learner = (call, cid, entry["gen"], entry["init"], entry["spawned"], True)
            log(f"learner {cid} reattached from {JOURNAL.name} (gen={entry['gen']} init={entry['init']})")
            continue
        if role == "arena" and arena is None:
            arena = (call, cid, entry["newer"], entry["older"], True)
            log(f"arena {cid} reattached from {JOURNAL.name}: {entry['newer']} vs {entry['older']}")
            continue
        reason = (f"it trains gen {entry.get('gen')} from {entry.get('init')}, not gen {GEN} from {INIT_MODEL}"
                  if role == "learner" else "a duplicate or unknown entry")
        try:
            with_timeout(60, call.cancel)
            log(f"{role} {cid} from {JOURNAL.name} cancelled: {reason}")
        except Exception as exc:
            log(f"could not cancel {role} {cid} ({reason}): {type(exc).__name__}: {str(exc)[:120]}")
    return actors, learner, arena


def discard_retained(name):
    """Delete the checkpoint and optimizer state that a failed learner left
    without lineage. The generation is retrained, so nothing will load them,
    and every failure would otherwise leave both on the Volume."""
    if not name.endswith(".pt") or "/" in name or "\\" in name:
        log(f"not removing {name!r}: not a checkpoint name")
        return []
    removed = []
    for path in (f"models/{name}", f"models/{name}.opt"):
        try:
            with_timeout(60, vol.remove_file, path)
            removed.append(path)
        except FileNotFoundError:
            continue
        except Exception as exc:
            log(f"could not remove {path}: {type(exc).__name__}: {str(exc)[:120]}; neural.prune can")
    return removed


def main():
    validate_selfplay(GAMES, SIMS, SHAPES, TARGET_SIMS, TARGET_SHARE)
    if K < 1 or STEPS < 1 or BATCH < 1 or WINDOW < 1 or MIN_NEW < 0:
        raise ValueError("Actor count, steps, batch and window must be positive; pacing nonnegative")
    if ARENA_EVERY < 0 or ARENA_LAG < 1 or not (0 < LR < float("inf")):
        raise ValueError("Invalid arena schedule or learning rate")
    # Everything the remote functions would reject, rejected here: a bad
    # value used to cost a container start per spawn, and a retry forever.
    if GEN < 0 or MAX_FAILURES < 1:
        raise ValueError("The first generation must be nonnegative and C4_MAX_FAILURES positive")
    if not INIT_MODEL.endswith(".pt") or "/" in INIT_MODEL or "\\" in INIT_MODEL:
        raise ValueError(f"the initial model must be a .pt file name under models/, not {INIT_MODEL!r}")
    if POLICY_TARGET not in ("visits", "gumbel"):
        raise ValueError(f"policy_target must be 'visits' or 'gumbel', not {POLICY_TARGET!r}")
    if Q_SEED is None:
        raise ValueError("q_seed must be 0 or 1")
    if not 0 <= REPLAY_FRACTION <= 1:
        raise ValueError("replay_fraction must be between 0 and 1")
    if not (0 <= ENTROPY_BONUS < float("inf") and 0 <= ROOT_VALUE_WEIGHT < float("inf")):
        raise ValueError("entropy_bonus and root_value_weight must be finite and nonnegative")
    if not EXACT_SUBDIR.strip("/ ") or ".." in EXACT_SUBDIR.split("/"):
        raise ValueError(f"exact_subdir must name a directory under the Volume, not {EXACT_SUBDIR!r}")
    if UNTIL_GEN and UNTIL_GEN < GEN:
        raise ValueError(f"until_gen {UNTIL_GEN} comes before the first generation {GEN}")
    if UNTIL_GEN < 0:
        raise ValueError("until_gen must be 0 (no limit) or a generation number")
    if not 0 <= GZIP_LEVEL <= 9:
        raise ValueError("C4_REPLAY_GZIP_LEVEL must be between 0 and 9")
    for tag in HOLDOUT_CONFIGS.split(","):
        if tag.strip() and parse_shape_spec(tag) is None:
            raise ValueError("DISTILL_HOLDOUT_CONFIGS must name specific configurations, not 'all'")
    require_initial_model()
    if MIRROR:
        REPLAY.mkdir(parents=True, exist_ok=True)
    log(f"mirroring {'on' if MIRROR else 'off'}: shards and checkpoints "
        f"{'are copied to ' + str(ROOT) if MIRROR else 'stay on the Volume'}")
    model = INIT_MODEL
    gen = GEN
    seed_base = (int(time.time()) % 10_000_000) * 100
    # Tracked calls: actors {id: (call, seed, model, spawned, restored)},
    # learner (call, id, gen, init, spawned, restored), arena (call, id,
    # newer, older, restored); `restored` marks a previous driver's call.
    actors, learner, arena = restore_journal()
    published = published_history()   # so a restart does not delay the next arena
    spawned = finished = 0
    # None = first generation, no pacing; a reattached learner is that one.
    new_positions = None if learner is None else 0
    waiting_logged = False
    failures = dict.fromkeys(ROLES, 0)
    log(f"loop start init={model} gen={gen} K={K} games={GAMES} steps={STEPS} batch={BATCH} "
        f"lr={LR} entropy={ENTROPY_BONUS} window={WINDOW} minNew={MIN_NEW} sims={SIMS} "
        f"targetSims={TARGET_SIMS} targetShare={TARGET_SHARE} qseed={int(Q_SEED)} "
        f"replay={REPLAY_FRACTION} target={POLICY_TARGET} rootValue={ROOT_VALUE_WEIGHT} "
        f"exact={EXACT_SUBDIR} holdouts={HOLDOUT_CONFIGS or '-'} gzip={GZIP_LEVEL} "
        f"maxFailures={MAX_FAILURES} untilGen={UNTIL_GEN or '-'} seedBase={seed_base}")
    stopping = False
    journaled = None
    ended_logged = False

    def trained_enough():
        # `gen` is the next generation to train; past UNTIL_GEN the run is done.
        return bool(UNTIL_GEN) and gen > UNTIL_GEN

    def stop_requested():
        # Network calls and mirrors can outlive the loop's initial check.
        # Recheck at each submission boundary; once observed, drain only.
        nonlocal stopping
        stopping = stopping or STOP.exists()
        return stopping

    def failed(role, restored=False):
        # MAX_FAILURES in a row in one role stop all submission, like a stop
        # request: what is in flight drains, nothing new is paid for.
        nonlocal stopping
        if restored:
            return
        failures[role] += 1
        if failures[role] >= MAX_FAILURES and not stopping:
            stopping = True
            log(f"{role} failed {failures[role]} times in a row: submitting nothing more and "
                f"draining the calls in flight (C4_MAX_FAILURES={MAX_FAILURES})")

    def record():
        # Journal the calls in flight whenever that set changes.
        nonlocal journaled
        tracked = (tuple(actors), learner and learner[1], arena and arena[1])
        if tracked != journaled:
            write_journal(actors, learner, arena)
            journaled = tracked

    try:
        record()
        while True:
            if trained_enough() and not ended_logged:
                ended_logged = True
                for cid, (call, *_rest) in list(actors.items()):
                    try:
                        call.cancel()
                    except Exception as exc:
                        log(f"actor {cid}: cancel failed: {type(exc).__name__}: {str(exc)[:120]}")
                    del actors[cid]
                record()
                log(f"generation {UNTIL_GEN} published: training done, "
                    f"{'waiting for its arena' if arena is not None else 'nothing left to wait for'}")
            if not stop_requested() and not trained_enough():
                ready = new_positions is None or new_positions >= MIN_NEW
                if learner is None and not ready and not waiting_logged:
                    log(f"learner pacing: {new_positions} of {MIN_NEW} fresh positions since gen {gen - 1}")
                    waiting_logged = True
                if learner is None and ready and not stop_requested():
                    try:
                        call = learn_fn.spawn(gen, model, STEPS, BATCH, LR, REPLAY_FRACTION, WINDOW,
                                              exact_subdir=EXACT_SUBDIR, entropy_bonus=ENTROPY_BONUS,
                                              root_value_weight=ROOT_VALUE_WEIGHT,
                                              holdout_configs=HOLDOUT_CONFIGS)
                    except Exception as exc:
                        log(f"learner spawn failed: {type(exc).__name__}: {str(exc)[:200]}; retry in 60 s")
                        if not is_transient(exc):
                            failed("learner")
                        time.sleep(60)
                    else:
                        learner = (call, call.object_id, gen, model, time.time(), False)
                        record()
                        log(f"learner spawned {call.object_id} gen={gen} init={model} "
                            f"(fresh positions since last spawn: {new_positions})")
                        new_positions = 0
                        waiting_logged = False
                while len(actors) < K and not stop_requested():
                    try:
                        spawned += 1
                        seed = seed_base + spawned
                        call = actor_fn.spawn(model, GAMES, SHAPES, seed, OUT_SUBDIR, SIMS,
                                              TARGET_SIMS, TARGET_SHARE, q_seed=Q_SEED,
                                              policy_target=POLICY_TARGET, gzip_level=GZIP_LEVEL)
                    except Exception as exc:
                        log(f"actor spawn failed: {type(exc).__name__}: {str(exc)[:200]}; retry in 60 s")
                        if not is_transient(exc):
                            failed("actor")
                        time.sleep(60)
                        break
                    actors[call.object_id] = (call, seed, model, time.time(), False)
                    record()
                    log(f"actor spawned {call.object_id} seed={seed} model={model}")
            for cid, (call, seed, used, t0, restored) in list(actors.items()):
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
                    failed("actor", restored)
                    continue
                del actors[cid]
                if result.get("exit") == 0 and result.get("shard"):
                    failures["actor"] = 0
                    # The Volume already holds the shard and the learner will
                    # train on it whether or not the local mirror succeeds, so it
                    # counts towards pacing before the mirror is attempted.
                    finished += 1
                    match = re.search(r"games, (\d+) positions", result.get("out") or "")
                    if match and new_positions is not None:
                        new_positions += int(match.group(1))
                    # Without a mirror the shard's own size is not known here;
                    # the actor reported its gzip size in bytes if it has one.
                    where, size = f"Volume {OUT_SUBDIR}/", result.get("shard_bytes")
                    if MIRROR:
                        try:
                            out, size = with_timeout(180, fetch_shard, result["shard"])
                            where = ""
                        except Exception as exc:
                            log(f"actor {cid} not mirrored: {type(exc).__name__}: {str(exc)[:160]}")
                            continue
                    summary = (result.get("out") or "").strip().splitlines()
                    compression = result.get('compression_seconds')
                    compression_note = f", gzip {compression}s" if compression is not None else ""
                    size_note = f"{size / 1e6:.1f} MB gz" if isinstance(size, (int, float)) else "on the Volume"
                    log(f"actor {cid} done {result['seconds']}s on {result.get('gpu')} -> {where}{result['shard']} "
                        f"({size_note}{compression_note}) {summary[-1] if summary else ''}")
                else:
                    log(f"actor {cid} exit={result.get('exit')} err={(result.get('err') or '')[-300:]!r}")
                    failed("actor", restored)
                    time.sleep(30)
            if learner is not None:
                call, lcid, lgen, init, t0, restored = learner
                try:
                    result = call.get(timeout=0)
                except TimeoutError:
                    result = None
                except Exception as exc:
                    if is_transient(exc):
                        log(f"learner {lcid}: {type(exc).__name__} while polling; still tracked")
                        result = None
                    else:
                        log(f"learner {lcid} failed: {type(exc).__name__}: "
                            f"{str(exc)[:200]}; retry in 120 s")
                        learner = None
                        failed("learner", restored)
                        time.sleep(120)
                        result = None
                if result is not None:
                    learner = None
                    # A run whose evaluation alone failed has saved everything
                    # and written its lineage: its checkpoint is used as usual.
                    adopted = bool(result.get("adopted"))
                    if result.get("model") and (result.get("exit") == 0 or adopted):
                        failures["learner"] = 0
                        model = result["model"]
                        gen = lgen + 1
                        if MIRROR:
                            try:
                                local = mirror_model(model)
                            except Exception as exc:
                                local = f"(not mirrored: {type(exc).__name__}: {str(exc)[:120]})"
                        else:
                            local = "(mirroring off; fetch with modal volume get)"
                        published.append(model)
                        if (ARENA_EVERY and arena is None and len(published) > ARENA_LAG
                                and lgen % ARENA_EVERY == 0 and not stop_requested()):
                            older = published[-1 - ARENA_LAG]
                            try:
                                call = arena_fn.spawn(model, older, ARENA_GAMES, ARENA_SIMS, "all", 7)
                            except Exception as exc:
                                log(f"arena spawn failed: {type(exc).__name__}: {str(exc)[:150]}")
                                if not is_transient(exc):
                                    failed("arena")
                            else:
                                arena = (call, call.object_id, model, older, False)
                                record()
                                log(f"arena spawned {call.object_id}: {model} vs {older}")
                        if adopted:
                            log(f"learner gen {lgen}: evaluation failed after training finished; adopted "
                                f"models/{model} with lineage. err={(result.get('err') or '')[-300:]!r}")
                        log(f"learner gen {lgen} done {result['seconds']}s on {result.get('gpu')} "
                            f"(staging {result.get('staging_seconds')}s, replay {result.get('replay_positions')} "
                            f"positions / {result.get('replay_shards')} shards, optimizer "
                            f"{'saved' if result.get('optimizer_state') else 'fresh'}) -> models/{model}; mirrored {local}")
                        for line in result.get("lines", []):
                            log(f"  gen {lgen} {line}")
                    else:
                        # What a failed run retained has no lineage and is
                        # retrained; the next attempt must not leave another.
                        retained = ""
                        if result.get("model"):
                            removed = discard_retained(result["model"])
                            retained = (f"; removed the retained {' and '.join(removed)}" if removed
                                        else f"; models/{result['model']} is still on the Volume")
                        log(f"learner gen {lgen} exit={result.get('exit')} err={(result.get('err') or '')[-400:]!r}; "
                            f"retry in 120 s{retained}")
                        failed("learner", restored)
                        time.sleep(120)
            if arena is not None:
                call, acid, newer, older, restored = arena
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
                        failed("arena", restored)
                if outcome is not None:
                    arena = None
                    if outcome.get("exit") == 0:
                        failures["arena"] = 0
                        for line in (outcome.get("out") or "").strip().splitlines():
                            log(f"  {line.strip()}")
                    else:
                        log(f"arena exit={outcome.get('exit')} {(outcome.get('err') or '')[-200:]!r}")
                        failed("arena", restored)
            record()
            if ((stop_requested() or trained_enough())
                    and not actors and learner is None and arena is None):
                break
            time.sleep(10)
    finally:
        # However the loop ends, the journal lists what is still running, so
        # the next start can reattach to it rather than pay for it twice.
        try:
            write_journal(actors, learner, arena)
        except Exception as exc:
            log(f"could not write {JOURNAL.name}: {type(exc).__name__}: {str(exc)[:160]}")
        in_flight = len(actors) + (learner is not None) + (arena is not None)
        if in_flight:
            log(f"driver exiting with {in_flight} calls in flight; {JOURNAL} lists them for the next start")
    log(f"loop end: actors spawned {spawned}, finished {finished}, next gen {gen}, model {model}")


if __name__ == "__main__":
    main()

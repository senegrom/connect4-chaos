"""Modal burst compute for the connect4-chaos program.

Everything CPU-heavy - exact solves, rank sidecars and exact-sample dataset
building - runs here as finite Functions over one
persistent Volume; GPU self-play actors (`selfplay_gpu`) and the learner
(`learn`, one generation per call) run on H100s. Everything is a Function,
never a Sandbox.
Burst Functions with a 24 h timeout, resumable through Volume checkpoints (the pair solver's block
files are ordinary checkpoints, so re-invoking a solve continues where the
previous call stopped).

Loop: `modal deploy neural/modal_app.py` once, then a local driver
(`python -m neural.modal_loop`, see scripts/launch-modal-loop.ps1) keeps K `selfplay_gpu` calls in
flight with the newest checkpoint in models/ on the Volume and one `learn`
call training the next generation from it; shards and checkpoints never
leave the Volume except for local mirrors.

Run from the modal environment, e.g.:
  D:/PyEnv/modal/Scripts/python.exe -m modal run neural/modal_app.py \
      --task solve --rows 6 --columns 7 --connect 4 --mode chaos \
      --discover-through 28 --threads 32
  ... --task sidecars --subdir chaos-6x7-c4
  ... --task dataset --subdir classic-5x7-c4 --rows 5 --columns 7 --connect 4 \
      --mode classic --samples 150000
  ... --task selfplay-gpu --model big1-abc123.pt --games 4096 \
      --shapes 6x7c4chaos,8x8c5chaos --seed 1
  ... --task learn --gen 4 --model big3-abc123.pt --steps 6000 --batch 1024
Results land in the Volume; fetch with `modal volume get connect4-tables ...`.
Without --out-subdir, selfplay-gpu writes replay-gpu (the learner's replay
default), while dataset/prepare write datasets. Explicit directories are preserved.
"""

from __future__ import annotations

import json
import os
from neural.training_config import DEFAULT_SIMS, validate_selfplay
import subprocess
import time
from pathlib import Path
from typing import Optional

import modal

APP_NAME = "connect4-chaos"
VOLUME_NAME = "connect4-tables"
TABLES = "/tables"

app = modal.App(APP_NAME)
tables = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
REPO = Path(__file__).resolve().parent.parent
# The pins CI tests (neural/requirements.txt), for both images: the CPU build
# of torch here, the CUDA build of the same version for the GPU image.
REQUIREMENTS = str(REPO / "neural" / "requirements.txt")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("g++", "make")
    .pip_install_from_requirements(REQUIREMENTS, extra_index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(str(REPO / "native"), "/repo/native", copy=True)
    .add_local_dir(str(REPO / "scripts"), "/repo/scripts", copy=True)
    .add_local_dir(str(REPO / "neural"), "/repo/neural", copy=True)
    .run_commands(
        "g++ -std=c++20 -O3 -pthread -o /opt/chaos-paired /repo/native/perfect-chaos-paired.cpp",
    )
    .workdir("/repo")
)

MOUNTS = {TABLES: tables}

# GPU actors: the default Linux torch wheel ships CUDA, so no index pin.
# One call = one batch of games on one GPU: the checkpoint is read from
# models/ on the Volume, the shard (uint8 planes) is gzipped into
# <out_subdir>/ on the Volume, and the driver pulls it home.
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(REQUIREMENTS)
    .workdir("/repo")
    .add_local_dir(str(REPO / "neural"), "/repo/neural")
    # 20 KB of recorded positions the GPU tests replay. They live with the
    # browser's fixtures because both sides check the same game, and the
    # CUDA half of that check can only run here.
    .add_local_dir(str(REPO / "tests" / "fixtures"), "/repo/tests/fixtures")
)
ACTOR_GPU = os.environ.get("C4_ACTOR_GPU", "H100")
LEARNER_GPU = os.environ.get("C4_LEARNER_GPU", "H100")


def _solver_args(rows, columns, connect, mode, threads, discover_through, out):
    args = ["/opt/chaos-paired", "--rows", str(rows), "--columns", str(columns),
            "--connect", str(connect), "--threads", str(threads), "--verbose",
            "--output", out]
    if mode == "classic":
        args.append("--classic")
    if discover_through is not None and discover_through >= 0:
        args += ["--discover-through", str(discover_through)]
    return args


def _run_solver(rows, columns, connect, mode, threads, discover_through, subdir):
    out = f"{TABLES}/{subdir}"
    os.makedirs(out, exist_ok=True)
    started = time.time()
    with open(f"{out}/solver.log", "a") as log:
        process = subprocess.run(
            _solver_args(rows, columns, connect, mode, threads, discover_through, out),
            stdout=subprocess.PIPE, stderr=log, text=True,
        )
    tables.commit()
    line = next((l for l in process.stdout.splitlines() if l.startswith("{")), None)
    return {"exit": process.returncode, "summary": json.loads(line) if line else None,
            "seconds": round(time.time() - started, 1), "subdir": subdir}


@app.function(image=image, cpu=8.0, memory=32 * 1024, timeout=24 * 60 * 60, volumes=MOUNTS)
def solve_8(rows: int, columns: int, connect: int, mode: str, discover_through: int, subdir: str):
    return _run_solver(rows, columns, connect, mode, 8, discover_through, subdir)


@app.function(image=image, cpu=32.0, memory=128 * 1024, timeout=24 * 60 * 60, volumes=MOUNTS)
def solve_32(rows: int, columns: int, connect: int, mode: str, discover_through: int, subdir: str):
    return _run_solver(rows, columns, connect, mode, 32, discover_through, subdir)


@app.function(image=image, cpu=4.0, memory=16 * 1024, timeout=24 * 60 * 60, volumes=MOUNTS)
def sidecars(subdir: str):
    tables.reload()                      # see tables a solve committed after start
    process = subprocess.run(
        ["python", "/repo/scripts/build-pair-rank-sidecars.py", f"{TABLES}/{subdir}"],
        capture_output=True, text=True,
    )
    tables.commit()
    return {"exit": process.returncode, "out": process.stdout[-2000:], "err": process.stderr[-2000:]}


@app.function(image=image, cpu=2.0, memory=16 * 1024, timeout=24 * 60 * 60, volumes=MOUNTS)
def dataset(subdir: str, rows: int, columns: int, connect: int, mode: str,
            samples: int, out_subdir: str, start_index: int = 0):
    """Builds exact shards for one config. start_index numbers the first
    shard, so extending a config never rewrites its held-out shard 0000."""
    tables.reload()
    spec = f"{TABLES}/{subdir}:{rows}:{columns}:{connect}:{mode}"
    process = subprocess.run(
        ["python", "-m", "neural.build_dataset", f"{TABLES}/{out_subdir}", str(samples), spec],
        capture_output=True, text=True, cwd="/repo",
        env=dict(os.environ, DATASET_START_INDEX=str(start_index)),
    )
    tables.commit()
    return {"exit": process.returncode, "out": process.stdout[-2000:], "err": process.stderr[-3000:]}


@app.function(image=image, cpu=2.0, memory=32 * 1024, timeout=24 * 60 * 60, volumes=MOUNTS)
def prepare(subdir: str, rows: int, columns: int, connect: int, mode: str,
            samples: int, out_subdir: str):
    """Sidecars, then exact-sample shards, for one solved table."""
    tables.reload()
    side = subprocess.run(
        ["python", "/repo/scripts/build-pair-rank-sidecars.py", f"{TABLES}/{subdir}"],
        capture_output=True, text=True,
    )
    if side.returncode != 0:
        return {"exit": side.returncode, "stage": "sidecars", "err": side.stderr[-2000:]}
    tables.commit()
    spec = f"{TABLES}/{subdir}:{rows}:{columns}:{connect}:{mode}"
    data = subprocess.run(
        ["python", "-m", "neural.build_dataset", f"{TABLES}/{out_subdir}", str(samples), spec],
        capture_output=True, text=True, cwd="/repo",
    )
    tables.commit()
    return {"exit": data.returncode, "stage": "dataset", "out": data.stdout[-1500:],
            "err": data.stderr[-2000:]}


@app.function(image=gpu_image, gpu=ACTOR_GPU, cpu=4.0, memory=16 * 1024,
              timeout=2 * 60 * 60, volumes=MOUNTS)
def selfplay_gpu(model_name: str, games: int, shapes: str, seed: int,
                 out_subdir: str = "replay-gpu", sims: int = DEFAULT_SIMS,
                 target_sims: int = 0, target_share: float = 0.25,
                 graphs: bool = True, profile: bool = False, channels_last: bool = True,
                 fused: bool = True, random_share: float = 0.5, random_plies: int = 4,
                 q_seed: bool = True, policy_target: str = "visits", gzip_level: int = 1):
    import gzip
    import shutil

    validate_selfplay(games, sims, shapes, target_sims, target_share)
    # An argument, not C4_REPLAY_GZIP_LEVEL: the container never sees the
    # caller's environment, so the variable was always the default here.
    if isinstance(gzip_level, bool) or not isinstance(gzip_level, int) or not 0 <= gzip_level <= 9:
        raise ValueError("gzip_level must be an integer between 0 and 9")
    started = time.time()
    tables.reload()                      # see checkpoints uploaded after container start
    model_path = f"{TABLES}/models/{model_name}"
    work = Path(f"/tmp/selfplay-{seed}")
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/repo", SELFPLAY_SIMS=str(sims),
               SELFPLAY_TARGET_SIMS=str(target_sims), SELFPLAY_TARGET_SHARE=str(target_share),
               SELFPLAY_GRAPHS="1" if graphs else "0", SELFPLAY_PROFILE="1" if profile else "",
               SELFPLAY_CHANNELS_LAST="1" if channels_last else "0",
               SELFPLAY_FUSED="1" if fused else "0",
               SELFPLAY_RANDOM_OPENING_SHARE=str(random_share),
               SELFPLAY_RANDOM_OPENING_PLIES=str(random_plies),
               MCTS_Q_SEED="1" if q_seed else "0", SELFPLAY_POLICY_TARGET=policy_target)
    process = subprocess.run(
        ["python", "-m", "neural.gpu_selfplay", model_path, str(work), str(games), shapes, str(seed)],
        capture_output=True, text=True, cwd="/repo", env=env,
    )
    shard = None
    produced = sorted(work.glob("*.pt"))
    if process.returncode == 0 and produced:
        dest_dir = Path(f"{TABLES}/{out_subdir}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        shard = produced[-1].name + ".gz"
        compression_started = time.time()
        target = dest_dir / shard
        staging = target.with_suffix(target.suffix + ".partial")
        with open(produced[-1], "rb") as src, gzip.open(staging, "wb", compresslevel=gzip_level) as dst:
            shutil.copyfileobj(src, dst)
        staging.replace(target)
        compression_seconds = time.time() - compression_started
        # The driver no longer mirrors shards by default, so the size it used
        # to read off its own copy is reported from here instead.
        shard_bytes = target.stat().st_size
        tables.commit()
    else:
        compression_seconds = 0.0
        shard_bytes = 0
    shutil.rmtree(work, ignore_errors=True)
    # The actor names its device (torch.cuda.get_device_name()). ACTOR_GPU is
    # what the deploy asked for, and the container re-reads it from an
    # environment without C4_ACTOR_GPU, so it always said "H100".
    gpu = next((line[5:] for line in process.stdout.splitlines() if line.startswith("gpu: ")), "unknown")
    return {"exit": process.returncode, "shard": shard, "seconds": round(time.time() - started, 1),
            "gpu": gpu, "sims": sims, "compression_seconds": round(compression_seconds, 1),
            "shard_bytes": shard_bytes,
            "out": process.stdout[-800:], "err": process.stderr[-1500:]}


@app.function(image=gpu_image, gpu=LEARNER_GPU, cpu=8.0, memory=40 * 1024,
              timeout=3 * 60 * 60, volumes=MOUNTS)
def learn(gen: int, init_model: str, steps: int = 6000, batch: int = 1024, lr: float = 4e-4,
          replay_fraction: float = 0.75, replay_window: int = 4_000_000,
          exact_subdir: str = "datasets-v3", replay_subdir: str = "replay-gpu",
          profile_steps: int = 0, entropy_bonus: float = 0.0, root_value_weight: float = 0.0,
          allow_no_exact: bool = False, holdout_configs: str = "", warmup_steps: int = -1):
    """One learner generation on one GPU: warm-starts from models/<init_model>,
    trains neural.distill on the exact shards in <exact_subdir>/ plus the
    newest replay_window self-play positions (gunzipped from <replay_subdir>/
    to local disk), and publishes models/big<gen>-<sha>.pt. Returns the
    trainer's key lines. A missing exact corpus is an error unless
    allow_no_exact asks for replay-only training. holdout_configs are the
    boards kept out of training (DISTILL_HOLDOUT_CONFIGS for a local run).
    warmup_steps >= 0 sets the learning-rate warm-up (DISTILL_WARMUP_STEPS);
    by default a warm start without optimizer state - the first generation
    after an ONNX import - warms up and any other run does not."""
    import hashlib
    import shutil

    from neural.checkpoint_lineage import write_lineage
    from neural.distill import training_holdouts
    from neural.replay_staging import stage_replay, validate_window

    validate_window(replay_window)
    if not exact_subdir.strip("/ ") or ".." in exact_subdir.split("/"):
        raise ValueError(f"exact_subdir must name a directory under the Volume, not {exact_subdir!r}")
    # An argument: this container's environment is not the caller's, so a
    # DISTILL_HOLDOUT_CONFIGS set where the driver ran never arrived here.
    holdout_spec = holdout_configs
    _, holdout_shapes = training_holdouts(holdout_spec)

    started = time.time()
    tables.reload()
    # Checked before staging replay: a missing corpus used to train on replay
    # alone, with the Q loss at zero and nothing else looking wrong.
    exact_dir = Path(f"{TABLES}/{exact_subdir}")
    if not exact_dir.is_dir() and not allow_no_exact:
        raise FileNotFoundError(f"exact corpus {exact_subdir}/ is not on the Volume (docs/NEURAL_CHAOS.md "
                                "has the recipe); pass allow_no_exact=True to train on replay alone")
    replay_dir = Path(f"/tmp/replay-{gen}")
    shutil.rmtree(replay_dir, ignore_errors=True)
    replay_dir.mkdir(parents=True)
    replay_stats = stage_replay(f"{TABLES}/{replay_subdir}", replay_dir, replay_window, holdout_shapes)
    positions, staged_shards = replay_stats["positions"], replay_stats["shards"]
    skipped, excluded = replay_stats["skipped"], replay_stats["excluded"]
    staged = time.time() - started
    out_dir = Path(f"/tmp/learn-{gen}")
    shutil.rmtree(out_dir, ignore_errors=True)
    # The generation seeds the row sampler: each generation draws other exact
    # rows, and rerunning one reproduces its draw.
    env = dict(os.environ, PYTHONPATH="/repo", DISTILL_INIT=f"{TABLES}/models/{init_model}",
               DISTILL_SEED=str(gen),
               DISTILL_LR=str(lr), DISTILL_REPLAY_FRACTION=str(replay_fraction),
               DISTILL_REPLAY_WINDOW=str(replay_window), DISTILL_PROFILE_STEPS=str(profile_steps),
               DISTILL_ENTROPY_BONUS=str(entropy_bonus), DISTILL_HOLDOUT_CONFIGS=holdout_spec,
               DISTILL_ROOT_VALUE_WEIGHT=str(root_value_weight),
               DISTILL_ALLOW_NO_EXACT="1" if allow_no_exact else "0")
    init_optimizer = Path(f"{TABLES}/models/{init_model}.opt")
    if init_optimizer.exists():
        env["DISTILL_INIT_OPT"] = str(init_optimizer)
    if warmup_steps >= 0:
        env["DISTILL_WARMUP_STEPS"] = str(warmup_steps)
    shard_dirs = ([str(exact_dir)] if exact_dir.is_dir() else []) + [str(replay_dir)]
    process = subprocess.run(
        ["python", "-m", "neural.distill", ";".join(shard_dirs), str(out_dir), str(steps), str(batch)],
        capture_output=True, text=True, cwd="/repo", env=env,
    )
    model = None
    optimizer_state = False
    checkpoint = out_dir / "distilled.pt"
    # The trainer prints this line once the checkpoint and its optimizer state
    # are both saved, just before the held-out evaluation.
    trained = f"saved {checkpoint}" in process.stdout.splitlines()
    adopted = False
    # The trainer atomically exposes this file after its final step, before
    # saving optional optimizer state and evaluating. Preserve completed work
    # even if either later stage fails; the nonzero exit still reports failure.
    if checkpoint.exists():
        data = checkpoint.read_bytes()
        model = f"big{gen}-{hashlib.sha1(data).hexdigest()[:10]}.pt"
        model_dir = Path(f"{TABLES}/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        staging = model_dir / f"{model}.partial"
        staging.write_bytes(data)
        staging.replace(model_dir / model)
        optimizer_checkpoint = out_dir / "optimizer.pt"
        if optimizer_checkpoint.exists():
            optimizer_staging = model_dir / f"{model}.opt.partial"
            shutil.copyfile(optimizer_checkpoint, optimizer_staging)
            optimizer_staging.replace(model_dir / f"{model}.opt")
            optimizer_state = True
        # A run that saved everything and failed only in the evaluation left
        # a checkpoint as good as any: it is adopted with lineage rather than
        # retrained. Anything that failed earlier stays without lineage, for a
        # person to inspect; the driver deletes it before retrying.
        adopted = process.returncode != 0 and trained
        if process.returncode == 0 or adopted:
            write_lineage(model_dir, model, init_model, gen)
        tables.commit()
    shutil.rmtree(replay_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    stdout = process.stdout.splitlines()
    gpu = next((line[5:] for line in stdout if line.startswith("gpu: ")), "unknown")
    profile = process.stdout.split("profile:", 1)[1].split("\nsaved ", 1)[0] if "profile:" in process.stdout else ""
    # Always keep the header lines (they say how much data trained) plus the
    # last few progress lines and the whole held-out report.
    lines = ([l for l in stdout if l.startswith(("sampler seed", "train samples", "replay window",
                                                 "warm start", "learning-rate warm-up"))]
             + [l for l in stdout if l.startswith("step ")][-4:]
             + [l for l in stdout if l.startswith("[held")])
    return {"exit": process.returncode, "gen": gen, "model": model, "init": init_model,
            "adopted": adopted, "replay_positions": positions, "replay_shards": staged_shards,
            "skipped_shards": skipped, "excluded_shards": excluded, "optimizer_state": optimizer_state, "profile": profile,
            "staging_seconds": round(staged, 1), "seconds": round(time.time() - started, 1),
            "gpu": gpu, "lines": lines[-40:], "err": process.stderr[-1500:]}


@app.function(image=gpu_image, gpu=ACTOR_GPU, cpu=4.0, memory=16 * 1024,
              timeout=2 * 60 * 60, volumes=MOUNTS)
def arena(model_a: str, model_b: str, games: int = 32, sims: int = 32,
          shapes: str = "", seed: int = 7, sims_b: int = -1):
    """Plays two checkpoints from models/ against each other over many board
    shapes, including ones the actors never play, and returns the report."""
    started = time.time()
    # One checkpoint per side. Output-averaging ensembles measured as a loss
    # and were removed, so a comma list is refused here rather than failing
    # as a missing file once the GPU container has started.
    paths = []
    for name in (model_a, model_b):
        name = name.strip()
        if not name or "," in name:
            raise ValueError(f"arena takes one checkpoint per side, not {name!r}")
        paths.append(f"{TABLES}/models/{name}")
    tables.reload()
    # Keep positional arguments aligned even when the caller uses all boards.
    # An omitted shape must not silently discard a seed or B's search budget.
    shapes = shapes.strip() or "all"
    command = ["python", "-m", "neural.arena", *paths, str(games), str(sims), shapes, str(seed)]
    if sims_b >= 0:
        command.append(str(sims_b))
    process = subprocess.run(command, capture_output=True, text=True, cwd="/repo",
                             env=dict(os.environ, PYTHONPATH="/repo"))
    return {"exit": process.returncode, "a": model_a, "b": model_b, "games": games,
            "sims": sims, "seconds": round(time.time() - started, 1),
            "out": process.stdout[-6000:], "err": process.stderr[-1500:]}


@app.function(image=gpu_image, gpu=ACTOR_GPU, cpu=4.0, memory=16 * 1024,
              timeout=2 * 60 * 60, volumes=MOUNTS)
def measure(model_name: str, sims: int = 128, positions: int = 2048,
            exact_subdir: str = "datasets-v3", q_seed: bool = True, holdout_configs: str = ""):
    """Blunder rates of one checkpoint - network plus search - on the
    held-out shard of every solved board, the positions the learner never
    trains on; the pooled chaos and classic rates are the numbers to
    compare checkpoints by (neural/search_quality.py). holdout_configs are
    the boards the checkpoint never trained on, scored whole."""
    started = time.time()
    name = model_name.strip()
    if not name or "," in name:
        raise ValueError(f"measure takes one checkpoint, not {name!r}")
    tables.reload()
    process = subprocess.run(
        ["python", "-m", "neural.search_quality", f"{TABLES}/models/{name}",
         f"{TABLES}/{exact_subdir}", str(sims), str(positions)],
        capture_output=True, text=True, cwd="/repo",
        env=dict(os.environ, PYTHONPATH="/repo", MCTS_Q_SEED="1" if q_seed else "0",
                 DISTILL_HOLDOUT_CONFIGS=holdout_configs))
    return {"exit": process.returncode, "model": model_name, "sims": sims, "q_seed": q_seed,
            "positions": positions, "seconds": round(time.time() - started, 1),
            "out": process.stdout[-6000:], "err": process.stderr[-1500:]}


@app.function(image=gpu_image, gpu=ACTOR_GPU, cpu=4.0, memory=16 * 1024,
              timeout=30 * 60, volumes=MOUNTS)
def gpu_test(module: str, args: str):
    """Runs one neural test module on a GPU, which CI does not have. `args`
    are the module's arguments, split on whitespace; models/ paths are on the
    Volume. The unittest modules take none: pass an empty string. There is
    no default checkpoint - the old one was deleted with the Volume."""
    tables.reload()
    arguments = [f"{TABLES}/{a}" if a.startswith("models/") else a for a in args.split()]
    process = subprocess.run(["python", "-m", f"neural.{module}", *arguments],
                             capture_output=True, text=True, cwd="/repo",
                             env=dict(os.environ, PYTHONPATH="/repo"))
    return {"exit": process.returncode, "module": module,
            "out": process.stdout[-6000:], "err": process.stderr[-3000:]}


@app.local_entrypoint()
def main(task: str, rows: int = 4, columns: int = 4, connect: int = 4, mode: str = "chaos",
         threads: int = 8, discover_through: int = -1, subdir: str = "",
         samples: int = 150000, out_subdir: Optional[str] = None, model: str = "",
         games: int = 256, shapes: str = "6x7c4chaos,6x7c4classic", seed: int = 1,
         gen: int = 0, steps: int = 6000, batch: int = 1024, lr: float = 4e-4,
         replay_window: Optional[int] = None, start_index: int = 0, sims: int = DEFAULT_SIMS,
         target_sims: int = 0, target_share: float = 0.25,
         spawn: bool = False, positions: int = 2048,
         graphs: bool = True, profile: bool = False, channels_last: bool = True,
         module: str = "test_graph_search", args: Optional[str] = None,
         entropy_bonus: float = 0.0, q_seed: bool = True,
         policy_target: str = "visits", root_value_weight: float = 0.0,
         replay_subdir: str = "replay-gpu", exact_subdir: str = "datasets-v3",
         allow_no_exact: bool = False,
         profile_steps: int = 0, fused: bool = True,
         random_share: float = 0.5, random_plies: int = 4, sims_b: int = -1):
    import sys

    # These two settings are read here, where the caller set them, and passed
    # as arguments: a Modal container does not inherit this environment.
    holdout_configs = os.environ.get("DISTILL_HOLDOUT_CONFIGS", "")
    gzip_level = int(os.environ.get("C4_REPLAY_GZIP_LEVEL", "1"))

    # Self-play feeds the same replay directory that learn reads by
    # default. Exact dataset/prepare tasks keep their historical destination.
    # Only omission selects a default; explicit paths remain untouched.
    if out_subdir is None:
        out_subdir = "replay-gpu" if task == "selfplay-gpu" else "datasets"

    # Omission keeps the learner's budget; an explicit zero is valid for
    # exact-only learning.
    if task == "learn":
        if replay_window is None:
            replay_window = 4_000_000
        if type(replay_window) is not int or replay_window < 0:
            raise ValueError("learn replay_window must be an integer >= 0")
    subdir = subdir or f"{mode}-{rows}x{columns}-c{connect}"
    if task == "solve":
        fn = solve_32 if threads > 8 else solve_8
        if spawn:
            # Fire and forget: with `modal run --detach` the app outlives this
            # client and the solve runs to completion (or its 24 h limit -
            # re-spawn to resume from the Volume checkpoints). Progress lives
            # in <subdir>/solver.log on the Volume.
            call = fn.spawn(rows, columns, connect, mode, discover_through, subdir)
            print(json.dumps({"spawned": call.object_id, "subdir": subdir}))
            return
        result = fn.remote(rows, columns, connect, mode, discover_through, subdir)
        print(json.dumps(result, indent=2))
    elif task == "sidecars":
        result = sidecars.remote(subdir)
        print(json.dumps(result, indent=2))
    elif task == "prepare":
        if spawn:
            call = prepare.spawn(subdir, rows, columns, connect, mode, samples, out_subdir)
            print(json.dumps({"spawned": call.object_id, "subdir": subdir}))
            return
        result = prepare.remote(subdir, rows, columns, connect, mode, samples, out_subdir)
        print(json.dumps(result, indent=2))
    elif task == "dataset":
        if spawn:
            call = dataset.spawn(subdir, rows, columns, connect, mode, samples,
                                 out_subdir, start_index)
            print(json.dumps({"spawned": call.object_id, "subdir": subdir,
                              "start_index": start_index}))
            return
        result = dataset.remote(subdir, rows, columns, connect, mode, samples,
                                out_subdir, start_index)
        print(json.dumps(result, indent=2))
    elif task == "selfplay-gpu":
        # One batch on one GPU; `model` names a checkpoint under models/ on
        # the Volume (the driver uploads them). Smoke test / manual use.
        validate_selfplay(games, sims, shapes, target_sims, target_share)
        result = selfplay_gpu.remote(model, games, shapes, seed, out_subdir, sims,
                                     target_sims, target_share, graphs, profile, channels_last, fused,
                                     random_share, random_plies, q_seed=q_seed,
                                     policy_target=policy_target, gzip_level=gzip_level)
        print(json.dumps({k: v for k, v in result.items() if k not in ("out", "err")}, indent=2))
        print(result["out"].strip() or result["err"][-600:])
    elif task == "learn":
        # One generation from models/<model> on the Volume (smoke test / manual).
        result = learn.remote(gen, model, steps, batch, lr, 0.75, replay_window,
                              exact_subdir=exact_subdir, replay_subdir=replay_subdir,
                              profile_steps=profile_steps, entropy_bonus=entropy_bonus,
                              root_value_weight=root_value_weight, allow_no_exact=allow_no_exact,
                              holdout_configs=holdout_configs)
        print(json.dumps({k: v for k, v in result.items() if k not in ("lines", "err", "profile")}, indent=2))
        print("\n".join(result["lines"]) or result["err"][-800:])
        if result.get("profile"):
            print("profile:" + result["profile"])
    elif task == "arena":
        result = arena.remote(model, subdir, games, sims, shapes, seed, sims_b)
        print(result["out"].strip() or result["err"][-800:])
    elif task == "measure":
        # Search blunder rates of models/<model> on the held-out exact shards.
        result = measure.remote(model, sims or 128, positions, exact_subdir=exact_subdir, q_seed=q_seed,
                                holdout_configs=holdout_configs)
        print(json.dumps({k: v for k, v in result.items() if k not in ("out", "err")}, indent=2))
        print(result["out"].strip() or result["err"][-800:])
    elif task == "gpu-test":
        # No default: the checkpoint the old default named went with the
        # Volume. Model tests take `--args models/<name>.pt ...`; the unittest
        # modules take no arguments, `--args=""`.
        if args is None:
            raise SystemExit("gpu-test needs --args: 'models/<checkpoint>.pt ...' for a model "
                             "test, or --args=\"\" for a unittest module")
        result = gpu_test.remote(module, args)
        print(result["out"].strip())
    else:
        raise SystemExit(f"unknown task {task}")

    # A completed RPC is not necessarily a successful subprocess. Apply one
    # exit contract to every synchronous task, after printing its diagnostics
    # (including any checkpoint retained after a learner evaluation failure).
    # Spawn-only paths return above: submission is not a completion result.
    if result["exit"] != 0:
        if result.get("err"):
            print(result["err"], file=sys.stderr)
        raise SystemExit(result["exit"])

"""Modal burst compute for the connect4-chaos program.

Everything CPU-heavy - exact solves, rank sidecars, exact-sample dataset
building and closure measurements - runs here as finite Functions over one
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
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "connect4-chaos"
VOLUME_NAME = "connect4-tables"
TABLES = "/tables"

app = modal.App(APP_NAME)
tables = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
REPO = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("g++", "make")
    .pip_install("numpy")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
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
    .pip_install("numpy", "torch")
    .workdir("/repo")
    .add_local_dir(str(REPO / "neural"), "/repo/neural")
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
                 out_subdir: str = "replay-gpu", sims: int = 0,
                 target_sims: int = 0, target_share: float = 0.25):
    import gzip
    import shutil

    started = time.time()
    tables.reload()                      # see checkpoints uploaded after container start
    model_path = f"{TABLES}/models/{model_name}"
    work = Path(f"/tmp/selfplay-{seed}")
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/repo", SELFPLAY_SIMS=str(sims),
               SELFPLAY_TARGET_SIMS=str(target_sims), SELFPLAY_TARGET_SHARE=str(target_share))
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
        with open(produced[-1], "rb") as src, gzip.open(dest_dir / shard, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        tables.commit()
    shutil.rmtree(work, ignore_errors=True)
    return {"exit": process.returncode, "shard": shard, "seconds": round(time.time() - started, 1),
            "gpu": ACTOR_GPU, "sims": sims, "out": process.stdout[-800:],
            "err": process.stderr[-1500:]}


@app.function(image=gpu_image, gpu=LEARNER_GPU, cpu=8.0, memory=40 * 1024,
              timeout=3 * 60 * 60, volumes=MOUNTS)
def learn(gen: int, init_model: str, steps: int = 6000, batch: int = 1024, lr: float = 4e-4,
          replay_fraction: float = 0.75, replay_window: int = 4_000_000,
          exact_subdir: str = "datasets-v3", replay_subdir: str = "replay-gpu"):
    """One learner generation on one GPU: warm-starts from models/<init_model>,
    trains neural.distill on the exact shards plus the newest replay_window
    self-play positions (gunzipped from <replay_subdir>/ to local disk), and
    publishes models/big<gen>-<sha>.pt. Returns the trainer's key lines."""
    import gzip
    import hashlib
    import shutil

    import torch

    started = time.time()
    tables.reload()
    replay_dir = Path(f"/tmp/replay-{gen}")
    shutil.rmtree(replay_dir, ignore_errors=True)
    replay_dir.mkdir(parents=True)
    shards = sorted(Path(f"{TABLES}/{replay_subdir}").glob("*.pt.gz"),
                    key=lambda path: path.stat().st_mtime, reverse=True)
    positions = 0
    skipped = 0
    for path in shards:
        if positions >= replay_window:
            break
        out = replay_dir / path.name[:-3]
        # A shard can be truncated if its actor's container was dropped
        # mid-write. One bad file used to kill the whole generation, so an
        # unreadable shard is dropped and counted instead.
        try:
            with gzip.open(path, "rb") as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            # The trainer orders replay shards by mtime. Staging newest-first
            # gave the newest shard the oldest mtime, so the trainer's own
            # window aged out exactly the freshest data.
            mtime = path.stat().st_mtime
            os.utime(out, (mtime, mtime))
            positions += len(torch.load(out, map_location="cpu",
                                        weights_only=True, mmap=True)["wdl"])
        except Exception:                                    # noqa: BLE001
            skipped += 1
            out.unlink(missing_ok=True)
    staged = time.time() - started
    out_dir = Path(f"/tmp/learn-{gen}")
    shutil.rmtree(out_dir, ignore_errors=True)
    env = dict(os.environ, PYTHONPATH="/repo", DISTILL_INIT=f"{TABLES}/models/{init_model}",
               DISTILL_LR=str(lr), DISTILL_REPLAY_FRACTION=str(replay_fraction),
               DISTILL_REPLAY_WINDOW=str(replay_window))
    process = subprocess.run(
        ["python", "-m", "neural.distill", f"{TABLES}/{exact_subdir};{replay_dir}",
         str(out_dir), str(steps), str(batch)],
        capture_output=True, text=True, cwd="/repo", env=env,
    )
    model = None
    checkpoint = out_dir / "distilled.pt"
    if process.returncode == 0 and checkpoint.exists():
        data = checkpoint.read_bytes()
        model = f"big{gen}-{hashlib.sha1(data).hexdigest()[:10]}.pt"
        Path(f"{TABLES}/models").mkdir(parents=True, exist_ok=True)
        staging = Path(f"{TABLES}/models/{model}.partial")
        staging.write_bytes(data)
        staging.replace(Path(f"{TABLES}/models/{model}"))
        tables.commit()
    shutil.rmtree(replay_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    stdout = process.stdout.splitlines()
    # Always keep the header lines (they say how much data trained) plus the
    # last few progress lines and the whole held-out report.
    lines = ([l for l in stdout if l.startswith(("train samples", "replay window", "warm start"))]
             + [l for l in stdout if l.startswith("step ")][-4:]
             + [l for l in stdout if l.startswith("[held")])
    return {"exit": process.returncode, "gen": gen, "model": model, "init": init_model,
            "replay_positions": positions, "replay_shards": len(shards),
            "skipped_shards": skipped,
            "staging_seconds": round(staged, 1), "seconds": round(time.time() - started, 1),
            "gpu": LEARNER_GPU, "lines": lines[-40:], "err": process.stderr[-1500:]}


@app.function(image=gpu_image, gpu=ACTOR_GPU, cpu=4.0, memory=16 * 1024,
              timeout=2 * 60 * 60, volumes=MOUNTS)
def arena(model_a: str, model_b: str, games: int = 32, sims: int = 32,
          shapes: str = "", seed: int = 7, sims_b: int = -1):
    """Plays two checkpoints from models/ against each other over many board
    shapes, including ones the actors never play, and returns the report."""
    started = time.time()
    tables.reload()
    command = ["python", "-m", "neural.arena", f"{TABLES}/models/{model_a}",
               f"{TABLES}/models/{model_b}", str(games), str(sims)]
    if shapes:
        command += [shapes, str(seed)]
        if sims_b >= 0:
            command.append(str(sims_b))
    process = subprocess.run(command, capture_output=True, text=True, cwd="/repo",
                             env=dict(os.environ, PYTHONPATH="/repo"))
    return {"exit": process.returncode, "a": model_a, "b": model_b, "games": games,
            "sims": sims, "seconds": round(time.time() - started, 1),
            "out": process.stdout[-6000:], "err": process.stderr[-1500:]}


@app.function(image=image, cpu=2.0, memory=32 * 1024, timeout=24 * 60 * 60, volumes=MOUNTS)
def closure(subdir: str, rows: int, columns: int, connect: int, cap: int):
    process = subprocess.run(
        ["python", "-m", "neural.winning_closure", f"{TABLES}/{subdir}",
         str(rows), str(columns), str(connect), str(cap)],
        capture_output=True, text=True, cwd="/repo",
    )
    return {"exit": process.returncode, "out": process.stdout[-3000:], "err": process.stderr[-2000:]}


@app.local_entrypoint()
def main(task: str, rows: int = 4, columns: int = 4, connect: int = 4, mode: str = "chaos",
         threads: int = 8, discover_through: int = -1, subdir: str = "",
         samples: int = 150000, out_subdir: str = "datasets", model: str = "",
         games: int = 256, shapes: str = "6x7c4chaos,6x7c4classic", seed: int = 1,
         gen: int = 0, steps: int = 6000, batch: int = 1024, lr: float = 4e-4,
         replay_window: int = 4_000_000, start_index: int = 0, sims: int = 0,
         target_sims: int = 0, target_share: float = 0.25,
         cap: int = 30_000_000, spawn: bool = False):
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
        print(json.dumps(sidecars.remote(subdir), indent=2))
    elif task == "prepare":
        if spawn:
            call = prepare.spawn(subdir, rows, columns, connect, mode, samples, out_subdir)
            print(json.dumps({"spawned": call.object_id, "subdir": subdir}))
            return
        print(json.dumps(prepare.remote(subdir, rows, columns, connect, mode, samples, out_subdir), indent=2))
    elif task == "dataset":
        if spawn:
            call = dataset.spawn(subdir, rows, columns, connect, mode, samples,
                                 out_subdir, start_index)
            print(json.dumps({"spawned": call.object_id, "subdir": subdir,
                              "start_index": start_index}))
            return
        print(json.dumps(dataset.remote(subdir, rows, columns, connect, mode, samples,
                                        out_subdir, start_index), indent=2))
    elif task == "selfplay-gpu":
        # One batch on one GPU; `model` names a checkpoint under models/ on
        # the Volume (the driver uploads them). Smoke test / manual use.
        result = selfplay_gpu.remote(model, games, shapes, seed, out_subdir, sims,
                                     target_sims, target_share)
        print(json.dumps({k: v for k, v in result.items() if k not in ("out", "err")}, indent=2))
        print(result["out"].strip() or result["err"][-600:])
    elif task == "learn":
        # One generation from models/<model> on the Volume (smoke test / manual).
        result = learn.remote(gen, model, steps, batch, lr, 0.75, replay_window)
        print(json.dumps({k: v for k, v in result.items() if k not in ("lines", "err")}, indent=2))
        print("\n".join(result["lines"]) or result["err"][-800:])
    elif task == "arena":
        result = arena.remote(model, subdir, games, sims, shapes, seed)
        print(result["out"].strip() or result["err"][-800:])
    elif task == "closure":
        print(json.dumps(closure.remote(subdir, rows, columns, connect, cap), indent=2))
    else:
        raise SystemExit(f"unknown task {task}")

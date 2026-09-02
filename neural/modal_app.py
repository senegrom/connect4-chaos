"""Modal burst compute for the connect4-chaos program.

Everything CPU-heavy - exact solves, rank sidecars, exact-sample dataset
building, closure measurements and MCTS self-play - runs here as
finite Functions over one persistent Volume; only GPU training stays on
the local machine. Mirrors E:/AI/Modal-Codex-Lean-Lab/src/heavy_cpu_jobs.py:
burst Functions, 24 h timeout, resumable through Volume checkpoints (the
pair solver's block files are ordinary checkpoints, so re-invoking a solve
continues where the previous call stopped).

Run from the modal environment, e.g.:
  D:/PyEnv/modal/Scripts/python.exe -m modal run neural/modal_app.py \
      --task solve --rows 6 --columns 7 --connect 4 --mode chaos \
      --discover-through 28 --threads 32
  ... --task sidecars --subdir chaos-6x7-c4
  ... --task dataset --subdir classic-5x7-c4 --rows 5 --columns 7 --connect 4 \
      --mode classic --samples 150000
  ... --task selfplay --model E:/.../distilled.pt --rows 6 --columns 7 \
      --connect 4 --mode chaos --games 512 --sims 128 --workers 16
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
            samples: int, out_subdir: str):
    spec = f"{TABLES}/{subdir}:{rows}:{columns}:{connect}:{mode}"
    process = subprocess.run(
        ["python", "-m", "neural.build_dataset", f"{TABLES}/{out_subdir}", str(samples), spec],
        capture_output=True, text=True, cwd="/repo",
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


@app.function(image=image, cpu=4.0, memory=8 * 1024, timeout=24 * 60 * 60, volumes=MOUNTS)
def selfplay(model_bytes: bytes, rows: int, columns: int, connect: int, mode: str,
             games: int, sims: int, out_subdir: str, seed: int):
    model_path = f"/tmp/model-{seed}.pt"
    Path(model_path).write_bytes(model_bytes)
    process = subprocess.run(
        ["python", "-m", "neural.selfplay", model_path, f"{TABLES}/{out_subdir}",
         str(rows), str(columns), str(connect), mode, str(games), str(sims),
         f"sp-{rows}x{columns}c{connect}{mode}-s{seed}", str(20260901 + 7919 * seed)],
        capture_output=True, text=True, cwd="/repo",
    )
    tables.commit()
    return {"exit": process.returncode, "out": process.stdout[-1000:], "err": process.stderr[-2000:]}


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
         games: int = 256, sims: int = 128, workers: int = 8, cap: int = 30_000_000,
         spawn: bool = False):
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
        print(json.dumps(dataset.remote(subdir, rows, columns, connect, mode, samples, out_subdir), indent=2))
    elif task == "selfplay":
        model_bytes = Path(model).read_bytes()
        per_worker = max(1, games // workers)
        results = list(selfplay.map(
            [model_bytes] * workers, [rows] * workers, [columns] * workers,
            [connect] * workers, [mode] * workers, [per_worker] * workers,
            [sims] * workers, [out_subdir] * workers, list(range(workers)),
        ))
        for result in results:
            print(result["out"].strip() or result["err"][-300:])
    elif task == "closure":
        print(json.dumps(closure.remote(subdir, rows, columns, connect, cap), indent=2))
    else:
        raise SystemExit(f"unknown task {task}")

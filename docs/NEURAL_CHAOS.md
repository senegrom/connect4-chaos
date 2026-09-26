# Neural play for variable boards and rules

One network plays Connect-k on any board up to 10×10, classic or Chaos.
It ships in the browser as the **Neural** opponent and is trained by an
AlphaZero-style loop on rented GPUs, anchored throughout by the exact
solver tables.

## What ships

- The network exported to ONNX in fp16 (106 MB), 20 residual blocks × 384
  channels, 53.2 million parameters. It is **not** in this repository: it
  lives in a Cloudflare R2 bucket, read through the Worker in
  `workers/model-cdn/`, and `assets/neural/model.json` names the object.
  `scripts/publish-model-r2.mjs` puts a new generation there under a key
  that carries its name, so responses are immutable and a rollback is one
  line of that manifest. Three reasons it moved: GitHub stores no file
  over 100 MB and Pages cannot serve Git LFS; each generation added its
  full size to git history for good; and at 53 MB a part it exceeded the
  ceiling Chromium puts on one disk-cache entry, so nothing was ever
  stored and every visit paid the whole download again - about 950 of
  them would have spent Pages' 100 GB monthly allowance. R2 charges
  nothing for egress. The object is stored gzipped (98.7 MB) because R2
  serves exactly the bytes it holds and compresses nothing on the fly.
- `src/neural-runtime.js` keeps the downloaded model in Cache Storage,
  which has no per-entry ceiling: 106 MB writes in about 0.8 s and reads
  back in under 0.1 s, so a returning visitor makes no request at all.
  The page asks before that one-time download and shows its progress
  (`src/download-gate.js`). The vendored ONNX runtime still ships here
  (WebGPU build plus its WebAssembly fallback, 25 MB, 6.4 MB gzipped).
- `src/neural-runtime.js` loads the model on WebGPU when the browser has
  a usable GPU and on WebAssembly otherwise, measures how fast one
  evaluation is, and sizes the search to about 1.5 s per move (up to 512
  simulations on a desktop GPU, about ten on WebAssembly). A GPU
  that is busy with other work, loses its device, or crashed the page
  last time is avoided.
- `src/neural-search.js` runs the PUCT search over `src/engine.js`
  moves, so the browser player uses the same rules as the game.
- `src/neural-planes.js` encodes a position exactly as the trainer does;
  `tests/fixtures/neural-planes.json` pins that encoding from Python.

## Network

- Input canvas **10×10** (shape mask for smaller boards; Chaos rotations
  swap rows and columns mid-game, the canvas holds both orientations).
- Planes: mover pieces, opponent pieces, on-board mask, connect-length
  encoding (k ≤ 10), classic/Chaos flag, two repetition planes (the
  threefold rule is part of the game). Mover-relative throughout.
- Action head: **13 masked actions** = 10 drop columns + flip + two
  rotations. Value head: 3-way win/draw/loss softmax. A per-action Q head
  gives the search a first estimate for untried moves. Board-relative
  mirror augmentation, the game's only symmetry.

## Training loop

`neural/modal_loop.py` drives Modal H100 Functions (`neural/modal_app.py`):

- **Actors** (`neural/gpu_selfplay.py`, batched PUCT in
  `neural/gpu_mcts.py` over `neural/gpu_env.py` boards) play thousands
  of games in lockstep across all 412 board shapes from 4×1 to 10×10,
  classic and Chaos. Playout-cap randomisation: a quarter of plies get
  the deep search (256 simulations in the recipe), the rest a cheap one
  (32). With `-PolicyTarget gumbel`, used since generation 332, every ply
  teaches the policy through Gumbel MuZero's improved policy; with the
  default `visits`, only the deep plies do.
- **Learner** (`neural/distill.py`) trains on the exact-table shards
  (from `neural/build_dataset.py`; 35% of each batch at the recipe's
  replay fraction of 0.65) plus a replay window of the newest self-play
  positions, warm-starting from the previous generation.
- **Arena** (`neural/arena.py`) plays each fifth generation against the
  one five back over every board shape. Games come in pairs: the second
  replays the first's opening with the colours swapped, so each network
  meets the same position from either side and a network against itself
  scores exactly 50%. Neither network picks the openings: every opening
  move is drawn from the mean of both networks' search policies. So naming
  the two the other way round plays the same games, and each board's result
  mirrors exactly (`neural/test_arena.py`). Until 2026-09-24 each side drew
  its own opening moves, so the first-named network chose every
  first-player move. A match also turns cuDNN autotuning off, so a seed
  replays the same games in any container. That costs up to 2% at 32
  simulations and 3-6% at 128. `neural/search_quality.py` measures blunder
  rates against the exact tables on held-out positions.

### Throughput

The training path keeps H100s busy rather than waiting on Python:

- **Search:** GPU MCTS runs in static workspaces padded to a few batch
  widths. A whole simulation is shape-static and sync-free, so it replays
  as one CUDA graph per width and depth bound. On an H100, 8,192 games
  took 240 s this way, 399 s eagerly and about 520 s before the rewrite.
- **Self-play records:** self-play keeps repetition history and replay
  records on the GPU, and compacts planes to uint8 before one host
  transfer per batch. Games that reach the ply cap are discarded, not
  scored as draws.
- **Shards:** replay shards omit the constant Q tensor (`q_default=3`) and
  store their train/validation partition once. Exact shards use the same
  uint8 planes. Actors gzip their shards at level 1.
- **Learner:** the learner keeps the corpus on the GPU when it fits, and
  otherwise pins it in host memory. It trains in bf16 with channels-last
  convolutions and fused AdamW. The whole step (forward, loss, backward,
  optimizer) replays as one CUDA graph; `torch.compile` measured slower.
- **Optimizer state:** AdamW moments go to `<model>.opt` beside the
  checkpoint, so the next generation keeps its optimizer history without
  enlarging what actors download.

For local runs, `neural.gpu_selfplay` and `neural.distill` read switches
from the environment: `SELFPLAY_GRAPHS`, `SELFPLAY_CHANNELS_LAST`,
`SELFPLAY_FUSED`, `SELFPLAY_PROFILE`, `DISTILL_GRAPH`,
`DISTILL_PROFILE_STEPS`, `DISTILL_GPU_DATA`, `DISTILL_GPU_RESERVE_GB`,
`DISTILL_PIN_MEMORY`, `DISTILL_CHANNELS_LAST`, `DISTILL_FUSED_ADAMW`,
`DISTILL_RESET_OPTIMIZER` and `DISTILL_PERSIST_OPTIMIZER`.

A Modal container does not inherit the caller's environment:
- `selfplay_gpu` sets the self-play switches from its `graphs`,
  `channels_last`, `fused` and `profile` arguments;
- `learn` sets `DISTILL_PROFILE_STEPS` from `profile_steps`;
- the other learner switches keep their defaults on Modal.

Every fast path has an eager or CPU fallback that the CPU tests exercise.

`neural/export_onnx.py` exports a checkpoint for the browser, and the
shipped model is replaced only at milestones.

## Resuming from the shipped network

Every PyTorch checkpoint went with the Modal Volume on 2026-09-15; the
newest weights left are the gen-504 export (`big504-808970a6d2.onnx`, fp16).
`neural/import_onnx.py` turns an export back into a trainable checkpoint.
The export has every BatchNorm folded into its convolution, so the import
rebuilds each one: the convolution keeps the folded weight, and the
normalisation after it is set from the per-channel mean and variance of that
convolution's output over positions from cheap tactical playouts on every
board shape, which keeps eval mode exact and makes train mode, which
normalises by batch statistics, stay close to it:

```sh
python -m neural.import_onnx .model-cache/big504-808970a6d2.onnx big504-808970a6d2.pt 4096
```

It checks all three heads against onnxruntime and reports how far train mode
moves them. On 2026-09-23 the gen-504 import (4,096 calibration positions,
about ten minutes on two CPU threads) matched onnxruntime to 1.0e-3 (policy),
1.2e-3 (W/D/L) and 2.3e-3 (Q) in probability, and matched the folded network
in eval mode to 1e-4 in logits. On a batch of 1,024 fresh positions, train
mode moved the probabilities by 0.5%, 1.2% and 1.1% on average and changed
the policy's top move on 2.3% of positions (0.7%, 1.6%, 1.8% and 4.7% on a
batch of 256); without the calibration, on the same 256, those figures were
18%, 44%, 33% and 77%.

The import carries no optimizer state, so the first generation starts AdamW
from nothing; `neural/distill.py` then ramps the learning rate up over a
fifth of the run (at most 1,000 steps) instead of taking full-size steps
before the moment estimates settle (`DISTILL_WARMUP_STEPS`, or
`learn(warmup_steps=...)`, overrides it).

The 505-555 run (2026-09-23/24) resumed from that import with the recipe
in step 3 below and stopped itself at 555. No generation beat 504 in the
arena:
- Against 504 at 32 simulations, each generation scored 48.5-49.7%; 555
  scored 49.5%, and 49.8% at 128 simulations.
- The held-out blunder rates fell by about half over the run.

So 504 stays shipped. The run's likely handicaps were the import's
optimizer reset and the missing Chaos 6×6 and 5×7 samples (next section).

On 2026-09-25 everything from that run was deleted, on Modal and off it:
its checkpoints (555 included), the exact corpus (`datasets-v3/`) and its
replay (`replay-gpu/`). The Volume holds only `models/big504-808970a6d2.pt`.
To resume from it:

1. Rebuild the exact corpus (next section). The driver refuses to start
   while `datasets-v3/` holds no training shards.
2. Deploy, then run the GPU tests on it (`--task gpu-test`), since CI has
   no GPU:
   - `test_search_settings` and `test_search_history` with `--args=""`;
   - `test_arena` with `--args cuda`;
   - `test_graph_search` with `--args models/big504-808970a6d2.pt`;
   - `test_gpu_mcts` with `--args "models/big504-808970a6d2.pt cuda 32"`.
3. Start the loop with the recipe that trained 332 to 504:
   `scripts/launch-modal-loop.ps1 -Init big504-808970a6d2.pt -Gen 505 -K 4
   -Games 8192 -Lr 2e-4 -MinNew 1000000 -Sims 32 -TargetSims 256 -QSeed 1
   -ReplayFraction 0.65 -PolicyTarget gumbel -RootValueWeight 0.5 -UntilGen
   555`.
   - With `replay-gpu/` empty, the first learner waits until the actors
     have written a whole replay window: 4 million positions, about twenty
     runs of 8192 games and $8. A learner with no replay would train on the
     exact rows alone.
   - `-UntilGen` stops the loop once that generation is published: the
     self-play still running is cancelled, and the arena due at that
     generation still plays.
   - An imported checkpoint has no lineage record, so it is a root, and the
     first arena comes five generations after it.

## The exact-table corpus

The learner reads its exact shards from one directory on the Modal Volume:
`datasets-v3` unless told otherwise (`learn(exact_subdir=...)`, the
driver's 21st argument, `scripts/launch-modal-loop.ps1 -ExactSubdir`,
`--exact-subdir` on `modal_app.py`). A learner whose directory is missing,
or holds no training rows, fails instead of quietly training on replay
alone with its Q loss at zero; `allow_no_exact` (`DISTILL_ALLOW_NO_EXACT=1`
for a local run) is the explicit way to do that on purpose.

The corpus that went with the Volume on 2026-09-15 was rebuilt on
2026-09-23/24 as `datasets-v3`, and deleted with the rest of the 505-555
run on 2026-09-25, so it has to be rebuilt before training resumes. That
rebuild covered thirteen of the fifteen solved boards, each sampled
uniformly over its reachable states, 25,000 positions to a shard:

| rule set | boards (Connect 4 unless marked) | shards |
| --- | --- | --- |
| classic | 4×4 c3, 4×4, 4×5, 4×6, 5×5, 5×6, 5×7, 6×6 | `-0000` to `-0015` each |
| chaos | 4×4 c3, 4×4, 4×5, 5×5, 5×6 | `-0000` to `-0015` each |

Chaos 6×6 (`-0000` to `-0015`) and 5×7 (`-0000` to `-0013`) were not drawn.
Sampling a 360-500 GB table through a 32 GB Modal container ran at a few
positions a second. To draw them locally from the solved tables:
1. Rebuild any stale rank sidecars with `scripts/build-pair-rank-sidecars.py`.
2. Run `python -m neural.build_dataset <out> <samples> <dir>:R:C:4:chaos`
   from local SSD. `DATASET_START_INDEX` numbers the first shard.

Shard `-0000` of each board is its held-out shard, sampled only from the
positions `neural/data_split.py` reserves; every later shard avoids them.
The solved tables were archived off Modal on 2026-09-24. On Modal, the
corpus is built per board in the order below. Every command is `modal run
neural/modal_app.py` from the Modal environment, and `M` is `chaos` or
`classic`:

1. Solve: `--task solve --rows R --columns C --connect K --mode M`, with
   `--threads 32` for the large boards. The pair tables land in
   `M-RxC-cK`. Chaos 5×6 took 13 minutes on 32 threads, chaos 6×6 71
   minutes and chaos 5×7 6.4 hours; the boards up to 5×5 take minutes.
2. Rank sidecars and the first 150,000 samples: `--task prepare --subdir
   M-RxC-cK --rows R --columns C --connect K --mode M --samples 150000
   --out-subdir datasets-v3` (shards `-0000` to `-0005`).
3. Ten more training shards: `--task dataset --subdir M-RxC-cK --rows R
   --columns C --connect K --mode M --samples 250000 --start-index 6
   --out-subdir datasets-v3` (shards `-0006` to `-0015`, 375,000 training
   positions a board), spawned for the fourteen boards at once by a local
   script.

A dataset call seeds its sampler from its start index, so the
same commands on the same tables draw the same positions. Without
`--out-subdir` both `prepare` and `dataset` write to `datasets/`, which the
learner does not read unless pointed at it.

## How well it plays

Blunder rate is the share of positions where the move chosen is not exactly
optimal, measured against the solved tables on held-out positions the
network never trained on. The distinction that matters is *what chooses the
move*: the policy head answers instantly from the current position, while
the search looks ahead, and only the search is what plays. On the same
network - the shipped generation 504 - on 2048 held-out positions per board
(the positions reserved by `neural/data_split.py`, which no generation has
trained on):

| board | policy head | 32 sims | 128 sims | 256 sims |
| --- | --- | --- | --- | --- |
| 6×6 classic | 0.34% | 0.00% | 0.00% | 0.00% |
| 5×7 classic | 0.88% | 0.15% | 0.05% | 0.05% |
| 5×6 classic | 0.34% | 0.05% | 0.05% | 0.00% |
| 4×6 classic | 0.24% | 0.00% | 0.00% | 0.00% |
| 6×6 chaos | 5.37% | 1.07% | 0.73% | 0.59% |
| 5×6 chaos | 3.71% | 0.59% | 0.20% | 0.20% |
| 5×5 chaos | 4.79% | 0.83% | 0.44% | 0.24% |
| 4×5 chaos | 4.54% | 0.88% | 0.44% | 0.34% |

Pooled over all fifteen solved boards the player misses 0.29% of positions
at 32 simulations, 0.16% at 128 and 0.11% at 256 (chaos 0.60 / 0.32 / 0.23%,
classic 0.02 / 0.01 / 0.01%). Chaos is harder for the same
network by roughly an order of magnitude, which is what the transforms
cost: they move material across the whole board, so a position's value can
turn on a line that a drop could never create.

These rates have barely moved for a hundred generations, and that is a
property of the boards they are measured on rather than of the training:
every solved board is small, and on small boards the arena finds nothing
left to win either - at 128 simulations generation 504 scores 50.1% against
the generation it replaced on boards of 30 cells or fewer, and 53.3% on
larger ones. Where the tables can see, the player is already close to
exact; the boards a person actually plays on are the ones only the arena
reaches.

Search depth grows with the simulation count but slowly, since each
doubling adds about one ply to the principal line: 6 plies at 16
simulations, 9 at 64, 11 at 128, 12 at 256, 14 at 512. Doubling from 128 to
256 is worth only 2.4 points of playing strength head to head, and the
exploration constant is flat anywhere from 1.5 upward.

How many simulations fit in the budget is a separate question, and the
answer changed: the search evaluates a batch of leaves per network call
rather than one. A single position leaves the GPU almost idle - one costs
18.1 ms in a browser here and eight cost 20.2 ms - so batching cut the cost
of a simulation about fourfold and the budget now runs 512 simulations
where it ran 99. Each leaf on a collected path carries a virtual loss until
its result arrives, so a batch explores several lines instead of eight
copies of one.

Batching is a GPU win only: WebAssembly is already busy with one position,
and a batch there merely makes a single call block that much longer. What
helps a phone is threads, and multi-threaded WebAssembly needs
SharedArrayBuffer, which only a cross-origin isolated page gets - declared
by headers GitHub Pages does not send. `cross-origin-isolation-worker.js`
is a service worker that adds them to what it serves, so a returning
visitor's page is isolated and inference spreads over four threads: one
position costs 410 ms on one thread and about 145 ms on four, which is
ten simulations a move instead of four. It is registered without a reload,
so the visit that installs it is never interrupted. `?coi=off` removes it
and keeps it off on later visits, until `?coi=on`.

`neural/search_quality.py` produces this table.

### The only-winning-move benchmark in CI

The table above is measured by hand, through the Python stack. What CI
measures is the network the page runs, through `src/neural-runtime.js` and
`src/neural-search.js`: `tests/strength/neural-strength.mjs` puts it to 60
positions in which exactly one legal move keeps a forced win and at least
one loses - 24 on standard 6×7, 12 on smaller classic boards (5×6, 6×5,
5×5) and 24 in Chaos (4×4 Connect 3; 4×4, 4×5, 5×4 and 5×5 Connect 4).
`scripts/neural-strength-positions.mjs` finds them in "sensible random"
games and proves every move's value with the exact solvers. It keeps a
position only when finding the winner takes lookahead: the move is not an
immediate win, nor merely the one move that parries a threat.

Each position goes to the policy head alone and to the search - 32
simulations, the leaves evaluated in batches of eight as on WebGPU - and
the search is scored twice: by the move it plays (the most visited) and by
the move its own values rate highest. Generation 504 finds:

| positions | policy head | search move | search values |
| --- | --- | --- | --- |
| 24 classic 6×7 | 19 | 19 | 22 |
| 12 smaller classic | 11 | 12 | 12 |
| 24 Chaos | 20 | 21 | 24 |
| all 60 | 50 | 52 | 58 |

The test fails when any figure in the last row drops by more than three, or
when the search plays fewer winning moves than the policy head finds, less
two. The move alone would not have caught the batch-packing slip of
September 2026. Put back, it still leaves the search playing 54 of these
moves: the root is evaluated on its own, so the prior steering the search
stays intact, and on tactical positions the wins the rules detect carry it.
The garbage reaches the search's values instead, which then rate the winner
highest in only 47, and the test fails.

`npm run test:strength` runs it. It needs the model (`NEURAL_MODEL` set to
the `.onnx`, or `NEURAL_MODEL_DOWNLOAD=1`) and takes about five minutes on
three WebAssembly threads; `NEURAL_STRENGTH_THREADS` sets the count, which
is otherwise up to four. CI runs it as the `neural-strength` job, and Pages
waits for it. `node scripts/neural-strength-positions.mjs` regenerates the
positions, the same file every time, in about a quarter of an hour on one
core, and `npm test` re-proves the cheapest 35 of them; new positions need
the numbers above measured again and written into the test's `CALIBRATED`.


## Open questions tracked

- Does ≤5×7 distillation + 6×7 self-play generalize, scored against the
  exact 6×6 table?
- Can the net serve as move-ordering for a future 6×7 winning-strategy
  certificate search (5×7's first-player win warns that 6×7 may be
  decided, in which case the draw-assuming certificate route cannot
  close it)?

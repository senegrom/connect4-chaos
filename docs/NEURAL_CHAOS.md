# Neural play for variable boards and rules

One network plays Connect-k on any board up to 10×10, classic or Chaos.
It ships in the browser as the **Neural** opponent and is trained by an
AlphaZero-style loop on rented GPUs, anchored throughout by the exact
solver tables.

## What ships

- `assets/neural/model.onnx.part1` and `.part2`: the network exported to
  ONNX in fp16 (106 MB), 20 residual blocks × 384 channels, 53.2 million
  parameters, with the vendored ONNX runtime (WebGPU build plus its
  WebAssembly fallback, 25 MB). GitHub stores no file over 100 MB and
  Pages cannot serve Git LFS, so the export is split into equal parts
  that `cat` - or the browser, streaming each into its own slice of one
  buffer - joins back byte for byte. The page asks before the one-time
  download and shows its progress (`src/download-gate.js`).
- `src/neural-runtime.js` loads the model on WebGPU when the browser has
  a usable GPU and on WebAssembly otherwise, measures how fast one
  evaluation is, and sizes the search to about 1.5 s per move (up to 512
  simulations on a desktop GPU, a handful on WebAssembly). A GPU
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
  the deep search and become policy targets, the rest a cheap search and
  teach only the value head.
- **Learner** (`neural/distill.py`) trains on the exact-table shards
  (a quarter of each batch, from `neural/build_dataset.py`) plus a
  replay window of the newest self-play positions, warm-starting from
  the previous generation.
- **Arena** (`neural/arena.py`) plays each fifth generation against the
  one five back over every board shape; `neural/search_quality.py`
  measures blunder rates against the exact tables on held-out positions.

`neural/export_onnx.py` exports a checkpoint for the browser, and the
shipped model is replaced only at milestones.

## How well it plays

Blunder rate is the share of positions where the move chosen is not exactly
optimal, measured against the solved tables on held-out positions the
network never trained on. The distinction that matters is *what chooses the
move*: the policy head answers instantly from the current position, while
the search looks ahead, and only the search is what plays. On the same
network - the shipped generation 453 - on 2048 held-out positions per board
(the positions reserved by `neural/data_split.py`, which no generation has
trained on):

| board | policy head | 32 sims | 128 sims | 256 sims |
| --- | --- | --- | --- | --- |
| 6×6 classic | 0.44% | 0.00% | 0.00% | 0.00% |
| 5×7 classic | 0.54% | 0.20% | 0.10% | 0.05% |
| 5×6 classic | 0.49% | 0.10% | 0.05% | 0.00% |
| 4×6 classic | 0.10% | 0.05% | 0.05% | 0.05% |
| 6×6 chaos | 5.13% | 0.83% | 0.49% | 0.44% |
| 5×6 chaos | 3.81% | 0.59% | 0.29% | 0.24% |
| 5×5 chaos | 4.88% | 1.03% | 0.44% | 0.29% |
| 4×5 chaos | 4.64% | 0.83% | 0.34% | 0.20% |

Pooled over all fifteen solved boards the player misses 0.30% of positions
at 32 simulations, 0.15% at 128 and 0.10% at 256 (chaos 0.59 / 0.29 / 0.21%,
classic 0.05 / 0.02 / 0.01%). Chaos is harder for the same
network by roughly an order of magnitude, which is what the transforms
cost: they move material across the whole board, so a position's value can
turn on a line that a drop could never create.

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

`neural/search_quality.py` produces this table.


## Open questions tracked

- Does ≤5×7 distillation + 6×7 self-play generalize, scored against the
  exact 6×6 table?
- Can the net serve as move-ordering for a future 6×7 winning-strategy
  certificate search (5×7's first-player win warns that 6×7 may be
  decided, in which case the draw-assuming certificate route cannot
  close it)?

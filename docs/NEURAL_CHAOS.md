# Neural play for variable boards and rules

One network plays Connect-k on any board up to 10×10, classic or Chaos.
It ships in the browser as the **Neural** opponent and is trained by an
AlphaZero-style loop on rented GPUs, anchored throughout by the exact
solver tables.

## What ships

- `assets/neural/model.onnx`: the network exported to ONNX in fp16
  (47 MB), 20 residual blocks × 256 channels, 23.7 million parameters,
  with the vendored ONNX runtime (WebGPU build plus its WebAssembly
  fallback, 25 MB). The page asks before the one-time download and shows
  its progress (`src/download-gate.js`).
- `src/neural-runtime.js` loads the model on WebGPU when the browser has
  a usable GPU and on WebAssembly otherwise, measures how fast one
  evaluation is, and sizes the search to about 1.5 s per move (roughly
  100 simulations on a desktop GPU, a handful on WebAssembly). A GPU
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
network (generation 60; the shipped model is generation 65), 1024 held-out positions per board:

| board | policy head | 32 sims | 128 sims | 512 sims |
| --- | --- | --- | --- | --- |
| 6×6 classic | 0.20% | 0.00% | 0.00% | 0.00% |
| 5×7 classic | 0.29% | 0.10% | 0.00% | 0.00% |
| 5×6 classic | 0.88% | 0.00% | 0.00% | 0.00% |
| 4×6 classic | 0.10% | 0.00% | 0.00% | 0.00% |
| 6×6 chaos | 3.52% | 0.88% | 0.49% | 0.29% |
| 5×6 chaos | 2.93% | 1.07% | 0.68% | — |
| 5×5 chaos | 3.12% | 0.49% | 0.29% | 0.20% |
| 4×5 chaos | 2.34% | 0.88% | 0.68% | 0.68% |

With 128 simulations the player chose an optimal move in every sampled
classic position, which bounds its blunder rate under about 0.3%, and
missed 0.3% to 0.7% of chaos positions. Chaos is harder for the same
network by roughly an order of magnitude, which is what the transforms
cost: they move material across the whole board, so a position's value can
turn on a line that a drop could never create.

Search depth grows with the simulation count but slowly, since each
doubling adds about one ply to the principal line: 6 plies at 16
simulations, 9 at 64, 11 at 128, 12 at 256, 14 at 512. Doubling from 128 to
256 is worth only 2.4 points of playing strength head to head, and the
exploration constant is flat anywhere from 1.5 upward, so the search itself
is at its plateau. Further gains have to come from the network.

`neural/search_quality.py` produces this table.


## Open questions tracked

- Does ≤5×7 distillation + 6×7 self-play generalize, scored against the
  exact 6×6 table?
- Can the net serve as move-ordering for a future 6×7 winning-strategy
  certificate search (5×7's first-player win warns that 6×7 may be
  decided, in which case the draw-assuming certificate route cannot
  close it)?

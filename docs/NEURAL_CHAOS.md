# AlphaZero-style neural play for variable boards and rules

Design for a single network that plays Connect-k on any board up to
10×10, classic or Chaos, trained first from the exact solver tables and
then by self-play. Hardware target: the local RTX 5070 Ti (16 GB) via
`D:/PyEnv/torch` (torch 2.13, CUDA 13); CPU-side work respects the
machine's 3-idle-core compute budget.

## Why this project is unusually well-armed

The pair/layered solver checkpoints are a supervised dataset of ~3×10¹¹
exactly-labelled positions across 25+ solved configurations. The `.bits`
files support uniform sampling over reachable states (pick a random set
bit; the value byte is one lookup away — `scripts/perfect-chaos-remote-lookup.mjs`
documents the decoding). That gives:

- **Perfect distillation** instead of a cold start: train on exact
  win/draw/loss labels and exact optimal-move sets (child lookups) for
  every solved board before any self-play.
- **Rigorous evaluation forever**: blunder rate against ground truth,
  with 6×6 c4 and 5×7 c4 usable as held-out generalization tests.

## Network

- Input canvas **10×10** (shape mask for smaller boards; Chaos rotations
  swap rows/columns mid-game, the canvas holds both orientations).
- Planes: mover pieces, opponent pieces, on-board mask, connect-length
  encoding (k ≤ 10), classic/Chaos flag, two repetition planes (the
  threefold rule is part of the game, as in AlphaZero's chess planes).
  Mover-relative throughout, matching the tables.
- Action head: **13 masked actions** = 10 drop columns + flip + two
  rotations. Value head: 3-way win/draw/loss softmax (draws dominate and
  the labels are exact). Mirror augmentation — the game's only symmetry.
- 6–10 residual blocks × 64–128 filters; small enough that the 16 GB
  card allows large batches and fast iteration.

## Game core

Python bitboards using native big ints — column stride (rows+1) as in
the native solvers, so a 10×10 board is a 110-bit word and the same
shift-chain line detection applies unchanged. Cross-validated against
the solved tables through the lookup module before any training. The
C++ solvers keep their ≤7×7 world; boards beyond it are self-play-only
territory (no oracles exist there — that is the point).

## Pipeline

1. Game core + tests (CPU-trivial; can run today).
2. Table sampler → streaming (position, WDL, optimal-move mask) batches.
3. Distillation on GPU — CPU-light, runs alongside the exact-solver jobs.
4. MCTS self-play fine-tuning on unsolved sizes (6×7 c4 first, then up
   to 10×10) — CPU-hungry, scheduled when solver jobs free cores. PUCT,
   root Dirichlet noise, replay buffer, league eval anchored by tables.
5. Export ONNX → a browser "Neural" opponent tier for boards where
   Perfect cannot exist. (The game UI currently offers boards up to 7×7;
   larger boards need UI enablement before the tier ships.)

## How well it plays

Blunder rate is the share of positions where the move chosen is not exactly
optimal, measured against the solved tables on held-out positions the
network never trained on. The distinction that matters is *what chooses the
move*: the policy head answers instantly from the current position, while
the search looks ahead, and only the search is what plays. On the same
network (generation 60), 1024 held-out positions per board:

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

# Perfect play in 6×7 Chaos Mode

Chaos Mode is not an ordinary game tree. Flip and rotation actions do not add a piece, so positions can lead back to earlier positions. The implementation also ends the round automatically when the same board, with the same player to move, appears for the third time.

This document records the exact model, the solver now in the repository, and the remaining proof work before the user interface may honestly label full 6×7 Chaos play as **Perfect**.

## Exact game model

A searchable position contains:

- the gravity-settled board;
- the side to move;
- the Connect length;
- the current orientation, represented by the board dimensions;
- the automatic threefold-repetition rule.

A drop increases the number of pieces. A flip or rotation keeps the piece count unchanged, changes the orientation or piece ordering, and reapplies gravity. A transform that creates a line for both players is lost by the transforming player.

The solver normalises the side to move as Red and folds horizontally mirrored positions together. Under horizontal reflection, clockwise and counter-clockwise rotations exchange roles; drops map to the reflected column and flips remain flips.

## Why unresolved cycles are draws

For win/draw/loss values from a fresh position, the threefold rule can be solved as a finite loopy game:

1. Positions with an immediate winning move enter the winning attractor.
2. Positions whose every move enters the opponent's winning attractor enter the losing attractor.
3. These implications are propagated backwards until no additional position can be classified.
4. Every remaining strongly connected region is a draw region.

The solver also records an attractor rank.

- A winning strategy chooses a losing child with a strictly smaller rank, so it reaches a terminal win without cycling.
- At a losing position, every move gives the opponent a smaller-ranked winning position.
- At a draw position, at least one move remains inside the draw region or ends in an immediate draw. Repeating that policy eventually triggers the real automatic threefold draw if neither player leaves the region.

The rank requirement is important. Merely finding a winning and losing cycle would not prove that the selected move actually reaches a terminal win.

## Implemented foundation

`src/chaos-solver.js` now provides a complete graph builder and ranked retrograde solver, independently mirrored by the compact C++20 engine in `native/perfect-chaos.cpp`. The JavaScript runtime:

- handles drops, flips, clockwise rotations and counter-clockwise rotations;
- applies the simultaneous-win loss rule exactly;
- canonicalises horizontal reflection and side-to-move colours;
- deduplicates equivalent actions;
- distinguishes proved wins, draws and losses;
- selects finite-progress winning moves;
- fails closed if a deterministic state-count safety limit is exceeded.

The normal AI automatically uses this solver for Chaos positions with six or fewer empty cells when every recorded position has occurred at most once. The solver has no wall-clock cutoff. Medium, Hard and Brutal fall back to their ordinary search only if the exact graph exceeds the configured state boundary. A direct Perfect request never falls back heuristically.

Before that frontier, the bounded Chaos engine now folds horizontally mirrored children together and reuses alpha-beta bounds only when the complete repetition multiset matches. Rotation actions are mirrored by exchanging clockwise and counter-clockwise, so the cache cannot return an orientation-invalid move. This improves practical depth without treating different repetition histories as interchangeable.

`node scripts/perfect-chaos.mjs verify` checks deterministic reference games, including:

- the complete 2×2 Connect-2 Chaos game;
- the complete 3×3 Connect-3 Chaos game;
- a 6×7 late-game position whose exact winning move is clockwise rotation.

`node scripts/perfect-chaos.mjs enumerate --depth 8` reproduces the canonical root layers through 212,379 states. `node scripts/perfect-chaos.mjs frontier` can solve newline-delimited frontier positions deterministically and supports sharding. The verified counts and current fail-closed runtime boundary are committed in `data/perfect-chaos-foundation.manifest.json`.

### Existing repetition history

The positional W/D/L result remains valid when earlier positions have occurred once. A ranked winning policy strictly decreases rank and therefore never revisits a state; a ranked losing state gives the opponent such an acyclic win; and an earlier repetition can only finish draw-region play as a draw sooner. Once any position has already occurred twice, however, the next visit is an immediate draw and can create a history-specific resource. The exact route therefore fails closed whenever a repetition count exceeds one.

### Independent native cross-check

`native/perfect-chaos.cpp` uses compact mover/opponent masks, the same horizontal canonicalisation and an independently written graph builder and retrograde implementation. `scripts/perfect-chaos-native.mjs` compiles it with a C++20 compiler and requires exact agreement with the JavaScript engine on the deterministic 2×2, 3×3 and 6×7 reference graphs, including the 2,585-state rotation fixture. This is not yet the separately designed algorithm required for a final root attestation, but it catches representation, transform and graph-construction drift while providing the base for larger sharded frontiers.

## Correctness tests

The automated suite covers:

- closed cycles resolving to draws;
- a finite terminal win taking priority over a cycle;
- losses being classified only after every action is proved losing;
- horizontal action symmetry, including rotation direction exchange;
- agreement with an independent literal threefold-history minimax on a complete tiny game;
- the exact value of the empty 3×3 Connect-3 Chaos game;
- strict rank reduction along selected winning moves;
- fail-closed graph limits;
- deterministic, shard-complete frontier output;
- JavaScript/native agreement on canonical state counts and the 6×7 action;
- exact 6×7 endgame routing through both the main AI entry point and a real browser worker.

## Why the full empty 6×7 board is not labelled Perfect yet

The empty 6×7 Chaos position has a much larger reachable graph than classic Connect Four. Transformations create large same-piece-count orbits, and rotations alternate between 6×7 and 7×6 orientations. A bounded breadth-first measurement already reaches hundreds of thousands of canonical states within nine plies; a complete root closure needs a compact native representation, resumable shards and a committed proof artifact.

The UI therefore still disables **Perfect** when Chaos Mode is selected. Enabling that label before the root closure is verified would silently overstate the result.

## Route to a complete Perfect Chaos release

1. Extend the compact native mask solver from verified frontier graphs to resumable disk-backed components.
2. Generate exact late-game frontier records in deterministic shards.
3. Run a counterexample-guided strategy search from both starting roles, exploring every opponent action while fixing only proved AI actions.
4. Use the exact frontier as the handoff and solve every remaining loopy component by ranked retrograde analysis.
5. Store strategy records with value, action, rank and orientation metadata.
6. Independently replay the complete adversarial closure under the literal threefold rule.
7. Extend the current real-browser exact-endgame worker fixture and require the final strategy loader to fail closed on any missing, malformed or illegal record.
8. Enable the Perfect option in Chaos Mode only after both starting-role closures pass in CI.

The existing classic Perfect strategy remains unchanged and independently verified.

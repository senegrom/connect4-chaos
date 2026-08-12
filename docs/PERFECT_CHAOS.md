# Perfect play in 6×7 Chaos Mode

Chaos Mode is not an ordinary game tree. Flip and rotation actions do not add a piece, so positions can lead back to earlier positions. The implementation also ends a round automatically when the same board, with the same player to move, appears for the third time.

This document records the exact model, the verified solvers now in the repository, the layered non-losing prefix certificate, and the remaining proof work before the interface may honestly label full 6×7 Chaos play as **Perfect**.

## Exact game model

A searchable position contains:

- the gravity-settled board;
- the side to move;
- the Connect length;
- the current orientation, represented by the board dimensions;
- the automatic threefold-repetition rule.

A drop increases the number of pieces. A flip or rotation keeps the piece count unchanged, changes the orientation or piece ordering, and reapplies gravity. A transform that creates a line for both players is lost by the transforming player.

The solvers normalise pieces relative to the side to move and fold horizontally mirrored positions together. Under horizontal reflection, clockwise and counter-clockwise rotations exchange roles; drops map to the reflected column and flips remain flips.

## Why unresolved cycles are draws

For win/draw/loss values from a fresh position, the threefold rule can be solved as a finite loopy game:

1. Positions with an immediate winning move enter the winning attractor.
2. Positions whose every move enters the opponent's winning attractor enter the losing attractor.
3. These implications propagate backwards until no additional position can be classified.
4. Every remaining closed region is a draw region.

The exact endgame solver also records an attractor rank.

- A winning strategy chooses a losing child with a strictly smaller rank, so it reaches a terminal win without cycling.
- At a losing position, every move gives the opponent a smaller-ranked winning position.
- At a draw position, at least one move remains inside the draw region or ends in an immediate draw. Repeating that policy eventually triggers the real automatic threefold draw if neither player leaves the region.

The rank requirement is important. Merely finding a winning and losing cycle would not prove that the selected move actually reaches a terminal win.

## Exact endgame foundation

`src/chaos-solver.js` provides a complete graph builder and ranked retrograde solver, independently mirrored by the compact C++20 engine in `native/perfect-chaos.cpp`. The runtime solver:

- handles drops, flips, clockwise rotations and counter-clockwise rotations;
- applies the simultaneous-win loss rule exactly;
- canonicalises horizontal reflection and side-to-move colours;
- deduplicates equivalent actions;
- distinguishes proved wins, draws and losses;
- selects finite-progress winning moves;
- fails closed if a deterministic state-count safety limit is exceeded.

The normal AI automatically uses this solver for standard 6×7 Chaos positions with six or fewer empty cells when every recorded position has occurred at most once. The solver has no wall-clock cutoff. Medium, Hard and Brutal fall back to ordinary bounded search only if the exact graph exceeds the configured state boundary. A direct Perfect request never falls back heuristically.

Before that frontier, the bounded Chaos engine folds horizontally mirrored children together and reuses alpha-beta bounds only when the complete repetition multiset matches. Rotation actions are mirrored by exchanging clockwise and counter-clockwise, so the cache cannot return an orientation-invalid move. This improves practical depth without treating different repetition histories as interchangeable.

`node scripts/perfect-chaos.mjs verify` checks deterministic reference games, including:

- the complete 2×2 Connect-2 Chaos game;
- the complete 3×3 Connect-3 Chaos game;
- a 6×7 late-game position whose exact winning move is clockwise rotation.

`node scripts/perfect-chaos.mjs enumerate --depth 8` reproduces the canonical root layers through 212,379 states. `node scripts/perfect-chaos.mjs frontier` solves newline-delimited frontier positions deterministically and supports sharding. The verified counts and fail-closed runtime boundary are committed in `data/perfect-chaos-foundation.manifest.json`.

### Existing repetition history

The positional W/D/L result remains valid when earlier positions have occurred once. A ranked winning policy strictly decreases rank and therefore never revisits a state; a ranked losing state gives the opponent such an acyclic win; and an earlier repetition can only finish draw-region play as a draw sooner. Once any position has already occurred twice, however, the next visit is an immediate draw and can create a history-specific resource. The exact route therefore fails closed whenever a repetition count exceeds one.

### Independent native cross-check

`native/perfect-chaos.cpp` uses compact mover/opponent masks, the same horizontal canonicalisation and an independently written graph builder and retrograde implementation. `scripts/perfect-chaos-native.mjs` compiles it with a C++20 compiler and requires exact agreement with the JavaScript engine on the deterministic 2×2, 3×3 and 6×7 reference graphs, including the 2,585-state rotation fixture.

## Layered prefix safety certificate

Version 1.10 adds `native/perfect-chaos-prefix.cpp` and `scripts/perfect-chaos-prefix.mjs`. They address the opposite end of the game: proving that a fixed strategy cannot lose from an empty board before a selected piece-count frontier.

For each starting role, a truncated safety game is solved with these rules:

- at an AI state, the state is losing only when every legal action is losing;
- at an opponent state, the state is losing when any legal action is losing;
- an AI win, terminal draw, or accepted frontier state is safe;
- states left outside the least losing attractor form the non-losing region;
- one deterministic safe action is stored for every reachable AI state;
- every legal opponent action remains in the replay closure.

The graph quotient includes horizontal reflection. A quotient cycle lifts to an actual orbit of size at most two: the concrete position is either the same canonical representative or its mirror. Repeating the quotient cycle therefore repeats an exact board and side to move after at most two traversals, so the real threefold rule ends the line as a draw. The certificate does not treat arbitrary search repetition as a win.

### Compositional boundaries

A monolithic ten-piece graph is unnecessarily large because it materialises positions that the chosen opening policy never reaches. The committed proof is instead split into exact, linked segments:

- empty board → 8 placed pieces;
- 8 → 10 placed pieces;
- 10 → 12 placed pieces.

The output frontier file from one segment is the exact binary input root set of the next segment. A later segment may discover that some incoming roots are losing before its next boundary. Those roots are written to a rejection file and fed back into the preceding safety game as losing boundary states. Synthesis repeats until every reachable boundary root extends safely.

This counterexample feedback is material:

- Red required no rejection at 8 pieces and 69 rejected states at 10 pieces.
- Yellow required 86 rejected states at 8 pieces and 758 rejected states at 10 pieces.

The rejected states are committed alongside the policies so the refinement is reproducible rather than hidden in a generation log.

### Verified twelve-piece closure

The committed reference is in `data/perfect-chaos-prefix/manifest.json`. It contains fixed-size binary policy, frontier and rejection tables plus a SHA-256 digest for every file.

For the AI playing Red:

| Segment | Input roots | Policy entries | Closure states | Output frontier |
|---|---:|---:|---:|---:|
| 0 → 8 | 1 | 1,299 | 3,161 | 1,477 |
| 8 → 10 | 1,477 | 5,058 | 13,397 | 6,912 |
| 10 → 12 | 6,912 | 22,715 | 57,390 | 28,378 |

For the AI playing Yellow:

| Segment | Input roots | Policy entries | Closure states | Output frontier |
|---|---:|---:|---:|---:|
| 0 → 8 | 1 | 3,863 | 9,581 | 4,522 |
| 8 → 10 | 4,522 | 15,109 | 40,188 | 20,585 |
| 10 → 12 | 20,585 | 66,944 | 172,472 | 86,223 |

Across the final segment, the independent replay follows 229,862 canonical closure states. Every policy record is reachable, every opponent action is explored, no AI-loss terminal is reachable, and the recomputed sorted frontier is byte-for-byte identical to the committed frontier.

The result is a **non-losing prefix certificate**, not a full game solution. Every adversarial line under the emitted strategy does one of four things before or at twelve placed pieces:

1. reaches an AI win;
2. reaches a terminal draw;
3. enters a proved repetition cycle that lifts to a real threefold draw; or
4. reaches one of the explicitly committed twelve-piece frontier states.

The fourth outcome is still unresolved and must be connected to later certified layers or to the exact endgame region.

### Verification commands

- `npm run chaos:prefix:verify` compiles the native solver, checks deterministic small cases, generates an eight-piece reference, and independently replays it in JavaScript.
- `npm run chaos:prefix:verify-reference` checks every committed artifact hash and independently replays the full twelve-piece reference without rerunning synthesis.
- `npm run chaos:prefix:generate` runs counterexample-guided generation to twelve pieces.
- `npm run chaos:prefix:reproduce` regenerates the committed reference and requires the complete manifest to match.

The JavaScript replay uses a separately written mask transition engine. It rejects malformed headers, wrong roles or boundaries, duplicate records, missing policy actions, unreachable policy records, AI-loss terminals, frontier mismatches and hash changes.

## Correctness coverage

The automated proof tooling covers:

- closed cycles resolving to draws;
- a finite terminal win taking priority over a cycle;
- losses being classified only after every action is proved losing;
- horizontal action symmetry, including rotation direction exchange;
- agreement with literal threefold-history minimax on complete tiny games;
- the exact value of empty 3×3 Connect-3 Chaos;
- strict rank reduction along selected endgame winning moves;
- fail-closed graph limits;
- deterministic and shard-complete frontier output;
- JavaScript/native agreement on canonical state counts and the 6×7 action;
- exact 6×7 endgame routing through the main AI entry point and a real browser worker;
- deterministic prefix-policy extraction for both starting roles;
- counterexample rejection propagation between piece-count layers;
- independent replay of every legal adversarial continuation in the committed prefix closure;
- exact frontier equality and artifact hashes.

## Why the empty 6×7 board is not labelled Perfect yet

The empty 6×7 Chaos position has a much larger reachable graph than classic Connect Four. Transformations create large same-piece-count orbits, and rotations alternate between 6×7 and 7×6 orientations. The committed prefix reaches twelve placed pieces; the exact runtime handoff begins at thirty-six placed pieces. The intervening frontier is not closed yet.

An exploratory twelve-to-fourteen classification found additional losing boundary roots for both roles, confirming that later counterexamples must continue to propagate backwards. That experiment is not committed as a certificate and is not counted as proved coverage.

The UI therefore still disables **Perfect** when Chaos Mode is selected. Enabling that label before both starting-role closures reach the exact endgame region would overstate the result.

## Route to a complete Perfect Chaos release

1. Extend the layered certificate from 12 to 14 placed pieces and continue in deterministic piece-count segments.
2. Shard large input frontier sets by canonical-state hash while retaining a single merged policy and exact rejection set.
3. Persist generation journals so interrupted counterexample passes resume without discarding completed layers.
4. Continue rejection propagation until every reachable segment root is non-losing.
5. Connect the final prefix frontier to exact ranked-retrograde endgame records, currently available from 36 placed pieces.
6. Independently replay both complete starting-role closures under the literal threefold rule and verify every policy/action lookup.
7. Add the compact policy loader to the browser worker with fail-closed handling for missing, malformed or illegal records.
8. Enable the Perfect option in Chaos Mode only after both complete closures pass CI and production integration tests.

The existing classic Perfect strategy remains unchanged and independently verified.

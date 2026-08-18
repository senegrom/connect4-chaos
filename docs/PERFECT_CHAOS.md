# Perfect play in Chaos Mode

Chaos Mode is not an ordinary game tree. Flip and rotation actions do not add a piece, so positions can lead back to earlier positions. The implementation also ends a round automatically when the same board, with the same player to move, appears for the third time.

This document records the exact model, the verified solvers now in the repository, the completely solved small boards, the layered non-losing prefix certificate for standard 6×7, and the remaining proof work before the interface may honestly label full 6×7 Chaos play as **Perfect**.

## Completely solved boards

Small enough boards do not need a bounded prefix at all: the whole reachable graph fits in memory and can be solved outright. `data/perfect-chaos-complete/` holds those solutions, and Perfect is enabled in Chaos Mode exactly where one exists.

### Solved values

| Board | Connect | Value | Canonical states | Maximum rank |
|---|---|---|---:|---:|
| 4×4 | 4 | Draw | 239,230 | 18 |
| 4×4 | 3 | First-player win | 31,523 | 13 |
| 4×5 / 5×4 | 4 | Draw | 8,312,306 | 22 |
| 4×5 / 5×4 | 3 | First-player win | 585,418 | 16 |
| 4×5 / 5×4 | 5 | Draw | 18,631,592 | 17 |
| 5×5 | 4 | Draw | 98,688,100 | 27 |
| 5×5 | 3 | First-player win | 3,017,155 | 17 |
| 5×5 | 5 | Draw † | 330,911,560 | 29 |
| 4×6 / 6×4 | 4 | Draw | 163,155,815 | 36 |
| 4×6 / 6×4 | 3 | First-player win | 6,080,450 | 19 |
| 4×6 / 6×4 | 5 | Draw † | 456,870,101 | 24 |
| 4×6 / 6×4 | 6 | Draw † | 524,136,151 | 21 |
| 4×7 / 7×4 | 3 | First-player win | 73,763,416 | 22 |
| 5×6 / 6×5 | 3 | First-player win | 67,692,003 | 23 |

† Solved and independently replayed like the rest, but each of these closures
emits certificates of 313–414 MB per board, past what the repository
publishes, so Perfect is not offered on those configurations.

Each was produced by ranked retrograde analysis over the mover-relative, mirror-canonical quotient graph — the same model `src/chaos-solver.js` uses for endgames. On 4×4 the two implementations agree exactly on the reachable-state, win, draw and loss counts for both connect lengths, and on 4×5 they agree on every sampled position, which is the only check that exercises the rotations that transpose the board.

A 4×5 board and a 5×4 board are the same game: either player can rotate at any time. One certificate therefore spans both orientations, and its records carry their own dimensions.

### Committed certificates

A certificate is the closure a starting role actually reaches: the AI's one action at each of its own turns, with every legal opponent reply explored.

| Board | Connect | Role | AI decisions | Closure states | Terminal AI losses | Bytes |
|---|---|---|---:|---:|---:|---:|
| 4×4 | 3 | 1 | 145 | 174 | 0 | 3,504 |
| 4×4 | 3 | 2 | 1,253 | 1,572 | 141 | 30,096 |
| 4×4 | 4 | 1 | 11,045 | 14,186 | 0 | 265,104 |
| 4×4 | 4 | 2 | 15,411 | 20,004 | 0 | 369,888 |
| 4×5 | 3 | 1 | 178 | 207 | 0 | 4,296 |
| 4×5 | 3 | 2 | 4,601 | 5,740 | 910 | 110,448 |
| 4×5 | 4 | 1 | 95,645 | 119,452 | 0 | 2,295,504 |
| 4×5 | 4 | 2 | 216,194 | 274,192 | 0 | 5,188,680 |
| 4×5 | 5 | 1 | 416,771 | 540,284 | 0 | 10,002,528 |
| 4×5 | 5 | 2 | 588,013 | 763,494 | 0 | 14,112,336 |
| 4×6 | 3 | 1 | 224 | 257 | 0 | 5,400 |
| 4×6 | 3 | 2 | 11,155 | 13,897 | 2,699 | 267,744 |
| 4×6 | 4 | 1 | 518,150 | 641,421 | 0 | 12,435,624 |
| 4×6 | 4 | 2 | 1,520,491 | 1,909,548 | 0 | 36,491,808 |
| 4×7 | 3 | 1 | 291 | 329 | 0 | 7,008 |
| 4×7 | 3 | 2 | 30,302 | 37,842 | 11,047 | 727,272 |
| 5×5 | 3 | 1 | 180 | 209 | 0 | 4,344 |
| 5×5 | 3 | 2 | 7,805 | 9,847 | 2,431 | 187,344 |
| 5×5 | 4 | 1 | 497,323 | 611,545 | 0 | 11,935,776 |
| 5×5 | 4 | 2 | 1,269,295 | 1,583,202 | 0 | 30,463,104 |
| 5×6 | 3 | 1 | 267 | 306 | 0 | 6,432 |
| 5×6 | 3 | 2 | 23,131 | 29,058 | 7,072 | 555,168 |

The Connect-3 second-player closures are lost games played optimally, which is why they record terminal losses. Every drawn certificate reaches none from either role. The largest file compresses to about 1.9 MB and is fetched only when a player actually selects Perfect on that board.

### Keeping a drawing closure small

A drawn position only needs an action that keeps it drawn, so every value-preserving action is admissible and the generator is free to pick among them. Choosing one whose successor is already inside the closure roughly halves the certificate, because the alternative wanders into fresh positions that then need records of their own:

| Certificate | AI decisions | Bytes |
|---|---|---|
| 4×4 Connect 4, role 2 | 32,502 → 15,364 | 780,072 → 368,760 |
| 4×5 Connect 4, role 2 | 496,911 → 216,228 | 11,925,888 → 5,189,496 |

This is applied only to drawn positions. A won position keeps the rank-reducing action the solver selected, because that is what makes the win finite, and the replay rejects a claimed win whose line can repeat.

4×5 Connect-5 is solved and drawn, but nearly its whole graph is drawn and therefore stays reachable under a drawing policy, so its certificates are far larger than the others. They are not committed: the size is out of proportion to a browser game, so Perfect stays unavailable for that configuration and the setup interface reports it as not installed.

### What the replay proves

`scripts/perfect-chaos-complete.mjs` replays every committed certificate using `src/engine.js` — `applyAction` and `resolveActionOutcome` — so the rules that check a policy are the rules the game plays by, not the solver's own copy. Only the record layout is shared with the generator. It requires:

- every reachable AI position has exactly one stored action, and that action is legal;
- the outcome the policy **forces** from each AI position equals the value stored in its record;
- a repetition cycle counts as a draw, so a position claiming a win whose line can repeat forever fails, which is what makes the finite-progress requirement checkable without trusting stored ranks;
- the replayed root value matches the header and the manifest;
- no record is unreachable, and the closure size matches the header.

Because the closure covers every opponent continuation, there is no frontier and no handoff: the runtime plays certified moves for the whole game and reports zero search nodes. A position the certificate does not cover is a defect, and `src/perfect-chaos-runtime.js` throws rather than reverting to search.

```bash
npm run chaos:complete:verify
```

### How the solver scales

`native/perfect-chaos-complete.cpp` sizes its memory by the number of *reachable* states rather than by the index space. A dense mixed-radix index over every gravity-valid arrangement is used only as a key; a rank/select bitset maps it to a compact ordinal, and values are resolved by rank iteration over compact successor lists, so no reverse-edge list is ever materialised. That is what moves the ceiling from 4×5 to 5×5 and beyond:

| Board family | Index space | Reachable canonical states | Peak memory |
|---|---:|---:|---:|
| 4×4 | 923,521 | 239,230 | trivial |
| 4×5 and 5×4 | 44,382,112 | 8,312,306 | ~0.3 GB |
| 5×5 | 992,436,543 | 98,688,100 | ~4.7 GB |
| 4×6 and 6×4 | ~1.15 billion | 163,155,815 | ~7 GB |

Beyond that, 5×6/6×5 and 6×6 lie in the tens of billions of index slots and would need the bitset itself to become sparse or the work to be sharded by piece count. That is future work; nothing above 5×5 is committed.

```bash
npm run chaos:complete:generate -- --rows 4 --columns 5 --connect 4
npm run chaos:complete:verify
```

The `generate` command compiles the native solver, solves the board, emits both role certificates, replays each through `engine.js`, and writes a per-board manifest that carries the generator summary and the independent replay side by side; an entry is written only when the two agree. `merge-manifests` assembles per-board manifests into the runtime catalog and rejects duplicate identities. Every committed certificate was produced this way from the committed source, so the catalog is reproducible rather than merely verifiable.

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
- 10 → 12 placed pieces;
- 12 → 14 placed pieces.

The output frontier file from one segment is the exact binary input root set of the next segment. A later segment may discover that some incoming roots are losing before its next boundary. Those roots are written to a rejection file and fed back into the preceding safety game as losing boundary states. Synthesis repeats until every reachable boundary root extends safely.

This counterexample feedback is material:

- Red required no rejection at 8 pieces, 74 rejected states at 10 pieces and 1,098 at 12 pieces.
- Yellow required 89 rejected states at 8 pieces, 862 at 10 pieces and 6,090 at 12 pieces.

The rejected states are committed alongside the policies so the refinement is reproducible rather than hidden in a generation log.

### Verified fourteen-piece closure

The committed reference is in `data/perfect-chaos-prefix/manifest.json`. It contains fixed-size binary policy, frontier and rejection tables plus a SHA-256 digest for every file.

`src/perfect-chaos-prefix.js` is the fail-closed runtime decoder for all four committed policy layers. The browser worker cross-checks the round starter against the recorded empty 6×7 initial position, lazy-loads only the role and layer matching the current piece count, validates its binary structure, mirrors actions correctly, and falls back to bounded search on an uncovered state. Brutal therefore follows the certified non-losing prefix through 14 placed pieces; this is not a claim that the complete game is solved.

For the AI playing Red:

| Segment | Input roots | Policy entries | Closure states | Output frontier |
|---|---:|---:|---:|---:|
| 0 → 8 | 1 | 1,299 | 3,161 | 1,477 |
| 8 → 10 | 1,477 | 5,058 | 13,397 | 6,912 |
| 10 → 12 | 6,912 | 22,800 | 57,579 | 28,494 |
| 12 → 14 | 28,494 | 91,493 | 219,861 | 104,251 |

For the AI playing Yellow:

| Segment | Input roots | Policy entries | Closure states | Output frontier |
|---|---:|---:|---:|---:|
| 0 → 8 | 1 | 3,863 | 9,581 | 4,522 |
| 8 → 10 | 4,522 | 15,124 | 40,257 | 20,638 |
| 10 → 12 | 20,638 | 67,486 | 173,736 | 86,845 |
| 12 → 14 | 86,845 | 278,371 | 689,361 | 334,185 |

Across the final segment, the independent replay follows 909,222 canonical closure states. Every policy record is reachable, every opponent action is explored, no AI-loss terminal is reachable, and the recomputed sorted frontier is byte-for-byte identical to the committed frontier.

The result is a **non-losing prefix certificate**, not a full game solution. Every adversarial line under the emitted strategy does one of four things before or at fourteen placed pieces:

1. reaches an AI win;
2. reaches a terminal draw;
3. enters a proved repetition cycle that lifts to a real threefold draw; or
4. reaches one of the explicitly committed fourteen-piece frontier states.

The fourth outcome is still unresolved and must be connected to later certified layers or to the exact endgame region.

### Deterministic sharding

Version 1.11 adds a memory-bounded extension path without changing the certified native transition engine. The JavaScript orchestrator splits a strictly sorted input frontier round-robin into deterministic binary shard files, invokes the unchanged native solver on each shard, and merges the resulting policies, output frontiers and rejection roots. Overlapping descendant states may admit more than one safe action; the merger chooses a stable action order, and the independent full-closure replay remains the acceptance gate.

A shard timeout, graph limit, malformed output or missing rejection certificate fails the complete extension. It is never interpreted as a safe result. Sharding reduces peak graph memory at the cost of recomputing descendant subgraphs shared by several root partitions.

### Verification commands

- `npm run chaos:prefix:verify` compiles the native solver, checks deterministic small cases, generates an eight-piece reference, and independently replays it in JavaScript.
- `npm run chaos:prefix:verify-reference` checks every committed artifact hash and independently replays the full fourteen-piece reference without rerunning synthesis.
- `npm run chaos:prefix:generate` runs counterexample-guided generation to fourteen pieces, using deterministic sharding for large extensions.
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
- exact frontier equality and artifact hashes;
- deterministic frontier splitting, shard policy/frontier merging and full merged-closure replay.

## Why the empty 6×7 board is not labelled Perfect yet

The empty 6×7 Chaos position has a much larger reachable graph than classic Connect Four. Transformations create large same-piece-count orbits, and rotations alternate between 6×7 and 7×6 orientations. The committed prefix reaches fourteen placed pieces; the exact runtime handoff begins at thirty-six placed pieces. The intervening frontier is not closed yet.

The committed twelve-to-fourteen layer found additional losing boundary roots for both roles and propagated them backward before acceptance. That confirms later segments must continue the same counterexample-guided refinement rather than assuming every frontier state is safe.

The UI therefore still disables **Perfect** when Chaos Mode is selected. Enabling that label before both starting-role closures reach the exact endgame region would overstate the result.

## Route to a complete Perfect Chaos release

1. Extend the layered certificate from 14 to 16 placed pieces and continue in deterministic piece-count segments.
2. Run large input frontier sets through the deterministic shard-and-merge path while retaining a single independently replayed policy and exact rejection set.
3. Persist generation journals so interrupted counterexample passes resume without discarding completed layers.
4. Continue rejection propagation until every reachable segment root is non-losing.
5. Connect the final prefix frontier to exact ranked-retrograde endgame records, currently available from 36 placed pieces.
6. Independently replay both complete starting-role closures under the literal threefold rule and verify every policy/action lookup.
7. Keep the browser loader fail-closed at the committed 14-piece frontier until the next independently replayed policy layers are accepted.
8. Enable the Perfect option in Chaos Mode only after both complete closures pass CI and production integration tests.

The existing classic Perfect strategy remains unchanged and independently verified.

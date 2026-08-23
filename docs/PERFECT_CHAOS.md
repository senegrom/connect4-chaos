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
| 4×7 / 7×4 | 4 | Draw † | 3,516,300,735 | 34 |
| 5×6 / 6×5 | 3 | First-player win | 67,692,003 | 23 |
| 5×6 / 6×5 | 4 | Draw ‡ | 5,422,925,373 | — |
| 5×6 / 6×5 | 5 | Draw ‡ | 26,560,696,869 | — |

† Solved and independently replayed like the rest, but these closures emit
certificate files past the 100 MB the repository can publish (up to
414 MB per board, 222 MB for the 4×7 connect 4 second role), so
Perfect is not offered on those configurations.

‡ Solved by `native/perfect-chaos-layered.cpp`, which decomposes the game
by piece count (drops add a piece, transformations never do, so every
repetition cycle is confined to one layer) and resolves layers backward with
two adjacent layers in memory. Its 5.4 billion states are past both this
machine's RAM and a 32-bit global ordinal, so no certificates are emitted and
no single maximum rank exists; the counts were produced by the same ranked
iteration validated count-exact against the monolithic solver on five smaller
boards (4×4 c3/c4, 4×5 c4, 5×5 c4, 4×6 c4).

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

`native/perfect-chaos-complete.cpp` sizes its memory by the number of *reachable* states rather than by the index space. A dense mixed-radix index over every gravity-valid arrangement is used only as a key; a rank/select bitset with 32-bit rank entries maps it to a compact ordinal, and rank iteration regenerates successor lists on demand each round, so neither a forward-edge nor a reverse-edge list is ever materialised. Long solves checkpoint the discovery bitset and every finished round, and sweeps run on multiple threads:

| Board family | Index space | Reachable canonical states | Peak memory |
|---|---:|---:|---:|
| 4×4 | 923,521 | 239,230 | trivial |
| 4×5 and 5×4 | 44,382,112 | 8,312,306 | ~0.1 GB |
| 5×5 | 992,436,543 | 98,688,100 | ~0.6 GB |
| 4×6 and 6×4 | ~1.15 billion | 163,155,815 | ~0.8 GB |
| 4×6 c5 / c6 | ~1.15 billion | 457–524 million | ~2 GB |
| 4×7 and 7×4 | ~31.7 billion | 3,516,300,735 | ~17 GB |

The practical ceiling on a 32 GB machine sits between 4×7 and 5×6: a
5×6 connect 4 discovery ran past the extrapolated state count for hours
and 46 GB into the page file before failing, so its reachable set exceeds
both the memory budget and the solver's 2^32 − 1 rank ceiling
(discovery now aborts at that ceiling in minutes instead). 5×6 connect 3
solves easily (67,692,003 states) because short lines end games long before
the board fills. 6×6 and larger lie further out still; those boards
belong to the layered prefix campaign below, not to exact enumeration.

The layered solver compiles and runs standalone:

```bash
g++ -O3 -std=c++20 -static -o chaos-layered native/perfect-chaos-layered.cpp
./chaos-layered --rows 5 --columns 6 --connect 4 --threads 2 --verbose --output solve-5x6
```

It creates the output directory, writes `layer-<k>.bits` and
`layer-<k>.values` checkpoints into it as layers finish, resumes from them
after any interruption, and prints one JSON solution line.
`tests/perfect-chaos-layered.test.js` locks its counts to the monolithic
solver's results on every test run.

```bash
npm run chaos:complete:generate -- --rows 4 --columns 5 --connect 4
npm run chaos:complete:verify
```

The `generate` command compiles the native solver, solves the board, emits both role certificates, replays each through `engine.js`, and writes a per-board manifest that carries the generator summary and the independent replay side by side; an entry is written only when the two agree. `merge-manifests` assembles per-board manifests into the runtime catalog and rejects duplicate identities. Every committed certificate was produced this way from the committed source, so the catalog is reproducible rather than merely verifiable.

## Layered prefix safety certificate

`native/perfect-chaos-prefix.cpp` and `scripts/perfect-chaos-prefix.mjs` prove that a fixed strategy cannot lose from the empty standard board before a selected exact piece-count frontier. The safety game uses these rules:

- at an AI state, at least one selected action must remain outside the least loss attractor;
- at an opponent state, every legal action is explored;
- terminal AI losses are forbidden;
- terminal AI wins, terminal draws and the next exact frontier are safe exits;
- quotient cycles lift to finite real-board mirror orbits and therefore end under the actual threefold-repetition rule.

### Compositional boundaries

The committed linked segments are `0→8`, `8→10`, `10→12`, `12→14`, `14→16`. The output frontier from one segment is the exact sorted input-root set of the next. When a later segment proves an incoming root losing, the root is written to a rejection table and propagated backward until the earlier closure can no longer reach it.

The committed rejection accounting is:

- Red: 0 at 8, 80 at 10, 1,266 at 12, 8,020 at 14.
- Yellow: 94 at 8, 941 at 10, 7,786 at 12, 44,737 at 14.

### Verified 16-piece closure

The reference in `data/perfect-chaos-prefix/manifest.json` carries a SHA-256 digest for every policy, frontier and rejection table. `src/perfect-chaos-prefix.js` validates each binary header, role, boundary, record size, gravity-valid canonical state and action before lookup. The browser loads only the role and segment needed for the current position.

For the AI playing Red:

| Segment | Input roots | Policy entries | Closure states | Output frontier |
|---|---:|---:|---:|---:|
| 0 → 8 | 1 | 1,299 | 3,161 | 1,477 |
| 8 → 10 | 1,477 | 5,058 | 13,404 | 6,919 |
| 10 → 12 | 6,919 | 22,831 | 57,688 | 28,561 |
| 12 → 14 | 28,561 | 92,200 | 221,708 | 105,254 |
| 14 → 16 | 105,254 | 326,031 | 747,775 | 339,682 |

For the AI playing Yellow:

| Segment | Input roots | Policy entries | Closure states | Output frontier |
|---|---:|---:|---:|---:|
| 0 → 8 | 1 | 3,863 | 9,581 | 4,522 |
| 8 → 10 | 4,522 | 15,112 | 40,223 | 20,619 |
| 10 → 12 | 20,619 | 67,605 | 174,087 | 87,073 |
| 12 → 14 | 87,073 | 281,707 | 696,282 | 337,197 |
| 14 → 16 | 337,197 | 1,059,068 | 2,498,257 | 1,164,120 |

Across the final segments, the independent replay follows 3,246,032 canonical closure states. Every policy record is reachable, every legal opponent action is explored, no AI-loss terminal is reachable, and both recomputed output frontiers are byte-for-byte identical to the committed tables.

The result is a **non-losing prefix certificate**, not by itself a full-game solution. Every adversarial line under the emitted strategy reaches an AI win, a terminal draw, a proved repetition draw, or an explicitly committed 16-piece frontier state. Beyond 16 pieces the runtime returns explicitly to bounded search; the complete standard 6×7 Chaos game is not yet claimed as solved.

### Deterministic sharding and exact repair

Large frontier sets are divided into deterministic shards. Missing or malformed shards, state-limit exits, policy conflicts and incomplete accounting fail the round. Once later counterexamples are known, the dependency partitioner reuses byte-identical unaffected policy slices and re-solves only affected or newly introduced roots. The assembled policy is then replayed as one complete closure; incremental repair is accepted only when it is equivalent to a full exact regeneration on the verification cases.

### Verification commands

- `npm run chaos:prefix:verify` checks the native solver on deterministic small references and cross-checks the JavaScript transition model.
- `npm run chaos:prefix:verify-reference` checks every committed artifact hash and independently replays the full 16-piece reference.
- `npm run chaos:prefix:generate` runs counterexample-guided generation through the configured frontier.
- `npm run chaos:prefix:reproduce` regenerates the committed reference from its rejection tables.

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

The empty standard Chaos position has a much larger reachable graph than classic Connect Four. Flip and rotation moves create large same-piece-count orbits, and rotations alternate between 6×7 and 7×6 orientations. The committed prefix reaches 16 placed pieces; the exact runtime endgame handoff begins at 36 pieces. The remaining certified gap runs from the committed 16-piece frontier to the exact ranked-retrograde endgame handoff at 36 placed pieces.

The final committed layer required 8,020 Red and 44,737 Yellow rejected roots at its incoming 14-piece boundary before both closures were safe. This is why later frontiers must continue exact counterexample-guided refinement rather than assuming every reachable state is safe.

The UI therefore keeps **Perfect** unavailable for standard 6×7 Chaos until both starting-role closures connect to the exact endgame region and pass the complete literal-threefold replay gate. Brutal uses the released certificate through 16 pieces and labels later computation as bounded search.

## Route to a complete Perfect Chaos release

1. Extend the independently audited prefix from 16 to 18 pieces for both starting roles.
2. Commit each role's exact counterexample state and continue deterministic sharded rounds until a zero-counterexample closure candidate is produced.
3. Re-download producer and independent-evidence artifacts by exact run, commit and digest; reproduce the closure decisions byte for byte.
4. Assemble a fresh two-role reference, replay every legal adversarial continuation, and promote the new runtime layer only after exact and browser release gates pass.
5. Repeat the same process over later even-piece boundaries until the prefix reaches the exact endgame handoff at 36 pieces.
6. Independently replay both complete starting-role closures under the literal threefold rule and verify every runtime lookup.
7. Enable the Perfect option for standard 6×7 Chaos only after the final full-game claim gate succeeds.

The existing classic Perfect strategy remains unchanged and independently verified.

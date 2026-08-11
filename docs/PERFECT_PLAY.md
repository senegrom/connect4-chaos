# Perfect-play proof

## Guarantee

The **Perfect** difficulty is game-theoretically optimal for standard 7×6 Connect Four when Connect 4 is selected and Chaos Mode is off. It is covered for both ways the AI can enter the game:

- the AI starts as the first player;
- the human starts and the AI plays second.

Every AI decision is supplied by one of two exact components:

1. `assets/perfect-strategy.bin`, a closed policy of exact AI moves before the handoff;
2. the browser's null-window bitboard outcome solver, which proves win, draw, or loss to the terminal game once at most 24 cells remain empty.

There is no heuristic fallback and no wall-clock cutoff on the Perfect path. If the strategy file is absent, malformed, incomplete for the required role, or contains an illegal move, the AI stops with a visible error instead of quietly selecting a weaker move.

Perfect play does not mean that the second player can overturn a theoretically lost position. Standard Connect Four is a first-player win. The policy therefore forces a win when the AI starts; when the AI plays second, it achieves the best game-theoretic result available after the human's opening.

## Closed strategy

The committed strategy has the following verified manifest:

- format: `C4PS` version 1;
- exact AI decisions: **470,494**;
- size: **4,704,952 bytes**;
- handoff: **24 empty cells**;
- starting roles: first and second (`roleFlags = 3`);
- SHA-256: `91de1bd2a5bef3805c19a018b9dcb3a11d240e0569086a03da2872b981363f7a`.

The generator stores one exact game-theoretic move on every AI turn and branches over **every legal opponent reply**. Horizontal reflection canonicalises equivalent positions. When several moves have the same win/draw/loss value, a deterministic policy first minimizes the next canonical opponent frontier and then prefers central columns.

The resulting closure proof is:

| AI role | Exact decisions | Solver handoffs | Earlier terminals | Canonical positions visited |
| --- | ---: | ---: | ---: | ---: |
| First player | 104,680 | 173,204 | 32,798 | 277,884 |
| Second player | 365,814 | 562,471 | 103,669 | 928,285 |
| **Total** | **470,494** | **735,675** | **136,467** | **1,206,169** |

The verifier starts from the empty board for the first-player role and from every legal human opening for the second-player role. At each covered AI position it requires an entry, maps mirrored moves back to the actual board, rejects illegal moves, follows every opponent continuation, and stops only at a terminal state or the exact-solver handoff. It also rejects unreachable surplus entries.

## Strategy format

`assets/perfect-strategy.bin` is deterministic:

| Bytes | Meaning |
| --- | --- |
| 0–3 | ASCII magic `C4PS` |
| 4 | format version |
| 5 | exact-solver handoff in remaining empty cells |
| 6 | entry size, currently 10 |
| 7 | role flags: first, second, or both |
| 8–11 | little-endian entry count |
| each entry | 64-bit canonical key, one 7-bit move, signed outcome |

The outcome is `1` for a forced win, `0` for a draw, and `-1` for a forced loss from the side-to-move's perspective. Keys are strictly ordered and each entry contains exactly one column bit. The runtime validates the header, length, ordering, roles, move bits, and outcomes before enabling lookup.

## Exact terminal solver

At the handoff, `ExactOutcomeSearch` searches directly in the three-valued domain `{-1, 0, 1}`. It uses:

- seven-bit column bitboards;
- immediate-win and non-losing-move pruning;
- null-window negamax;
- horizontal symmetry canonicalisation;
- a fixed-size lower/upper-bound transposition table;
- no elapsed-time abort.

The solver has independent regression coverage against an array-based exhaustive minimax implementation across 250 deterministic late-game positions, in addition to tactical, mirrored-position, mutation-safety, and no-timeout tests.

## Opening book for other levels

`assets/perfect-book.bin` is separate from the Perfect strategy. Medium, Hard, and Brutal consult it before ordinary search. It contains all **129,498 canonical non-terminal positions through ply 8**, occupies **1,294,992 bytes**, and has SHA-256 `7d9e4e39f469083b1297671c015309ede049515716b7d0cbae0a8ddb5e8ced13`.

The opening-book format is `C4PB` version 1 and can store multiple equally strong moves per position. It improves lower difficulty levels but is not part of the Perfect closure requirement.

## Reproducible generation

Two persistent GitHub Actions workflows reproduce the exact data:

- **Generate perfect-play book** enumerates canonical positions by ply, distributes them with a mixed 64-bit shard hash, scores every legal move, and packs the opening book.
- **Generate perfect-play strategy** builds the closed two-role policy to a selected exact-solver handoff and runs its adversarial verifier before an optional commit.

Both use Pascal Pons' exact solver pinned to commit `d6ba50d8aaf2308c769d9bf2abd42d90f34baf41`. The downloaded `7x6_small.book` must match SHA-256 `38f9834317c37d9516e45a21da598569a5d1556595686593d14c2e63f59c1f38`. The generated manifests record the source, policy, byte length, checksum, role statistics, and closure counts.

## CI proof boundary

The committed-artifact test performs all of the following on every push:

1. hashes `assets/perfect-strategy.bin` and compares it with the manifest;
2. validates the runtime binary decoder and both role flags;
3. checks the solved empty-board centre move and first-player win outcome;
4. traverses the complete first- and second-player adversarial closure;
5. compares the observed closure counts with the committed manifest;
6. runs the exact endgame cross-checks and all classic, custom-board, and Chaos tests;
7. launches real Chrome or Chromium at a 390×844 viewport, exercises Perfect as first and second player, rejects browser console and runtime errors, verifies zero board-position movement during a turn, and checks the 320/420 ms flip and 280/360 ms rotate phases.

This proves that the committed policy is structurally closed and that the runtime consumes the same verified artifact. The zero-dependency browser smoke talks directly to the Chrome DevTools Protocol, so the deployed interface, Web Worker, lazy strategy fetch, CSS layout, and animation timing are checked together. Exactness of the generated policy values is rooted in the pinned oracle used during generation.

## Further assurance work

The coverage gap is closed. Remaining work would strengthen independent assurance or improve performance rather than change the game-theoretic policy:

1. implement a second independent native solver and require agreement during regeneration;
2. produce a signed release attestation for the binary strategy and manifests;
3. move the terminal solver to WebAssembly or a more compact table if worst-case browser solve times need reducing;
4. extend the current two-role browser smoke into longer multi-move games and additional browser engines.

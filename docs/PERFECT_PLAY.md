# Perfect-play programme

## Current guarantee

The standard 7×6 Connect Four engine has two exact components:

1. A generated opening book. Every stored entry contains a game-theoretically exact outcome and the complete mask of strong-optimal moves for that position.
2. A dedicated bitboard outcome solver. At the configured handoff it proves win, draw, or loss to the end of the game with null-window search, non-losing-move pruning, symmetry, and no wall-clock cutoff.

The committed book covers every canonical, non-terminal position through ply 8:

- 129,498 exact positions;
- 1,294,992 bytes;
- 44,025 positions with more than one equally strong optimal move;
- SHA-256 `7d9e4e39f469083b1297671c015309ede049515716b7d0cbae0a8ddb5e8ced13`.

The runtime consults the opening book before allocating a transposition table. A book hit therefore returns an exact move with zero searched nodes. A book miss falls back to the existing bitboard search. The game is already perfect for every stored opening and every position that reaches exact terminal search, but the midgame gap between those regions is not yet globally solved.

## Book format

`assets/perfect-book.bin` is deterministic and deliberately simple:

| Bytes | Meaning |
| --- | --- |
| 0–3 | ASCII magic `C4PB` |
| 4 | format version |
| 5 | maximum stored ply |
| 6 | entry size, currently 10 |
| 7 | reserved |
| 8–11 | little-endian entry count |
| each entry | 64-bit canonical key, 7-bit optimal-move mask, signed outcome |

The outcome is `1` for a forced win, `0` for a draw, and `-1` for a forced loss from the side-to-move's perspective. The move mask is based on strong scores, so among equally winning or losing outcomes it preserves the fastest win or longest resistance.

Keys are sorted and horizontally canonicalised. The browser validates the header, length, ordering, move masks, and outcomes before enabling lookup. A malformed or unavailable book is ignored by the worker and cannot override the search engine.

## Generation

The persistent **Generate perfect-play book** GitHub Actions workflow:

1. Validates a requested depth and shard count.
2. Enumerates each canonical legal position through the requested ply and partitions positions deterministically across shards.
3. Builds a pinned exact Connect Four oracle in every scoring job.
4. Scores every legal move exactly, with up to 16 shards running in parallel.
5. Merges the score artifacts and packs a deterministic runtime book.
6. Runs the repository's independent tests.
7. Uploads the book as an artifact and optionally commits only the binary and manifest.

The oracle is Pascal Pons' solver pinned to commit
`d6ba50d8aaf2308c769d9bf2abd42d90f34baf41`. The workflow also verifies the downloaded `7x6_small.book` against SHA-256
`38f9834317c37d9516e45a21da598569a5d1556595686593d14c2e63f59c1f38` before using it. The generated manifest records the oracle commit, checksum, and shard count.

The generator itself is repository-owned code:

```bash
node scripts/perfect-book.mjs enumerate \
  --depth 8 \
  --shard-count 8 \
  --shard-index 0 \
  > positions-0.txt

node scripts/perfect-book.mjs pack \
  --input merged-scores.txt \
  --output assets/perfect-book.bin \
  --manifest data/perfect-book.manifest.json \
  --max-ply 8 \
  --source "exact oracle provenance"
```

## Verification

CI verifies:

- binary structure and strictly ordered keys;
- the solved empty-board centre move;
- book-first runtime routing with zero search nodes;
- legal and exact immediate tactics;
- exact late-game results against an independent array minimax sample;
- worker loading and message handling;
- all existing classic and Chaos rules.

Book generation additionally refuses conflicting scores for the same canonical key and rejects missing or illegal moves. Its mixed 64-bit shard assignment is regression-tested for completeness, disjointness, and load balance. The deployed Pages artifact is checked against the committed manifest checksum during release review.

## Adversarial strategy compression

`perfect-strategy.mjs` builds a much smaller exact policy than an all-position opening book. On AI turns it records one exact game-theoretic move; on opponent turns it branches over every legal reply. Horizontal symmetry deduplicates positions, and a deterministic tie-break chooses the exact move whose immediate opponent frontier is smallest before preferring central columns.

The binary strategy contains one legal move bit and the exact outcome for each covered AI decision. Its verifier traverses both starting-player roles, follows every opponent reply, rejects unreachable entries, and requires every continuation to terminate or reach the configured exact-solver handoff. This closure check is the machine-checkable bridge needed before exposing a **Perfect** difficulty in the interface.

## Route to global perfect play

Completed foundations:

1. Exact all-position opening coverage through ply 8.
2. A no-clock terminal outcome solver for the last 16, 20, or 24 empty cells by difficulty.
3. Deterministically mixed, regression-tested book sharding.
4. An adversarial strategy format, generator, and closure verifier for both starting-player roles.

Remaining proof work:

1. Generate and commit the closed exact strategy down to the Brutal 24-empty-cell handoff.
2. Load that policy only for a new **Perfect** difficulty and verify every returned move against the strategy role and board orientation.
3. Add a full runtime traversal proving that every legal opponent reply reaches another strategy decision, a terminal position, or the exact solver.
4. Add a second independently implemented native solver and require both solvers to agree before regenerating published policy data.
5. Expose **Perfect** only after CI passes the complete adversarial traversal from both starting-player choices.

The final proof must fail on the first uncovered, contradictory, illegal, unreachable, or heuristic-only decision.

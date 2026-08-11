# Perfect-play programme

## Current guarantee

The standard 7×6 Connect Four engine has two exact components:

1. A generated opening book. Every stored entry contains a game-theoretically exact outcome and the complete mask of strong-optimal moves for that position.
2. The bitboard solver's exact terminal search. Once it searches every remaining ply, its result is proved rather than evaluated heuristically.

The committed book covers every canonical, non-terminal position through ply 6:

- 11,094 exact positions;
- 110,952 bytes;
- 3,963 positions with more than one equally strong optimal move;
- SHA-256 `595a8531e72337e7e3f881ff135f282880610cff7f53c2e7fed912e695f8e562`.

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

Book generation additionally refuses conflicting scores for the same canonical key and rejects missing or illegal moves. The deployed Pages artifact is checked against the committed manifest checksum during release review.

## Route to global perfect play

The next milestones are:

1. Run the sharded generator through ply 8 and commit the validated book.
2. Measure ply-9 and ply-10 state counts, generation time, artifact size, and the best shard configuration.
3. Add resumable shard outputs and deterministic partial-merge validation so interrupted deep generations do not restart from zero.
4. Add a second independently implemented native solver and require both solvers to agree before publishing new entries.
5. Expand the exact strategy frontier until every book continuation reaches the browser's exact-search region.
6. Mark Brutal as **Perfect** only after a full adversarial traversal proves that every reachable AI decision is covered by either the book or terminal solving.

The final proof must be machine-checkable: traverse every opponent reply from both starting-player choices, require every AI move to preserve the game-theoretic optimum, and fail on the first uncovered, contradictory, or heuristic-only decision.

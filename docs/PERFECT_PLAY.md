# Perfect-play programme

## Current guarantee

The standard 7×6 Connect Four engine has two exact components:

1. A generated opening book. Every stored entry contains a game-theoretically exact outcome and the complete mask of strong-optimal moves for that position.
2. The bitboard solver's exact terminal search. Once it searches every remaining ply, its result is proved rather than evaluated heuristically.

The runtime consults the opening book before allocating a transposition table. A book miss falls back to the existing bitboard search. This means the game is already perfect for every stored opening and every position that reaches exact terminal search, but the gap between those regions is not yet globally solved.

The committed bootstrap entry proves the empty-board centre opening. The generation workflow expands that seed to every canonical, non-terminal position through a selected ply.

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

Keys are sorted and horizontally canonicalised. The browser validates the header, length, ordering, move masks, and outcomes before enabling lookup. A malformed book is ignored by the worker and cannot override the search engine.

## Generation

The persistent **Generate perfect-play book** GitHub Actions workflow:

1. Enumerates each canonical legal position through the requested ply.
2. Builds a pinned exact Connect Four oracle.
3. Scores every legal move exactly.
4. Packs the results into the binary runtime format.
5. Runs the repository's independent tests.
6. Uploads the book as an artifact and optionally commits it.

The oracle is Pascal Pons' solver pinned to commit
`d6ba50d8aaf2308c769d9bf2abd42d90f34baf41`. Its AGPL source and opening-book file are used only as build-time tools and are not redistributed in this repository. The generated manifest records the oracle commit and downloaded book checksum.

The generator itself is repository-owned code:

```bash
node scripts/perfect-book.mjs enumerate --depth 8 > positions.txt
node scripts/perfect-book.mjs pack \
  --input scored.txt \
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

Book generation additionally refuses conflicting scores for the same canonical key and rejects missing or illegal moves.

## Route to global perfect play

The next milestones are:

1. Grow the committed exact book from the bootstrap frontier to ply 8.
2. Add shard-and-merge support so deeper frontiers can be generated in parallel.
3. Replace the build-time oracle with a second independently implemented native solver, then require both solvers to agree before publishing entries.
4. Expand the strategy frontier until every book continuation reaches the browser's exact-search region.
5. Mark Brutal as **Perfect** only after a full adversarial traversal proves that every reachable AI decision is covered by either the book or terminal solving.

The final proof should be machine-checkable: traverse every opponent reply from both starting-player choices, require every AI move to be book-backed or exactly solved, and fail on the first uncovered position.

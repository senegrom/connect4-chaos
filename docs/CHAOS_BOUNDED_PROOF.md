# Bounded loopy proofs for Chaos AI

The committed Perfect Chaos work solves two ends of the standard 6×7 game:

- a certified non-losing policy through 16 placed pieces; and
- complete ranked-retrograde endgames from 36 placed pieces.

Ordinary minimax still has to cover the large middle interval. `src/chaos-proof.js` adds a sound proof layer for that interval without presenting a fixed-depth heuristic as perfect play.

## Proof graph

A proof node contains:

- the gravity-settled board, canonicalised by side-to-move colours and horizontal reflection;
- whether the side to move is the root AI;
- the number of future drops remaining in the proof horizon.

Drops consume one unit of horizon. Flips and rotations consume none. Every transformation orbit and transform-only cycle inside a layer is therefore represented exactly instead of being truncated after an arbitrary number of transform plies.

A non-terminal drop that reaches the horizon becomes an unknown frontier edge. The graph is solved twice with the existing ranked loopy-game retrograde engine:

1. **Lower solve:** every frontier is treated as an AI loss.
2. **Upper solve:** every frontier is treated as an AI win.

The AI-turn bit is part of the graph key, so those assumptions remain globally consistent after odd or even transform chains. The true value must lie between the two W/D/L results.

- Equal lower and upper values are an exact proof.
- An action whose upper value is a loss is losing even under the most optimistic continuation and can be rejected safely.
- An action whose lower value is a draw or win has a certified non-losing continuation within the bounded graph.
- Different bounds are reported as unresolved; they are never labelled solved.

Closed regions that do not reach a terminal or frontier remain draws under ranked retrograde. With a fresh repetition history, those quotient cycles lift to the real threefold-draw rule in the same way as the complete exact solver.

## Runtime use

`src/ai-worker.js` runs the proof layer before ordinary Chaos search for Medium, Hard and Brutal play when no stronger route applies. The certified opening policy and complete exact endgame solver retain priority.

Default proof settings are:

| Difficulty | Drop horizon | State limit |
|---|---:|---:|
| Medium | 1 | 10,000 |
| Hard | 2 | 50,000 |
| Brutal | 2 | 100,000 |

The proof fails open to the existing search when its deterministic state limit is reached. It only replaces a searched move when that move's optimistic upper bound is still a loss. Fixed-depth diagnostic searches retain their prior behaviour unless `useChaosProof: true` is supplied explicitly.

Available worker options:

- `useChaosProof`: enable or disable the layer;
- `chaosProofDropDepth`: number of future placements, with `0` disabling it;
- `chaosProofMaximumStates`: deterministic graph-state limit.

The runtime skips bounded proofs once any recorded position has already appeared twice, because the next visit then becomes a history-specific immediate draw.

## Rejection seeds

The 14-piece rejection sets behind the committed 14→16 certificate came from a bridge scanner, `scripts/perfect-chaos-bridge.mjs`, which ran this proof over the layered solver's frontier files and wrote the proved-loss roots in the `C4CFRN1` format the prefix synthesiser reads. With no layer beyond 16 in progress it was retired on 2026-09-26; commit 0fedaa3 has it.

A proved-loss rejection is conservative when a concrete play history could trigger a draw sooner: it may exclude an otherwise usable route, but it cannot make an unsafe policy pass verification. The full independent closure replay remains the acceptance gate.

The committed certificate regenerates from its own seeds (`npm run chaos:prefix:generate`):

```bash
node scripts/perfect-chaos-prefix.mjs generate \
  --frontier-pieces 16 \
  --seed-rejections data/perfect-chaos-prefix \
  --shards 8 \
  --shard-from-pieces 14 \
  --output generated/perfect-chaos-prefix-16
```

## Verification

The automated tests:

- compare complete bounded graphs with exact 2×2 and 3×3 games;
- exhaustively check every reachable 2×3 state and every legal action, requiring the exact value to remain inside the reported bounds;
- verify horizontal action reflection;
- cover deterministic graph limits; and
- reproduce the known bounded-search horizon regression, where a two-drop proof rejects seven losing root actions.

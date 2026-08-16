# Perfect classic play on boards through 7×7

This project ships game-theoretically exact, non-Chaos Connect Four infrastructure for every app-supported board from 4×4 through 7×7. The existing standard 6×7 strategy remains unchanged; the variable-board pipeline adds exact solving, compact selected-policy closures, independent replay, lazy runtime loading and configuration-gated UI support for the other dimensions.

The generalized solver supports every connect length that fits a board. The committed root-value matrix, production policy catalog and browser setup currently focus on ordinary Connect Four (`connect = 4`).

## Exact position model

`src/classic-solver.js` stores a position with two mover-relative bit masks:

- `current`: pieces belonging to the side to move;
- `mask`: every occupied cell.

Each column reserves one sentinel bit, so every board through 7×7 fits in at most 56 bits. The representation retains the board dimensions and connect length outside the position key. Horizontal reflection is canonicalised before transposition lookup.

The exact search has no heuristic leaf evaluation. It reaches one of three proved values:

- `1`: the side to move can force a win;
- `0`: the side to move can force a draw but not a win;
- `-1`: the opponent can force a win.

The search uses mover-relative null-window alpha-beta, immediate-win detection, exact forced blocks and double threats, horizontal symmetry, bounded-memory lower/upper transposition records, and deterministic move ordering. Reaching a configured node boundary throws `CLASSIC_EXACT_NODE_LIMIT`; an unfinished search is never converted into a draw or Perfect result.

```js
import { solveClassicPosition } from './src/classic-solver.js';

const result = solveClassicPosition({
  board,
  currentPlayer,
  connect: 4,
  chaosMode: false,
});
```

The returned object uses the same telemetry shape as the other engines and includes `value`, `action`, `nodes`, transposition statistics and `solved: true`.

## Independent native engines

Two separately maintained C++20 programs support reproducible proof work:

- `native/perfect-classic.cpp` solves arbitrary positions and empty-board roots.
- `native/perfect-classic-policy.cpp` selects one exact AI action at each reachable AI decision while retaining every legal opponent continuation.

Both programs take dimensions and connect length at runtime. The policy generator reuses an exact W/D/L transposition search across the complete selected closure, but its output is not accepted merely because generation completed. Every candidate policy must pass the independent JavaScript replay described below.

```bash
npm run classic:verify
npm run classic:solve -- --rows 4 --columns 6 --connect 4
npm run classic:policy:verify
```

## Selected optimal-policy closures

A complete policy is generated separately for the AI playing the first and second starting roles.

At an AI state:

1. the native exact solver determines the state value and every optimal move;
2. a deterministic optimal move is selected, preferring the move with the smallest distinct opponent continuation frontier;
3. the canonical position key, one-column move mask and exact outcome are written to the policy.

At an opponent state, every legal column is retained. Terminal wins and draws are recorded by the closure replay, while non-terminal descendants continue until an AI-turn handoff state reaches the configured number of remaining cells. The handoff is solved exactly at runtime and during verification.

This is not an opening book containing only likely play. It is a strategy closure against every legal opponent response reachable under the selected AI strategy.

## Binary policy format

`src/perfect-classic-policy.js` decodes the fail-closed `C4VPOL1` format. Its header records:

- rows, columns and connect length;
- first-player or second-player AI role;
- exact-solver handoff boundary;
- empty-board value;
- policy entry count and replay closure count.

Each fixed-size record contains a horizontally canonical mover-relative position key, exactly one selected column bit and its exact mover-relative outcome. The decoder rejects malformed dimensions, unsupported roles, ambiguous moves, out-of-range keys, unsorted records, duplicates and metadata mismatches.

`data/perfect-classic/manifest.json` is the runtime catalog. Each accepted policy record includes its file path, SHA-256 digest, byte length, closure statistics and independent replay summary. A configuration is advertised only when its committed catalog entry and binary policy are present.

## Independent replay and exact handoff

`scripts/perfect-classic-policy.mjs` independently replays each candidate policy:

- policy actions are decoded and applied by a separate transition implementation;
- every legal opponent action is explored;
- every policy record must be reachable, and every reachable pre-handoff AI state must have exactly one record;
- terminal outcomes and stored record values must agree;
- every reached handoff position is re-solved by a separately written JavaScript null-window solver;
- the recomputed empty-board value and closure-state count must match the binary header;
- the policy file hash must match its manifest entry.

The replay exact solver uses a fixed-size direct-mapped transposition table. Replacement collisions can increase work but cannot create false hits. Memory is deterministic, and a configured node limit fails the proof rather than weakening it.

Useful commands:

```bash
node scripts/perfect-classic-policy.mjs generate \
  --rows 7 --columns 7 --connect 4 \
  --role both --handoff-remaining 24 \
  --output generated/perfect-classic-7x7-c4

node scripts/perfect-classic-policy.mjs verify-reference \
  --reference data/perfect-classic/manifest.json
```

## Browser runtime

The worker routing has three exact paths for non-Chaos Perfect play:

1. the existing verified 6×7 policy and bitboard endgame solver;
2. the matching variable-board `C4VPOL1` policy above its handoff;
3. the generalized exact solver at and below the handoff.

The matching policy is lazy-loaded using rows, columns, Connect Four rules and whether the AI is the first or second player. Missing, malformed or uncovered early policy data produces an explicit error; it never falls back to bounded or heuristic search. The setup interface exposes Perfect for non-Chaos Connect Four dimensions from 4×4 through 7×7.

## Published Connect Four root values

`data/perfect-classic-root-values.json` records the published game-theoretic values for all sixteen boards from 4×4 through 7×7.

| Rows × columns | 4 columns | 5 columns | 6 columns | 7 columns |
|---|---:|---:|---:|---:|
| 4 rows | Draw | Draw | Second-player win | Draw |
| 5 rows | Draw | Draw | Draw | Draw |
| 6 rows | Draw | Draw | Second-player win | First-player win |
| 7 rows | Draw | Draw | First-player win | Draw |

The repository stores values from the first player's perspective: `1` for a first-player win, `0` for a draw and `-1` for a second-player win. Policy generation requires the first-player role to match this matrix and the second-player role to match its negation.

## Generation workflows

The manual **Solve classic boards through 7x7** workflow produces exact empty-board values and diagnostic search statistics.

The **Generate perfect classic policies** workflow performs the production pipeline:

1. dispatch every non-standard board from 4×4 through 7×7 as a matrix;
2. generate both starting-role policies with no clock cutoff;
3. independently replay each complete selected closure and exact handoff;
4. verify the published root value;
5. hash and upload each board artifact;
6. merge accepted board manifests into one deterministic runtime catalog.

Standard 6×7 is excluded from this matrix because its existing strategy is independently generated and verified. A timeout, node-limit exit, closure-limit exit, replay mismatch or missing matrix artifact fails the catalog job. No partial or heuristic result is promoted as Perfect.

## Correctness coverage

Automated tests and proof jobs cover:

- exhaustive comparison with independent array minimax on every reachable 2×3 Connect-2 and 3×3 Connect-3 state;
- exact late-game samples across all sixteen 4×4–7×7 dimensions;
- horizontal reflection and input immutability;
- malformed-position and deterministic node-limit rejection;
- binary policy validation and mirrored move lookup;
- policy use without allocating a search table;
- exact runtime handoff;
- native policy synthesis on complete small games;
- independent replay of both starting roles;
- published root-value agreement, including the 4×6 second-player win.

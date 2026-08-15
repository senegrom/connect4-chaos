# Exact classic play on boards through 7×7

This project already ships an instant verified strategy for ordinary 6×7 Connect Four. The generalized classic solver extends the exact model to every gravity-settled, non-Chaos board with at most seven rows and seven columns.

The solver supports every connect length that fits the selected board. The published root-value reference and the batch matrix workflow currently focus on Connect Four (`connect = 4`).

## Exact position model

`src/classic-solver.js` stores a position with two mover-relative bit masks:

- `current`: pieces belonging to the side to move;
- `mask`: every occupied cell.

Each column reserves one sentinel bit, so every board through 7×7 fits in at most 56 bits. The representation retains the board dimensions and connect length outside the position key. Horizontal reflection is canonicalised before transposition lookup.

The exact search has no heuristic leaf evaluation. It reaches one of three proved values:

- `1`: the side to move can force a win;
- `0`: the side to move can force a draw but not a win;
- `-1`: the opponent can force a win.

The search uses:

- mover-relative negamax;
- null-window alpha-beta searches over win/draw/loss values;
- horizontal symmetry;
- immediate-win detection;
- exact forced-block and double-threat detection;
- a bounded-memory transposition table with lower and upper results;
- future-threat, centre and history move ordering.

A caller may set a deterministic node boundary for batch jobs. Reaching it throws `CLASSIC_EXACT_NODE_LIMIT`; the solver never converts an unfinished search into a perfect result. Omitting that option leaves the search depth and node count unrestricted.

## JavaScript API

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

The browser still uses the existing precomputed 6×7 strategy for instant opening play. Large empty variant boards require generated policy layers before they should be exposed as instant Perfect choices in the interface. The generic solver is nevertheless exact for every individual supported position and is the verifier and handoff engine for those future policies.

## Independent native solver

`native/perfect-classic.cpp` is a separately implemented C++20 solver for deterministic batch work. It accepts runtime dimensions, connect length, a move sequence, transposition-table size and an optional node boundary.

```bash
npm run classic:verify
npm run classic:solve -- --rows 4 --columns 6 --connect 4
npm run classic:solve -- --rows 7 --columns 7 --connect 4 --table-bits 26
```

`classic:verify` compiles the native implementation and requires agreement on complete small games, including the 4×6 Connect Four second-player win. The JavaScript tests independently compare every reachable state of complete tiny games with array minimax and sample exact late positions across all sixteen dimensions from 4×4 through 7×7.

## Published Connect Four root values

`data/perfect-classic-root-values.json` records the published game-theoretic values for the sixteen boards from 4×4 through 7×7. The source is John Tromp's Connect Four Playground, whose medium-board results were produced with the Fhourstones solver.

| Rows × columns | 4 columns | 5 columns | 6 columns | 7 columns |
|---|---:|---:|---:|---:|
| 4 rows | Draw | Draw | Second-player win | Draw |
| 5 rows | Draw | Draw | Draw | Draw |
| 6 rows | Draw | Draw | Second-player win | First-player win |
| 7 rows | Draw | Draw | First-player win | Draw |

The repository stores values from the first player's perspective: `1` for a first-player win, `0` for a draw and `-1` for a second-player win.

## Full-board proof workflow

The manual **Solve classic boards through 7x7** workflow can solve one board or dispatch all sixteen dimensions as a matrix. Each board is solved from the empty position with no clock cutoff, checked against the published root value when `connect = 4`, and uploaded with node and transposition statistics. A final artifact combines successful board results into one exact matrix manifest.

The workflow is deliberately manual because the largest dimensions are substantial proof jobs. A timeout or configured node boundary is a failed proof, not a draw and not a strategy record.

## Route to instant Perfect play for every variant

The exact engines establish the position evaluator needed for all dimensions. The remaining production work is data generation and replay:

1. Generate deterministic optimal-policy closures for each rows/columns/connect configuration.
2. Stop each policy at a verified remaining-cell handoff where the JavaScript exact solver is practical.
3. Store one canonical move and exact outcome for every reachable AI decision state, for both starting roles.
4. Independently replay every opponent action and verify that the policy preserves the recorded game-theoretic value.
5. Hash and commit accepted binary tables plus a configuration manifest.
6. Lazy-load the matching table in the worker and fail closed on any missing or malformed record.
7. Enable Perfect in the setup interface only for configurations whose complete policy and exact handoff pass CI.

This separates mathematical correctness from response time: the generic solver is exact now, while instant browser play will be enabled configuration by configuration as replayable policy artifacts are accepted.

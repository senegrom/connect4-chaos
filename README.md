# Connect 4: Chaos Edition

A polished, dependency-light browser implementation of Connect Four with configurable boards, optional flip-and-rotate Chaos moves, accessible controls, several search-based AI levels, and exact-play tooling.

[Play the current build](https://senegrom.github.io/connect4-chaos/)

![Connect 4: Chaos Edition preview](assets/game-preview.svg)

## Highlights

- **Game-first interface** — the board and primary controls stay prominent on desktop and mobile, while advanced settings remain available without crowding play.
- **Configurable rules** — choose the number of rows, columns and pieces needed to connect.
- **Chaos Mode** — players may drop a piece, flip the board, rotate clockwise or rotate counter-clockwise. Gravity is reapplied after every transformation.
- **Local and computer play** — play against another person or against Easy, Medium, Hard, Brutal or Perfect AI where supported.
- **Exact classic play** — standard 6×7 Connect Four uses a verified deterministic policy covering 470,494 AI decisions and both starting roles.
- **Certified Chaos prefix** — standard 6×7 Chaos Mode now has an independently replayed non-losing policy certificate for both starting roles through **14 placed pieces**.
- **Exact Chaos endgames** — eligible late-game Chaos positions with six or fewer empty cells are solved as complete loopy game graphs rather than ordinary depth-limited trees.
- **Transparent telemetry** — search depth, nodes, principal variation and exact proof status are shown without presenting bounded search as solved play.
- **Accessible interaction** — keyboard support, touch guidance, ARIA labels, live announcements, strong focus states and reduced-motion support are built in.
- **No runtime framework** — the shipped game is plain HTML, CSS and JavaScript; development tooling uses Node.js only.

## Quick start

A current Node.js installation is recommended for local checks and the development server.

```bash
npm install
npm run dev
```

Open the address printed by the server, normally `http://127.0.0.1:4173`.

The static application can also be served by any ordinary web server. ES modules and web workers should be loaded over HTTP rather than by opening `index.html` directly from the filesystem.

## Rules

Players alternate turns. A turn may be one of the following:

1. Drop a piece into a non-full column.
2. Flip the board vertically, then let every piece fall under gravity.
3. Rotate the board clockwise, then reapply gravity.
4. Rotate the board counter-clockwise, then reapply gravity.

The first player to connect the configured number of pieces wins. A Chaos transformation that creates winning lines for both players is lost by the player who made that transformation. A full board with no winner is a draw. The same settled board with the same player to move appearing for the third time is also an automatic draw.

## AI levels

| Level | Behaviour |
|---|---|
| Easy | Immediate tactical wins and blocks, then a legal move with controlled randomness. |
| Medium | Bounded iterative-deepening search with tactical extensions. |
| Hard | Deeper search with larger transposition tables. |
| Brutal | The strongest bounded profile. In Chaos, placement depth is preserved across a bounded number of transforms, quiet root transforms are verified one layer deeper, and certified exact endgames are used automatically. |
| Perfect | Game-theoretically exact play for standard classic 6×7 Connect Four. |

Perfect is deliberately unavailable at the beginning of a Chaos round. The project does not enable that label until every adversarial continuation from the empty board is connected to a verified policy or an exact solved region.

## Exact classic play

Classic 6×7 Connect Four uses three verified layers:

- A solved opening book.
- A deterministic strategy covering both possible starting roles.
- An exact late-game bitboard solver.

The policy is replayed against every legal opponent continuation. Missing, malformed or ambiguous records fail closed instead of falling back to heuristic play.

```bash
npm run strategy:verify
```

See [docs/PERFECT_PLAY.md](docs/PERFECT_PLAY.md) for the proof boundary, binary formats and verification process.

## Perfect Chaos work

Chaos Mode is a directed graph rather than an ordinary game tree because flips and rotations can revisit earlier positions. The exact model therefore includes board orientation, the side to move, transformation outcomes and the real threefold-repetition rule.

### Exact endgame layer

`src/chaos-solver.js` constructs the reachable graph, canonicalises horizontal reflection and side-to-move colours, and performs ranked retrograde analysis. Closed unresolved cycles are draws; ranked winning choices must make finite progress toward a terminal win. A separately implemented C++20 engine in `native/perfect-chaos.cpp` cross-checks deterministic reference games.

### Layered non-losing prefix certificate

Version 1.10 introduced `native/perfect-chaos-prefix.cpp` and `scripts/perfect-chaos-prefix.mjs`; version 1.11 extends their committed certificate and adds memory-bounded deterministic sharding. They solve finite safety games between exact piece-count frontiers:

- At an AI state, at least one selected action must remain outside the loss attractor.
- At an opponent state, every legal action is explored.
- Terminal AI losses are forbidden.
- Terminal AI wins, terminal draws and the next exact frontier are safe exits.
- Quotient cycles lift to finite real-board orbits and therefore trigger the actual threefold draw if repeated.

The committed certificate covers both starting roles through 14 placed pieces. It is split into the boundaries `0→8`, `8→10`, `10→12` and `12→14`. Counterexamples discovered in a later layer are propagated backward as explicit rejection sets until the earlier policy no longer reaches them.

The independent JavaScript verifier checks binary headers, canonical ordering, hashes, exact frontier equality, policy reachability and every opponent action. It replayed 909,222 states in the final `12→14` closures without reaching an AI-loss terminal. Large extensions can split an exact sorted input frontier into deterministic shards, merge all safe policies and rejection sets, and then replay the merged certificate as one closure.

```bash
npm run chaos:verify
npm run chaos:prefix:verify-reference
```

The remaining gap is from the committed 14-piece frontier to the exact endgame handoff at 36 placed pieces. Work beyond 14 pieces is experimental until a complete replayable certificate is committed. See [docs/PERFECT_CHAOS.md](docs/PERFECT_CHAOS.md) for the theorem, exact counts, rejection sets and continuation plan.

## Commands

| Command | Purpose |
|---|---|
| `npm run dev` | Start the local static server. |
| `npm run check` | Parse-check application, solver and proof-tooling source files. |
| `npm test` | Run the Node.js test suite. |
| `npm run ci` | Run source checks and unit/integration tests. |
| `npm run test:coverage` | Run tests with Node's coverage report. |
| `npm run test:browser` | Exercise the built application in a real Chromium browser. |
| `npm run strategy:verify` | Replay the committed exact classic strategy. |
| `npm run chaos:verify` | Cross-check exact Chaos reference games and the small prefix solver. |
| `npm run chaos:prefix:verify-reference` | Independently replay and hash-check the committed 14-piece certificate. |
| `npm run chaos:prefix:reproduce` | Regenerate the committed prefix manifest from its rejection seeds. |

## Project structure

```text
.
├── index.html
├── styles.css
├── assets/
│   ├── perfect-book.bin
│   └── perfect-strategy.bin
├── data/
│   ├── perfect-book.manifest.json
│   ├── perfect-strategy.manifest.json
│   ├── perfect-chaos-foundation.manifest.json
│   └── perfect-chaos-prefix/
│       ├── manifest.json
│       ├── red/
│       └── yellow/
├── docs/
│   ├── PERFECT_PLAY.md
│   └── PERFECT_CHAOS.md
├── native/
│   ├── perfect-chaos.cpp
│   └── perfect-chaos-prefix.cpp
├── scripts/
│   ├── browser-smoke.mjs
│   ├── perfect-book.mjs
│   ├── perfect-strategy.mjs
│   ├── perfect-chaos.mjs
│   ├── perfect-chaos-native.mjs
│   ├── perfect-chaos-prefix.mjs
│   └── serve.mjs
└── src/
    ├── app.js
    ├── engine.js
    ├── ai.js
    ├── ai-worker.js
    ├── bitboard.js
    ├── chaos-solver.js
    ├── perfect-book.js
    ├── perfect-strategy.js
    └── exact-table.js
```

## Testing and release discipline

The repository checks tactical play, board transformations, repetition handling, exact table validation, classic strategy closure, loopy-game retrograde behaviour, native/JavaScript agreement, binary certificate replay, keyboard/touch interaction and responsive layout.

GitHub Actions runs ordinary CI, the dedicated Perfect Chaos prefix verifier and the Pages deployment. Generator workflows are manual so large proof jobs are explicit and their artifacts can be reviewed before promotion.

## Licence

Copyright © 2026 senegrom.

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).

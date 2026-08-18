# Connect 4: Chaos Edition

A polished, dependency-light browser implementation of Connect Four with configurable boards, optional flip-and-rotate Chaos moves, accessible controls, several search-based AI levels, and exact-play tooling.

[Play the current build](https://senegrom.github.io/connect4-chaos/)

![Connect 4: Chaos Edition preview](assets/game-preview.svg)

## Highlights

- **Game-first interface** — the board and primary controls stay prominent on desktop and mobile, while advanced settings remain available without crowding play.
- **Configurable rules** — choose the number of rows, columns and pieces needed to connect.
- **Chaos Mode** — players may drop a piece, flip the board, rotate clockwise or rotate counter-clockwise. Gravity is reapplied after every transformation.
- **Local and computer play** — play against another person or against Easy, Medium, Hard, Brutal or Perfect AI where supported.
- **Perfect classic variants** — non-Chaos Connect Four boards from 4×4 through 7×6 use verified role-specific policies with an exact endgame handoff; the existing standard 6×7 strategy remains independently verified. Only 7×7 is still uncertified.
- **Perfect Chaos on solved boards** — 4×4 and 4×5 Chaos Mode are solved completely for both starting roles at Connect 4 and Connect 3, so Perfect is available there with no search and no handoff.
- **Certified Chaos prefix** — standard 6×7 Chaos Mode has an independently replayed non-losing policy certificate for both starting roles through **16 placed pieces**; Brutal lazy-loads only the matching certified layer during live play.
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
| Brutal | Certified standard-board Chaos play through 16 placed pieces, transform-aware bounded search beyond it, and automatic use of the exact Chaos endgame frontier. |
| Perfect | Game-theoretically exact play wherever a certificate exists: non-Chaos Connect Four on the 15 verified boards from 4×4 through 7×6 plus standard 6×7, and Chaos Mode on 4×4 or 4×5 at Connect 4 or Connect 3. |

Perfect is enabled only where every adversarial continuation from the empty board is connected to a verified policy or an exact solved region. In Chaos Mode that condition is met on 4×4 and 4×5, whose complete solutions are committed below; every larger Chaos board still falls back to Brutal.

## Perfect classic play through 7×7

The classic exact engine supports every gravity-valid board with at most seven rows and seven columns. Production policies currently target ordinary Connect Four (`connect = 4`) on fourteen of the fifteen non-standard dimensions from 4×4 through 7×7; 7×7 is the only one still missing, so `data/perfect-classic/manifest.json` records `"complete": false`.

Each non-standard board has two selected optimal-policy closures:

- one for the AI playing the first starting role;
- one for the AI playing the second starting role.

At an AI decision, the native generator solves the position exactly and stores one deterministic optimal move. At an opponent decision, every legal reply remains in the closure. The policy continues until an AI-turn endgame reaches the configured exact-search boundary.

Candidate policies are independently replayed in JavaScript. The verifier checks every reachable policy record, every legal opponent continuation, every stored outcome, the complete closure count, the binary hash, and every exact endgame handoff. Missing, malformed, uncovered or hash-mismatched records fail closed instead of falling back to heuristic play.

The browser lazy-loads only the policy matching the current board dimensions and whether the AI is the first or second player. Standard 6×7 keeps its existing verified strategy and compact bitboard endgame solver.

```bash
npm run classic:verify
npm run classic:policy:verify
```

See [docs/PERFECT_CLASSIC_VARIANTS.md](docs/PERFECT_CLASSIC_VARIANTS.md) for the binary format, root-value matrix, generation workflow and independent replay theorem.

## Exact standard 6×7 play

Classic 6×7 Connect Four retains three independently verified layers:

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

### Completely solved small boards

`data/perfect-chaos-complete/` holds full solutions rather than bounded prefixes. Every position reachable from the empty board under the committed policy is covered, so Perfect needs no search and no handoff there. A rotation transposes the board, so 4×5 and 5×4 are the same game and one certificate spans both orientations.

| Board | Connect | Value | AI decisions (role 1 / role 2) |
|---|---|---|---|
| 4×4 | 4 | Draw | 11,045 / 15,411 |
| 4×4 | 3 | First-player win | 145 / 1,253 |
| 4×5 | 5 | Draw | 416,771 / 588,013 |
| 4×5 | 4 | Draw | 95,645 / 216,194 |
| 4×5 | 3 | First-player win | 178 / 4,601 |
| 4×6 | 4 | Draw | 518,150 / 1,520,491 |
| 4×6 | 3 | First-player win | 224 / 11,155 |
| 4×7 | 3 | First-player win | 291 / 30,302 |
| 5×5 | 4 | Draw | 497,323 / 1,269,295 |
| 5×5 | 3 | First-player win | 180 / 7,805 |
| 5×6 | 3 | First-player win | 267 / 23,131 |

Three larger variants are solved as draws as well — 5×5 connect 5, 4×6 connect 5 and 4×6 connect 6 — but their certificates (313–414 MB each) are too large to publish, so Perfect is not offered there; see [docs/PERFECT_CHAOS.md](docs/PERFECT_CHAOS.md).

The whole catalog is 8.3 MB, and only the file matching the selected board and starting role is fetched. Drawn certificates are kept small by preferring actions that stay inside the closure already built, which roughly halves them.

4×5 Connect-5 is also solved — a draw over 18,631,592 states — but nearly its whole graph is drawn and stays reachable under a drawing policy, so its certificates are far too large to commit.

`scripts/perfect-chaos-complete.mjs` replays each certificate through `src/engine.js` itself, so the rules that check a policy are the rules the game plays by. It requires that every reachable AI position has exactly one legal stored action, that the outcome the policy forces from each position equals the value stored in its record, and that a claimed win cannot be reached by repeating forever — a repetition cycle counts as a draw, which is the real drawing rule. Both drawn certificates reach zero terminal AI losses across their complete closures.

```bash
npm run chaos:complete:verify
```

The complete solver is `native/perfect-chaos-complete.cpp`; `npm run chaos:complete:generate` compiles it, solves a board, and replays the resulting certificates before writing a manifest, so every committed certificate is reproducible from the committed source. The solver and `src/chaos-solver.js` agree exactly on 4×4, including the reachable-state, win, draw and loss counts, and on every sampled 4×5 position.

### Layered non-losing prefix certificate

The released standard 6×7 Chaos policy is a compositional finite-safety-game certificate. At an AI state it stores one action outside the least loss attractor; at an opponent state every legal action remains in the closure. Terminal AI losses are forbidden, while terminal wins, terminal draws, proved repetition cycles and the next exact frontier are safe exits.

The committed boundaries are `0→8`, `8→10`, `10→12`, `12→14`, `14→16`. A later layer may prove an incoming frontier root losing, in which case that root is committed as a rejection and propagated backward until the earlier policy can no longer reach it.

| Role | Final segment | Input roots | Rejected incoming roots | Policy entries | Closure states | Output frontier |
|---|---|---:|---:|---:|---:|---:|
| Red | 14 → 16 | 105,254 | 8,020 | 326,031 | 747,775 | 339,682 |
| Yellow | 14 → 16 | 337,197 | 44,737 | 1,059,068 | 2,498,257 | 1,164,120 |

The final two role segments contain 3,246,032 independently replayed canonical closure states. Every stored AI record is reachable, every opponent continuation is explored, and each recomputed sorted frontier must be byte-identical to the committed table. Artifact hashes and binary metadata are checked before runtime loading.

The remaining certified gap runs from the committed 16-piece frontier to the exact ranked-retrograde endgame handoff at 36 placed pieces. Beyond 16 pieces the runtime returns explicitly to bounded search; the complete standard 6×7 Chaos game is not yet claimed as solved.

```bash
npm run chaos:verify
npm run chaos:prefix:verify-reference
```

## Commands

| Command | Purpose |
|---|---|
| `npm run dev` | Start the local static server. |
| `npm run check` | Parse-check application, solver and proof-tooling source files. |
| `npm test` | Run the Node.js test suite. |
| `npm run ci` | Run source checks, tests and compact exact solver verification. |
| `npm run test:coverage` | Run tests with Node's coverage report. |
| `npm run test:browser` | Exercise the built application in a real Chromium browser. |
| `npm run strategy:verify` | Replay the committed exact standard 6×7 strategy. |
| `npm run classic:verify` | Cross-check the generalized JavaScript and native classic solvers. |
| `npm run classic:solve` | Solve an arbitrary classic board through 7×7 with the native engine. |
| `npm run classic:policy:verify` | Generate and independently replay complete small policy references. |
| `npm run classic:policy:generate` | Generate both role policies for a selected classic board. |
| `npm run classic:policy:verify-reference` | Hash-check and independently replay a generated or committed policy catalog. |
| `npm run chaos:verify` | Cross-check exact Chaos reference games and the small prefix solver. |
| `npm run chaos:prefix:verify-reference` | Independently replay and hash-check the committed 16-piece Chaos certificate. |
| `npm run chaos:prefix:reproduce` | Regenerate the committed Chaos prefix manifest from its rejection seeds. |
| `npm run chaos:complete:generate` | Compile the native complete Chaos solver, solve one board, emit and replay both role certificates. |
| `npm run chaos:complete:verify` | Independently replay the committed complete Chaos certificates. |

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
│   ├── perfect-classic-root-values.json
│   ├── perfect-classic/
│   │   ├── manifest.json
│   │   └── *.bin
│   ├── perfect-chaos-foundation.manifest.json
│   ├── perfect-chaos-prefix/
│   │   ├── manifest.json
│   │   ├── red/
│   │   └── yellow/
│   └── perfect-chaos-complete/
│       ├── manifest.json
│       └── *.bin
├── docs/
│   ├── PERFECT_PLAY.md
│   ├── PERFECT_CLASSIC_VARIANTS.md
│   └── PERFECT_CHAOS.md
├── native/
│   ├── perfect-classic.cpp
│   ├── perfect-classic-policy.cpp
│   ├── perfect-chaos.cpp
│   ├── perfect-chaos-prefix.cpp
│   └── perfect-chaos-complete.cpp
├── scripts/
│   ├── browser-smoke.mjs
│   ├── perfect-book.mjs
│   ├── perfect-strategy.mjs
│   ├── perfect-classic.mjs
│   ├── perfect-classic-policy.mjs
│   ├── perfect-chaos.mjs
│   ├── perfect-chaos-native.mjs
│   ├── perfect-chaos-prefix.mjs
│   ├── perfect-chaos-complete.mjs
│   └── serve.mjs
└── src/
    ├── app.js
    ├── engine.js
    ├── ai.js
    ├── ai-worker.js
    ├── bitboard.js
    ├── classic-solver.js
    ├── perfect-classic-policy.js
    ├── perfect-classic-runtime.js
    ├── perfect-classic-verified.js
    ├── chaos-solver.js
    ├── perfect-chaos-prefix.js
    ├── perfect-chaos-complete.js
    ├── perfect-chaos-runtime.js
    ├── perfect-book.js
    ├── perfect-strategy.js
    └── exact-table.js
```

## Testing and release discipline

The repository checks tactical play, board transformations, repetition handling, exact table validation, classic strategy closure, variable-board policy replay, hash-verified runtime loading, loopy-game retrograde behaviour, native/JavaScript agreement, binary certificate replay, keyboard/touch interaction and responsive layout.

GitHub Actions runs ordinary CI, dedicated Perfect classic and Perfect Chaos policy verifiers, generation workflows, and the Pages deployment. Large generators are manual so proof jobs remain explicit and their artifacts can be reviewed before promotion.

## Licence

Copyright © 2026 senegrom.

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).

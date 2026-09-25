# Connect 4: Chaos Edition

A polished, dependency-light browser implementation of Connect Four with configurable boards, optional flip-and-rotate Chaos moves, accessible controls, several search-based AI levels, a neural opponent, and exact-play tooling.

[Play the current build](https://senegrom.github.io/connect4-chaos/)

![Connect 4: Chaos Edition preview](assets/game-preview.svg)

## Highlights

- **Game-first interface** — the board and primary controls stay prominent on desktop and mobile, while advanced settings remain available without crowding play.
- **Configurable rules** — choose the number of rows, columns and pieces needed to connect.
- **Chaos Mode** — players may drop a piece, flip the board, rotate clockwise or rotate counter-clockwise. Gravity is reapplied after every transformation.
- **Local and computer play** — play against another person, against Easy, Medium, Hard or Brutal search, against Perfect play where a certificate exists, or against the Neural opponent.
- **Perfect classic variants** — non-Chaos Connect Four boards from 4×4 through 7×6 use verified role-specific policies with an exact endgame handoff; standard 6×7 keeps its oracle-generated strategy, whose closure is replayed. Only 7×7 is still uncertified.
- **Perfect Chaos on solved boards** — eleven Chaos Mode configurations from 4×4 to 5×6, at Connect 3, 4 and 5, are solved completely for both starting roles, so Perfect is available there with no search and no handoff.
- **Neural opponent** — an AlphaZero-style network with a look-ahead search runs in the browser on WebGPU, or on WebAssembly where there is no usable GPU, on any board up to 10×10. It is an approximately 133 MB download that the page asks about first, normally cached by your browser.
- **Certified Chaos prefix** — standard 6×7 Chaos Mode has an independently replayed non-losing policy certificate for both starting roles through **16 placed pieces**; Brutal lazy-loads only the matching certified layer during live play.
- **Exact Chaos endgames** — eligible late-game Chaos positions with six or fewer empty cells are solved as complete loopy game graphs rather than ordinary depth-limited trees.
- **Transparent telemetry** — search depth, nodes, principal variation and exact proof status are shown without presenting bounded search as solved play.
- **Accessible interaction** — keyboard support, touch guidance, ARIA labels, live announcements, strong focus states and reduced-motion support are built in.
- **No runtime framework** — the shipped game is plain HTML, CSS and JavaScript. The only vendored library is the ONNX runtime behind the neural opponent, loaded on demand. Development checks need Node.js and Python 3; the exact-table generators need a C++20 compiler.

## Quick start

The development server needs only Node.js. `npm run check` also parses the Python sources and `npm run ci` runs Python 3 tests, so `python3` must be on the path, and the table generators compile C++20 sources.

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
| Perfect | Game-theoretically exact play wherever a certificate exists: non-Chaos Connect Four on the 14 verified boards from 4×4 through 7×6 plus standard 6×7, and Chaos Mode on the eleven completely solved configurations listed below. |
| Neural | A trained network with a look-ahead search, run on your device after an approximately 133 MB download, normally cached by your browser. The strongest general opponent on large boards, but not exact. |

Perfect is enabled only where every adversarial continuation from the empty board is connected to a verified policy or an exact solved region. In Chaos Mode that condition is met on the eleven completely solved configurations listed below, in the orientation each certificate was solved from; every other Chaos board falls back to Brutal. Tables over 8 MB are downloaded once, after an explicit prompt.

## Perfect play

Perfect uses only proved results. The proofs, formats and verification commands are in the docs:

- **Classic Connect Four from 4×4 through 7×6.** The fourteen non-standard boards play verified role-specific policies with an exact endgame handoff. Each policy is replayed independently, and the two starting roles of every board must prove opposite values, which pins the exact game value. See [PERFECT_CLASSIC_VARIANTS](docs/PERFECT_CLASSIC_VARIANTS.md).
- **Standard 6×7.** A solved opening book, a deterministic strategy for both starting roles and an exact late-game solver; the strategy's closure is replayed against every legal opponent continuation. See [PERFECT_PLAY](docs/PERFECT_PLAY.md).
- **Chaos Mode.** Flips and rotations can revisit positions, so the exact model is a game graph rather than a tree. Eleven configurations are solved completely for both starting roles: 4×4 (connect 3 and 4), 4×5 (3, 4 and 5), 4×6 (3 and 4), 4×7 (3), 5×5 (3 and 4) and 5×6 (3). Their certificates are replayed through the game engine itself. On standard 6×7 a layered non-losing certificate covers the opening for both roles, and late positions with six or fewer empty cells are solved exactly. See [PERFECT_CHAOS](docs/PERFECT_CHAOS.md).

## Commands

| Command | Purpose |
|---|---|
| `npm run dev` | Start the local static server. |
| `npm run check` | Parse-check every tracked JavaScript and Python file. |
| `npm test` | Run the Node.js test suite. |
| `npm run ci` | Run source checks, tests and compact exact solver verification. |
| `npm run test:coverage` | Run tests with Node's coverage report. |
| `npm run test:browser` | Exercise the built application in a real Chromium browser. |
| `npm run test:strength` | Score the shipped network and its search on positions with exactly one winning move; needs the model (`NEURAL_MODEL` or `NEURAL_MODEL_DOWNLOAD=1`). |
| `npm run strategy:verify` | Replay the committed exact standard 6×7 strategy. |
| `npm run classic:verify` | Cross-check the generalized JavaScript and native classic solvers. |
| `npm run classic:solve -- --rows R --columns C --connect 4` | Solve one classic board through 7×7 with the native engine; without the flags it solves standard 6×7. |
| `npm run classic:policy:verify` | Generate and independently replay complete small policy references. |
| `npm run classic:policy:generate -- --rows R --columns C --connect 4` | Generate both role policies for one classic board; without the flags it targets standard 6×7. |
| `npm run classic:policy:verify-reference` | Hash-check and independently replay a generated or committed policy catalog. |
| `npm run chaos:verify` | Cross-check exact Chaos reference games and the small prefix solver. |
| `npm run chaos:prefix:verify-reference` | Independently replay and hash-check the committed 16-piece Chaos certificate. |
| `npm run chaos:prefix:reproduce` | Regenerate the committed Chaos prefix certificates from their rejection seeds and compare the files and summaries with the committed ones; the last yellow segment builds a 57-million-state graph. |
| `npm run chaos:complete:generate` | Compile the native complete Chaos solver, solve one board, emit and replay both role certificates. |
| `npm run chaos:complete:verify` | Independently replay the committed complete Chaos certificates. |

The WDL solver and its cross-check, the claim gate a 6×7 Chaos Perfect label would need, and the prefix bridge scanner run directly from `scripts/`; [PERFECT_CHAOS_OPTIMALITY](docs/PERFECT_CHAOS_OPTIMALITY.md) and [CHAOS_BOUNDED_PROOF](docs/CHAOS_BOUNDED_PROOF.md) describe them.

## Documentation

| Document | What it covers |
| --- | --- |
| [PERFECT_PLAY](docs/PERFECT_PLAY.md) | The Perfect guarantee for standard 6×7 play, its proof boundary and binary formats |
| [PERFECT_CLASSIC_VARIANTS](docs/PERFECT_CLASSIC_VARIANTS.md) | Exact classic play on boards from 4×4 through 7×7: policies, root values, replay |
| [PERFECT_CHAOS](docs/PERFECT_CHAOS.md) | Chaos Mode's exact model, its solvers, the solved boards and the 6×7 prefix certificate |
| [PERFECT_CHAOS_OPTIMALITY](docs/PERFECT_CHAOS_OPTIMALITY.md) | What a non-losing certificate proves, and what a Perfect label needs beyond it |
| [CHAOS_BOUNDED_PROOF](docs/CHAOS_BOUNDED_PROOF.md) | The bounded loopy proofs behind the Chaos AI, and how to extend the prefix |
| [NEURAL_CHAOS](docs/NEURAL_CHAOS.md) | The neural opponent: network, search, training loop and datasets |
| [NEURAL_MODEL_RELEASES](docs/NEURAL_MODEL_RELEASES.md) | How a trained network is verified, published to R2 and pinned by the page |
| [neural-worker](docs/neural-worker.md) | Where neural inference runs in the browser, and how it recovers from failures |
| [neural-benchmark](docs/neural-benchmark.md) | The Neural versus Brutal benchmark and how to run it |
| [browser-testing](docs/browser-testing.md) | The browser regression suites and how to run them locally |
| [TRAINING_REVIEW_NOTES](docs/TRAINING_REVIEW_NOTES.md) | Notes from the review rounds of the training, dataset and Modal code |

## Project structure

```text
.
├── index.html, styles.css, manifest.json, favicon.svg, favicon.ico
├── cross-origin-isolation-worker.js   service worker that lets WebAssembly use threads
├── assets/
│   ├── game-preview.svg
│   ├── perfect-book.bin, perfect-strategy.bin
│   └── neural/                        model.json (the R2 model's identity) and the vendored ONNX runtime
├── data/
│   ├── perfect-book.manifest.json, perfect-strategy.manifest.json
│   ├── perfect-classic-root-values.json
│   ├── perfect-classic/               manifest and 28 role policies
│   ├── perfect-chaos-complete/        manifest and 22 complete certificates
│   └── perfect-chaos-prefix/          manifest, red/, yellow/, provenance/
├── docs/                              the eleven documents listed above
├── icons/                             app icons
├── native/                            C++20 solvers: perfect-classic*, perfect-chaos*
├── neural/                            training stack: GPU self-play, batched search, trainer,
│                                      arena, Modal app and loop driver
├── scripts/                           perfect-*.mjs generators and verifiers, Chaos table
│                                      tooling (*.py), site build, browser suites, dev server,
│                                      model publishing
├── src/                               engine, AI worker and search, exact-play runtimes,
│                                      neural runtime, download gate
├── tests/                             node --test suites
└── workers/model-cdn/                 Cloudflare Worker that serves the model from R2
```

## Testing and release discipline

The repository checks tactical play, board transformations, repetition handling, exact table validation, classic strategy closure, variable-board policy replay, hash-verified runtime loading, loopy-game retrograde behaviour, native/JavaScript agreement, binary certificate replay, keyboard/touch interaction and responsive layout.

GitHub Actions runs ordinary CI (Node, browser, native, security, training and network-strength jobs), the replay of the Perfect classic catalog and the replay of the Chaos prefix certificate. Pages deploys only after all of them pass on the same commit. Table generation runs on demand from the command line, so proof jobs remain explicit and their artifacts can be reviewed before promotion.

## Licence

Copyright © 2026 senegrom.

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).

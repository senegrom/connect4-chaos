# Connect 4: Chaos Edition

A polished browser game that keeps classic Connect Four intact while adding configurable boards, five AI levels, and an optional **Chaos Mode** where flipping or rotating the board is a legal move.

[Play the game](https://senegrom.github.io/connect4-chaos/) · [View the source](https://github.com/senegrom/connect4-chaos)

![Connect 4: Chaos Edition preview](assets/game-preview.svg)

## Highlights

- Classic two-player Connect Four or play against Easy, Medium, Hard, Brutal, or game-theoretically **Perfect** AI.
- Board sizes from 4×4 to 10×10 and connect lengths from 3 to 6.
- Optional flip, clockwise rotation, and counter-clockwise rotation moves.
- Responsive, game-first layout with progressive setup controls and an in-page rules guide.
- Live AI depth, position count, elapsed time, and search-rate feedback.
- Undo that returns to the previous human decision in AI games.
- Keyboard, mouse, touch, reduced-motion, forced-colour, and screen-reader support.
- Persistent settings and match scores using local storage.
- No runtime dependencies, tracking, adverts, or external network calls.
- Pure game-engine and AI modules with proof tests, a real-Chromium smoke test, and GitHub Actions CI.

## AI

The AI runs in a Web Worker, so deeper searches do not freeze the interface. Standard 7×6 Connect Four uses a dedicated BigInt bitboard engine; configurable classic boards use the mutable array search, while Chaos Mode retains the fully general transformation-aware search.

Strength improvements include:

- A closed exact strategy makes every Perfect-AI decision on the standard board.
- A generated exact opening book is checked before ordinary standard-board search.
- Seven-bit column bitboards with make/unmake-free move generation on the standard board.
- Iterative-deepening alpha-beta search with forced-block extensions and non-losing move pruning.
- A fixed-size typed-array transposition table with horizontal symmetry canonicalisation.
- A dedicated null-window outcome solver proves standard-board endgames without a clock cutoff.
- Gravity-aware evaluation that distinguishes playable threats from floating shapes.
- Principal-variation, killer-move, history, and centre-first move ordering.
- Reusable search information between completed depths.
- Repetition-aware search for Chaos Mode.
- A final tactical-safety invariant: when at least one legal move avoids an immediate loss on the opponent's next move, the AI will not return an unsafe move.

On a standard 7×6 board, Medium completes 10 plies, Hard 14, and Brutal 16. Medium solves positions with at most 16 empty cells exactly, Hard 20, and Brutal 24. Custom boards retain the general 6/9/12-ply engine. There is no wall-clock cutoff: a worker finishes the selected depth or exact endgame unless the player explicitly cancels it by restarting, undoing, or changing the game.

The endgame thresholds use a specialised win/draw/loss solver with symmetry-aware bound caching. It searches to terminal positions directly rather than running the heuristic engine at a nominal remaining depth.

### Perfect play

**Perfect** is available only for classic 6×7 Connect Four with Connect 4 and Chaos Mode off. It does not use heuristic search. A committed strategy contains **470,494 exact AI decisions** covering both possible starting roles and every legal opposing reply until the board has at most 24 empty cells. From there, the browser's exact outcome solver proves the rest of the game tree.

The strategy's closure proof visits 1,206,169 canonical positions across the two roles and reaches 735,675 exact-solver handoffs plus 136,467 earlier terminal continuations. CI validates the binary checksum, metadata, legal moves, reachability, both role flags, and the full adversarial closure before deployment.

“Perfect” means game-theoretically optimal, not automatically unbeatable from every position. Standard Connect Four is a first-player win: the Perfect AI forces a win when it starts, while as the second player it returns the best achievable result against the human's chosen opening. It never substitutes a bounded heuristic move; a missing or invalid strategy stops the AI visibly instead of silently weakening play.

Lower levels continue to use `assets/perfect-book.bin`, which covers all **129,498 canonical positions through ply 8** in 1,294,992 bytes before falling back to their configured search. The 4,704,952-byte Perfect strategy is loaded lazily and only inside the AI worker, so ordinary games, custom boards, and Chaos Mode do not pay its download or memory cost.

The persistent **Generate perfect-play book** and **Generate perfect-play strategy** workflows reproduce both exact datasets from a pinned oracle, validate deterministic binary output, run the complete test suite, and record provenance in `data/`. See [`docs/PERFECT_PLAY.md`](docs/PERFECT_PLAY.md) for the formats, proof boundary, generation process, and remaining independent-verification work.

The exact endgame path is regression-tested against a separate array-based minimax implementation across 250 deterministic late-game positions. Winning-square generation also has dedicated tests for both line ends and internal gaps.

## Chaos Mode rules

A flip or rotation consumes the current player's turn. After the transformation, gravity is applied downward in the board's new orientation.

- If the transformation creates a line for one player, that player wins.
- If it creates winning lines for both players, the player who transformed the board loses the tie.
- A full board without a winner is a draw.
- The same board position with the same player to move for a third time is a draw by repetition.

Rotating a non-square board swaps its row and column counts for the rest of that round.

## Controls

| Action | Mouse / touch | Keyboard |
| --- | --- | --- |
| Choose a column | Point at a cell or column marker | <kbd>←</kbd> / <kbd>→</kbd>, <kbd>Home</kbd>, <kbd>End</kbd> |
| Drop a piece | Click or tap | <kbd>Enter</kbd> / <kbd>Space</kbd> |
| Undo | Undo button | <kbd>U</kbd> |
| New round | New round button | <kbd>N</kbd> |
| Flip | Flip button | <kbd>F</kbd> |
| Rotate clockwise | Rotate right | <kbd>R</kbd> |
| Rotate counter-clockwise | Rotate left | <kbd>Shift</kbd> + <kbd>R</kbd> |

## Run locally

Node.js 22 or newer is recommended.

```bash
npm install
npm run dev
```

Open `http://127.0.0.1:4173`. A local server is needed during development because browsers restrict ES modules and Web Workers when an HTML file is opened directly from disk.

Run all engine, proof, and format checks:

```bash
npm run ci
```

Run the zero-dependency real-Chromium smoke test:

```bash
npm run test:browser
```

The browser test exercises Perfect AI from both starting roles, watches for console and runtime errors, frame-samples the board position, and verifies the deliberate transform timings. It auto-detects Google Chrome or Chromium; `CHROME_BIN` can select a specific executable.

An optional coverage report is available with `npm run test:coverage`.

## Project structure

```text
.
├── index.html              Semantic interface, metadata, and security policy
├── styles.css              Responsive visual design and animations
├── assets/perfect-book.bin Exact standard-board opening entries
├── assets/perfect-strategy.bin Closed Perfect-AI decision policy
├── data/                   Exact-data provenance, checksums, and proof metadata
├── src/
│   ├── engine.js           Pure rules, gravity, wins, transforms, repetition keys
│   ├── bitboard.js         Standard 7×6 bitboard search and exact endgame solver
│   ├── perfect-book.js     Validated binary opening-book loader and lookup
│   ├── perfect-strategy.js Validated closed-strategy loader and lookup
│   ├── ai.js               General-board and Chaos alpha-beta searches
│   ├── ai-worker.js        Background AI entry point and progress messages
│   └── app.js              UI state, rendering, persistence, input, and undo
├── tests/
│   ├── engine.test.js      Rules and transform tests
│   ├── ai.test.js          Routing, tactical, fixed-depth, and mutation-safety tests
│   ├── bitboard.test.js    Bitboard conversion, safety, and exact-solve cross-checks
│   ├── perfect-book.test.js Binary validation and book-routing tests
│   ├── perfect-book-generator.test.js Shard completeness and balance tests
│   ├── perfect-strategy.test.js Strategy format and synthetic closure tests
│   ├── perfect-strategy-runtime.test.js Runtime decoder and routing tests
│   ├── perfect-strategy-file.test.js Committed checksum and full closure proof
│   └── worker.test.js      Browser-worker protocol and progress test
└── scripts/
    ├── perfect-book.mjs    Canonical enumeration and deterministic packing
    ├── perfect-strategy.mjs Adversarial exact-policy generation and proof
    ├── browser-smoke.mjs   Zero-dependency Chrome DevTools Protocol smoke test
    └── serve.mjs           Dependency-free local static server
```

The rules engine does not depend on the DOM, which makes game behaviour deterministic and straightforward to test. UI state is kept separately. The bitboard engine works with immutable position values, while the configurable classic engine mutates only its private search copy and restores it after every branch.

## Deployment

CI runs the full engine/proof suite and a real-Chromium smoke test on every push and pull request. The Pages build repeats both gates before a successful push to `main` is deployed.

## Origin

This project is a ground-up refactor of a single-file HTML prototype supplied by email. The visual direction and unusual flip/rotate mechanics were retained; the code was separated into maintainable modules and the game gained background AI, undo, persistence, accessibility improvements, tests, CI, and progressively stronger search.

## License

[MIT](LICENSE)

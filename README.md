# Connect 4: Chaos Edition

A polished browser game that keeps classic Connect Four intact while adding configurable boards, four AI levels, and an optional **Chaos Mode** where flipping or rotating the board is a legal move.

![Connect 4: Chaos Edition preview](assets/game-preview.svg)

## Highlights

- Classic two-player Connect Four or play against Easy, Medium, Hard, or Brutal AI.
- Board sizes from 4×4 to 10×10 and connect lengths from 3 to 6.
- Optional flip, clockwise rotation, and counter-clockwise rotation moves.
- Iterative-deepening alpha-beta AI with move ordering, repetition awareness, and a transposition table for classic games.
- AI search runs in a Web Worker, so deeper searches do not freeze the interface.
- Undo that returns to the previous human decision in AI games.
- Keyboard, mouse, touch, reduced-motion, and screen-reader support.
- Persistent settings and match scores using local storage.
- No runtime dependencies, tracking, adverts, or external network calls.
- Pure game-engine and AI modules with automated tests and GitHub Actions CI.

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
| New round | Start new round | <kbd>N</kbd> |
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

Run all checks:

```bash
npm run ci
```

An optional coverage report is available with `npm run test:coverage`.

## Project structure

```text
.
├── index.html              Semantic interface and security policy
├── styles.css              Responsive visual design and animations
├── src/
│   ├── engine.js           Pure rules, gravity, wins, transforms, repetition keys
│   ├── ai.js               Evaluation and iterative-deepening alpha-beta search
│   ├── ai-worker.js        Background AI entry point
│   └── app.js              UI state, rendering, persistence, input, and undo
├── tests/
│   ├── engine.test.js      Rules and transform tests
│   ├── ai.test.js          Tactical and search tests
│   └── worker.test.js      Browser-worker protocol test
└── scripts/serve.mjs       Dependency-free local static server
```

The rules engine does not depend on the DOM, which makes game behaviour deterministic and straightforward to test. UI state is kept separately, and every move produces a new board rather than mutating the previous one.

## GitHub Pages

The repository includes a Pages deployment workflow. After selecting **GitHub Actions** as the Pages source in the repository settings once, every push to `main` deploys the static game automatically.

## Origin

This project is a ground-up refactor of a single-file HTML prototype supplied by email. The visual direction and unusual flip/rotate mechanics were retained; the code was separated into maintainable modules and the game gained background AI, undo, persistence, accessibility improvements, tests, and CI.

## License

[MIT](LICENSE)

from pathlib import Path
import base64
import json


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def write_text(path, content):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


write_text('src/perfect-book.js', "const MAGIC = 'C4PB';\nconst FORMAT_VERSION = 1;\nconst HEADER_SIZE = 12;\nconst ENTRY_SIZE = 10;\n\nfunction bytesFrom(input) {\n  if (input instanceof Uint8Array) return input;\n  if (input instanceof ArrayBuffer) return new Uint8Array(input);\n  if (ArrayBuffer.isView(input)) {\n    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);\n  }\n  throw new TypeError('Perfect-book data must be an ArrayBuffer or typed array.');\n}\n\nfunction ascii(bytes, offset, length) {\n  let value = '';\n  for (let index = 0; index < length; index += 1) {\n    value += String.fromCharCode(bytes[offset + index]);\n  }\n  return value;\n}\n\nexport function decodePerfectBook(input) {\n  const bytes = bytesFrom(input);\n  if (bytes.byteLength < HEADER_SIZE) throw new Error('Perfect-book data is truncated.');\n  if (ascii(bytes, 0, 4) !== MAGIC) throw new Error('Perfect-book magic is invalid.');\n\n  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);\n  const version = view.getUint8(4);\n  const maxPly = view.getUint8(5);\n  const entrySize = view.getUint8(6);\n  const entryCount = view.getUint32(8, true);\n\n  if (version !== FORMAT_VERSION) {\n    throw new Error(`Unsupported perfect-book version ${version}.`);\n  }\n  if (entrySize !== ENTRY_SIZE) {\n    throw new Error(`Unsupported perfect-book entry size ${entrySize}.`);\n  }\n\n  const expectedLength = HEADER_SIZE + entryCount * entrySize;\n  if (bytes.byteLength !== expectedLength) {\n    throw new Error(\n      `Perfect-book length mismatch: expected ${expectedLength}, found ${bytes.byteLength}.`,\n    );\n  }\n\n  let previousKey = -1n;\n  for (let index = 0; index < entryCount; index += 1) {\n    const offset = HEADER_SIZE + index * ENTRY_SIZE;\n    const key = view.getBigUint64(offset, true);\n    const moveMask = view.getUint8(offset + 8);\n    const outcome = view.getInt8(offset + 9);\n    if (key <= previousKey) throw new Error('Perfect-book keys must be strictly increasing.');\n    if (moveMask === 0 || (moveMask & 0x80) !== 0) {\n      throw new Error('Perfect-book move masks must contain at least one of seven columns.');\n    }\n    if (outcome < -1 || outcome > 1) {\n      throw new Error('Perfect-book outcomes must be -1, 0, or 1.');\n    }\n    previousKey = key;\n  }\n\n  return Object.freeze({\n    version,\n    maxPly,\n    entryCount,\n    byteLength: bytes.byteLength,\n    lookup(key) {\n      if (typeof key !== 'bigint' || key < 0n) return null;\n\n      let low = 0;\n      let high = entryCount - 1;\n      while (low <= high) {\n        const middle = (low + high) >> 1;\n        const offset = HEADER_SIZE + middle * ENTRY_SIZE;\n        const candidate = view.getBigUint64(offset, true);\n\n        if (candidate === key) {\n          return Object.freeze({\n            key: candidate,\n            moveMask: view.getUint8(offset + 8),\n            outcome: view.getInt8(offset + 9),\n          });\n        }\n        if (candidate < key) low = middle + 1;\n        else high = middle - 1;\n      }\n      return null;\n    },\n  });\n}\n\nasync function readBookBytes(url) {\n  if (url.protocol === 'file:' && typeof process !== 'undefined' && process.versions?.node) {\n    const { readFile } = await import('node:fs/promises');\n    return readFile(url);\n  }\n\n  const response = await fetch(url);\n  if (!response.ok) {\n    throw new Error(`Could not load perfect-play book (${response.status}).`);\n  }\n  return new Uint8Array(await response.arrayBuffer());\n}\n\nlet defaultBookPromise = null;\n\nexport function loadPerfectBook(\n  url = new URL('../assets/perfect-book.bin', import.meta.url),\n) {\n  if (!defaultBookPromise) {\n    defaultBookPromise = readBookBytes(url).then(decodePerfectBook);\n  }\n  return defaultBookPromise;\n}\n")
write_text('scripts/perfect-book.mjs', "#!/usr/bin/env node\n\nimport { createHash } from 'node:crypto';\nimport { readFile, writeFile } from 'node:fs/promises';\nimport process from 'node:process';\n\nconst WIDTH = 7;\nconst HEIGHT = 6;\nconst STRIDE = HEIGHT + 1;\nconst COLUMN_BITS = (1n << BigInt(HEIGHT)) - 1n;\nconst COLUMN_WITH_SENTINEL = (1n << BigInt(STRIDE)) - 1n;\nconst BOTTOM_MASKS = Array.from(\n  { length: WIDTH },\n  (_, column) => 1n << BigInt(column * STRIDE),\n);\nconst COLUMN_MASKS = BOTTOM_MASKS.map((bottom) => bottom * COLUMN_BITS);\nconst BOTTOM_MASK = BOTTOM_MASKS.reduce((mask, bit) => mask | bit, 0n);\nconst BOARD_MASK = BOTTOM_MASK * COLUMN_BITS;\nconst COLUMN_ORDER = Object.freeze([3, 2, 4, 1, 5, 0, 6]);\nconst HEADER_SIZE = 12;\nconst ENTRY_SIZE = 10;\n\nfunction fail(message) {\n  console.error(message);\n  process.exitCode = 1;\n}\n\nfunction parseOptions(values) {\n  const options = new Map();\n  for (let index = 0; index < values.length; index += 1) {\n    const token = values[index];\n    if (!token.startsWith('--')) throw new Error(`Unexpected argument: ${token}`);\n    const key = token.slice(2);\n    const next = values[index + 1];\n    if (next === undefined || next.startsWith('--')) options.set(key, true);\n    else {\n      options.set(key, next);\n      index += 1;\n    }\n  }\n  return options;\n}\n\nfunction integerOption(options, name, fallback, minimum = 0) {\n  const raw = options.get(name);\n  if (raw === undefined) return fallback;\n  const value = Number.parseInt(String(raw), 10);\n  if (!Number.isInteger(value) || value < minimum) {\n    throw new Error(`--${name} must be an integer of at least ${minimum}.`);\n  }\n  return value;\n}\n\nfunction possibleMoves(mask) {\n  return (mask + BOTTOM_MASK) & BOARD_MASK;\n}\n\nfunction moveForColumn(mask, column) {\n  return (mask + BOTTOM_MASKS[column]) & COLUMN_MASKS[column];\n}\n\nfunction play(position, move) {\n  return {\n    current: position.current ^ position.mask,\n    mask: position.mask | move,\n    moves: position.moves + 1,\n  };\n}\n\nfunction hasAlignment(bits) {\n  for (const direction of [1, HEIGHT, STRIDE, HEIGHT + 2]) {\n    const shift = BigInt(direction);\n    const pair = bits & (bits >> shift);\n    if ((pair & (pair >> (2n * shift))) !== 0n) return true;\n  }\n  return false;\n}\n\nfunction mirrorBits(bits) {\n  let mirrored = 0n;\n  for (let column = 0; column < WIDTH; column += 1) {\n    const columnBits = (bits >> BigInt(column * STRIDE)) & COLUMN_WITH_SENTINEL;\n    mirrored |= columnBits << BigInt((WIDTH - 1 - column) * STRIDE);\n  }\n  return mirrored;\n}\n\nfunction mirrorMoveMask(mask) {\n  let mirrored = 0;\n  for (let column = 0; column < WIDTH; column += 1) {\n    if ((mask & (1 << column)) !== 0) mirrored |= 1 << (WIDTH - 1 - column);\n  }\n  return mirrored;\n}\n\nfunction canonicalPosition(position) {\n  const normal = position.current + position.mask;\n  const mirrored = mirrorBits(position.current) + mirrorBits(position.mask);\n  return normal <= mirrored\n    ? { key: normal, mirrored: false }\n    : { key: mirrored, mirrored: true };\n}\n\nfunction positionFromSequence(sequence) {\n  let position = { current: 0n, mask: 0n, moves: 0 };\n\n  for (let index = 0; index < sequence.length; index += 1) {\n    const column = Number(sequence[index]) - 1;\n    if (!Number.isInteger(column) || column < 0 || column >= WIDTH) {\n      throw new Error(`Invalid column \"${sequence[index]}\" in sequence \"${sequence}\".`);\n    }\n    const move = moveForColumn(position.mask, column);\n    if (move === 0n) throw new Error(`Full column in sequence \"${sequence}\".`);\n    if (hasAlignment(position.current | move) && index + 1 !== sequence.length) {\n      throw new Error(`Sequence continues after a win: \"${sequence}\".`);\n    }\n    position = play(position, move);\n  }\n\n  return position;\n}\n\nasync function enumerate(options) {\n  const depth = integerOption(options, 'depth', 6, 0);\n  const shardCount = integerOption(options, 'shard-count', 1, 1);\n  const shardIndex = integerOption(options, 'shard-index', 0, 0);\n  if (shardIndex >= shardCount) throw new Error('--shard-index must be below --shard-count.');\n\n  const visited = new Set();\n  const lines = [];\n\n  function explore(position, sequence) {\n    const canonical = canonicalPosition(position);\n    if (visited.has(canonical.key)) return;\n    visited.add(canonical.key);\n\n    if (Number(canonical.key % BigInt(shardCount)) === shardIndex) lines.push(sequence);\n    if (position.moves >= depth) return;\n\n    const possible = possibleMoves(position.mask);\n    for (const column of COLUMN_ORDER) {\n      const move = moveForColumn(position.mask, column);\n      if ((possible & move) === 0n) continue;\n      if (hasAlignment(position.current | move)) continue;\n      explore(play(position, move), `${sequence}${column + 1}`);\n    }\n  }\n\n  explore({ current: 0n, mask: 0n, moves: 0 }, '');\n  process.stdout.write(`${lines.join('\\n')}\\n`);\n  console.error(\n    `Enumerated ${visited.size} canonical positions; emitted ${lines.length} for shard `\n      + `${shardIndex + 1}/${shardCount} through ply ${depth}.`,\n  );\n}\n\nfunction parseScoredLine(line, lineNumber) {\n  const tokens = line.trim().split(/\\s+/).filter(Boolean);\n  let sequence;\n  let scoreTokens;\n\n  if (tokens.length === WIDTH) {\n    sequence = '';\n    scoreTokens = tokens;\n  } else if (tokens.length === WIDTH + 1 && /^[1-7]*$/.test(tokens[0])) {\n    [sequence] = tokens;\n    scoreTokens = tokens.slice(1);\n  } else {\n    throw new Error(`Invalid scored line ${lineNumber}: ${line}`);\n  }\n\n  const scores = scoreTokens.map((token) => Number.parseInt(token, 10));\n  if (scores.some((score) => !Number.isInteger(score))) {\n    throw new Error(`Non-integer score on line ${lineNumber}.`);\n  }\n  return { sequence, scores };\n}\n\nfunction encode(entries, maxPly) {\n  const bytes = Buffer.alloc(HEADER_SIZE + entries.length * ENTRY_SIZE);\n  bytes.write('C4PB', 0, 4, 'ascii');\n  bytes.writeUInt8(1, 4);\n  bytes.writeUInt8(maxPly, 5);\n  bytes.writeUInt8(ENTRY_SIZE, 6);\n  bytes.writeUInt8(0, 7);\n  bytes.writeUInt32LE(entries.length, 8);\n\n  entries.forEach((entry, index) => {\n    const offset = HEADER_SIZE + index * ENTRY_SIZE;\n    bytes.writeBigUInt64LE(entry.key, offset);\n    bytes.writeUInt8(entry.moveMask, offset + 8);\n    bytes.writeInt8(entry.score, offset + 9);\n  });\n  return bytes;\n}\n\nasync function pack(options) {\n  const inputPath = options.get('input');\n  const outputPath = options.get('output') ?? 'assets/perfect-book.bin';\n  const manifestPath = options.get('manifest') ?? 'data/perfect-book.manifest.json';\n  const maxPly = integerOption(options, 'max-ply', 6, 0);\n  const source = String(options.get('source') ?? 'exact solver output');\n\n  if (!inputPath || inputPath === true) throw new Error('--input is required.');\n\n  const text = await readFile(inputPath, 'utf8');\n  const entriesByKey = new Map();\n\n  text.split(/\\r?\\n/).forEach((line, index) => {\n    if (!line.trim()) return;\n    const { sequence, scores } = parseScoredLine(line, index + 1);\n    if (sequence.length > maxPly) return;\n\n    const position = positionFromSequence(sequence);\n    const possible = possibleMoves(position.mask);\n    let bestScore = -Infinity;\n    let moveMask = 0;\n\n    for (let column = 0; column < WIDTH; column += 1) {\n      const move = moveForColumn(position.mask, column);\n      if ((possible & move) === 0n) continue;\n      const score = scores[column];\n      if (score > bestScore) {\n        bestScore = score;\n        moveMask = 1 << column;\n      } else if (score === bestScore) {\n        moveMask |= 1 << column;\n      }\n    }\n\n    if (!Number.isFinite(bestScore) || bestScore < -127 || bestScore > 127 || moveMask === 0) {\n      throw new Error(`Invalid best move on scored line ${index + 1}.`);\n    }\n\n    const canonical = canonicalPosition(position);\n    if (canonical.mirrored) moveMask = mirrorMoveMask(moveMask);\n    const existing = entriesByKey.get(canonical.key);\n    if (existing) {\n      if (existing.strongScore !== bestScore) {\n        throw new Error(`Conflicting scores for canonical key ${canonical.key}.`);\n      }\n      existing.moveMask |= moveMask;\n    } else {\n      entriesByKey.set(canonical.key, {\n        key: canonical.key,\n        moveMask,\n        score: Math.sign(bestScore),\n        strongScore: bestScore,\n        ply: sequence.length,\n      });\n    }\n  });\n\n  const entries = [...entriesByKey.values()].sort((first, second) => (\n    first.key < second.key ? -1 : first.key > second.key ? 1 : 0\n  ));\n\n  for (let index = 1; index < entries.length; index += 1) {\n    if (entries[index - 1].key >= entries[index].key) {\n      throw new Error('Perfect-book keys are not strictly increasing.');\n    }\n  }\n\n  const bytes = encode(entries, maxPly);\n  const sha256 = createHash('sha256').update(bytes).digest('hex');\n  const metadata = {\n    format: 1,\n    maxPly,\n    entryCount: entries.length,\n    source,\n    scoreSemantics: 'game-theoretic outcome; move mask contains strong-optimal moves',\n    generatedAt: null,\n    sha256,\n  };\n  const manifest = {\n    ...metadata,\n    byteLength: bytes.length,\n    optimalMoveEntries: entries.reduce((count, entry) => (\n      count + ((entry.moveMask & (entry.moveMask - 1)) !== 0 ? 1 : 0)\n    ), 0),\n  };\n\n  await writeFile(outputPath, bytes);\n  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\\n`);\n  console.error(\n    `Packed ${entries.length} exact positions through ply ${maxPly} `\n      + `(${bytes.length} bytes, sha256 ${sha256}).`,\n  );\n}\n\nasync function main() {\n  const [command, ...rest] = process.argv.slice(2);\n  const options = parseOptions(rest);\n\n  if (command === 'enumerate') await enumerate(options);\n  else if (command === 'pack') await pack(options);\n  else {\n    throw new Error(\n      'Usage:\\n'\n        + '  node scripts/perfect-book.mjs enumerate --depth 6\\n'\n        + '  node scripts/perfect-book.mjs pack --input scored.txt --output assets/perfect-book.bin --max-ply 6',\n    );\n  }\n}\n\nmain().catch((error) => fail(error instanceof Error ? error.message : String(error)));\n")
write_text('tests/perfect-book.test.js', "import test from 'node:test';\nimport assert from 'node:assert/strict';\nimport { readFile } from 'node:fs/promises';\nimport { chooseMove } from '../src/ai.js';\nimport { RED } from '../src/engine.js';\nimport { decodePerfectBook, loadPerfectBook } from '../src/perfect-book.js';\n\nfunction emptyBoard() {\n  return Array.from({ length: 6 }, () => Array(7).fill(0));\n}\n\ntest('the committed perfect-play book is valid and contains the solved root move', async () => {\n  const book = await loadPerfectBook();\n  assert.equal(book.version, 1);\n  assert.ok(book.entryCount >= 1);\n\n  const root = book.lookup(0n);\n  assert.ok(root);\n  assert.equal(root.outcome, 1);\n  assert.notEqual(root.moveMask & (1 << 3), 0);\n});\n\ntest('standard play uses an exact book move before allocating a search table', async () => {\n  const book = await loadPerfectBook();\n  const result = chooseMove({\n    board: emptyBoard(),\n    currentPlayer: RED,\n    connect: 4,\n    chaosMode: false,\n  }, {\n    difficulty: 'brutal',\n    aiPlayer: RED,\n    perfectBook: book,\n  });\n\n  assert.equal(result.solver, 'perfect-book');\n  assert.equal(result.solved, true);\n  assert.deepEqual(result.action, { type: 'drop', column: 3 });\n  assert.equal(result.nodes, 0);\n});\n\ntest('the decoder rejects malformed books instead of making an unverified move', async () => {\n  const bytes = new Uint8Array(\n    await readFile(new URL('../assets/perfect-book.bin', import.meta.url)),\n  );\n\n  assert.throws(() => decodePerfectBook(bytes.slice(0, bytes.length - 1)), /length mismatch/);\n\n  const invalidOutcome = bytes.slice();\n  invalidOutcome[invalidOutcome.length - 1] = 9;\n  assert.throws(() => decodePerfectBook(invalidOutcome), /outcomes must be/);\n});\n")
write_text('docs/PERFECT_PLAY.md', '# Perfect-play programme\n\n## Current guarantee\n\nThe standard 7×6 Connect Four engine has two exact components:\n\n1. A generated opening book. Every stored entry contains a game-theoretically exact outcome and the complete mask of strong-optimal moves for that position.\n2. The bitboard solver\'s exact terminal search. Once it searches every remaining ply, its result is proved rather than evaluated heuristically.\n\nThe runtime consults the opening book before allocating a transposition table. A book miss falls back to the existing bitboard search. This means the game is already perfect for every stored opening and every position that reaches exact terminal search, but the gap between those regions is not yet globally solved.\n\nThe committed bootstrap entry proves the empty-board centre opening. The generation workflow expands that seed to every canonical, non-terminal position through a selected ply.\n\n## Book format\n\n`assets/perfect-book.bin` is deterministic and deliberately simple:\n\n| Bytes | Meaning |\n| --- | --- |\n| 0–3 | ASCII magic `C4PB` |\n| 4 | format version |\n| 5 | maximum stored ply |\n| 6 | entry size, currently 10 |\n| 7 | reserved |\n| 8–11 | little-endian entry count |\n| each entry | 64-bit canonical key, 7-bit optimal-move mask, signed outcome |\n\nThe outcome is `1` for a forced win, `0` for a draw, and `-1` for a forced loss from the side-to-move\'s perspective. The move mask is based on strong scores, so among equally winning or losing outcomes it preserves the fastest win or longest resistance.\n\nKeys are sorted and horizontally canonicalised. The browser validates the header, length, ordering, move masks, and outcomes before enabling lookup. A malformed book is ignored by the worker and cannot override the search engine.\n\n## Generation\n\nThe persistent **Generate perfect-play book** GitHub Actions workflow:\n\n1. Enumerates each canonical legal position through the requested ply.\n2. Builds a pinned exact Connect Four oracle.\n3. Scores every legal move exactly.\n4. Packs the results into the binary runtime format.\n5. Runs the repository\'s independent tests.\n6. Uploads the book as an artifact and optionally commits it.\n\nThe oracle is Pascal Pons\' solver pinned to commit\n`d6ba50d8aaf2308c769d9bf2abd42d90f34baf41`. Its AGPL source and opening-book file are used only as build-time tools and are not redistributed in this repository. The generated manifest records the oracle commit and downloaded book checksum.\n\nThe generator itself is repository-owned code:\n\n```bash\nnode scripts/perfect-book.mjs enumerate --depth 8 > positions.txt\nnode scripts/perfect-book.mjs pack \\\n  --input scored.txt \\\n  --output assets/perfect-book.bin \\\n  --manifest data/perfect-book.manifest.json \\\n  --max-ply 8 \\\n  --source "exact oracle provenance"\n```\n\n## Verification\n\nCI verifies:\n\n- binary structure and strictly ordered keys;\n- the solved empty-board centre move;\n- book-first runtime routing with zero search nodes;\n- legal and exact immediate tactics;\n- exact late-game results against an independent array minimax sample;\n- worker loading and message handling;\n- all existing classic and Chaos rules.\n\nBook generation additionally refuses conflicting scores for the same canonical key and rejects missing or illegal moves.\n\n## Route to global perfect play\n\nThe next milestones are:\n\n1. Grow the committed exact book from the bootstrap frontier to ply 8.\n2. Add shard-and-merge support so deeper frontiers can be generated in parallel.\n3. Replace the build-time oracle with a second independently implemented native solver, then require both solvers to agree before publishing entries.\n4. Expand the strategy frontier until every book continuation reaches the browser\'s exact-search region.\n5. Mark Brutal as **Perfect** only after a full adversarial traversal proves that every reachable AI decision is covered by either the book or terminal solving.\n\nThe final proof should be machine-checkable: traverse every opponent reply from both starting-player choices, require every AI move to be book-backed or exactly solved, and fail on the first uncovered position.\n')
write_text('.github/workflows/generate-perfect-book.yml', 'name: Generate perfect-play book\n\non:\n  workflow_dispatch:\n    inputs:\n      depth:\n        description: Exact opening-book ply to generate (0-10)\n        required: true\n        default: \'8\'\n        type: string\n      commit_book:\n        description: Commit the generated book to main\n        required: true\n        default: false\n        type: boolean\n\npermissions:\n  actions: write\n  contents: write\n\nconcurrency:\n  group: perfect-play-book\n  cancel-in-progress: false\n\njobs:\n  generate:\n    runs-on: ubuntu-latest\n    timeout-minutes: 90\n\n    steps:\n      - name: Check out repository\n        uses: actions/checkout@v6\n        with:\n          fetch-depth: 2\n\n      - name: Set up Node.js\n        uses: actions/setup-node@v6\n        with:\n          node-version: 24\n          cache: npm\n\n      - name: Select generation depth\n        shell: bash\n        run: |\n          set -euo pipefail\n          depth="${{ inputs.depth }}"\n          if ! [[ "$depth" =~ ^([0-9]|10)$ ]]; then\n            echo "Depth must be an integer from 0 through 10." >&2\n            exit 1\n          fi\n          echo "BOOK_DEPTH=$depth" >> "$GITHUB_ENV"\n\n      - name: Install project\n        run: npm ci\n\n      - name: Build pinned exact oracle\n        shell: bash\n        env:\n          ORACLE_COMMIT: d6ba50d8aaf2308c769d9bf2abd42d90f34baf41\n        run: |\n          set -euo pipefail\n          git clone --quiet https://github.com/PascalPons/connect4.git /tmp/connect4-oracle\n          git -C /tmp/connect4-oracle checkout --quiet "$ORACLE_COMMIT"\n          make -C /tmp/connect4-oracle -j2 c4solver\n          curl --fail --location --retry 3 \\\n            https://github.com/PascalPons/connect4/releases/download/book/7x6_small.book \\\n            --output /tmp/connect4-oracle/7x6_small.book\n          echo "ORACLE_BOOK_SHA=$(sha256sum /tmp/connect4-oracle/7x6_small.book | cut -d\' \' -f1)" \\\n            >> "$GITHUB_ENV"\n\n      - name: Enumerate canonical openings\n        shell: bash\n        run: |\n          set -euo pipefail\n          node scripts/perfect-book.mjs enumerate --depth "$BOOK_DEPTH" > /tmp/perfect-positions.txt\n          echo "Generated $(wc -l < /tmp/perfect-positions.txt) canonical positions."\n\n      - name: Score every opening exactly\n        shell: bash\n        run: |\n          set -euo pipefail\n          /tmp/connect4-oracle/c4solver \\\n            -a \\\n            -b /tmp/connect4-oracle/7x6_small.book \\\n            < /tmp/perfect-positions.txt \\\n            > /tmp/perfect-scores.txt\n          test "$(wc -l < /tmp/perfect-scores.txt)" -eq "$(wc -l < /tmp/perfect-positions.txt)"\n\n      - name: Pack deterministic runtime book\n        shell: bash\n        env:\n          ORACLE_COMMIT: d6ba50d8aaf2308c769d9bf2abd42d90f34baf41\n        run: |\n          set -euo pipefail\n          mkdir -p assets data\n          node scripts/perfect-book.mjs pack \\\n            --input /tmp/perfect-scores.txt \\\n            --output assets/perfect-book.bin \\\n            --manifest data/perfect-book.manifest.json \\\n            --max-ply "$BOOK_DEPTH" \\\n            --source "PascalPons/connect4@$ORACLE_COMMIT; 7x6_small.book sha256=$ORACLE_BOOK_SHA"\n\n      - name: Validate project and committed-book contract\n        run: |\n          npm run ci\n          git diff --check\n\n      - name: Upload generated book\n        uses: actions/upload-artifact@v4\n        with:\n          name: perfect-play-book-ply-${{ env.BOOK_DEPTH }}\n          path: |\n            assets/perfect-book.bin\n            data/perfect-book.manifest.json\n          if-no-files-found: error\n          retention-days: 30\n\n      - name: Commit generated book\n        if: inputs.commit_book\n        shell: bash\n        run: |\n          set -euo pipefail\n          git config user.name "github-actions[bot]"\n          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"\n          git add assets/perfect-book.bin data/perfect-book.manifest.json\n          if git diff --cached --quiet; then\n            echo "The committed book is already current."\n            exit 0\n          fi\n          git commit -m "Grow exact opening book to ply $BOOK_DEPTH"\n          git push origin HEAD:main\n          GH_TOKEN="${{ github.token }}" gh workflow run ci.yml --ref main\n          GH_TOKEN="${{ github.token }}" gh workflow run pages.yml --ref main\n')

Path('assets').mkdir(exist_ok=True)
Path('assets/perfect-book.bin').write_bytes(base64.b64decode('QzRQQgEACgABAAAAAAAAAAAAAAAIAQ=='))
write_text('data/perfect-book.manifest.json', '{\n  "format": 1,\n  "maxPly": 0,\n  "entryCount": 1,\n  "source": "verified root seed: centre opening is a first-player win",\n  "scoreSemantics": "game-theoretic outcome; move mask contains strong-optimal moves",\n  "generatedAt": null,\n  "sha256": "278e77c177cc387292720c0490ef2213fa52e79221bb101f78438c740c7b6556",\n  "byteLength": 22,\n  "optimalMoveEntries": 0\n}\n')

# Route exact book entries through the standard bitboard engine without loading
# the book on the main UI thread.
bitboard_path = Path('src/bitboard.js')
bitboard = bitboard_path.read_text()

canonical_anchor = '''function canonicalPosition(position) {
  const normal = position.current + position.mask;
  const mirrored = mirrorBits(position.current) + mirrorBits(position.mask);
  return normal <= mirrored
    ? { key: normal, mirrored: false }
    : { key: mirrored, mirrored: true };
}

export function boardToBitboard(board, currentPlayer) {
'''
canonical_replacement = '''function canonicalPosition(position) {
  const normal = position.current + position.mask;
  const mirrored = mirrorBits(position.current) + mirrorBits(position.mask);
  return normal <= mirrored
    ? { key: normal, mirrored: false }
    : { key: mirrored, mirrored: true };
}

function mirrorMoveMask(mask) {
  let mirrored = 0;
  for (let column = 0; column < WIDTH; column += 1) {
    if ((mask & (1 << column)) !== 0) mirrored |= 1 << (WIDTH - 1 - column);
  }
  return mirrored;
}

function columnFromMoveMask(position, moveMask) {
  const possible = possibleMoves(position.mask);
  for (const column of COLUMN_ORDER) {
    const move = moveForColumn(position.mask, column);
    if ((moveMask & (1 << column)) !== 0 && (possible & move) !== 0n) return column;
  }
  return -1;
}

export function boardToBitboard(board, currentPlayer) {
'''
bitboard = replace_once(
    bitboard,
    canonical_anchor,
    canonical_replacement,
    'bitboard book helpers',
)

search_anchor = '''  const targetDepth = Math.max(1, Math.min(remaining, requestedDepth));
  const search = new BitboardSearch(options.tableBits ?? config.tableBits);
  const start = now();

  let completedDepth = 0;
'''
search_replacement = '''  const targetDepth = Math.max(1, Math.min(remaining, requestedDepth));
  const start = now();

  const perfectBook = options.perfectBook;
  if (options.useBook !== false
      && options.maximumDepth === undefined
      && perfectBook
      && typeof perfectBook.lookup === 'function') {
    const canonical = canonicalPosition(bitboard);
    const stored = perfectBook.lookup(canonical.key);
    if (stored) {
      const moveMask = canonical.mirrored
        ? mirrorMoveMask(stored.moveMask)
        : stored.moveMask;
      const column = columnFromMoveMask(bitboard, moveMask);
      if (column >= 0) {
        const action = { type: ACTION_DROP, column };
        const relativeOutcome = position.currentPlayer
          === (options.aiPlayer ?? position.currentPlayer)
          ? stored.outcome
          : -stored.outcome;
        const result = {
          action,
          score: relativeOutcome === 0
            ? 0
            : relativeOutcome * (MATE_SCORE - bitboard.moves),
          depth: 0,
          nodes: 0,
          elapsedMs: now() - start,
          tableHits: 0,
          cutoffs: 0,
          tableResets: 0,
          tableStores: 0,
          tableCollisions: 0,
          principalVariation: [{ ...action }],
          solved: true,
          solver: 'perfect-book',
          bookPly: bitboard.moves,
          bookMaxPly: perfectBook.maxPly ?? null,
          bookEntryCount: perfectBook.entryCount ?? null,
        };
        if (typeof options.onIteration === 'function') {
          try {
            options.onIteration(result);
          } catch {
            // Telemetry must never affect an exact move.
          }
        }
        return result;
      }
    }
  }

  const search = new BitboardSearch(options.tableBits ?? config.tableBits);
  let completedDepth = 0;
'''
bitboard = replace_once(
    bitboard,
    search_anchor,
    search_replacement,
    'bitboard exact-book routing',
)
bitboard_path.write_text(bitboard)

# The worker owns the potentially large book. Custom boards and Chaos Mode never
# download it, and the synchronous emergency fallback remains lightweight.
write_text('src/ai-worker.js', '''import { chooseMove } from './ai.js';

let perfectBookPromise = null;

function canUsePerfectBook(position) {
  return Boolean(
    position
      && !position.chaosMode
      && position.connect === 4
      && Array.isArray(position.board)
      && position.board.length === 6
      && position.board.every((row) => Array.isArray(row) && row.length === 7),
  );
}

async function perfectBookFor(position) {
  if (!canUsePerfectBook(position)) return null;
  if (!perfectBookPromise) {
    perfectBookPromise = import('./perfect-book.js')
      .then((module) => module.loadPerfectBook())
      .catch(() => null);
  }
  return perfectBookPromise;
}

self.addEventListener('message', async (event) => {
  const { requestId, position, options } = event.data ?? {};
  try {
    const perfectBook = await perfectBookFor(position);
    const result = chooseMove(position, {
      ...options,
      perfectBook,
      onIteration(progress) {
        self.postMessage({ requestId, kind: 'progress', progress });
      },
    });
    self.postMessage({ requestId, kind: 'result', result });
  } catch (error) {
    self.postMessage({
      requestId,
      kind: 'error',
      error: error instanceof Error ? error.message : String(error),
    });
  }
});
''')

app_path = Path('src/app.js')
app = app_path.read_text()

app = replace_once(
    app,
    '''  medium: 'Standard 7×6 uses a fast 10-ply bitboard search.',
  hard: 'Standard 7×6 uses 14 plies and exact late-game solving.',
  brutal: 'Standard 7×6 uses 16 plies and solves up to 24 empty cells exactly.',
''',
    '''  medium: 'Checks the exact opening book, then searches 10 plies.',
  hard: 'Uses exact book moves, 14-ply search, and late-game solving.',
  brutal: 'Uses exact book moves, 16-ply search, and the largest exact endgame.',
''',
    'difficulty hints',
)

animation_anchor = '''const ANIMATION_CLASSES = [
  'anim-flip-out',
  'anim-flip-in',
  'anim-cw-out',
  'anim-cw-in',
  'anim-ccw-out',
  'anim-ccw-in',
];

const elements = {
'''
animation_replacement = '''const ANIMATION_CLASSES = [
  'anim-flip-out',
  'anim-flip-in',
  'anim-cw-out',
  'anim-cw-in',
  'anim-ccw-out',
  'anim-ccw-in',
];
const TRANSFORM_ANIMATIONS = Object.freeze({
  [ACTION_FLIP]: Object.freeze({
    outClass: 'anim-flip-out',
    inClass: 'anim-flip-in',
    outMs: 320,
    inMs: 420,
  }),
  [ACTION_ROTATE_CW]: Object.freeze({
    outClass: 'anim-cw-out',
    inClass: 'anim-cw-in',
    outMs: 280,
    inMs: 360,
  }),
  [ACTION_ROTATE_CCW]: Object.freeze({
    outClass: 'anim-ccw-out',
    inClass: 'anim-ccw-in',
    outMs: 280,
    inMs: 360,
  }),
});

const elements = {
'''
app = replace_once(app, animation_anchor, animation_replacement, 'transform timings')

status_anchor = '''  const turn = `${playerName(state.currentPlayer)} to move`;
  return state.lastAction ? `${state.lastAction} · ${turn}` : turn;
}
'''
status_replacement = '''  return `${playerName(state.currentPlayer)} to move`;
}
'''
app = replace_once(app, status_anchor, status_replacement, 'stable status headline')

thinking_anchor = '''  elements.statusText.textContent = statusMessage();
  elements.thinkingIndicator.hidden = !state.aiThinking;
  if (state.aiThinking && state.liveSearch) {
    elements.thinkingProgress.textContent = `Depth ${state.liveSearch.depth} · ${numberFormatter.format(state.liveSearch.nodes)} positions`;
  } else {
    elements.thinkingProgress.textContent = '';
  }
'''
thinking_replacement = '''  elements.statusText.textContent = statusMessage();
  elements.thinkingIndicator.hidden = false;
  elements.thinkingIndicator.classList.toggle('is-idle', !state.aiThinking);
  elements.thinkingIndicator.setAttribute('aria-hidden', String(!state.aiThinking));
  if (state.aiThinking && state.liveSearch) {
    elements.thinkingProgress.textContent = state.liveSearch.solver === 'perfect-book'
      ? 'Exact opening-book move'
      : `Depth ${state.liveSearch.depth} · ${numberFormatter.format(state.liveSearch.nodes)} positions`;
  } else {
    elements.thinkingProgress.textContent = '';
  }
'''
app = replace_once(app, thinking_anchor, thinking_replacement, 'stable thinking slot')

telemetry_anchor = '''  } else if (search) {
    const seconds = search.elapsedMs / 1_000;
    const rate = search.elapsedMs > 0 ? Math.round(search.nodes / seconds) : 0;
    const details = [];
    if (search.solver === 'bitboard') details.push(search.solved ? 'Bitboard solved' : 'Bitboard');
    else if (search.solved) details.push('Solved');
    details.push(
      `Depth ${search.depth}`,
      `${numberFormatter.format(search.nodes)} positions`,
      `${seconds.toFixed(seconds >= 1 ? 1 : 2)}s`,
    );
    if (rate > 0) details.push(`${numberFormatter.format(rate)}/s`);
    elements.searchInfo.textContent = details.join(' · ');
'''
telemetry_replacement = '''  } else if (search) {
    const details = [];
    if (search.solver === 'perfect-book') {
      details.push('Perfect book', 'Exact move');
      if (search.bookEntryCount) {
        details.push(`${numberFormatter.format(search.bookEntryCount)} solved openings`);
      }
    } else {
      const seconds = search.elapsedMs / 1_000;
      const rate = search.elapsedMs > 0 ? Math.round(search.nodes / seconds) : 0;
      if (search.solver === 'bitboard') details.push(search.solved ? 'Bitboard solved' : 'Bitboard');
      else if (search.solved) details.push('Solved');
      details.push(
        `Depth ${search.depth}`,
        `${numberFormatter.format(search.nodes)} positions`,
        `${seconds.toFixed(seconds >= 1 ? 1 : 2)}s`,
      );
      if (rate > 0) details.push(`${numberFormatter.format(rate)}/s`);
    }
    elements.searchInfo.textContent = details.join(' · ');
'''
app = replace_once(app, telemetry_anchor, telemetry_replacement, 'book telemetry')

animation_function_anchor = '''function animationNames(action) {
  if (action.type === ACTION_FLIP) return ['anim-flip-out', 'anim-flip-in'];
  if (action.type === ACTION_ROTATE_CW) return ['anim-cw-out', 'anim-cw-in'];
  if (action.type === ACTION_ROTATE_CCW) return ['anim-ccw-out', 'anim-ccw-in'];
  return [null, null];
}
'''
animation_function_replacement = '''function animationPlan(action) {
  return TRANSFORM_ANIMATIONS[action.type] ?? null;
}
'''
app = replace_once(
    app,
    animation_function_anchor,
    animation_function_replacement,
    'animation plan',
)

perform_out_anchor = '''  const [outAnimation, inAnimation] = animationNames(action);
  if (outAnimation) {
    clearBoardAnimations();
    elements.boardFrame.classList.add(outAnimation);
    await pause(180);
    if (roundVersion !== state.version) return;
  }
'''
perform_out_replacement = '''  const animation = animationPlan(action);
  if (animation) {
    clearBoardAnimations();
    elements.boardFrame.classList.add(animation.outClass);
    await pause(animation.outMs);
    if (roundVersion !== state.version) return;
  }
'''
app = replace_once(app, perform_out_anchor, perform_out_replacement, 'transform out phase')

perform_in_anchor = '''  if (outAnimation) {
    elements.boardFrame.classList.remove(outAnimation);
    elements.boardFrame.classList.add(inAnimation);
    await pause(220);
    elements.boardFrame.classList.remove(inAnimation);
  } else {
'''
perform_in_replacement = '''  if (animation) {
    elements.boardFrame.classList.remove(animation.outClass);
    elements.boardFrame.classList.add(animation.inClass);
    await pause(animation.inMs);
    elements.boardFrame.classList.remove(animation.inClass);
  } else {
'''
app = replace_once(app, perform_in_anchor, perform_in_replacement, 'transform in phase')

search_state_anchor = '''        solved: Boolean(result.solved),
        solver: result.solver ?? 'general',
        principalVariation: Array.isArray(result.principalVariation)
'''
search_state_replacement = '''        solved: Boolean(result.solved),
        solver: result.solver ?? 'general',
        bookPly: result.bookPly ?? null,
        bookMaxPly: result.bookMaxPly ?? null,
        bookEntryCount: result.bookEntryCount ?? null,
        principalVariation: Array.isArray(result.principalVariation)
'''
app = replace_once(app, search_state_anchor, search_state_replacement, 'book search state')
app_path.write_text(app)

index_path = Path('index.html')
index = index_path.read_text()
index = replace_once(
    index,
    '<div id="thinkingIndicator" class="thinking-indicator" role="status" hidden>',
    '<div id="thinkingIndicator" class="thinking-indicator is-idle" role="status" aria-hidden="true">',
    'thinking indicator markup',
)
index_path.write_text(index)

styles_path = Path('styles.css')
styles = styles_path.read_text()
marker = '/* Stable status geometry and deliberate transform timing */'
if marker not in styles:
    styles += '''

/* Stable status geometry and deliberate transform timing */
.turn-status {
  min-height: 3.6rem;
}

.turn-status > div {
  min-width: 0;
}

.move-info {
  min-height: 1em;
  font-variant-numeric: tabular-nums;
}

#statusText {
  display: flex;
  align-items: center;
  min-height: 2.4em;
  line-height: 1.2;
}

.thinking-indicator {
  height: 2rem;
  min-height: 2rem;
  overflow: hidden;
  white-space: nowrap;
}

.thinking-indicator.is-idle {
  visibility: hidden;
}

.thinking-progress {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}

.board-frame.anim-flip-out {
  animation-duration: 320ms;
}

.board-frame.anim-flip-in {
  animation-duration: 420ms;
}

.board-frame.anim-cw-out,
.board-frame.anim-ccw-out {
  animation-duration: 280ms;
}

.board-frame.anim-cw-in,
.board-frame.anim-ccw-in {
  animation-duration: 360ms;
}
'''
styles_path.write_text(styles)

package_path = Path('package.json')
package = json.loads(package_path.read_text())
package['version'] = '1.4.0'
package['scripts']['check'] = (
    'node --check src/engine.js && '
    'node --check src/perfect-book.js && '
    'node --check src/bitboard.js && '
    'node --check src/ai.js && '
    'node --check src/ai-worker.js && '
    'node --check src/app.js && '
    'node --check scripts/perfect-book.mjs && '
    'node --check scripts/serve.mjs'
)
package['scripts']['book:enumerate'] = 'node scripts/perfect-book.mjs enumerate --depth 8'
package_path.write_text(json.dumps(package, indent=2) + '\n')

lock_path = Path('package-lock.json')
lock = json.loads(lock_path.read_text())
lock['version'] = '1.4.0'
lock['packages']['']['version'] = '1.4.0'
lock_path.write_text(json.dumps(lock, indent=2) + '\n')

readme_path = Path('README.md')
readme = readme_path.read_text()
readme = replace_once(
    readme,
    '- Seven-bit column bitboards with make/unmake-free move generation on the standard board.\n',
    '- A generated exact opening book is checked before any standard-board search.\n'
    '- Seven-bit column bitboards with make/unmake-free move generation on the standard board.\n',
    'README perfect-book bullet',
)
difficulty_paragraph = (
    'On a standard 7×6 board, Medium completes 10 plies, Hard 14, and Brutal 16. '
    'Medium solves positions with at most 16 empty cells exactly, Hard 20, and Brutal 24. '
    'Custom boards retain the general 6/9/12-ply engine. There is no wall-clock cutoff: '
    'a worker finishes the selected depth or exact endgame unless the player explicitly '
    'cancels it by restarting, undoing, or changing the game.\n'
)
perfect_section = difficulty_paragraph + '''

### Perfect-play book

Standard play now checks `assets/perfect-book.bin` before allocating a search table. Every stored entry has an exact game-theoretic outcome and a mask of strong-optimal moves. The book is loaded only inside the AI worker, so the main interface and Chaos games do not pay its download or memory cost.

The manual **Generate perfect-play book** workflow can reproducibly grow the committed frontier. It enumerates canonical openings, scores every legal move with a pinned exact oracle, packs a deterministic binary book, reruns all tests, and records provenance in `data/perfect-book.manifest.json`.

The global perfect-play proof is deliberately tracked separately from ordinary AI strength. See [`docs/PERFECT_PLAY.md`](docs/PERFECT_PLAY.md) for the current guarantee, binary format, verification rules, and remaining coverage work.
'''
readme = replace_once(
    readme,
    difficulty_paragraph,
    perfect_section,
    'README perfect-play section',
)
readme = replace_once(
    readme,
    '├── styles.css              Responsive visual design and animations\n',
    '├── styles.css              Responsive visual design and animations\n'
    '├── assets/perfect-book.bin Exact standard-board opening entries\n'
    '├── data/                   Perfect-book provenance and generation metadata\n',
    'README asset tree',
)
readme = replace_once(
    readme,
    '│   ├── bitboard.js         Standard 7×6 bitboard search and exact endgame solver\n',
    '│   ├── bitboard.js         Standard 7×6 bitboard search and exact endgame solver\n'
    '│   ├── perfect-book.js     Validated binary-book loader and lookup\n',
    'README source tree',
)
readme = replace_once(
    readme,
    '│   ├── ai.test.js          Tactical, fixed-depth, and mutation-safety tests\n'
    '│   └── worker.test.js      Browser-worker protocol and progress test\n'
    '└── scripts/serve.mjs       Dependency-free local static server\n',
    '│   ├── ai.test.js          Tactical, fixed-depth, and mutation-safety tests\n'
    '│   ├── perfect-book.test.js Binary validation and book-routing tests\n'
    '│   └── worker.test.js      Browser-worker protocol and progress test\n'
    '└── scripts/\n'
    '    ├── perfect-book.mjs    Canonical enumeration and deterministic packing\n'
    '    └── serve.mjs           Dependency-free local static server\n',
    'README test tree',
)
readme_path.write_text(readme)

Path('.github/workflows/apply-perfect-play-foundation.yml').unlink()
Path('.automation/start-perfect-play.py').unlink()

# Perfect Chaos prefix certificate

This directory contains the verified 6×7 Chaos Mode non-losing prefix certificate through sixteen placed pieces.

- `manifest.json` records the theorem, exact solver summaries, independent replay summaries, rejection counts, file sizes and SHA-256 hashes.
- `red/` contains the strategy when the AI moves first.
- `yellow/` contains the strategy when the AI moves second.
- `*.policy.bin` files contain one deterministic action for every reachable AI state in that segment.
- `*.frontier.bin` files contain the exact sorted canonical boundary passed to the next segment.
- `reject-*.bin` files contain later-proved losing roots that were fed back into the preceding segment.

The fixed-size little-endian formats are validated by `scripts/perfect-chaos-prefix.mjs`:

- policy header magic: `C4CPOL1\0`; 20-byte records;
- frontier header magic: `C4CFRN1\0`; 19-byte records.

Run `npm run chaos:prefix:verify-reference` to check every digest and replay every adversarial continuation in the committed closure.

## Seed rejections for the next layer

`red/reject-16.bin` is not part of the verified sixteen-piece manifest and is not referenced by it. It holds 16,242 sixteen-piece frontier states that the exact solver's first sixteen-to-eighteen pass proved losing, in the ordinary frontier format, so a later `generate --frontier-pieces 18 --seed-rejections data/perfect-chaos-prefix` run starts from them instead of rediscovering them.

Every one of the 16,242 states was checked to carry exactly sixteen pieces in strictly sorted canonical order, and an independently written bounded lower/upper proof engine confirmed a sample of 204 as losses with no disagreement, no draw and no unresolved case.

The set is sound but deliberately incomplete: later refinement passes discover further losing roots. Seeding can only ever make synthesis more conservative, because a rejection removes a boundary state rather than asserting one is safe, and the independent full-closure replay remains the acceptance gate.

This is not a full empty-board solution. The sixteen-piece output frontier remains to be connected to later certified layers and ultimately to the exact endgame solver.

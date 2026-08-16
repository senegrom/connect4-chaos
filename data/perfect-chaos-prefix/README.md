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

## Rejection seeds for later targets

Extending the certificate past sixteen pieces grows the rejection sets of the
earlier layers, and `reject-8.bin` through `reject-14.bin` here are hashed
artifacts of this manifest. Those larger sets are therefore kept in
`../perfect-chaos-prefix-seeds-18/` so this directory stays exactly as verified.

This is not a full empty-board solution. The sixteen-piece output frontier remains to be connected to later certified layers and ultimately to the exact endgame solver.

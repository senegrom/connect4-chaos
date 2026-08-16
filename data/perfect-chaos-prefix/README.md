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

`red/reject-16.bin` is not part of the verified sixteen-piece manifest and is not referenced by it. It accumulates sixteen-piece frontier states that exact sixteen-to-eighteen passes proved losing, in the ordinary frontier format, so a later `generate --frontier-pieces 18 --seed-rejections data/perfect-chaos-prefix` run starts from them instead of rediscovering them. Two sweeps have contributed so far: 16,242 states from the first and 9,183 from the second, for 25,425 in total.

Rejecting a boundary state does not shrink the next frontier. It forces the preceding layer to route around that state, which yields a different reachable boundary set of a similar size, whose own losing states the following sweep then proves. That is why the count accumulates rather than converging in one pass.

Every record is checked to carry exactly sixteen pieces in strictly sorted canonical order with no overlapping pieces, and no state is ever removed once added. An independently written bounded lower/upper proof engine has classified 404 sampled states across both sweeps as losses, with no disagreement, no draw and no unresolved or state-limited case.

The set is sound but deliberately incomplete: later refinement passes discover further losing roots. Seeding can only ever make synthesis more conservative, because a rejection removes a boundary state rather than asserting one is safe, and the independent full-closure replay remains the acceptance gate.

This is not a full empty-board solution. The sixteen-piece output frontier remains to be connected to later certified layers and ultimately to the exact endgame solver.

# Rejection seeds for the eighteen-piece target

This directory holds accumulated losing-root sets for an in-progress
`--frontier-pieces 18` synthesis. It is **not** part of any verified certificate
and no manifest references it.

```bash
node scripts/perfect-chaos-prefix.mjs generate \
  --frontier-pieces 18 \
  --seed-rejections data/perfect-chaos-prefix-seeds-18 \
  --shards 32 --shard-from-pieces 18 \
  --journal generated/perfect-chaos-prefix-journal \
  --output generated/perfect-chaos-prefix-18
```

## Why these are separate from the committed certificate

`reject-8.bin` through `reject-14.bin` are hashed artifacts of the verified
sixteen-piece manifest in `../perfect-chaos-prefix/`. Targeting eighteen pieces
forbids the sixteen-piece boundary states that later proved losing, which makes
the earlier layers strictly more constrained and grows their rejection sets.
Writing those larger sets back into the certificate directory would break its
artifact digests and invalidate a complete, independently replayed proof that
remains correct for its own claim. They therefore live here instead.

Both roles are present so the directory can seed a run on its own. Yellow's
files are the committed sixteen-piece values, unchanged, because the
eighteen-piece cascade has not reached that role yet.

## Current contents

| Boundary | Red | Yellow | Committed sixteen-piece value |
|---|---:|---:|---|
| 8 | 0 | 94 | red 0, yellow 94 |
| 10 | 80 | 941 | red 80, yellow 941 |
| 12 | 1,297 | 7,786 | red 1,266, yellow 7,786 |
| 14 | 9,207 | 44,737 | red 8,020, yellow 44,737 |
| 16 | 39,607 | — | not applicable |

Every set is a strict superset of its committed counterpart; no state is ever
removed once added.

## Soundness

A rejection removes a boundary state rather than asserting one is safe, so
seeding can only make a later synthesis more conservative, and the independent
full-closure replay remains the sole acceptance gate.

Every file is validated as strictly sorted canonical frontier records whose
states carry exactly the boundary's piece count with no overlapping pieces. An
independently written bounded lower/upper proof engine has additionally
classified samples as losses with no disagreement: 604 of the sixteen-piece
states across three sweeps, plus 72 of 198 sampled new fourteen-piece states and
4 of 31 new twelve-piece states. The remainder of the deeper samples are
reported unresolved rather than contradictory, because a forced loss inherited
through the sixteen-piece boundary lies beyond a two-drop horizon. No sampled
state at any boundary was ever classified an AI win.

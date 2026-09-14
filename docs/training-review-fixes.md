# Training review fixes (September 2026)

## Replay calibration

`learn()` and remote `soup()` share `neural.replay_staging.stage_replay`.
The replay window counts eligible positions after stable validation and
whole-board holdout exclusions, not raw archive rows. Corrupt and excluded
archives are skipped while older eligible archives can fill the window.
Original modification times and lexical ties preserve the learner's ordering.
Calibration also enforces the newest-tail position cap across shard directories.

Remote soup reads the holdout partition from its source checkpoints, uses a
unique temporary staging directory with cleanup on exceptions, and requires
eligible replay both before launching calibration and inside the calibration
process. It cannot silently recalibrate on exact tables alone. Local
`calibration_data(..., require_replay=False)` still supports deliberate
exact-only calibration. `DISTILL_REPLAY_WINDOW` bounds eligible calibration
replay as well as learner replay.

## Arena ancestry and migration

A successful learner writes `models/<checkpoint>.lineage.json` alongside the
checkpoint, before committing the Volume. It records its parent checkpoint and
generation. A checkpoint retained after a failed post-training evaluation is
still recoverable but does not receive successful lineage. Existing records
cannot be overwritten with a different parent.

The driver restores only the initial checkpoint's explicit ancestry. Duplicate
generation files, experiments, partial writes and unrelated branches cannot
change the arena opponent. `ARENA_LAG` counts successful parent links, not files
or numeric filename differences. There is no special generation-900 cutoff.

**Migration:** redeploy `neural/modal_app.py` and restart the driver from the
chosen checkpoint. Checkpoints created before lineage metadata existed are
roots; their ancestry is intentionally not guessed. The arena waits for enough
new successful ancestors to satisfy its lag. A lineage read failure also starts
a new in-memory history at the explicitly selected initial checkpoint and is
logged; training can continue without comparing unrelated models.

## Shape coverage and driver regression tests

`all_shapes()` covers Connect 3, 4 and 5 on every board the browser offers, plus
the narrower training boards, in both rule sets. Connect-6 was removed from the
play options on 2026-09-14 (the settings clamp at 5), so the enumerator omits it
too rather than spending self-play on boards nobody can play. Its `all` default
is shared by self-play and arena. The Node regression enumerates the actual
browser `normalizeConfig()` outputs, checks every supported neural-opponent
configuration against the Python enumerator, and asserts neither side includes
Connect-6.

Driver tests exercise mirroring on, explicitly off, and unset (the default),
including fresh-position pacing, transient polling errors, exact arena
opponents, generation advancement and clean stop-file shutdown. Mirror-off
runs assert neither mirror hook nor mirror storage is used.

Run focused checks:

```sh
python -m neural.test_checkpoint_lineage
python -m neural.test_modal_loop
python -m neural.test_soup_replay
python -m neural.test_replay_staging
node --test tests/neural-training-shapes.test.js
```

The new Python regressions run in the CPU training CI job; the shape coverage
regression runs in the normal Node test suite. These checks do not require paid
Modal jobs or a CUDA device. They do not measure playing strength.

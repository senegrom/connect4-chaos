# Modal command status, replay options and shutdown

## Completion is different from submission

Every synchronous task in `neural/modal_app.py` prints its diagnostics and then
checks the remote subprocess's `exit` status. A nonzero status raises
`SystemExit` with that code and also prints the returned error text to stderr,
even when stdout already contains progress. This applies to solver, sidecar,
dataset, self-play, learner, arena, measurement, soup, GPU-test and closure tasks.

A learner can retain a completed checkpoint after a later evaluation failure.
The checkpoint name remains in the printed result; retaining it does not turn
the failed command into a success. Remote exceptions still propagate.

The supported `--spawn` paths (solve, prepare and dataset) only acknowledge
submission and print a call ID. They do not claim that the remote work finished.

## Task-specific replay defaults

An omitted `--replay-window` uses 4,000,000 eligible positions for `learn` and
400,000 for `soup`, preserving the existing remote defaults. Soup now forwards
both `--replay-window` and `--replay-subdir` rather than silently ignoring them.
An explicit zero is allowed for exact-only learning, but remote soup requires a
positive window. Invalid windows are rejected before remote submission.

For example, from an environment with Modal configured:

```sh
modal run neural/modal_app.py --task soup \
  --models a.pt,b.pt --out-name mix.pt --batches 3 \
  --replay-window 17 --replay-subdir experiment-only
```

The remote call receives a 17-position eligible window from `experiment-only`.
The existing filtering, recency and checkpoint holdout rules still apply.

## Drain-only stop requests

Create `<C4_NEURAL_ROOT>/modal-loop.stop` to request driver shutdown. The driver
checks the file at each learner, actor and arena submission boundary, including
after polling, retry backoff or a mirror operation. Once it observes the stop,
shutdown remains latched for that invocation even if the file is removed.

Already-submitted work is still collected: finished shards are counted,
checkpoints are published/mirrored as configured, and an existing arena is
allowed to finish. A learner finishing on an arena generation does not launch
a new arena after the stop is observed. A submission already in progress when
the file is created cannot be retroactively prevented; its result is tracked
and drained like other in-flight work.

Use the updated local driver/entrypoint after merging. These changes do not
alter the game rules, deployed model, checkpoint contents or replay archives,
and do not start or stop any live Modal jobs by themselves.

## Regression tests

```sh
python -m neural.test_modal_control
python -m neural.test_modal_loop
python -m neural.test_modal_arena
```

The control tests execute the actual entrypoint and driver function bodies with
remote calls, clocks and mirrors mocked. They cover successful and failed task
results, retained checkpoints, spawn-only responses, replay defaults and custom
options, and stop requests during submissions, polls, mirrors and retry delays.
They also check that an existing arena drains and transient polling failures do
not lose tracked work. The tests run in the CPU training CI job; no GPU or paid
Modal work is needed.

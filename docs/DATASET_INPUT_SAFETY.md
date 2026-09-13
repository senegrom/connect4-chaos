# Exact-table identity and eligible replay staging

## Reading pair tables

`neural.pair_tables.PairTable` checks the C4PAIR2 magic, requested geometry
(including orientation), Connect length, Classic/Chaos kind, layer, pair index,
and expected payload length. It also verifies every rank sidecar prefix against
the bitset, rejects bits outside the geometry's slots, and requires one resolved
WDL byte (0, 1, or 2) per reachable state. The native solver legitimately omits
`.values` for zero-state blocks; a nonempty unresolved block is not a dataset.

The dataset builder preflights **every supplied bits block and its companions**
before sampling or publishing its first shard. A mixed directory therefore fails
even when the first sampled state would have come from a valid block. Standalone
lookups validate their requested blocks as well. Invalid rule-mode names are
rejected rather than interpreted as Chaos. Existing shards are not rewritten.

Preflight performs one sequential validation pass in bounded chunks. At most 32
blocks remain mapped, bounding open descriptors. Reopening an evicted block can
reuse its scan result only when the opened files retain the same device, inode,
size, modification time and change time; headers are always checked on opening.
Use the reader as a context manager or call `close()` to release its maps.
The source directory must remain immutable while a reader is active: do not
truncate or rewrite a live mmap in place. Publish completed tables into a new
directory and create a new reader instead.

These checks establish table identity and structural consistency, not a fresh
proof of every WDL value. They cannot authenticate deliberately forged headers
or detect every same-format change to a valid WDL byte. Keep the existing solver
verification and proof gates. No table files or training datasets are migrated
or relabelled by this fix; regenerate any dataset known to use the wrong rules.

## Staging a learner's replay window

The Modal learner uses `distill.training_holdouts` and the production
`filtered_chunks` predicate, just like `load_shards`. It counts eligible rows,
not physical rows, before deciding whether to stop staging compressed files.
Whole-board exclusions (including rotated Chaos boards), the legacy position
hash partition, current validation flags, newest-tail selection and the remaining
window cap therefore agree between staging and training.

Zero-eligible files are removed and counted as `excluded_shards`; unreadable or
invalid replay files remain separately counted as `skipped_shards`. The wrapper
continues to older files until the eligible window is full or the archive is
exhausted. `replay_positions` reports the eligible count capped to that window,
while `replay_shards` counts retained files. Original mtimes are preserved and
equal mtimes use filename order, matching the trainer. A zero window stages no
replay. Invalid window/holdout configuration fails before remote volume work.

Run `python -m neural.test_pair_tables` and
`python -m neural.test_replay_staging`. The former builds real small native
Classic/Chaos tables; the latter runs the production Modal wrapper with only its
remote execution boundaries replaced, then performs real CPU training and
checkpoint IO. Both are in the required training CI job.

# Native builds and neural evaluation regressions

## Native linking

The GCC-style native build entry points (GCC or Clang) and their tests use
`scripts/native-toolchain.mjs`. Native Windows/MinGW builds keep `-static` so
an unrelated libstdc++ DLL earlier on PATH cannot crash file IO. Darwin and
Linux use ordinary linking; macOS requires its dynamic system runtime.
These wrappers build for their host, not for cross-compilation targets.

The required `native-portability` CI gate compiles and exercises the native
fixtures on macOS, including a real iostream file-writing test. The existing
Linux sanitizer jobs are unchanged. Windows flag selection has unit coverage;
the new Darwin gate is not a substitute for a Windows-host execution test.
The classic-policy replay fingerprint includes the shared linker helper.

## Evaluation budgets

`neural.search_quality` normalizes budgets once, preserving their order. A
budget is evaluated once per shard, including when the requested budget is
16 or 32. Each pooled rate uses its own accumulated position count; a budget
with no positions is reported as `n/a`, not as a measured zero error rate.
Requesting zero simulations scores only the raw policy for either CLI mode.

## Captured search settings

A workspace freezes its exploration constant and Q-head seeding setting.
Both settings, as well as eager/captured mode, are part of cache identity.
Returning to an earlier setting reuses only its matching workspace. Eager
selection and expansion read the same frozen values that graph capture uses.
This prevents evaluation order from changing the meaning of a search setting.

Run `python -m neural.test_search_quality` and
`python -m neural.test_search_settings` for the CPU regressions. On a CUDA
machine the latter also executes successive graph-backed searches and compares
them against fresh eager searches; on CPU-only CI that test explicitly skips.
No performance claim or CUDA validation should be inferred from a CPU skip.

## Portable atomic reads

The Darwin gate also exposed a C++20 libc++ incompatibility in
`atomic_ref<const T>::load`. The three parallel Chaos solvers now share a
load-only helper instantiated as `atomic_ref<T>` over their mutable storage.
Acquire/relaxed ordering and all release writes are unchanged. The native
tests cover both byte and word loads, solver counts, and tiny policy output;
the existing Linux ThreadSanitizer checks remain required.

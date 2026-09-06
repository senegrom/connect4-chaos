# Neural versus Brutal benchmark

Run `node scripts/neural-vs-brutal.mjs [simulations] [games-per-board] [model.onnx]`.
Defaults remain 64 simulations, 12 games per board and the shipped model.
The model is loaded only by the command-line entry point and released on exit.

The comparator shares `choosePreparedMove` with the browser worker, including
Classic opening-book preparation and bounded Chaos proofs. Four randomized
opening plies deliberately do not follow the certified Chaos opening policy;
those matches use the same general Brutal route as a mid-game opponent handoff.
Zero-opening diagnostic matches can use the strict certified route. A missing
record in a genuinely policy-following game remains an error, not an implicit
fallback to a weaker opponent.

Both searches receive complete repetition counts, including the initial
position. Counts use the engine's rule-aware key and the next player to move.
Neural input includes repetition planes on every evaluation. An empty Chaos
board ends in a threefold-repetition draw after four consecutive flips.

Invalid opponent moves fail the benchmark instead of becoming random moves or
draws. The 300-ply safety cap is reported as unresolved and excluded from the
completed-game score; counts of capped games are always shown. Random openings
and finite search budgets make this a diagnostic, not an optimality proof or a
retroactive correction of results produced by earlier benchmark versions.

In the app, a handoff to general Brutal persists through reload and Undo. Restart
begins a new certified-policy-eligible Brutal round. Legacy saved rounds without
policy provenance conservatively use general search.

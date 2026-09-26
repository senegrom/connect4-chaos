# Perfect Chaos: safety versus optimality

The layered 6×7 Chaos certificate and a game-theoretically perfect policy are different proof objects.

## What the current prefix certificate proves

The prefix theorem is:

```text
finite-safety-game-with-quotient-cycles-lifting-to-threefold-draws
```

For a fixed AI role, it proves that the stored policy does not reach an AI loss before the certified frontier. A line may instead reach an AI win, a terminal draw, a repetition cycle that becomes a real threefold draw, or the next explicit frontier.

That is a strong **non-losing safety certificate**. It is not an optimality certificate.

At an AI node the safety solver may select any action outside the losing attractor. Therefore it is permitted to select a drawing cycle even when another legal move forces a win. Extending that policy all the way to the exact endgame would prove “the AI cannot lose,” but would still not prove “the AI always chooses the best game-theoretic result.”

## What a 6×7 Perfect label requires

A Perfect release for standard 6×7 Chaos must optimise outcomes in this order:

```text
win > draw > loss
```

For both starting roles, the release proof must establish all of the following:

1. Every state reachable from the empty board under the stored policy is assigned an exact win/draw/loss value from the AI role’s perspective.
2. At every AI decision, the stored action attains the maximum available value.
3. At every opponent decision, every legal action is included and the state value is the minimum of those continuations.
4. Every prefix frontier is connected to another exact value layer or to an exact ranked-retrograde endgame value.
5. Winning policy edges make finite ranked progress to a terminal win rather than remaining in a favourable-looking cycle.
6. Draw policy edges remain in a closed draw region or end in an immediate draw.
7. Literal threefold repetition is independently verified.
8. At least two independently implemented verifiers agree on the values and selected actions.

The exact fixed-role propagation rules are:

| State owner | Win | Loss | Otherwise |
|---|---|---|---|
| AI | At least one child is Win | Every child is Loss | Draw |
| Opponent | Every child is Win | At least one child is Loss | Draw |

Unresolved closed components are draws. Ranked winning states select finite-progress children; losing ranks may be used only to choose how long an unavoidable loss is delayed, not to change its value.

## Where Perfect Chaos is enabled today

Perfect is enabled in Chaos Mode on the eleven completely solved configurations in `data/perfect-chaos-complete/` (docs/PERFECT_CHAOS.md), on a narrower footing than the list above. Their certificates cover the complete adversarial closure from the empty board, so there is no frontier, and `scripts/perfect-chaos-complete.mjs` replays each one through `src/engine.js`. Against the list:

- Established by the replay: every reachable AI decision has one legal stored action (1); every opponent action is explored (3); there is no frontier to connect (4); a stored win must reach a terminal win through finitely many replies, recomputed without trusting stored ranks (5); a stored draw is never a position the opponent can force into a loss (6).
- Established by the replay together with the role pair: the root value of each role is the exact game value. Each replay proves that its policy forces at least its root value, and the release check requires the two roles of every board to prove opposite values, which only the exact value satisfies.
- Resting on the native solver alone: that every stored action attains the maximum available value (2). The replay checks that each stored value is what the policy forces from that position, not that no better action exists, so the policy's play after an opponent's mistake - where the position is worth more than the root value - is as good as the solver's exact retrograde values, which agree with `src/chaos-solver.js` on the complete 4×4 graphs and on sampled 4×5 positions but are not re-solved by a second implementation (8).
- Treated by the model rather than replayed literally: repetition. A cycle in the mirror-canonical quotient graph counts as a draw, which the threefold rule makes of it; histories are not replayed (7).

## Claim boundary

Certified non-loss is not **Perfect**. A 6×7 Chaos claim of Perfect needs a separate exact W/D/L optimality manifest, bound to the safety manifest by hash, that records complete empty-board coverage, exact frontier handoffs, literal-threefold verification, independent implementation agreement, both root values - each the other's negation, since a Red win is a Yellow loss - and complete adversarial closure. A gate that checked such a manifest's form, `scripts/perfect-chaos-claim-gate.py`, was retired on 2026-09-26 with nothing producing its input; commit 0fedaa3 has it. The completely solved boards above never passed through it.

`scripts/perfect-chaos-wdl.py` is the first exact objective layer. It solves a closed fixed-role graph by minimax W/D/L retrograde propagation, assigns winning ranks, treats unresolved closed cycles as draws, and emits an optimal AI action for every AI node. Its regressions include a position where one action is safely drawing while another wins; the solver must select the win.

## Remaining route for 6×7

1. Finish counterexample-guided non-loss closure for each segment.
2. Export the complete policy-reachable graph with exact frontier value references.
3. Run the fixed-role W/D/L solver over that closed graph.
4. Implement an independent native W/D/L solver and require byte-identical values and optimal policy decisions.
5. Build the exact optimality manifest, and a gate that checks it (the retired claim gate is a start).
6. Only then add a browser policy loader and enable the **Perfect** label for standard 6×7 Chaos.

The cloud campaign that drove step 1 was retired on 2026-08-26; the exact pair-scheduled solver is now the route to larger boards. Until these conditions are met, the UI keeps Perfect disabled for standard 6×7 Chaos even if the non-losing prefix reaches the endgame handoff. The eleven completely solved configurations above are the only Chaos boards where it is enabled.

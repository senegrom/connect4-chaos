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

## What the Perfect label requires

A Perfect Chaos release must optimise outcomes in this order:

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

## Enforced claim boundary

`scripts/perfect-chaos-claim-gate.py` verifies the distinction.

A safety-only check is valid:

```bash
python3 scripts/perfect-chaos-claim-gate.py \
  --claim safety \
  --safety-manifest data/perfect-chaos-prefix/manifest.json
```

The resulting allowed label is **Non-losing certified**.

A Perfect claim fails closed unless a separate exact W/D/L optimality manifest is supplied:

```bash
python3 scripts/perfect-chaos-claim-gate.py \
  --claim perfect \
  --safety-manifest path/to/complete-safety-manifest.json \
  --optimality-manifest path/to/exact-wdl-optimality-manifest.json
```

The optimality manifest is cryptographically bound to the safety manifest and must record complete empty-board coverage, exact frontier handoffs, literal-threefold verification, independent implementation agreement, both root values, complete adversarial closure, and artifact hashes.

`scripts/perfect-chaos-wdl.py` is the first exact objective layer. It solves a closed fixed-role graph by minimax W/D/L retrograde propagation, assigns winning ranks, treats unresolved closed cycles as draws, and emits an optimal AI action for every AI node. Its regressions include a position where one action is safely drawing while another wins; the solver must select the win.

## Remaining route

1. Finish counterexample-guided non-loss closure for each segment.
2. Export the complete policy-reachable graph with exact frontier value references.
3. Run the fixed-role W/D/L solver over that closed graph.
4. Implement an independent native W/D/L solver and require byte-identical values and optimal policy decisions.
5. Build the exact optimality manifest and pass the claim gate.
6. Only then add a browser policy loader and enable the **Perfect** label in Chaos Mode.

Until those conditions are met, the UI must continue to disable Perfect Chaos even if the non-losing safety campaign reaches the endgame handoff.

# Livnium Native Pile

This is a minimal, falsifiable prototype of the architecture:

```text
neural model
  └── trainable Livnium action head
        └── exact reversible observer operations
              └── persistent 27 × 27 Livnium data pile
```

The pile is a registered model buffer, participates directly in every forward
pass, and is saved in the model checkpoint. It is not an external search API.

## What the experiment tests

The pile contains 729 unique payload values. A query supplies starting observers at
the macro and micro levels plus arbitrary instruction tokens. The model is **not**
told what those tokens mean. It selects among six exact Livnium observer shifts
(`Ox±`, `Oy±`, `Oz±`) plus two observer-frame reorientations (`Rx±`) and reads
the payload at the resulting hierarchical address. The cyclic shifts can reach
every cell; reorientation makes operation order matter. Rigid cube rotations
alone cannot reach every cell because they preserve center/face/edge/corner
exposure classes.

The observer shifts are an explicit **memory-interface extension** to Livnium
Core. They move the read address, not the stored symbols, so the pile and its
invariants remain unchanged. They are not being presented as members of the
canonical 24-rotation group.

Training uses only the final payload-answer loss. The decisive tests are:

1. **Unseen pile:** replace every payload after training with values from a
   disjoint range and recompute answers. Success means the action head navigates
   memory instead of memorizing training values.
2. **Longer paths:** train on 1–3 moves and test on 4–6 moves.
3. **Wrong-pile control:** silently replace the pile without changing answers.
   Accuracy must collapse toward chance (1/729).
4. **Instruction shuffle:** use a guaranteed token derangement to break the
   learned action language.
5. **Order reversal:** reverse paths containing non-commuting operations.
6. **No-flow control:** read the starting address without moving.
7. **Target intervention:** swap the addressed payload with one non-target
   payload; the answer must follow the changed target content.
8. **Non-target intervention:** alter other cells; the answer must remain fixed.
9. **Reversibility:** every observer-operation sequence followed by its inverse must
   return exactly to the starting coordinate.

Training uses soft distributions over exact permutations so gradients can flow.
Those mixtures are not themselves reversible. Every reported retrieval accuracy
uses hard one-action execution; exact reversibility applies to that hard trace
and the Livnium substrate, not to the neural controller or its training process.

## Run

```bash
cd /Users/chetanpatil/Desktop/test/livnium-native-pile
python3 -m pytest
python3 run_experiment.py
```

The experiment writes:

- `results/metrics.json`
- `results/model.pt`

The checkpoint contains the trained action head and its original training pile.
Held-out and intervention piles are used temporarily for evaluation and are not
silently saved as the model's deployment state.

The checked three-seed result is recorded in [`RESULTS.md`](RESULTS.md).

## Honest scope

A pass demonstrates that the pile works inside the model's forward pass and that
its neural action head can learn an obfuscated token-to-operation dictionary,
then compose those operations over a pile containing unseen payload values.

The read algorithm and the requirement to answer through memory are fixed in the
architecture. This prototype does not show that a model autonomously decided to
use Livnium or learned the read algorithm itself.

It does **not** yet prove:

- language understanding;
- content-based addressing from natural-language queries;
- writable or self-organizing memory;
- an advantage over a matched flat-memory model;
- useful scaling beyond this 729-leaf prototype.

Those are later experiments, after this mechanism survives its first kill tests.

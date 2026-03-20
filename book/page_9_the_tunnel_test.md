# Page 9 — The Tunnel Test

## Chapter 1: Discovering the Universal Fixed Point

Every good theory has a failure mode that teaches you something. Livnium's most revealing failure was the universal fixed point.

The discovery happened during diagnostic runs. After training, the collapse engine was run on a large batch of diverse sentence pairs — entailments, contradictions, neutrals, easy examples, hard examples, short sentences, long sentences. The traces were plotted: alignment to each anchor over 6 collapse steps.

The plot should have shown three clusters of trajectories, each converging to a different anchor. What it showed instead was startling: **after approximately 3 steps, every trajectory converged to the same point.**

Not the same basin. The same *point*. Regardless of whether the input was entailment, contradiction, or neutral. Regardless of the encoder quality or the initial h₀. After step 3, all states collapsed to a single fixed point in R^768.

This is the universal fixed point. And it is the primary bottleneck of the system.

---

## Chapter 2: What a Fixed Point Is

In dynamical systems, a fixed point is a state that maps to itself under the dynamics:

```
h* is a fixed point if:  h* − α∇V(h*) = h*
which means:              ∇V(h*) = 0
```

A fixed point is where the gradient of the energy function is zero — no force, no movement. In a well-designed system, there should be three fixed points (one per anchor basin), and each input should converge to the correct one.

A *universal* fixed point is a single point that *all* trajectories converge to, regardless of their starting position. This happens when the energy landscape has effectively one deep well instead of three — or more precisely, when the dynamics are so strongly contractive that all states are pulled to the same region before the weaker per-basin forces can differentiate them.

---

## Chapter 3: The Tunnel Test

The tunnel test was designed to diagnose this. It works as follows:

1. Take a trained model.
2. Construct a set of test sentence pairs with known, unambiguous labels.
3. Compute h₀ for each.
4. Run the collapse dynamics and record the full 6-step trace.
5. At each step, compute the cosine similarity between all pairs of states.

If the system is working correctly, states from different classes should *diverge* over time — E states should move toward anchor_E while C states move toward anchor_C, and the distance between them should grow.

What the tunnel test revealed:

```
Step 0: mean pairwise cosine = 0.12  (low — states start in diverse directions)
Step 1: mean pairwise cosine = 0.34  (states begin converging)
Step 2: mean pairwise cosine = 0.67  (rapid convergence)
Step 3: mean pairwise cosine = 0.94  (nearly identical)
Step 4: mean pairwise cosine = 0.97  (effectively the same point)
Step 5: mean pairwise cosine = 0.98  (stuck)
```

By step 3, all states are within 6° of each other in angular distance. The system has "tunnelled" — all trajectories pass through a narrow bottleneck in state space before the basin-specific forces can separate them.

---

## Chapter 4: Why It Happens

The universal fixed point is caused by a competition between two forces in the dynamics:

**Force 1: The learned residual `delta(h)`.**
This is a neural network (Linear → Tanh → Linear) that takes h and produces a correction vector. It is the same network applied at every collapse step, to every sample. During training, it learns to push states toward a region where the physics works well — a "sweet spot" where the three basin forces are roughly balanced and the energy landscape is well-conditioned.

**Force 2: The basin-specific forces.**
These pull each state toward its correct anchor, proportional to the divergence from equilibrium.

The problem is that Force 1 dominates Force 2 in the early steps. The residual network `delta(h)` is a contraction mapping — it maps diverse inputs to a small region. This happens because the network parameters are shared across all samples and all steps. The Tanh nonlinearity saturates for large inputs, squashing everything toward a bounded region. And the network was trained on all samples simultaneously, so it learns a compromise: a single region that is "okay" for all three classes rather than good for any one.

Once all states have been pulled to this region by step 2–3, the basin-specific forces are too weak to separate them. The divergence from each anchor is similar for all states (because they are all near the same point), so the forces are similar, and the states stay bundled.

This is the tunnel: a narrow passage in state space that all trajectories are forced through, after which differentiation is nearly impossible.

---

## Chapter 5: The Head's Workaround

The system still achieves 82.79% accuracy despite the tunnel. How?

The SNLIHead compensates. Remember its input:

```python
x = cat([h_final, diff, prod])
```

`h_final` is the collapsed state — near-identical for all samples due to the tunnel. But `diff = v_h − v_p` and `prod = v_h * v_p` are computed directly from the BERT vectors, before any collapse dynamics. They retain all of BERT's discriminative information.

The head has learned to rely primarily on `diff` and `prod` for its classification and to use `h_final` as a minor additional signal. This explains the head-dynamics agreement number from Page 8:

- 73.9% agreement means: for 73.9% of samples, the basin that h_final is nearest to matches the head's prediction. The head is partially using the dynamics.
- 26.1% disagreement means: for 26.1% of samples, the head overrides the dynamics because h_final has tunnelled to the wrong basin (or to a basin that does not correspond to any specific class).

The system works because the head is powerful enough to classify from raw BERT features. The dynamics contribute when they work (73.9% of the time) and are overridden when they don't. This is functional but inelegant. The whole point of Livnium is that the dynamics *are* the classification. If the head is doing the work, the physics is decorative.

---

## Chapter 6: What the Fixed Point Reveals

The universal fixed point is not a bug. It is a diagnostic. It reveals something fundamental about the current state of the system:

**The dynamics are over-constrained and under-differentiated.**

Over-constrained: the shared residual network imposes a strong prior on where states should go. This prior is so strong that it overrides the basin-specific forces.

Under-differentiated: the three basins are not different enough. In the current energy landscape, the three wells are too similar in depth and shape. The gradient differences between "heading toward E" and "heading toward C" are small compared to the gradient of the residual network's contraction.

The fix must address one or both of these:

1. **Weaken the residual network.** Use a smaller `delta(h)` or apply it only at early steps, allowing the basin forces to dominate in later steps. This is a hyperparameter problem — reduce the learning rate or capacity of the update network.

2. **Strengthen the basin forces.** Increase the force strengths (`strength_entail`, etc.) or reduce the barrier so that the basins pull harder. This is a landscape shaping problem — make the energy wells deeper and the ridges between them sharper.

3. **Use step-dependent dynamics.** Instead of applying the same `delta(h)` at every step, use different networks for early steps (where coarse positioning matters) and late steps (where basin differentiation matters). This breaks the symmetry that causes the tunnel.

4. **Initialize closer to the basins.** If BERT is jointly trained to produce h₀ vectors that already point toward the correct anchor, the basin forces dominate from step 1 and the residual network never has a chance to create the tunnel.

---

## Chapter 7: The Fixed Point as a Phase Transition

There is a beautiful theoretical framing for the tunnel.

In physics, a phase transition is a point where a system changes qualitative behavior. Water freezes. Magnets align. Symmetry breaks.

The universal fixed point is a sign that the system is *at* a phase transition but has not crossed it. The three basins exist in the energy landscape, but they are separated by barriers that are too low for the dynamics to distinguish. The system is like a magnet at exactly its critical temperature — it has the structure for ordered behavior but not quite enough asymmetry to choose a direction.

Training pushes the system toward the ordered phase. Joint training with BERT pushes harder — head-dynamics agreement went from 29.7% to 73.9%, meaning the tunnel narrowed and more trajectories escaped it. But the system has not yet crossed the critical point where all trajectories cleanly separate.

Crossing that critical point — achieving near-100% head-dynamics agreement — is the goal of future training experiments. The three laws are correct. The energy landscape is well-defined. What remains is making the landscape *sharp enough* that the dynamics can do their job without the head's help.

---

## Chapter 8: Living With the Bottleneck

Until the tunnel is resolved, the system operates in a hybrid mode:

- The collapse dynamics contribute when the initial geometry is favorable (BERT produces an h₀ that already points roughly toward the correct anchor).
- The SNLIHead fills in when the dynamics fail (h₀ starts in a contested region and the tunnel captures it).
- The head-dynamics agreement metric tracks progress: as it approaches 100%, the system is moving toward a pure physics-based classification.

This hybrid mode is not a failure. It is a realistic intermediate state of a system that is still learning to fully inhabit its own theory. The three laws define what the system *should* do. The tunnel reveals what it *cannot yet* do. The gap between theory and practice is the work that remains.

Every experimental improvement — better joint training schedules, deeper collapse steps, step-dependent dynamics, stronger basin separation — is tested against the tunnel test. If the pairwise cosine at step 3 drops from 0.94 to 0.70, the tunnel is opening. If it drops below 0.50, the basins are starting to differentiate. If it drops below 0.30, the system has crossed the critical point and the three laws are governing the classification.

That is the destination. The tunnel is the map that shows how far we have to go.

---

*Next: Page 10 — Cortex v1 (a cubic lattice, 24 rotations, and a quantum simulator that reads text)*

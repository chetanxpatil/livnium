# Page 7 — The Three Laws of Livnium

## Chapter 1: When Physics Stops Being Analogy

For the first six pages of this book, Livnium has been described as "inspired by physics" or "physics-like." The anchors act *like* gravitational wells. The collapse dynamics *resemble* a dissipative system. The basin field *behaves as if* it were a potential landscape.

But at some point during v2 development, something changed. The equations stopped being analogies. They became the system's actual laws — not metaphors borrowed from physics, but first principles from which the entire architecture could be derived.

This page is about the moment those laws crystallised and what they mean.

---

## Chapter 2: Law I — The Relational State

```
h₀ = v_h − v_p
```

The first law is deceptively simple. It says: *the object that Livnium reasons about is not a sentence. It is the relationship between two sentences.*

`v_p` is a vector encoding the premise. `v_h` is a vector encoding the hypothesis. Neither of these, by itself, is the input to the system. The input is their *difference* — the directed displacement from premise to hypothesis in semantic space.

This is not a preprocessing step. It is a foundational claim about what NLI is. NLI is not "classify this sentence pair." It is "characterise the geometric relationship between these two points in meaning-space."

Consider what `h₀` encodes:

- If the hypothesis is a paraphrase of the premise, `v_h ≈ v_p`, so `h₀ ≈ 0`. The relationship is near the origin — small, undramatic, entailing.
- If the hypothesis contradicts the premise, `v_h` and `v_p` point in conflicting directions. `h₀` is large and points away from the regions both sentences share.
- If the hypothesis is unrelated, `v_h` and `v_p` are orthogonal. `h₀` has moderate magnitude but no consistent direction relative to the shared content.

The key insight is that `h₀` lives in the same vector space as the sentences themselves, but it is not a sentence. It is a *displacement*. It has magnitude (how different are these sentences?) and direction (how are they different?). That displacement is the raw material the dynamics operate on.

Every encoder change in Livnium's history — from legacy BoW to pretrained BoW to BERT to the native encoder — changes the quality of `v_p` and `v_h`. But it never changes the first law. The initial state is always the difference. The dynamics always start from there.

---

## Chapter 3: Law II — The Energy Landscape

```
V(h) = −logsumexp(β · cos(h, anchors))
```

The second law defines the terrain the state moves through. It says: *the space has an energy function, and that function has discrete minima at the anchor positions.*

In the earlier pages, this was described as "three gravitational wells." That description is correct but incomplete. The second law makes the landscape precise.

`cos(h, A_i)` is the cosine similarity between the current state and the i-th anchor. When h is near an anchor, the cosine is high. When h is far from all anchors, all cosines are low.

`β` is a temperature-like parameter that controls how sharp the basins are. High β means the wells are deep and narrow — once you are close to an anchor, the energy drops steeply and you are trapped. Low β means the landscape is gentle — the state can drift between basins.

The `logsumexp` is the crucial mathematical choice. It is a smooth approximation to the maximum function. What it gives us:

1. **Differentiability everywhere.** Unlike a hard `max`, `logsumexp` has gradients at every point in space. This means the dynamics can compute a force direction at any state.

2. **Competition between basins.** When h is equidistant between two anchors, both contribute to the energy. The landscape forms a saddle — a ridge between two valleys. The state is unstable on the ridge and will fall to one side or the other.

3. **Emergent boundaries.** The decision boundaries between E, N, and C are not learned directly. They emerge as the ridges of the energy landscape — the set of points where two basins have equal pull. Move the anchors, and the boundaries move with them.

The negative sign in front of `logsumexp` flips the landscape: high cosine similarity (near an anchor) corresponds to low energy (a valley). The state wants to minimise energy, which means it wants to maximise its alignment with the nearest anchor.

The earlier `BARRIER = 0.38` from the physics equations is the equilibrium point of this landscape — the cosine value at which the gradient of V with respect to alignment is zero. It is where the slope of the valley floor levels off. Above 0.38, the state is pushed back (too close to the anchor). Below 0.38, the state is pulled in. The shell at 0.38 is the bottom of the valley.

---

## Chapter 4: Law III — The Collapse Dynamics

```
h_{t+1} = h_t − α∇V(h_t)
```

The third law is gradient descent on the energy landscape. It says: *the state moves downhill.*

This is the equation that connects everything. Given the energy function from Law II, take its gradient with respect to h. That gradient points uphill — toward higher energy. Subtract it. The state moves downhill — toward lower energy, toward the nearest basin.

`α` is the step size. In Livnium it is controlled by the learned force strengths (`strength_entail`, `strength_contra`, `strength_neutral`) and the residual network `delta(h)`. The effective step size varies per dimension, per step, per sample. But the principle is always the same: move in the direction that reduces energy.

The magic of this law is that it turns classification into simulation. You do not need to learn a decision boundary. You do not need to learn a mapping from input to output. You just need to define the landscape (Law II), set the initial condition (Law I), and let the dynamics run (Law III). The answer is wherever the state ends up.

The number of collapse steps is finite — six by default. This is not gradient descent on a loss function during training (which runs for thousands of steps). This is gradient descent *as the forward pass*. The model's prediction is the result of six steps of energy minimisation. Each step is a matrix multiplication, a cosine computation, and a vector update. Fast, deterministic, interpretable.

---

## Chapter 5: Why Three Laws and Not More

Physicists have a tradition of reducing systems to their minimal set of governing equations. Newton had three laws of motion. Thermodynamics has four laws. Maxwell unified electromagnetism into four equations.

Livnium has three laws, and that number is not accidental. Each law answers one question:

| Law | Question | Answer |
|---|---|---|
| I | What is the state? | The displacement between hypothesis and premise |
| II | What is the landscape? | An energy function with minima at the anchors |
| III | How does the state evolve? | Gradient descent on the energy landscape |

Remove any one law and the system breaks:

- Without Law I, there is no initial condition. The dynamics have nothing to act on.
- Without Law II, there is no landscape. The state has no reason to move toward any anchor.
- Without Law III, there is no dynamics. The state just sits at h₀ forever.

Add a fourth law and you have redundancy. Every other component of Livnium — the basin field, the locking zones, the rotation matrix, the null endpoint, the neutral boost — is a refinement or elaboration of one of the three laws. The basin field modifies Law II (more minima). The locking zones modify Law III (stronger forces inside basins). The rotation matrix modifies Law III (adds a curl component to the gradient). None of them are new laws. They are engineering choices within the framework the three laws define.

---

## Chapter 6: The Laws as a Unification

Before the three laws were articulated explicitly, Livnium was a collection of engineering decisions that happened to work. The barrier was 0.38 because grid search said so. The collapse ran for 6 steps because more steps didn't help. The forces pointed toward anchors because that's what attractors do.

After the laws, every decision has a derivation:

**Why is barrier 0.38?** Because that is the cosine value at which ∇V = 0 for the default β. It is the equilibrium of Law II.

**Why 6 steps?** Because empirically, h converges to within 1% of its final position in 3–5 steps of Law III. Six gives a safety margin.

**Why does neutral get a boost at the E-C boundary?** Because the energy landscape from Law II has a saddle between the E and C wells. The neutral boost reshapes Law II to add a well at the saddle point, giving neutral territory that would otherwise be contested.

**Why does the SNLIHead use `[h_final, diff, prod]`?** Because `h_final` is the output of Laws I–III (the collapsed state). `diff` is h₀ itself (Law I, before dynamics). `prod` captures element-wise agreement that the displacement alone discards. The head has access to both the dynamics and the raw geometry.

The laws do not make the system simpler. They make it *derivable*. Every component can be traced back to one of three principles. That is what a theory gives you that engineering alone does not — the ability to say not just "this works" but "this works because."

---

## Chapter 7: The Laws in Code

For anyone reading the codebase, here is where each law lives:

**Law I** — `encoding_snli.py` → `build_initial_state()`
```python
h0 = v_h - v_p
```

**Law II** — `physics_laws.py` + `vector_collapse_engine.py`
```python
BARRIER = 0.38                           # equilibrium of V
divergence = BARRIER - cos(h, anchor)    # gradient of V at this point
```

The explicit `logsumexp` form of V is implicit in the code. The forces computed from `divergence` are exactly the gradient of V with respect to h. You can verify this by computing `−∂V/∂h` analytically — it yields the same update equations.

**Law III** — `vector_collapse_engine.py` → `_collapse_static()`
```python
for step in range(num_layers):
    h = h + delta(h) - forces       # gradient descent + learned residual
```

The learned residual `delta(h)` is the only part of the dynamics that is not derivable from Laws I–III. It is the model's degree of freedom — the part that learns from data what the physics alone cannot capture. In a purely physics-based system, `delta(h)` would be zero. In a hybrid system like Livnium, it accounts for the gap between the idealised energy landscape and the actual geometry of natural language.

---

## Chapter 8: What the Laws Predict

A theory is tested by its predictions. Here are three that the three laws make, all of which have been confirmed:

**Prediction 1: Convergence is fast.**
Law III says the dynamics are gradient descent on a smooth landscape. For a smooth, strongly convex landscape (which V approximately is near each minimum), gradient descent converges exponentially. Empirically: h reaches 95% of its final value in 3 steps, regardless of the initial condition.

**Prediction 2: The decision boundaries are Voronoi-like.**
Law II says the basins are the regions where each anchor has the lowest energy. For the `logsumexp` form, these regions are smooth approximations to Voronoi cells on the hypersphere. The boundaries are the ridges where two anchors have equal pull. This was confirmed visually in 3D projections of the Livnium basis.

**Prediction 3: Ambiguous samples oscillate near boundaries.**
Law III says the force is weak near saddle points (where ∇V ≈ 0 because two basins cancel). Samples whose h₀ starts near a saddle should oscillate — attracted by both basins, committed to neither. The trace logs confirm this: hard neutral examples show oscillating alignment scores before finally settling, and their collapse takes all 6 steps instead of the 3 that easy examples need.

---

*Next: Page 8 — Joint Retraining (the breakthrough from BoW to BERT, 76% → 82.79%, and what neutral recall reveals)*

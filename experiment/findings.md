# Research Brief: Noun Model Potential & Discrete Group Benchmark

## 1. The Noun Model Force Has No Exact Scalar Potential (and why that matters)

In the NLI paper (*"A Trained Iterative Classifier Reduces to Energy Gradient Descent"*), we showed that an iterative update head trained with an MLP can be replaced by the analytical gradient of a logsumexp energy potential:
$$V(h) = -\log \sum_k \exp\big(\beta \cos(h, A_k)\big)$$

We tried to apply the same methodology to the **Noun Collapse model** (`noun_collapse_pure.py`), whose hand-designed read step is an attraction force
$$F(h) = -s\,(1 - \cos(h, c))\,\frac{h - c}{\|h - c\|}.$$

The result is a **negative** one, and it is the interesting part: this force is **not** the gradient of the cosine-style potential a naive derivation suggests. It is verified numerically in `scratch/verify_noun_potential.py`.

### The tempting (wrong) derivation
Let $r = \|h - c\|$. If $h$ and $c$ both lay exactly on the unit sphere, then $r^2 = 2 - 2\cos(h,c)$, so $1-\cos = r^2/2$, and the force becomes a Euclidean central force $F = -\tfrac{s}{2} r\,(h-c)$. Integrating the magnitude gives a cubic-in-distance well
$$E_{\text{euc}}(h) = \tfrac{s}{6}\,\|h-c\|^{3},$$
and one is tempted to rewrite it via the same substitution as $E(h) = \tfrac{\sqrt 2}{3}\, s\,(1-\cos(h,c))^{1.5}$.

### Why it fails
Substituting $r^2 = 2-2\cos$ **inside the potential and then differentiating freely in $\mathbb R^{d}$ is not valid** — the identity only holds on the constraint surface $\|h\|=1$. Numerically, even *on* the unit sphere:

$$-\nabla E_{\text{euc}} \text{ vs } F : \text{angle } 0.00^\circ,\ \text{mag-ratio } 1.000 \qquad
-\nabla E_{\cos} \text{ vs } F : \text{angle } 43.9^\circ,\ \text{mag-ratio } 0.72$$

So the cubic **Euclidean** well $E_{\text{euc}} = \tfrac{s}{6}\|h-c\|^3$ reproduces the force *on the sphere* (direction exact everywhere, magnitude exact at $\|h\|=1$), but the $(1-\cos)^{1.5}$ **cosine** form does **not** — its gradient is $\sim\!44^\circ$ off, because $\nabla(\cos\text{-form})$ points along the *tangential* cosine-gradient $(c-\cos\hat h)$ while the force points along the *Euclidean chord* $(h-c)$.

### The real conclusion
As a field in $\mathbb R^{d}$ the noun force is **non-conservative**: its Jacobian is asymmetric ($\max|J-J^\top| = 2.3\times10^{-2}$ on the sphere), so **no exact global scalar potential exists**. The magnitude uses the *normalized* cosine while the direction uses the *unnormalized* chord $(h-c)$, and because the walk norm-clamps rather than renormalizes, $h$ leaves the sphere and the two disagree.

This is precisely the **Euclidean-radial vs. cosine-gradient mismatch** the NLI paper identified in Livnium v1 (there $\approx135^\circ$; here $\approx44^\circ$). In other words, the noun model runs the *un-corrected* force — the very thing the trained SNLI MLP had to learn to fix.

### The correction (already live in `pure_reply.py`)
Replacing the chord direction with the **true cosine gradient** makes the system conservative with an *exact* closed-form potential:
$$V(h) = -\cos(h,c), \qquad \nabla_h V = -\frac{c - \cos(h,c)\,\hat h}{\|h\|}, \qquad h \leftarrow h + s\,\frac{c - \cos(h,c)\,\hat h}{\|h\|}.$$
This is exactly `collapse_step` in `experiment/pure_reply.py`, and the single-anchor special case of the NLI energy $V(h) = -\log\sum_k \exp(\beta\cos(h,A_k))$. Porting it into the noun read is the honest "grad-V" upgrade; the best that can be said for the *current* noun force is that it approximates a cubic distance well $E_{\text{euc}} \approx \tfrac{s}{6}\|h-c\|^3$ near the unit sphere.

---

## 2. Speed Benchmark: Continuous vs. Discrete Collapse

We compared the throughput (operations per second) of the continuous floating-point updates (gradient descent on $\mathbb{R}^{256}$) against the discrete hypercube updates (matrix multiplication of 5-dimensional integers) on CPU over 200,000 steps.

### Results:
* **Continuous Collapse Step:** `106,762.96` operations/sec
* **Discrete Group Collapse Step:** `3,026,804.84` operations/sec

### Verdict:
**Discrete Group Collapse is 28.4x FASTER**, running at over **3 Million steps/sec** on CPU.

Because hypercube rotations are simple coordinate swaps and sign changes, they require zero floating-point division or norm evaluations. This makes them ideal for edge processors and analog memristor execution, while maintaining 100% reversible, lossless history.

---

## 3. Precomputed Group Table Lookup Benchmark

To achieve maximum execution speed, we precomputed the 24x24 octahedral group multiplication table ($G \times G \to G$) and replaced all matrix multiplications in the sequence loop with a chain of simple integer lookups.

### Benchmark Results (100,000 runs):
* **Matrix Multiply Version:** `188,020.46` runs/sec
* **Table Lookup Version:** `620,747.51` runs/sec

### Verdict:
**Precomputed Table Lookup is 3.30x FASTER** than the matrix-multiplication version. 

By precomputing the multiplication table, we reduce sequence composition to a chain of integer lookups, deferring the single matrix-vector multiplication until label readout. This makes discrete sequence collapse extremely lightweight and fast.

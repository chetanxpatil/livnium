# Livnium v3 — Paper, Visuals, and Validation

## Status
v2.0 is stable and tagged. The three laws are established empirically.
v3 is about proving them rigorously and publishing.

---

## The Claim We Are Making

> NLI can be formulated as a dynamical system governed by three laws:
>
> **Law 1 (Relational State):**   `h₀ = v_h − v_p`
> **Law 2 (Energy Landscape):**  `V(h) = −logsumexp(β · cos(h, A_E), β · cos(h, A_C), β · cos(h, A_N))`
> **Law 3 (Collapse Dynamics):** `h_{t+1} = h_t − α∇V(h_t)`
>
> Laws 2 and 3 were not assumed. They were recovered from a trained system.
> This is the main result.

---

## Tunnel Test Findings (v2 trajectory analysis, 2000 samples)

Neutral errors breakdown:
- TYPE-1 (bad h₀, Law 1): 29 errors (14.4%)
- TYPE-2 (mid-diversion, Law 3): 43 errors (21.3%)
- **TYPE-3 (boundary stall, Law 2): 129 errors (63.9%)** ← dominant
- TYPE-?: 1 (0.5%) — head overrides correct collapse

Energy: error avg V drop = −0.0017 (flat), correct avg V drop = +0.0455 (ascending)
Misclassification: 67.3% neutral→entailment, 32.7% neutral→contradiction

**Root cause identified (not "narrow neutral basin" but "universal fixed point"):**

Steps 4–6 of collapse produce nearly identical trajectories across completely different
inputs: cos_E ≈ −0.129, cos_C ≈ −0.124, cos_N ≈ +0.172. The dynamics converge to a
universal fixed point attractor after ~3 steps, discarding input information. This fixed
point is weakly neutral-biased, explaining 71% neutral recall (better than chance,
worse than entailment/contra). TYPE-3 is input-agnostic convergence, not semantic
boundary ambiguity.

Additional finding: many TYPE-1/TYPE-2 errors end with correct dominant basin (N) at
step 6 but head still predicts wrong. The SNLIHead uses v_p/v_h features that can
override the collapse outcome — some "dynamics errors" are actually head failures.

**Three concrete v3 targets from tunnel test:**
- Target A (Law 2, 63.9%): break the universal fixed point — add input-dependent
  residual forcing to keep h_final sensitive to h_0 beyond step 3
- Target B (head–dynamics disconnect): add consistency loss between head prediction
  and dominant-basin at final step during training
- Target C (entailment pull): regularize anchor angles so A_N is more orthogonal to
  A_E and A_C (currently neutral→entail errors are 2× more common than neutral→contra)

---

## Priority 1 — Statistical Validation

### 1a. Full dev set gradient flow run
- Current result (+0.003 accuracy, +1.3pp neutral) is on 2000 samples
- Need: run on full SNLI dev set (9842 samples)
- Command: `python3 test_gradient_collapse.py --checkpoint ../../pretrained/bert-joint/best_model.pt --snli-dev ../../data/snli/snli_1.0_dev.jsonl --mode grad-v --beta 1.0 --alpha 0.2`
- Expected: result stays within ±0.3% — if so, claim is clean

### 1b. Multi-seed gradient flow test
- Train 3 independent BERT-joint runs from scratch (different random seeds)
- Run grad-V on each checkpoint
- Report: does the law emerge consistently, or is it checkpoint-specific?
- If consistent across seeds → strongest version of the claim

### 1c. Per-class breakdown of grad-V improvement
- Already have: neutral +1.3pp, overall +0.3%
- Need: entail and contra breakdown across full dev set
- Hypothesis: entail recall drops slightly (trade-off), contra stays flat

---

## Priority 2 — Trajectory Visualizations

### 2a. PCA trajectory plots
- Run collapse on 100 correctly-classified SNLI examples
- Record `h₀, h₁, h₂, ..., h_L` for each
- PCA to 2D or 3D using trained anchor geometry as projection basis
- Plot: trajectories flowing toward E/N/C basins
- Expected result: clean convergence toward correct basin; show near-boundary examples spiraling

### 2b. Energy descent visualization
- For each trajectory step, compute `V(h_t)` using Law 2
- Plot: energy as function of step number
- Expected: monotone decrease under grad-V, approximate decrease under trained system
- This directly validates Law 3 visually

### 2c. Attractor basin map
- Sample random 2D slices of the semantic space (anchored around BERT h₀ distribution)
- For each point, run collapse → record final label
- Plot: basin boundaries in 2D projection
- Show: E, N, C basins are separable, with identifiable boundary regions

### 2d. Wrong predictions as near-boundary points
- Overlay wrong predictions on basin map
- Hypothesis: they cluster at basin boundaries / saddle regions
- Connects stability result (3.9× flip rate) to geometry

---

## Priority 3 — Cross-Encoder Experiment

### 3a. Complete cross-encoder BERT training
- Cross-encoder processes (P, H) jointly — should handle role-reversal better
- Run: `--encoder-type crossbert --epochs 10`
- Target: beat 82.06% bi-encoder (expect ~83–84%)

### 3b. Apply Three Laws to cross-encoder
- Does grad-V work the same on cross-encoder anchors?
- If yes: the laws are encoder-independent (structural, not BERT-specific)
- This would be a strong generalization claim

---

## Priority 4 — Energy Landscape Characterization

### 4a. Anchor orthogonality vs accuracy
- Current: cos(A_E, A_C) = −0.56, cos(A_E, A_N) = −0.77, cos(A_C, A_N) = −0.09
- Question: does more orthogonal anchor geometry → better accuracy?
- Experiment: regularize anchor angles during training, measure effect
- Connects geometry to performance systematically

### 4b. β sensitivity curve
- Already have sweep β ∈ {1, 2, 5, 10, 20, 50}: all within ±1%
- Full curve: β ∈ {0.1, 0.5, 1, 2, 5, 10, 20, 50, 100}
- Plot: accuracy vs β, showing plateau region
- This characterizes the energy landscape's "sharpness"

### 4c. Why β=1 is optimal
- β=1 gives softmax over anchors → equal weighting near boundaries
- β>>1 becomes argmax → selects single nearest anchor (loses gradient information)
- Theoretical explanation: low β preserves multi-basin gradient signal at boundaries

---

## Priority 5 — Livnium-Native Completion

### 5a. Resume training from epoch 9
- Best checkpoint saved at epoch 9 (81.05%), cancelled at epoch 11
- Resume: `--resume ../../../pretrained/livnium-native/best_model.pt`
- Target: 20 epochs → expect ~82%+

### 5b. Apply Three Laws to native encoder
- If native encoder converges to same laws (β=1.0, α=0.2 optimal)
- → Laws are representation-agnostic
- Key test: does grad-V also help the 772K-param encoder?

### 5c. Native encoder ablation
- Test: remove cross-encoder component (`--livnium-cross-encoder` off)
- Measure: accuracy delta
- This isolates the contribution of cross-attention to the native representation

---

## Priority 6 — Paper Structure (Draft Outline)

### Title (candidate)
*"Three Laws of Semantic Collapse: NLI as Gradient Descent on an Anchor Energy Landscape"*

or

*"From Dynamics to Laws: Discovering the Equation of Motion of a Neural Inference System"*

### Sections
1. **Introduction** — NLI as dynamical system; gap in current literature; contribution summary
2. **Background** — attractor dynamics in ML, Lyapunov theory, energy-based models
3. **The Livnium System** — architecture, anchor geometry, collapse engine
4. **Discovering the Laws**
   - Section 4.1: Law 1 (h₀ design choice, ablation)
   - Section 4.2: Law 2 (energy landscape, anchor structure, β analysis)
   - Section 4.3: Law 3 (trained vs grad-V comparison, full dev set results)
5. **Empirical Validation**
   - Accuracy (multi-seed, full dev)
   - Trajectory visualization
   - Basin stability (3.9× flip ratio)
   - Speed (142× fewer params, 5.3× pipeline)
6. **Analysis**
   - Why the MLP converges to ∇V
   - What β=1 means (maximum gradient signal at basin boundaries)
   - Contradiction bias in stability (1.82× vs 7.18× for entail)
7. **Related Work** — attractor networks, energy-based NLI, neural ODEs
8. **Conclusion** — three laws, what they mean, limitations, v3 directions

### Key claims (need to defend each)
- ✅ "The trained MLP approximates ∇V" — grad-V ≥ full pipeline accuracy
- ⬜ "This holds consistently" — multi-seed test (Priority 1b)
- ⬜ "This is encoder-independent" — cross-encoder test (Priority 3b)
- ✅ "The geometry is Lyapunov-stable" — proven + basin stability empirics
- ⬜ "Trajectories visualizably flow toward basins" — Priority 2a

---

## Priority 7 — Release and Dissemination

### 7a. GitHub release v2.0
- Tag: v2.0 (commit + push)
- Release notes: Three Laws, grad-V discovery, stability results, speed table
- Attach: README as release description

### 7b. HuggingFace model card update
- Update model card to reflect v2.0 results
- Add: equation of motion, three laws, accuracy table

### 7c. Zenodo preprint update
- Upload revised paper once written
- Link to v2.0 GitHub tag

### 7d. ArXiv submission (post v3 experiments)
- Target: after multi-seed validation is complete
- Venue: EMNLP 2026 (short paper) or NAACL 2026 findings

---

## Immediate Next Steps (this week)

1. **Run full dev set grad-V** (Priority 1a) — 30 mins, resolves the noise question
2. **Plot energy descent curves** for 10 examples (Priority 2b) — strongest visual for the paper
3. **Resume Livnium-native training** to epoch 20 (Priority 5a) — already almost there
4. **Draft Section 4 of the paper** — the core discovery section, while results are fresh

---

## Open Questions

**Q1: Are the Three Laws universal or BERT-specific?**
Testing on Livnium-native and cross-encoder answers this.

**Q2: Why does β=1 work best?**
Theoretical answer: log-softmax gradient at β=1 preserves maximum multi-basin signal near boundaries. At β→∞, gradient collapses to argmax (single anchor), losing boundary sensitivity.

**Q3: Is Law 1 (h₀ = v_h − v_p) actually necessary?**
Could train with h₀ = [v_h; v_p] or h₀ = v_h + v_p and test. The subtraction might be what creates the semantic directionality that Laws 2 and 3 operate on.

**Q4: Is V(h) minimal?**
Could a simpler energy (e.g., single-anchor L2 distance) work as well? The logsumexp formulation is the simplest smooth function over multiple class anchors — but this hasn't been ablated.

**Q5: Does gradient flow generalize beyond NLI?**
In principle, any classification task where the head uses anchor-cosine geometry could have this property. Sentiment analysis, textual similarity, etc.

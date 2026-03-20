# Page 8 — Joint Retraining

## Chapter 1: The Wall at 76%

The pretrained bag-of-words encoder brought Livnium from 56% to 76.32%. That was a triumph — a 20-point gain from better embeddings alone, without touching the dynamics. But 76% is not good enough. Human performance on SNLI is 89%. A simple BERT + linear head reaches 90%. Livnium at 76% was interesting. It was not competitive.

The diagnosis was clear from Page 5: bag-of-words destroys information. Mean pooling erases word order, dilutes negation, and forbids cross-sentence attention. No amount of attractor dynamics can recover what the encoder threw away.

The obvious fix was BERT. But the obvious fix failed.

---

## Chapter 2: The Frozen BERT Disaster

The first attempt was the simplest possible: take a pretrained BERT model, freeze its weights, extract its 768-dimensional CLS vectors for premise and hypothesis, compute `h₀ = v_h − v_p`, and feed that into the Livnium collapse engine.

Result: **~61% accuracy.**

Worse than the pretrained BoW model. Worse than a model with 13 million parameters training from scratch. A 110-million-parameter transformer, pretrained on all of English Wikipedia and BooksCorpus, scored 15 points lower than a mean-pooled word embedding table.

What happened?

BERT's internal geometry is not aligned with Livnium's attractor space. BERT was trained to predict masked tokens — its CLS vectors encode information useful for that task. The directions that matter in BERT-space (where "dog" is close to "puppy" and far from "quantum") are not the directions that matter in Livnium-space (where "entailment" has a specific anchor direction and "contradiction" has another).

When you compute `h₀ = v_h − v_p` using frozen BERT vectors and feed that into a collapse engine, the engine is trying to classify points in a space whose geometry it does not control. The anchors learn positions in R^768, but they can only attract along directions that BERT happens to use for semantic relationship encoding. BERT may not use those directions at all — its geometry was shaped by a completely different objective.

The collapse engine was learning to navigate a foreign landscape. It could not reshape the landscape to fit its physics because BERT was frozen. So the physics broke.

---

## Chapter 3: The Joint Training Idea

The fix required a paradigm shift: let BERT learn alongside Livnium.

Joint training means unfreezing the BERT encoder and computing gradients through the entire pipeline:

```
BERT(premise) → v_p
BERT(hypothesis) → v_h
h₀ = v_h − v_p
collapse(h₀) → h_final
head(h_final, v_p, v_h) → logits
loss = CrossEntropy(logits, label)

Backpropagate through: head → collapse engine → h₀ → BERT
```

The critical line is the last one. When loss gradients flow back through the collapse dynamics and into BERT, they carry a message: "reshape your embedding space so that the difference vectors collapse correctly." BERT's 110 million parameters shift — not much, just enough — so that its geometry becomes compatible with Livnium's physics.

The BERT encoder is no longer a passive feature extractor. It is a partner in the dynamics, co-adapting its geometry with the collapse engine.

---

## Chapter 4: The Bi-Encoder Architecture

The joint training used a bi-encoder setup:

```
BERT(premise)    → CLS → v_p ∈ R^768
BERT(hypothesis) → CLS → v_h ∈ R^768
h₀ = v_h − v_p
```

Premise and hypothesis are encoded separately through BERT. Each gets its own CLS vector. The two vectors meet only at the subtraction — the first law.

This is different from a cross-encoder (where both sentences go into BERT together as `[CLS] premise [SEP] hypothesis [SEP]`). The bi-encoder is less powerful in theory — BERT cannot attend from hypothesis tokens to premise tokens — but it preserves the clean geometric interpretation of `h₀` as a displacement vector.

The bi-encoder was the first experiment because it changes the least. The collapse engine and SNLIHead stay exactly the same. Only the source of `v_p` and `v_h` changes: BERT instead of mean-pooled embeddings. If accuracy improves, the improvement is purely from better input geometry.

---

## Chapter 5: The Numbers

After 5 epochs of joint training with a learning rate of 2e-5 for BERT and 1e-3 for the collapse engine and head:

```
Overall accuracy:     82.79%    (was 76.32% with pretrained BoW)
Entailment recall:    89.2%     (was 86.1%)
Contradiction recall: 84.6%     (was 81.4%)
Neutral recall:       74.5%     (was 68.4%)

Neutral recall gain:  +6.1 percentage points
```

The overall gain is 6.47 points. That does not sound dramatic. But look at where the gains come from.

Entailment and contradiction both improved, but modestly — these are the "easy" classes where even bag-of-words gets directional information right. The hypothesis is either semantically close to the premise (entailment) or semantically opposed (contradiction). Cosine similarity of mean-pooled vectors already captures most of this.

Neutral is where BERT earns its keep. A 6.1-point gain in neutral recall means BERT is providing information that bag-of-words structurally cannot: the difference between "related but not entailing" and "related and entailing." That distinction requires understanding the nature of the relationship, not just the degree of overlap.

---

## Chapter 6: Head-Dynamics Agreement

The most revealing metric was not accuracy. It was **head-dynamics agreement** — the percentage of samples where the collapse engine's basin assignment (based on which anchor h_final is closest to) matches the SNLIHead's classification (based on the logits).

```
Before joint training:  29.7%
After joint training:   73.9%
```

This number deserves a long pause.

Before joint training, the collapse dynamics and the classification head disagreed on 70% of samples. The engine was collapsing h into one basin, and the head was ignoring that and making its own decision based on the raw `v_p`, `v_h`, and `h_final` features. The dynamics were doing work that the head disregarded.

This makes sense: with frozen BERT, the collapse engine could not shape the geometry properly. The anchor positions were a poor fit for BERT's native space. So the SNLIHead learned to use `diff` and `prod` (which came directly from BERT's informative vectors) and to mostly ignore `h_final` (which came from broken dynamics).

After joint training, agreement jumped to 73.9%. The dynamics and the head now agree on nearly three-quarters of samples. The head has learned to *trust* the collapsed state because the collapsed state now carries real information — BERT's geometry has been reshaped so that the attractors work properly.

This is the metric that matters most. Not just "is the system accurate?" but "is the system working *the way it was designed to work*?" High head-dynamics agreement means the collapse is doing its job. Low agreement means the head is routing around the dynamics. At 73.9%, the system is genuinely using its physics for most samples.

The remaining 26.1% disagreement is the next frontier — these are samples where the dynamics and the head see different things. Understanding why they disagree is the path to further improvement.

---

## Chapter 7: What BERT Changed in the Geometry

During joint training, BERT's internal representations shifted. The shift was small in absolute terms (the weights moved by <1% on average) but large in geometric terms. The key change:

BERT learned to encode sentence pairs so that their difference vectors align with Livnium's anchor directions.

Before training, the direction from `v_p` to `v_h` for an entailment pair might point in any direction in R^768. After training, it points roughly toward `anchor_entail`. Not perfectly — the residual network `delta(h)` handles the fine adjustments — but close enough that the collapse dynamics can finish the job in 3–4 steps.

This is visible in the trace logs. Before joint training, h₀ starts with low alignment to all three anchors (typically cos < 0.1). The collapse has to work hard, taking all 6 steps and often ending in a contested region. After joint training, h₀ starts with clear directional bias (cos to the correct anchor typically > 0.3). The collapse confirms and sharpens what BERT already suggested.

The system moved from "BERT provides a starting point and the dynamics do all the work" to "BERT provides a strong hint and the dynamics refine it." This division of labor is healthier and more robust.

---

## Chapter 8: The Confusion Matrix

The confusion matrix before and after joint training tells a clean story:

**Before (pretrained BoW, 76.32%):**
```
              Predicted
              E      C      N
Actual E    86.1   4.2    9.7
Actual C     5.8  81.4   12.8
Actual N    14.3  17.3   68.4
```

**After (joint BERT, 82.79%):**
```
              Predicted
              E      C      N
Actual E    89.2   2.8    8.0
Actual C     3.1  84.6   12.3
Actual N    10.7  14.8   74.5
```

Three things stand out:

1. **E↔C confusion dropped.** E misclassified as C went from 4.2% to 2.8%. C misclassified as E went from 5.8% to 3.1%. These are the errors caused by word-order blindness ("dog bites man" vs "man bites dog") and BERT's attention mechanism directly addresses them.

2. **N→E and N→C errors dropped.** Neutral misclassified as entailment went from 14.3% to 10.7%. Neutral misclassified as contradiction went from 17.3% to 14.8%. Neutral is still the hardest class, but BERT gives the dynamics better starting geometry to work with.

3. **C→N errors barely changed.** Contradiction misclassified as neutral stayed at ~12%. This is the stubborn error — cases where the contradiction signal is subtle and the dynamics cannot distinguish "these sentences are about different things" (neutral) from "these sentences say opposite things" (contradiction). This requires deeper semantic understanding than even BERT with Livnium dynamics currently achieves.

---

## Chapter 9: What 82.79% Means

82.79% is not state-of-the-art. A BERT + linear head with fine-tuning scores ~90%. Livnium with joint BERT is still 7 points behind the baseline it is trying to surpass.

But the comparison is misleading. A BERT + linear head is a 110-million-parameter black box. You put in a sentence pair and get out three logits. You cannot ask "why" in any meaningful geometric sense.

Livnium at 82.79% gives you:

- The full collapse trajectory (6 states, each with alignment scores to all anchors)
- The basin assignment at each step (which attractor is currently winning)
- The tension at each step (how far from equilibrium)
- The head-dynamics agreement (does the physics match the classification)
- The energy at each step (is the system actually descending the landscape)

This is interpretability that no attention map can provide. Attention tells you which tokens the model looked at. The Livnium trace tells you *what the model decided at each step and why the geometry favored that decision*.

The question is not "can Livnium beat BERT+linear?" — that would miss the point. The question is "how close to the black-box ceiling can Livnium get while remaining fully interpretable?" 82.79% says: close. Closer than anyone expected from a physics-based system that was at 56% six months ago.

---

*Next: Page 9 — The Tunnel Test (the universal fixed point, and why every trajectory converges to the same place)*

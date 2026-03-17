# Page 1 — What is Livnium?

## Chapter 1: The Big Idea

Most neural networks classify things by learning a boundary.
You train a model, it learns a line (or a curve, or a surface) that separates class A from class B, and at inference time you check which side of the line a new input falls on.

Livnium does something different.

**Livnium classifies by simulating movement.**

Instead of learning a boundary, it builds a space with three gravitational wells — one for *Entailment*, one for *Contradiction*, one for *Neutral*. When you give it a sentence pair, it converts it into a point in that space and then lets the point *fall* toward whichever well is pulling it hardest. Where it lands is the answer.

This isn't metaphor. The update equations literally look like physics:

```
h_{t+1} = h_t + learned_delta - force_toward_anchor
```

The "force" is computed from how misaligned the current state is with the nearest attractor. The point moves, step by step, until it converges. The label is wherever it ends up.

---

## Chapter 2: The Task — Natural Language Inference (SNLI)

The dataset Livnium is trained and evaluated on is called **SNLI** (Stanford Natural Language Inference). It contains ~550,000 sentence pairs. Each pair has:

- A **premise** — a sentence describing a scene
- A **hypothesis** — a statement about that scene
- A **label** — one of three:

| Label | Meaning | Example |
|---|---|---|
| Entailment | If premise is true, hypothesis must be true | Premise: *A man is walking a dog.* / Hypothesis: *A person is outside.* |
| Contradiction | If premise is true, hypothesis must be false | Premise: *A man is walking a dog.* / Hypothesis: *There is no one outdoors.* |
| Neutral | Premise neither confirms nor denies hypothesis | Premise: *A man is walking a dog.* / Hypothesis: *The man is enjoying his walk.* |

The model reads premise and hypothesis and predicts one of those three labels. This is the task. Human-level performance is around 89%. A strong BERT + linear head baseline sits around 90%.

---

## Chapter 3: Why Physics?

The natural question is: *why build a physics simulator when you could just train a classifier?*

The answer is about **interpretability and control**.

A standard neural classifier is a black box. You put in a vector, you get out logits, and you have no idea what happened in between. You cannot look at the computation and say "this sentence pair is near the E-C boundary" or "this example needs 3 more collapse steps to stabilize."

Livnium exposes the internal state. At every step you can see:

- **Alignment**: how close is the current state to each anchor? (`cos(h, A_entail)`, etc.)
- **Divergence**: how far is the state from its equilibrium? (`0.38 - alignment`)
- **Tension**: how hard is the state being pulled? (`|divergence|`)
- **Trace**: the full trajectory from h₀ to h_final across all collapse steps

This is the **trace**. It is a record of the geometry the model used to make its decision. No standard neural network gives you this.

Beyond interpretability, the physics framing gives you levers to tune. You can adjust how strong the entailment attractor is (`strength_entail`), where the equilibrium sits (`barrier`), whether states that are near the E-C boundary get pulled toward neutral (`strength_neutral_boost`). Each has a clear geometric meaning.

---

## Chapter 4: The Full Pipeline (One Sentence Pair)

Here is exactly what happens when Livnium processes one sentence pair:

```
Premise:    "A man is sleeping on a bench."
Hypothesis: "A person is resting outdoors."
Label:       Entailment
```

**Step 1 — Encode**

The encoder converts each sentence to a vector.

```
v_p = encode("A man is sleeping on a bench.")     → [256-dim or 768-dim vector]
v_h = encode("A person is resting outdoors.")     → [256-dim or 768-dim vector]
```

**Step 2 — Difference**

The initial state is the *difference* between hypothesis and premise:

```
h0 = v_h - v_p
```

This is the core geometric idea. If the hypothesis is semantically close to the premise, `h0` is near zero. If they point in opposite directions, `h0` is large. The direction of `h0` encodes *how* the hypothesis relates to the premise.

**Step 3 — Collapse**

`h0` is fed into the VectorCollapseEngine, which runs 6 steps of attractor dynamics:

```
For step in 1..6:
    compute alignment with E, N, C anchors
    compute divergence from equilibrium
    apply learned update + anchor forces
    h moves closer to whichever basin is pulling it
```

**Step 4 — Classify**

The final state `h_final`, plus `v_p` and `v_h`, go into the SNLIHead:

```
logits = SNLIHead(h_final, v_p, v_h)
prediction = argmax(logits)   →   Entailment ✓
```

---

## Chapter 5: The Code Map

```
livnium-main/
│
├── data/snli/                    ← SNLI dataset (train/dev/test JSONL)
│
├── system/snli/
│   ├── embed/                    ← Embedding pretraining system (mini-Livnium)
│   │   ├── text_encoder.py       ← PretrainedTextEncoder (loads embeddings_final.pt)
│   │   ├── collapse_engine.py    ← Collapse dynamics used during embedding pretraining
│   │   └── basin_field.py        ← Basin field used during embedding pretraining
│   │
│   └── model/                    ← Main SNLI training system
│       ├── train.py              ← Training loop, all --flags, data loading
│       ├── eval.py               ← Test set evaluation
│       ├── precompute_embeddings.py  ← Pre-cache BERT embeddings for speed
│       │
│       ├── core/
│       │   ├── vector_collapse_engine.py  ← THE ENGINE (attractor dynamics)
│       │   ├── basin_field.py             ← Dynamic basin management
│       │   ├── physics_laws.py            ← divergence, tension, barrier
│       │   └── vector_state.py            ← State vector utilities
│       │
│       └── tasks/snli/
│           ├── encoding_snli.py   ← All encoder types (BoW, pretrained, BERT, llama.cpp)
│           └── head_snli.py       ← SNLIHead (classification from h_final)
│
├── pretrained/                   ← Saved checkpoints (not in git)
└── book/                         ← You are here
```

---

*Next: Page 2 — The Physics (divergence, barrier, tension, and what 0.38 actually means)*

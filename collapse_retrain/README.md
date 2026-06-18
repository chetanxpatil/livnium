# Supervised Collapse Model (v1) — Manual & Guide

This folder contains the label-supervised NLI collapse model, the training scripts, evaluation harnesses, and ablation tools.

---

## 1. How It's Made (The Pipeline)

The model is trained end-to-end on the SNLI dataset (549,367 training pairs) to learn a semantic representation without any transformer attention layers.

1. **Word Embeddings**: Words are mapped to a learned 256-dimensional vector space.
2. **Mean Pooling**: The word vectors for the premise ($u$) and hypothesis ($v$) are mean-pooled.
3. **Difference Vector**: We construct the pair representation as `pair = u - v`.
4. **Attractor Collapse**: The difference vector is evolved through a 4-layer collapse engine (`VectorCollapseEngine`) which warps the vector toward one of three learned semantic anchors: **Entailment (E)**, **Neutral (N)**, and **Contradiction (C)**.
5. **Geometric Classification**: The logits are the cosine similarity (dot product) of the normalized collapsed vector against the three normalized anchors. No dense head classifier is used.
6. **Separation Loss**: Cross-entropy classification loss is combined with an explicit separation penalty that keeps the E, N, and C anchors separated.

---

## 2. Where Are the Checkpoints?

We have packaged the official checkpoint inside the repository:
* **`model_nli_v1/nli_epoch20.pt`**: The optimal dev-selected checkpoint from our 100-epoch training run (retained Epoch 23, mapped to `nli_epoch20.pt` for compatibility) that achieves **68.87% SNLI test accuracy** (and **69.76% dev accuracy**), clearing the hypothesis-only artifact baseline (61.5%).

---

## 3. How to Use It (CLI Commands)

Make sure you have installed the dependencies:
```bash
pip install "livnium-core[experimental]"
```

### Run a Single Prediction
To run a classification for a single sentence pair:
```bash
python3 predict.py \
    --ckpt model_nli_v1/nli_epoch20.pt \
    --premise "A man is playing a guitar on stage." \
    --hypothesis "A person is performing music."
```

### Run Full Evaluation
To evaluate the checkpoint accuracy and print the confusion matrix on a local SNLI dataset:
```bash
python3 eval_nli.py \
    --ckpt model_nli_v1/nli_epoch20.pt \
    --data /path/to/snli_1.0_test.jsonl
```

### Run the Ablation Study
To run the ablation suite comparing the collapse engine against standard linear/MLP classifiers, randomized anchors, and random embedding baselines:
```bash
python3 ablate_nli.py \
    --ckpt model_nli_v1/nli_epoch20.pt \
    --train-data /path/to/snli_1.0_train.jsonl \
    --dev-data /path/to/snli_1.0_dev.jsonl \
    --test-data /path/to/snli_1.0_test.jsonl
```

### Extract Failure Cases
To find and log all test examples classified incorrectly by the model (writes to `failed_examples.json`):
```bash
python3 save_failures.py
```

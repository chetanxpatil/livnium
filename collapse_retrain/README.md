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

---

## 4. Benchmark & Performance Statistics

The model is highly efficient due to its attention-free architecture (linear $O(L)$ pooling and constant $O(1)$ collapse warping). Measured on macOS (Apple Silicon M-series):

### Throughput & Latency

| Batch Size | CPU Latency | CPU Throughput | MPS GPU Latency | MPS GPU Throughput |
| :---: | :---: | :---: | :---: | :---: |
| **1** | **0.33 ms** | 3,015 pairs/sec | **1.92 ms** | 521 pairs/sec |
| **16** | **0.66 ms** | 24,089 pairs/sec | **2.32 ms** | 6,903 pairs/sec |
| **64** | **1.05 ms** | 61,113 pairs/sec | **2.17 ms** | 29,474 pairs/sec |
| **256** | **4.38 ms** | 58,503 pairs/sec | **2.32 ms** | 110,595 pairs/sec |
| **1024** | **8.35 ms** | 122,677 pairs/sec | **4.74 ms** | **215,886 pairs/sec** |
| **4096** | **48.54 ms** | 84,381 pairs/sec | **19.15 ms** | **213,867 pairs/sec** |

*Note: For single-pair inference, CPU is faster because it avoids GPU memory copying and kernel launch overhead. For large batches, GPU acceleration scales throughput to >215k pairs/sec.*

### Compute Breakdown (Batch Size 256)
- **Embedding Lookup & Pooling**: ~20-30% of total time
- **Vector Collapse Warping (4 layers)**: ~70% of total time
- **Cosine Similarity Classification**: ~1-8% of total time

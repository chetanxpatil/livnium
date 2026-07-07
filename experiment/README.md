# Experiment: Pure Geometric Chat Collapse Engine

This folder contains a 100% parameter-free, MLP-free, and linear-free chat generation pipeline based on your NLI paper's findings.

## Mathematical Core

### 1. Analytical Gradient Collapse
Instead of learning updates using a parameter-heavy MLP, the hidden state $h$ is updated directly by moving along the conservative gradient of the cosine energy potential $V(h) = -\cos(h, T)$ defined over target vectors:

$$\nabla_h V(h) = -\frac{T - h_n \cos(h, T)}{\|h\|}$$

$$h_{next} = h - \alpha \nabla_h V(h)$$

This eliminates all parameter updates in the trajectory path transitions.

### 2. Zero-Parameter Writing Query
The autoregressive decoding step builds the search query vector $q_t$ purely from vector additions and raw cosine softmax alignments:

$$q_t = \text{normalize}\big(h_t + z + \text{pos\_anchor}[t] + \text{align\_context}_t\big)$$

* **$h_t$**: Current generator state (representing already generated tokens).
* **$z$**: Topic vector, set directly to the final context state ($z = h_{read}$).
* **$\text{pos\_anchor}$**: Static position coordinates (prevents local bigram looping).
* **$\text{align\_context}$**: Context alignment vector computed by raw cosine similarities against context word wells.

---

## Running the Pipeline

### 1. Link Context Data
Ensure the chat database is linked to the experiment folder:
```bash
python3 experiment/prep_data.py
```

### 2. Train the Model (Pure Mode)
Train the Pure Geometric generator:
```bash
python3 experiment/pure_reply.py --epochs 50 --device mps
```
This will save checkpoints locally to `experiment/model/chat_reply_pure.pt`.

### 3. Talk to the Model
Run the interactive chat console:
```bash
python3 experiment/pure_reply.py --chat --ckpt experiment/model/chat_reply_pure.pt
```

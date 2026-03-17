"""
Pre-compute BERT embeddings for all SNLI sentences via llama.cpp.

Run this ONCE before training. Every sentence in the dataset is embedded
and saved to a .pt cache file. Training then does pure tensor lookups —
no BERT forward passes in the training loop.

Why: BERT is frozen, so its outputs are identical every epoch. Pre-computing
saves 3x (or more) the embedding work, turning a 5-hour training run into
~15 minutes.

Usage:
    python precompute_embeddings.py \\
        --snli-train  ../../../data/snli/snli_1.0_train.jsonl \\
        --snli-dev    ../../../data/snli/snli_1.0_dev.jsonl \\
        --llamacpp-model ~/models/bert-gguf/bert-base-uncased-Q8_0.gguf \\
        --output      ../../../pretrained/bert-llamacpp/embeddings_cache.pt

Then train with:
    python train.py --encoder-type llamacpp \\
                    --llamacpp-model ~/models/bert-gguf/bert-base-uncased-Q8_0.gguf \\
                    --embed-cache ../../../pretrained/bert-llamacpp/embeddings_cache.pt \\
                    ...
"""

import json
import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm


def load_unique_sentences(jsonl_paths):
    """Collect all unique sentence strings from one or more SNLI JSONL files."""
    sentences = set()
    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                gold = data.get("gold_label", "").strip()
                if gold not in ("entailment", "contradiction", "neutral"):
                    continue
                s1 = data.get("sentence1", "").strip()
                s2 = data.get("sentence2", "").strip()
                if s1:
                    sentences.add(s1)
                if s2:
                    sentences.add(s2)
    return sorted(sentences)  # sorted for reproducibility


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute BERT embeddings for all SNLI sentences."
    )
    parser.add_argument("--snli-train", required=True, help="Path to SNLI train JSONL")
    parser.add_argument("--snli-dev",   required=True, help="Path to SNLI dev JSONL")
    parser.add_argument("--snli-test",  default=None,  help="Path to SNLI test JSONL (optional)")
    parser.add_argument(
        "--llamacpp-model", required=True,
        help="Path to BERT GGUF file (e.g. bert-base-uncased-Q8_0.gguf)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to save the cache .pt file"
    )
    parser.add_argument(
        "--n-ctx", type=int, default=512,
        help="Context window for llama.cpp (default: 512)"
    )
    args = parser.parse_args()

    # ── Collect all unique sentences ──────────────────────────────────────────
    paths = [Path(args.snli_train), Path(args.snli_dev)]
    if args.snli_test:
        paths.append(Path(args.snli_test))

    print("Scanning SNLI files for unique sentences...")
    sentences = load_unique_sentences(paths)
    print(f"  Found {len(sentences):,} unique sentences across {len(paths)} splits")

    # ── Load BERT via llama.cpp ───────────────────────────────────────────────
    print(f"\nLoading BERT from {args.llamacpp_model} ...")
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python is not installed.\n"
            "Install with: pip install llama-cpp-python"
        )

    llm = Llama(
        model_path=args.llamacpp_model,
        embedding=True,
        n_ctx=args.n_ctx,
        n_batch=args.n_ctx,   # match n_ctx so full sequences fit in one decode
        verbose=False,
    )

    # Probe dim
    _probe = llm.embed("probe")
    dim = len(_probe[0])
    print(f"  BERT embedding dim = {dim}")

    # ── Embed all sentences ───────────────────────────────────────────────────
    print(f"\nEmbedding {len(sentences):,} sentences (one-time cost) ...")
    t0 = time.time()

    cache = {}
    for sent in tqdm(sentences, desc="Embedding", unit="sent"):
        token_embs = llm.embed(sent)          # (n_tokens, dim) as nested lists
        cls_vec = torch.tensor(token_embs[0], dtype=torch.float16)  # save as fp16 to halve disk size
        cache[sent] = cls_vec

    elapsed = time.time() - t0
    rate = len(sentences) / elapsed
    print(f"\n  Done in {elapsed/60:.1f} min  ({rate:.0f} sentences/sec)")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)

    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved to {out_path}  ({size_mb:.0f} MB,  {len(cache):,} entries)")
    print("\nNow train with:  --embed-cache", out_path)


if __name__ == "__main__":
    main()

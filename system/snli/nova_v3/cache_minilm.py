"""
cache_minilm.py — One-time MiniLM embedding precomputation.
============================================================
Runs sentence-transformers/all-MiniLM-L6-v2 ONCE over SNLI train+dev,
saves v_p and v_h as float16 tensors to disk.

After this, training never touches MiniLM again — it just reads cached
tensors. This makes the training loop as fast as the quantum-cache runs.

Usage:
    cd ~/Documents/livnium-sacred
    python system/snli/nova_v3/cache_minilm.py \\
        --snli-train data/snli/snli_1.0_train.jsonl \\
        --snli-dev   data/snli/snli_1.0_dev.jsonl \\
        --out-dir    minilm-cache

Output files:
    minilm-cache/train.pt  — { 'v_p': (N,384), 'v_h': (N,384), 'labels': (N,) }
    minilm-cache/dev.pt    — same

Time: ~5-10 min on CPU  (batch-size=256, vectorized)
Size: ~500 MB total (train=440k samples × 384 × 2 × 2 bytes)
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ── path setup ─────────────────────────────────────────────────────────────────
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LABEL_MAP = {"entailment": 0, "contradiction": 1, "neutral": 2}


# ── Data loading ───────────────────────────────────────────────────────────────
def load_snli_data(path: Path, max_samples: Optional[int] = None) -> List[Dict]:
    samples, seen = [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            lbl = d.get("gold_label", "").strip()
            if lbl not in LABEL_MAP:
                continue
            prem = d.get("sentence1", "").strip()
            hyp  = d.get("sentence2", "").strip()
            if not prem or not hyp:
                continue
            key = (prem, hyp)
            if key in seen and seen[key] != lbl:
                continue
            seen[key] = lbl
            samples.append({"premise": prem, "hypothesis": hyp, "label": LABEL_MAP[lbl]})
            if max_samples and len(samples) >= max_samples:
                break
    return samples


# ── Encoding ───────────────────────────────────────────────────────────────────
def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings over non-padding positions."""
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-8)


@torch.no_grad()
def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 256,
    max_length: int = 128,
) -> torch.Tensor:
    """Encode a list of strings → (N, hidden_dim) float16 tensor."""
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  encoding", leave=False):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        vecs = mean_pool(out.last_hidden_state, enc["attention_mask"])
        all_vecs.append(vecs.cpu().to(torch.float16))
    return torch.cat(all_vecs, dim=0)


# ── Main ───────────────────────────────────────────────────────────────────────
def cache_split(
    split_name: str,
    samples: List[Dict],
    tokenizer,
    model,
    device: torch.device,
    out_dir: Path,
    batch_size: int,
):
    print(f"\n── {split_name}  ({len(samples):,} samples) ──────────────────────")

    premises    = [s["premise"]    for s in samples]
    hypotheses  = [s["hypothesis"] for s in samples]
    labels      = torch.tensor([s["label"] for s in samples], dtype=torch.int16)

    print("  Encoding premises...")
    v_p = encode_texts(premises,   tokenizer, model, device, batch_size)
    print("  Encoding hypotheses...")
    v_h = encode_texts(hypotheses, tokenizer, model, device, batch_size)

    hidden_dim = v_p.shape[1]
    out_path = out_dir / f"{split_name}.pt"
    torch.save({
        "v_p":        v_p,       # (N, hidden_dim) float16
        "v_h":        v_h,       # (N, hidden_dim) float16
        "labels":     labels,    # (N,)            int16
        "hidden_dim": hidden_dim,
        "n":          len(samples),
        "model":      MINILM_MODEL,
    }, out_path)

    size_mb = out_path.stat().st_size / 1e6
    print(f"  ✓ Saved {out_path}  ({size_mb:.0f} MB)  dim={hidden_dim}")


def main():
    parser = argparse.ArgumentParser(description="Cache MiniLM embeddings for SNLI")
    parser.add_argument("--snli-train",  required=True)
    parser.add_argument("--snli-dev",    default=None)
    parser.add_argument("--out-dir",     default="minilm-cache")
    parser.add_argument("--batch-size",  type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit train samples (for quick testing)")
    parser.add_argument("--model",       default=MINILM_MODEL,
                        help="HuggingFace model name (default: all-MiniLM-L6-v2)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {args.model} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModel.from_pretrained(args.model).to(device).eval()

    hidden_dim = model.config.hidden_size
    print(f"Model hidden dim: {hidden_dim}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SNLI train...")
    train_samples = load_snli_data(Path(args.snli_train), args.max_samples)
    cache_split("train", train_samples, tokenizer, model, device, out_dir, args.batch_size)

    if args.snli_dev:
        print("Loading SNLI dev...")
        dev_samples = load_snli_data(Path(args.snli_dev))
        cache_split("dev", dev_samples, tokenizer, model, device, out_dir, args.batch_size)

    print(f"\n✓ Done. Cache ready in: {out_dir}/")
    print("Now run train_snli_minilm_fast.py with --minilm-cache", out_dir)


if __name__ == "__main__":
    main()

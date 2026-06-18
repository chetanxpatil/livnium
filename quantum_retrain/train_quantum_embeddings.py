"""
Quantum-Inspired Embedding Trainer (Livnium v0.1) [M5 Optimized]

Trains word embeddings on WikiText-103 using a Livnium-style energy.
Optimized for Apple Silicon (MPS) with memory pinning and parallel loading.

UPDATED: Supports "Static Collapse" mode where physics runs (on GPU) 
but dynamic spawning (CPU bottleneck) is disabled.
"""

import os
import argparse
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from vector_collapse import VectorCollapseEngine
from basin_field import BasinField


class Vocab:
    def __init__(self, max_size: int = 50000, min_freq: int = 1):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []
        self.freqs: Dict[str, int] = {}
        self.special_tokens = ["<pad>", "<unk>"]
        for tok in self.special_tokens:
            self._add(tok)

    def _add(self, token: str):
        if token not in self.word2idx:
            idx = len(self.idx2word)
            self.word2idx[token] = idx
            self.idx2word.append(token)

    def add_tokens_from_line(self, line: str):
        for tok in line.strip().split():
            if not tok:
                continue
            self.freqs[tok] = self.freqs.get(tok, 0) + 1

    def build(self):
        sorted_items = sorted(self.freqs.items(), key=lambda x: -x[1])
        for tok, freq in sorted_items:
            if freq < self.min_freq:
                continue
            if tok in self.word2idx:
                continue
            if len(self.idx2word) >= self.max_size:
                break
            self._add(tok)

    @property
    def pad_idx(self) -> int:
        return self.word2idx["<pad>"]

    @property
    def unk_idx(self) -> int:
        return self.word2idx["<unk>"]

    def __len__(self) -> int:
        return len(self.idx2word)

    def encode_line(self, line: str) -> List[int]:
        return [self.word2idx.get(tok, self.unk_idx) for tok in line.strip().split() if tok]


class SkipGramDataset(Dataset):
    def __init__(self, sequences: List[List[int]], window_size: int = 2):
        self.pairs: List[Tuple[int, int]] = []
        print(f"[SkipGramDataset] generating pairs from {len(sequences)} sequences...")
        for seq in sequences:
            for i, c in enumerate(seq):
                if c == 0:
                    continue
                left = max(0, i - window_size)
                right = min(len(seq), i + window_size + 1)
                for j in range(left, right):
                    if j == i:
                        continue
                    self.pairs.append((c, seq[j]))
        print(f"[SkipGramDataset] total pairs: {len(self.pairs)}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        return self.pairs[idx]


class QuantumEmbeddingModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256, pad_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=pad_idx)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.05)

    def forward(self, idxs: torch.Tensor) -> torch.Tensor:
        return self.emb(idxs)


def livnium_energy_loss(
    model: QuantumEmbeddingModel,
    centers: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    d_margin: float = 0.4,
    neg_weight: float = 5.0,
    norm_reg_weight: float = 1e-4,
    collapse_engine: VectorCollapseEngine = None,
    basin_field: BasinField = None,
    use_dynamic_basins: bool = False,
    global_step: int = 0,
    spawn_new: bool = True,
    prune_every: int = 0,
) -> torch.Tensor:
    
    # 1. Look up raw embeddings
    v_c = model(centers)
    v_p = model(positives)
    v_n = model(negatives)

    # 2. Apply Collapse Physics (If engine exists)
    if use_dynamic_basins and collapse_engine is not None and basin_field is not None:
        # --- DYNAMIC MODE (Slower, Adaptive Topology) ---
        labels_pos = torch.zeros(centers.size(0), dtype=torch.long, device=centers.device)
        labels_neg = torch.ones(centers.size(0), dtype=torch.long, device=centers.device)

        v_c_pos, _ = collapse_engine.collapse_dynamic(
            v_c, labels_pos, basin_field, global_step=global_step,
            spawn_new=spawn_new, prune_every=prune_every, update_anchors=True,
        )
        v_p, _ = collapse_engine.collapse_dynamic(
            v_p, labels_pos, basin_field, global_step=global_step,
            spawn_new=spawn_new, prune_every=0, update_anchors=True,
        )
        v_c_neg, _ = collapse_engine.collapse_dynamic(
            v_c.detach(), labels_neg, basin_field, global_step=global_step,
            spawn_new=spawn_new, prune_every=0, update_anchors=True,
        )
        v_n, _ = collapse_engine.collapse_dynamic(
            v_n, labels_neg, basin_field, global_step=global_step,
            spawn_new=spawn_new, prune_every=prune_every, update_anchors=True,
        )
        v_c = v_c_pos
        v_c_for_neg = v_c_neg

    elif collapse_engine is not None:
        # --- STATIC MODE (Fast, GPU-friendly Physics) ---
        # No basin field, just pure Vector Collapse mechanics
        # Collapses vectors towards the 3 learned static anchors (E/N/C)
        v_c, _ = collapse_engine(v_c)
        v_p, _ = collapse_engine(v_p)
        v_n, _ = collapse_engine(v_n)
        v_c_for_neg = v_c
        
    else:
        # --- RAW MODE (Standard Skip-Gram) ---
        v_c_for_neg = v_c

    # 3. Compute Energy
    v_c_n = F.normalize(v_c, dim=-1)
    v_p_n = F.normalize(v_p, dim=-1)
    v_n_n = F.normalize(v_n, dim=-1)
    v_c_neg_n = F.normalize(v_c_for_neg, dim=-1)

    a_pos = (v_c_n * v_p_n).sum(dim=-1)
    a_neg = (v_c_neg_n * v_n_n).sum(dim=-1)

    d_pos = 0.38 - a_pos
    d_neg = 0.38 - a_neg

    E_pos = d_pos.pow(2)
    diff = torch.clamp(d_margin - d_neg, min=0.0)
    E_neg = diff.pow(2)

    norm = v_c.norm(dim=-1) + v_p.norm(dim=-1) + v_n.norm(dim=-1)
    norm_reg = norm.mean()

    loss = E_pos.mean() + neg_weight * E_neg.mean() + norm_reg_weight * norm_reg
    return loss


# ---------------------------------------------------------------------------
# NLI-supervised training.
#
# The skip-gram objective above never sees an NLI label, so the E/C/N anchors
# cannot specialize (proven forward-only: the static collapse is label-blind and
# anchor_E / anchor_C are interchangeable). This path fixes that: each example
# carries its gold label, the loss is cross-entropy of the collapsed pair vector
# against the three anchors (which forces label conditioning AND distinguishes
# E from C), plus an explicit separation penalty on the anchors themselves.
# Anchor order is fixed as [Entailment, Neutral, Contradiction] everywhere here.
# ---------------------------------------------------------------------------

NLI_LABEL_TO_IDX = {"entailment": 0, "neutral": 1, "contradiction": 2}


def read_nli_jsonl(path: str, max_lines: int = 0):
    """Yield (sentence1, sentence2, label_idx) from an SNLI/MultiNLI .jsonl,
    skipping examples whose gold_label is '-' (no annotator consensus)."""
    import json
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and len(out) >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            lab = d.get("gold_label", "-")
            if lab not in NLI_LABEL_TO_IDX:
                continue
            s1 = (d.get("sentence1") or "").lower()
            s2 = (d.get("sentence2") or "").lower()
            if s1 and s2:
                out.append((s1, s2, NLI_LABEL_TO_IDX[lab]))
    return out


class NLIDataset(Dataset):
    def __init__(self, examples, vocab: Vocab):
        self.data = []
        for s1, s2, y in examples:
            p = vocab.encode_line(s1) or [vocab.unk_idx]
            h = vocab.encode_line(s2) or [vocab.unk_idx]
            self.data.append((p, h, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def nli_collate(batch, pad_idx: int):
    pms, hys, ys = zip(*batch)
    mp = max(len(x) for x in pms)
    mh = max(len(x) for x in hys)
    def pad(seqs, m):
        return torch.tensor([s + [pad_idx] * (m - len(s)) for s in seqs], dtype=torch.long)
    return pad(pms, mp), pad(hys, mh), torch.tensor(ys, dtype=torch.long)


def _meanpool(model, ids, pad_idx):
    emb = model.emb(ids)                          # (B, T, dim)
    mask = (ids != pad_idx).float().unsqueeze(-1)
    return (emb * mask).sum(1) / mask.sum(1).clamp(min=1.0)


def _anchor_matrix(collapse_engine):
    """Stack the three anchors in fixed [E, N, C] order, unit-normalized."""
    return torch.stack([
        F.normalize(collapse_engine.anchor_entail, dim=0),
        F.normalize(collapse_engine.anchor_neutral, dim=0),
        F.normalize(collapse_engine.anchor_contra, dim=0),
    ])


def nli_supervised_loss(model, collapse_engine, prem, hyp, gold, pad_idx,
                        lambda_sep: float = 1.0, sep_margin: float = 0.0,
                        temp: float = 0.1):
    """Cross-entropy of the collapsed pair vector against the 3 anchors, plus an
    explicit anchor-separation penalty. Returns (loss, logits)."""
    u = _meanpool(model, prem, pad_idx)
    v = _meanpool(model, hyp, pad_idx)
    pair = u - v                                   # hypothesis relative to premise
    if collapse_engine is not None:
        pair, _ = collapse_engine(pair)            # warp toward the anchor geometry
    pair_n = F.normalize(pair, dim=-1)

    anchors = _anchor_matrix(collapse_engine)      # (3, dim) -> [E, N, C]
    logits = (pair_n @ anchors.t()) / temp         # (B, 3)
    ce = F.cross_entropy(logits, gold)

    # Explicit separation: penalize pairwise cosine above the margin.
    cE, cN, cC = anchors[0], anchors[1], anchors[2]
    sep = (torch.relu((cE * cN).sum() - sep_margin)
           + torch.relu((cE * cC).sum() - sep_margin)
           + torch.relu((cN * cC).sum() - sep_margin))
    return ce + lambda_sep * sep, logits


def build_vocab_and_sequences(path: str, max_lines: int, max_size: int) -> Tuple[Vocab, List[List[int]]]:
    vocab = Vocab(max_size=max_size, min_freq=2)
    print(f"[build_vocab] scanning {path}...")
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            vocab.add_tokens_from_line(line)
    vocab.build()
    print(f"[build_vocab] vocab size: {len(vocab)}")
    sequences: List[List[int]] = [vocab.encode_line(line) for line in lines]
    print(f"[build_vocab] sequences: {len(sequences)}")
    return vocab, sequences


def sample_negative(batch_size: int, vocab_size: int, pad_idx: int, device: torch.device) -> torch.Tensor:
    neg = torch.randint(low=0, high=vocab_size, size=(batch_size,), device=device)
    neg = torch.where(neg == pad_idx, (neg + 1) % vocab_size, neg)
    return neg


def verify_engine_trained(collapse_engine, init_anchors, sep_threshold: float, epoch: int):
    """Tripwire: prove the collapse engine is (a) actually being optimized and
    (b) forming distinct E/C/N geometry. Raises RuntimeError on failure so a
    silent regression (e.g. the engine detached from the optimizer, or the loss
    not rewarding separation) cannot pass unnoticed. See COLLAPSE_ENGINE_VERDICT.md.
    """
    import torch.nn.functional as _F
    sd = collapse_engine.state_dict()
    names = ("anchor_entail", "anchor_contra", "anchor_neutral")

    # (a) anchors must have moved from initialization -> optimizer is connected
    moved = {n: float((sd[n] - init_anchors[n]).abs().max()) for n in names}
    print(f"[verify] epoch {epoch}: max|Δanchor| = "
          + ", ".join(f"{n.split('_')[1]}={moved[n]:.5f}" for n in names))
    if max(moved.values()) < 1e-6:
        raise RuntimeError(
            "Collapse engine did not change during epoch %d — its parameters are "
            "NOT being optimized. Check that collapse_engine.parameters() were "
            "passed to the optimizer." % epoch
        )

    # (b) anchors must be separating, not clumping into a degenerate point
    e, c, n = (_F.normalize(sd[k], dim=0) for k in names)
    cec = float((e * c).sum()); cen = float((e * n).sum()); ccn = float((c * n).sum())
    print(f"[verify] epoch {epoch}: cos(E,C)={cec:+.3f} cos(E,N)={cen:+.3f} cos(C,N)={ccn:+.3f} "
          f"(threshold {sep_threshold:+.2f})")
    worst = max(cec, cen, ccn)
    if worst >= sep_threshold:
        raise RuntimeError(
            "E/C/N anchors are not separating (max pairwise cos=%.3f >= %.3f) at "
            "epoch %d. The loss is not rewarding anchor separation." % (worst, sep_threshold, epoch)
        )


def train(
    train_path: str, output_dir: str, dim: int = 256, max_vocab: int = 50000,
    max_lines: int = 200000, window_size: int = 2, batch_size: int = 1024,
    epochs: int = 3, lr: float = 3e-4, device: str = "auto",
    disable_dynamic_basins: bool = False, collapse_layers: int = 4,
    strength_entail: float = 0.1, strength_contra: float = 0.1,
    strength_neutral: float = 0.05, basin_max_per_label: int = 64,
    basin_tension_threshold: float = 0.15, basin_align_threshold: float = 0.6,
    basin_anchor_lr: float = 0.05, basin_prune_every: int = 0,
    basin_prune_min_count: int = 10, basin_merge_cos_threshold: float = 0.97,
    verify_engine: bool = False, sep_threshold: float = 0.5,
):
    os.makedirs(output_dir, exist_ok=True)
    if device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Device Activated: {device}")
    else:
        device = torch.device(device)

    vocab, sequences = build_vocab_and_sequences(train_path, max_lines=max_lines, max_size=max_vocab)
    dataset = SkipGramDataset(sequences, window_size=window_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    model = QuantumEmbeddingModel(vocab_size=len(vocab), dim=dim, pad_idx=vocab.pad_idx).to(device)
    # NOTE: optimizer is built AFTER the collapse engine so its parameters are
    # actually optimized. Previously Adam(model.parameters()) silently excluded
    # the collapse engine (a separate module), leaving its anchors + MLP frozen
    # at random init for the entire run. See COLLAPSE_ENGINE_VERDICT.md.

    use_dynamic_basins = not disable_dynamic_basins
    basin_field = BasinField(max_basins_per_label=basin_max_per_label).to(device) if use_dynamic_basins else None
    
    # Initialize Collapse Engine ALWAYS if layers > 0
    # This allows "Static Collapse" even if Dynamic Basins are disabled
    collapse_engine = None
    if collapse_layers > 0:
        collapse_engine = VectorCollapseEngine(
            dim=dim, num_layers=collapse_layers,
            strength_entail=strength_entail, strength_contra=strength_contra,
            strength_neutral=strength_neutral,
            basin_tension_threshold=basin_tension_threshold,
            basin_align_threshold=basin_align_threshold,
            basin_anchor_lr=basin_anchor_lr,
            basin_prune_min_count=basin_prune_min_count,
            basin_prune_merge_cos=basin_merge_cos_threshold,
        ).to(device)
        print(f"[init] Collapse Engine Active (Layers: {collapse_layers}, Dynamic: {use_dynamic_basins})")

    # Build the optimizer over ALL learnable parameters: the embedding model
    # AND the collapse engine (anchors + residual MLP). BasinField holds only
    # buffers (centers/active/counts), so it has no parameters to optimize.
    trainable = list(model.parameters())
    if collapse_engine is not None:
        trainable += list(collapse_engine.parameters())
    optimizer = torch.optim.Adam(trainable, lr=lr)
    print(f"[init] optimizer tracking {sum(p.numel() for p in trainable):,} params "
          f"across {len(trainable)} tensors")

    # Snapshot initial anchors so the tripwire can confirm they actually moved.
    init_anchors = None
    if verify_engine and collapse_engine is not None:
        sd0 = collapse_engine.state_dict()
        init_anchors = {n: sd0[n].detach().clone()
                        for n in ("anchor_entail", "anchor_contra", "anchor_neutral")}

    print(f"[train] device={device}, vocab={len(vocab)}, dim={dim}, batches={len(loader)}")

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=True, unit="batch")

        for step, (centers, contexts) in enumerate(pbar):
            centers = centers.to(device, non_blocking=True)
            contexts = contexts.to(device, non_blocking=True)
            negatives = sample_negative(centers.size(0), len(vocab), vocab.pad_idx, device)

            optimizer.zero_grad()
            loss = livnium_energy_loss(
                model, centers, contexts, negatives,
                collapse_engine=collapse_engine,
                basin_field=basin_field,
                use_dynamic_basins=use_dynamic_basins,
                global_step=global_step,
                spawn_new=use_dynamic_basins,
                prune_every=basin_prune_every,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += centers.size(0)
            pbar.set_postfix(loss=f"{total_loss / (step + 1):.4f}", step=global_step)

        ckpt_path = os.path.join(output_dir, f"quantum_embed_epoch{epoch}.pt")
        torch.save({
            "state_dict": model.state_dict(),
            "vocab": {"idx2word": vocab.idx2word, "pad_idx": vocab.pad_idx, "unk_idx": vocab.unk_idx},
            "dim": dim,
            "use_dynamic_basins": use_dynamic_basins,
            "collapse_engine": collapse_engine.state_dict() if collapse_engine else None,
            "basin_field": basin_field.state_dict() if basin_field else None,
            "collapse_config": {
                "num_layers": collapse_layers, "strength_entail": strength_entail,
                "strength_contra": strength_contra, "strength_neutral": strength_neutral
            },
        }, ckpt_path)
        print(f"[train] saved {ckpt_path}")

        # Tripwire: after the first epoch, prove the engine is training and the
        # E/C/N anchors are separating. Fails loud rather than silently.
        if verify_engine and collapse_engine is not None and init_anchors is not None:
            verify_engine_trained(collapse_engine, init_anchors, sep_threshold, epoch)

        if use_dynamic_basins and basin_field:
            counts = {l: len(basin_field.anchors[l]) for l in ["E", "N", "C"]}
            print(f"[train] basin counts E/N/C: {counts['E']} / {counts['N']} / {counts['C']}")

    final_path = os.path.join(output_dir, "quantum_embeddings_final.pt")
    torch.save({
        "embeddings": model.emb.weight.detach().cpu(),
        "vocab": {"idx2word": vocab.idx2word, "pad_idx": vocab.pad_idx, "unk_idx": vocab.unk_idx},
        "dim": dim,
        "use_dynamic_basins": use_dynamic_basins,
        "collapse_engine": collapse_engine.state_dict() if collapse_engine else None,
        "basin_field": basin_field.state_dict() if basin_field else None,
        # Persist config in the final checkpoint too, so the encoder can rebuild
        # the engine with the right geometry (num_layers, strengths).
        "collapse_config": {
            "num_layers": collapse_layers, "strength_entail": strength_entail,
            "strength_contra": strength_contra, "strength_neutral": strength_neutral,
        },
    }, final_path)
    print(f"[train] saved final embeddings to {final_path}")


def train_nli(
    nli_path: str, output_dir: str, dim: int = 256, max_vocab: int = 50000,
    max_lines: int = 0, batch_size: int = 512, epochs: int = 3, lr: float = 3e-4,
    device: str = "auto", collapse_layers: int = 4,
    strength_entail: float = 0.1, strength_contra: float = 0.1,
    strength_neutral: float = 0.05, lambda_sep: float = 1.0,
    sep_margin: float = 0.0, temp: float = 0.1,
    verify_engine: bool = True, sep_threshold: float = 0.5,
):
    """Label-supervised NLI fine-tune: the anchors learn E/N/C because the loss
    is conditioned on the gold label. This is the path that makes the Epoch-1
    separation tripwire meaningful."""
    os.makedirs(output_dir, exist_ok=True)
    if device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"🚀 Device: {device}")

    examples = read_nli_jsonl(nli_path, max_lines=max_lines)
    print(f"[nli] loaded {len(examples)} labeled examples from {nli_path}")
    if not examples:
        raise SystemExit("No usable NLI examples (check the path / gold labels).")

    vocab = Vocab(max_size=max_vocab, min_freq=1)
    for s1, s2, _ in examples:
        vocab.add_tokens_from_line(s1)
        vocab.add_tokens_from_line(s2)
    vocab.build()
    print(f"[nli] vocab size: {len(vocab)}")

    dataset = NLIDataset(examples, vocab)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=lambda b: nli_collate(b, vocab.pad_idx),
    )

    model = QuantumEmbeddingModel(vocab_size=len(vocab), dim=dim, pad_idx=vocab.pad_idx).to(device)
    collapse_engine = VectorCollapseEngine(
        dim=dim, num_layers=collapse_layers,
        strength_entail=strength_entail, strength_contra=strength_contra,
        strength_neutral=strength_neutral,
    ).to(device)

    # Correct optimizer wiring: BOTH the embeddings and the collapse engine.
    trainable = list(model.parameters()) + list(collapse_engine.parameters())
    optimizer = torch.optim.Adam(trainable, lr=lr)
    print(f"[init] optimizer tracking {sum(p.numel() for p in trainable):,} params "
          f"across {len(trainable)} tensors")

    init_anchors = None
    if verify_engine:
        sd0 = collapse_engine.state_dict()
        init_anchors = {n: sd0[n].detach().clone()
                        for n in ("anchor_entail", "anchor_contra", "anchor_neutral")}

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train(); collapse_engine.train()
        tot, correct, seen = 0.0, 0, 0
        pbar = tqdm(loader, desc=f"NLI epoch {epoch}/{epochs}", unit="batch")
        for prem, hyp, gold in pbar:
            prem, hyp, gold = prem.to(device), hyp.to(device), gold.to(device)
            optimizer.zero_grad()
            loss, logits = nli_supervised_loss(
                model, collapse_engine, prem, hyp, gold, vocab.pad_idx,
                lambda_sep=lambda_sep, sep_margin=sep_margin, temp=temp,
            )
            loss.backward()
            optimizer.step()
            global_step += 1
            tot += float(loss.item())
            correct += int((logits.argmax(dim=-1) == gold).sum().item())
            seen += gold.size(0)
            pbar.set_postfix(loss=f"{tot/global_step:.4f}", acc=f"{correct/max(seen,1):.3f}")

        print(f"[nli] epoch {epoch}: mean_loss={tot/max(global_step,1):.4f} "
              f"train_acc={correct/max(seen,1):.3f}")

        ckpt = os.path.join(output_dir, f"nli_epoch{epoch}.pt")
        torch.save({
            "embeddings": model.emb.weight.detach().cpu(),
            "vocab": {"idx2word": vocab.idx2word, "pad_idx": vocab.pad_idx, "unk_idx": vocab.unk_idx},
            "dim": dim,
            "use_dynamic_basins": False,
            "collapse_engine": collapse_engine.state_dict(),
            "basin_field": None,
            "collapse_config": {
                "num_layers": collapse_layers, "strength_entail": strength_entail,
                "strength_contra": strength_contra, "strength_neutral": strength_neutral,
            },
        }, ckpt)
        print(f"[nli] saved {ckpt}")

        if verify_engine and init_anchors is not None:
            verify_engine_trained(collapse_engine, init_anchors, sep_threshold, epoch)


def main():
    parser = argparse.ArgumentParser(description="Quantum-Inspired Embedding Trainer [M5 Optimized]")
    parser.add_argument("--train-path", type=str, default=None,
                        help="WikiText-style corpus (required for --task skipgram).")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--max-vocab", type=int, default=50000)
    parser.add_argument("--max-lines", type=int, default=200000)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--disable-dynamic-basins", action="store_true", help="Disable dynamic spawning.")
    parser.add_argument("--collapse-layers", type=int, default=4, help="Number of collapse steps.")
    # ... other args same as before ...
    # Simplified parser block for brevity, ensure all args from before are included
    parser.add_argument("--strength-entail", type=float, default=0.1)
    parser.add_argument("--strength-contra", type=float, default=0.1)
    parser.add_argument("--strength-neutral", type=float, default=0.05)
    parser.add_argument("--basin-max-per-label", type=int, default=64)
    parser.add_argument("--basin-tension-threshold", type=float, default=0.15)
    parser.add_argument("--basin-align-threshold", type=float, default=0.6)
    parser.add_argument("--basin-anchor-lr", type=float, default=0.05)
    parser.add_argument("--basin-prune-every", type=int, default=0)
    parser.add_argument("--basin-prune-min-count", type=int, default=10)
    parser.add_argument("--basin-merge-cos-threshold", type=float, default=0.97)
    parser.add_argument("--verify-engine-trained", dest="verify_engine", action="store_true",
                        help="After each epoch, assert the collapse engine is training "
                             "and the E/C/N anchors are separating (fails loud otherwise).")
    parser.add_argument("--sep-threshold", type=float, default=0.5,
                        help="Max allowed pairwise cosine between E/C/N anchors for the tripwire.")
    # Task selection: skip-gram (default) or label-supervised NLI.
    parser.add_argument("--task", choices=["skipgram", "nli"], default="skipgram",
                        help="skipgram = original WikiText objective; "
                             "nli = label-supervised E/N/C training on an SNLI/MultiNLI .jsonl.")
    parser.add_argument("--nli-path", type=str, default=None,
                        help="Path to an SNLI/MultiNLI .jsonl (required for --task nli).")
    parser.add_argument("--lambda-sep", type=float, default=1.0,
                        help="Weight on the anchor-separation penalty (nli task).")
    parser.add_argument("--sep-margin", type=float, default=0.0,
                        help="Allowed pairwise anchor cosine before the separation penalty bites.")
    parser.add_argument("--temp", type=float, default=0.1,
                        help="Temperature for the anchor-logit softmax (nli task).")

    args = parser.parse_args()
    d = vars(args)
    task = d.pop("task")
    nli_path = d.pop("nli_path")
    lambda_sep = d.pop("lambda_sep"); sep_margin = d.pop("sep_margin"); temp = d.pop("temp")

    if task == "nli":
        if not nli_path:
            parser.error("--task nli requires --nli-path pointing to an SNLI/MultiNLI .jsonl")
        train_nli(
            nli_path=nli_path, output_dir=d["output_dir"], dim=d["dim"],
            max_vocab=d["max_vocab"], max_lines=d["max_lines"], batch_size=d["batch_size"],
            epochs=d["epochs"], lr=d["lr"], device=d["device"], collapse_layers=d["collapse_layers"],
            strength_entail=d["strength_entail"], strength_contra=d["strength_contra"],
            strength_neutral=d["strength_neutral"], lambda_sep=lambda_sep, sep_margin=sep_margin,
            temp=temp, verify_engine=d["verify_engine"], sep_threshold=d["sep_threshold"],
        )
    else:
        if not d.get("train_path"):
            parser.error("--task skipgram requires --train-path")
        # skip-gram path: drop NLI-only knobs that train() doesn't accept.
        train(**d)

if __name__ == "__main__":
    main()
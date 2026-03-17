"""
Livnium SNLI Training Script
=============================
Pipeline:
  pretrained/collapse4/embeddings_final.pt
      └─► PretrainedSNLIEncoder  (mean-pool premise/hypothesis)
              └─► h0 = v_h - v_p  (difference vector)
                      └─► VectorCollapseEngine  (attractor dynamics → E/N/C basin)
                              └─► SNLIHead  (logits)

Run from this directory (model/):
  python train.py \\
      --snli-train  /path/to/snli_1.0_train.jsonl \\
      --snli-dev    /path/to/snli_1.0_dev.jsonl \\
      --encoder-type pretrained \\
      --embed-ckpt /path/to/pretrained/collapse4/embeddings_final.pt \\
      --dim 256 --batch-size 32 --epochs 3 \\
      --output-dir /path/to/output
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

# Self-contained path setup — works from any location
_here = Path(__file__).resolve().parent          # model/
_repo = _here.parent                              # snli/  (contains embed/)
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_repo))

from core import VectorCollapseEngine, BasinField
from tasks.snli import SNLIEncoder, PretrainedSNLIEncoder, QuantumSNLIEncoder, BERTSNLIEncoder, SNLIHead, LinearSNLIHead
from embed.text_encoder import PretrainedTextEncoder, QuantumTextEncoder
from utils.vocab import build_vocab_from_snli


class SNLIDataset(Dataset):
    """SNLI dataset."""
    
    def __init__(self, samples: List[Dict], vocab=None, max_len: int = 128, encode_fn=None):
        """
        Args:
            samples: SNLI examples
            vocab: vocabulary object with .encode(...) (optional if encode_fn supplied)
            max_len: max sequence length for padding/truncation
            encode_fn: callable(text: str, max_len: int) -> List[int] (overrides vocab.encode)
        """
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len
        self.encode_fn = encode_fn
        
        # Label mapping
        self.label_map = {
            'entailment': 0,
            'contradiction': 1,
            'neutral': 2
        }
        if self.vocab is None and self.encode_fn is None:
            raise ValueError("SNLIDataset needs either a vocab or encode_fn")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Encode premise and hypothesis
        encode = self.encode_fn if self.encode_fn is not None else self.vocab.encode
        prem_ids = encode(sample['premise'], max_len=self.max_len)
        hyp_ids = encode(sample['hypothesis'], max_len=self.max_len)
        
        # Label
        label = self.label_map.get(sample['gold_label'], 2)  # Default to neutral
        
        return {
            'prem_ids': torch.tensor(prem_ids, dtype=torch.long),
            'hyp_ids': torch.tensor(hyp_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'premise': sample['premise'],
            'hypothesis': sample['hypothesis'],
            'gold_label': sample['gold_label']
        }


def load_snli_data(jsonl_path: Path, max_samples: Optional[int] = None) -> List[Dict]:
    """Load SNLI data from JSONL file."""
    samples = []
    label_by_pair = {}
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            data = json.loads(line.strip())
            gold_label = data.get('gold_label', '').strip()
            
            # Skip invalid labels
            if gold_label not in ['entailment', 'contradiction', 'neutral'] or gold_label == '-':
                continue
            
            premise = data.get('sentence1', '').strip()
            hypothesis = data.get('sentence2', '').strip()
            
            if not premise or not hypothesis:
                continue
            
            pair = (premise, hypothesis)
            # Skip ambiguous pairs that appear with conflicting labels
            if pair in label_by_pair and label_by_pair[pair] != gold_label:
                continue
            label_by_pair[pair] = gold_label
            
            samples.append({
                'premise': premise,
                'hypothesis': hypothesis,
                'gold_label': gold_label
            })
            
            # Stop once we have collected the desired number of VALID samples
            if max_samples and len(samples) >= max_samples:
                break
    
    return samples


def train_epoch(
    model,
    encoder,
    head,
    dataloader,
    optimizer,
    criterion,
    device,
    *,
    use_dynamic_basins: bool = False,
    basin_field: Optional[BasinField] = None,
    spawn_new: bool = True,
    prune_every: int = 0,
    start_step: int = 0,
    scheduler=None,
    lambda_traj: float = 0.0,
    lambda_fn: float = 0.0,
    lambda_rep: float = 0.0,
    margin_rep: float = 0.3,
):
    """Train for one epoch."""
    model.train()
    encoder.train()
    head.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    global_step = start_step
    
    for batch in tqdm(dataloader, desc="Training"):
        labels = batch['label'].to(device)
        if getattr(encoder, 'is_bert', False):
            h0, v_p, v_h = encoder.build_initial_state(
                batch['premise'], batch['hypothesis'], device=device
            )
        else:
            prem_ids = batch['prem_ids'].to(device)
            hyp_ids = batch['hyp_ids'].to(device)
            h0, v_p, v_h = encoder.build_initial_state(prem_ids, hyp_ids)
        
        if use_dynamic_basins and basin_field is not None:
            # Step 1: update basin geometry using label-guided routing (no gradient).
            # This builds/moves basin centers toward the correct class regions.
            with torch.no_grad():
                model.collapse_dynamic(
                    h0,
                    labels,
                    basin_field,
                    global_step=global_step,
                    spawn_new=spawn_new,
                    prune_every=prune_every,
                    update_anchors=True,
                )
            # Step 2: forward pass using label-free inference routing.
            # Gradient flows through this path so the model learns to work
            # correctly WITHOUT being told the label — closing the train/eval gap.
            h_final, trace = model.collapse_inference(h0, basin_field)
        else:
            h_final, trace = model.collapse(h0)

        # Classify with directional signals
        logits = head(h_final, v_p, v_h)

        # Base cross-entropy loss
        loss = criterion(logits, labels)

        # --- Trajectory direction loss ---
        # Rewards collapse trajectories that moved toward the correct basin anchor.
        # delta = total displacement during collapse.
        # We want cos(delta, correct_anchor) to be high.
        # label_map: 0=entailment, 1=contradiction, 2=neutral
        if lambda_traj > 0.0:
            anchors = torch.stack([
                F.normalize(model.anchor_entail, dim=0),
                F.normalize(model.anchor_contra, dim=0),
                F.normalize(model.anchor_neutral, dim=0),
            ], dim=0)  # (3, dim)
            correct_anchors = anchors[labels]  # (B, dim)
            delta = h_final - h0  # direction traveled during collapse
            L_traj = 1.0 - F.cosine_similarity(delta, correct_anchors.detach(), dim=-1)
            loss = loss + lambda_traj * L_traj.mean()

        # --- False-neutral penalty ---
        # Penalizes placing probability mass on neutral when true label is E or C.
        # This breaks the N-as-hedge behavior: model can no longer park
        # ambiguous E/C cases in neutral for free.
        # neutral = label 2  (N_ID = 2)
        if lambda_fn > 0.0:
            probs = F.softmax(logits, dim=-1)
            p_neutral = probs[:, 2]                        # P(neutral) per sample
            non_neutral_mask = (labels != 2).float()       # 1 for E and C samples
            L_falseN = non_neutral_mask * p_neutral
            loss = loss + lambda_fn * L_falseN.mean()

        # --- Basin repulsion loss (anti-gravity) ---
        # Pulls h_final toward the correct anchor AND pushes it away from
        # wrong anchors. Without repulsion, classes that live close together
        # (neutral vs contradiction) can collapse into each other over time.
        #
        # For each sample: sim(h, correct_anchor) should exceed
        # sim(h, wrong_anchor) by at least `margin_rep`.
        # If not, pay a penalty proportional to the violation.
        #
        # label_map: 0=entailment, 1=contradiction, 2=neutral
        if lambda_rep > 0.0:
            # Build normalized anchor matrix (3, dim)
            if lambda_traj == 0.0:  # reuse if already computed above
                anchors = torch.stack([
                    F.normalize(model.anchor_entail, dim=0),
                    F.normalize(model.anchor_contra, dim=0),
                    F.normalize(model.anchor_neutral, dim=0),
                ], dim=0)
            h_norm = F.normalize(h_final, dim=-1)          # (B, dim)
            sims = h_norm @ anchors.T                      # (B, 3)
            sim_correct = sims.gather(1, labels.unsqueeze(1))  # (B, 1)
            # wrong_mask: 1 for the two wrong anchors, 0 for the correct one
            wrong_mask = torch.ones(labels.size(0), 3, device=device)
            wrong_mask.scatter_(1, labels.unsqueeze(1), 0.0)
            # Margin violation: penalise when wrong anchor is too close
            violations = F.relu(margin_rep + sims - sim_correct) * wrong_mask
            L_rep = violations.sum(dim=1) / 2.0            # mean over 2 wrong anchors
            loss = loss + lambda_rep * L_rep.mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        global_step += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy, global_step


def evaluate(
    model,
    encoder,
    head,
    dataloader,
    device,
    *,
    use_dynamic_basins: bool = False,
    basin_field: Optional[BasinField] = None,
):
    """Evaluate model."""
    model.eval()
    encoder.eval()
    head.eval()
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            labels = batch['label'].to(device)
            if getattr(encoder, 'is_bert', False):
                h0, v_p, v_h = encoder.build_initial_state(
                    batch['premise'], batch['hypothesis'], device=device
                )
            else:
                prem_ids = batch['prem_ids'].to(device)
                hyp_ids = batch['hyp_ids'].to(device)
                h0, v_p, v_h = encoder.build_initial_state(prem_ids, hyp_ids)
            
            if use_dynamic_basins and basin_field is not None:
                # collapse_inference: label-free routing across all basins.
                # collapse_dynamic must NOT be called here — it takes true labels
                # as input to route each sample to its correct basin, which leaks
                # the answer and produces artificially perfect dev accuracy.
                h_final, trace = model.collapse_inference(h0, basin_field)
            else:
                h_final, trace = model.collapse(h0)

            # Classify with directional signals
            logits = head(h_final, v_p, v_h)
            pred = logits.argmax(dim=-1)
            
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Confusion matrix
    confusion = np.zeros((3, 3), dtype=int)
    for p, l in zip(all_predictions, all_labels):
        confusion[l, p] += 1
    
    return accuracy, confusion


def main():
    parser = argparse.ArgumentParser(description='Train SNLI with Livnium Core v1.0')
    parser.add_argument('--snli-train', type=str, required=True,
                       help='Path to SNLI training JSONL file')
    parser.add_argument('--snli-dev', type=str, default=None,
                       help='Path to SNLI dev JSONL file')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of training samples')
    parser.add_argument('--dim', type=int, default=256,
                       help='Vector dimension')
    parser.add_argument('--num-layers', type=int, default=6,
                       help='Number of collapse layers')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--strength-entail', type=float, default=0.1,
                       help='Force strength for entail anchor')
    parser.add_argument('--strength-contra', type=float, default=0.1,
                       help='Force strength for contradiction anchor')
    parser.add_argument('--strength-neutral', type=float, default=0.05,
                       help='Force strength for neutral anchor')
    parser.add_argument('--disable-dynamic-basins', action='store_true',
                       help='Use legacy static anchors instead of dynamic basin field')
    parser.add_argument('--basin-max-per-label', type=int, default=64,
                       help='Maximum number of basins per label')
    parser.add_argument('--basin-tension-threshold', type=float, default=0.15,
                       help='Tension threshold to trigger basin spawn')
    parser.add_argument('--basin-align-threshold', type=float, default=0.6,
                       help='Alignment threshold to allow basin spawn')
    parser.add_argument('--basin-anchor-lr', type=float, default=0.05,
                       help='EMA rate for basin center updates')
    parser.add_argument('--basin-prune-every', type=int, default=0,
                       help='If >0, prune/merge basins every N steps')
    parser.add_argument('--basin-prune-min-count', type=int, default=10,
                       help='Minimum count before keeping a basin during prune')
    parser.add_argument('--basin-merge-cos-threshold', type=float, default=0.97,
                       help='Cosine threshold to merge similar basins during prune')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for model')
    parser.add_argument('--max-len', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                       help='Label smoothing for CrossEntropyLoss (e.g., 0.05)')
    parser.add_argument('--neutral-weight', type=float, default=1.0,
                       help='Class weight multiplier for neutral to emphasize that class')
    parser.add_argument('--neutral-oversample', type=float, default=1.0,
                       help='>1.0 to oversample neutral examples (e.g., 1.5)')
    parser.add_argument('--encoder-type', choices=['legacy', 'pretrained', 'quantum', 'bert'], default='pretrained',
                       help='Sentence encoder: pretrained (pretrained embeddings), legacy (vocab mean-pool), '
                            'or bert (frozen bert-base-uncased, dim=768). '
                            '"quantum" accepted as alias for "pretrained" for checkpoint compatibility.')
    parser.add_argument('--bert-model', type=str, default='bert-base-uncased',
                       help='HuggingFace model name for BERT encoder (default: bert-base-uncased)')
    parser.add_argument('--head-type', choices=['attractor', 'linear'], default='attractor',
                       help='Classification head: attractor (full SNLIHead with geometry features) '
                            'or linear (single Linear(dim, 3) baseline). '
                            'Use linear + bert to establish the A/B baseline.')
    parser.add_argument('--embed-ckpt', type=str, default=None,
                       help='Path to embeddings_final.pt (required if encoder-type=pretrained)')
    # Geometric encoder knobs
    parser.add_argument('--geom-disable-transformer', action='store_true',
                       help='Disable transformer interaction layer in geometric encoder')
    parser.add_argument('--geom-disable-attn-pool', action='store_true',
                       help='Disable attention pooling in geometric encoder (use masked mean)')
    parser.add_argument('--geom-nhead', type=int, default=4,
                       help='Attention heads for geometric encoder transformer')
    parser.add_argument('--geom-num-layers', type=int, default=1,
                       help='Transformer layers for geometric encoder')
    parser.add_argument('--geom-ff-mult', type=int, default=2,
                       help='Feedforward multiplier for geometric encoder transformer')
    parser.add_argument('--geom-dropout', type=float, default=0.1,
                       help='Dropout for geometric encoder projection/transformer')
    parser.add_argument('--geom-token-norm-cap', type=float, default=3.0,
                       help='Per-token norm cap after projection (set <=0 to disable)')
    parser.add_argument('--strength-neutral-boost', type=float, default=0.05,
                       help='Extra neutral pull at E-C boundary (Fix B). '
                            'When the state is equidistant between E and C anchors, '
                            'this additional force pulls it toward the neutral basin. '
                            'Set 0.0 to disable.')
    parser.add_argument('--barrier', type=float, default=0.38,
                       help='Livnium barrier constant (default 0.38). '
                            'divergence = barrier - cos(h, anchor). '
                            'Set 0.0 to test no-barrier (targets orthogonality), '
                            '1.0 for full alignment target.')
    parser.add_argument('--warmup-ratio', type=float, default=0.06,
                       help='Fraction of total training steps used for linear LR warmup '
                            '(default 0.06 = 6%%). After warmup, LR decays linearly to 0.')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='AdamW weight decay (default 0.01). AdamW decouples weight '
                            'decay from the adaptive gradient update, improving generalisation.')
    parser.add_argument('--alpha-first-token', type=float, default=0.0,
                       help='Symmetry-breaking first-token injection strength. '
                            'Adds alpha*(embed(hyp[0])-embed(prem[0])) to h0, '
                            'breaking bag-of-words symmetry so the field can '
                            'distinguish "A bites B" from "B bites A". '
                            '0.0=disabled, 0.1=weak, 0.25=stronger.')
    parser.add_argument('--lambda-traj', type=float, default=0.0,
                       help='Weight for trajectory direction loss. '
                            'Rewards collapse trajectories that moved toward the '
                            'correct basin anchor, giving partial credit to '
                            'near-misses where the path was right but the head '
                            'tipped wrong. 0.0=disabled, 0.1=weak start.')
    parser.add_argument('--lambda-fn', type=float, default=0.0,
                       help='Weight for false-neutral penalty. '
                            'Penalizes P(neutral) when true label is E or C, '
                            'breaking the model habit of parking ambiguous cases '
                            'in neutral as a cheap hedge. 0.0=disabled, 0.05=start.')
    parser.add_argument('--lambda-rep', type=float, default=0.0,
                       help='Weight for basin repulsion loss (anti-gravity). '
                            'Requires sim(h_final, correct_anchor) to exceed '
                            'sim(h_final, wrong_anchor) by margin-rep. '
                            'Prevents wrong basins from pulling nearby states '
                            'across class boundaries. 0.0=disabled, 0.1=start.')
    parser.add_argument('--margin-rep', type=float, default=0.3,
                       help='Cosine margin for basin repulsion loss. '
                            'Correct anchor must be closer than any wrong anchor '
                            'by this margin (cosine units, range roughly 0–2). '
                            '0.3 is a reasonable starting point.')
    parser.add_argument('--rot-rank', type=int, default=0,
                       help='Rank of the low-rank skew-symmetric rotation matrix '
                            'added to collapse dynamics (v6 rotational stabilisation). '
                            'Adds a circulating component so vectors spiral toward '
                            'attractors instead of falling straight in — prevents '
                            'basin collapse into blobs. 0=disabled, 8=start.')
    parser.add_argument('--rot-strength', type=float, default=0.01,
                       help='Strength of the rotational component applied at each '
                            'collapse step. Small values (0.01–0.05) add gentle '
                            'circulation without disrupting convergence.')
    parser.add_argument('--lock-threshold', type=float, default=0.0,
                       help='Alignment threshold for basin locking (v7). '
                            'When cos(h, basin) > lock_threshold, the attractive '
                            'force is multiplied by (1 + lock_gain * sigmoid_gate). '
                            'Creates a capture zone: once inside, the vector is held '
                            'firmly. 0.0=disabled, 0.5=engages when clearly aligned.')
    parser.add_argument('--lock-gain', type=float, default=2.0,
                       help='Strength multiplier inside the locking zone. '
                            '2.0 = up to 3x base attraction when fully locked.')
    parser.add_argument('--lock-temp', type=float, default=0.1,
                       help='Temperature for the locking sigmoid. '
                            'Smaller = sharper transition into the capture zone.')
    parser.add_argument('--strength-null', type=float, default=0.0,
                       help='Strength of the virtual null endpoint (v8). '
                            'A 4th attractor with no label — gives genuinely ambiguous '
                            'vectors somewhere to go that is not N. Breaks the habit of '
                            'N acting as the default uncertainty dump. '
                            '0.0=disabled, 0.03=gentle, 0.05=stronger.')
    parser.add_argument('--adaptive-metric', action='store_true',
                       help='Enable adaptive diagonal metric tensor (v9). '
                            'Learns 256 per-dimension scale factors so alignment '
                            'is measured in a warped space — stretching dimensions '
                            'that help separate E/C/N and suppressing noise. '
                            'The geometry bends where resolution is needed.')
    parser.add_argument('--freeze-encoder', action='store_true',
                       help='Freeze encoder weights. Only collapse engine and head '
                            'are trained. Used for A/B experiment isolating encoder effect.')

    args = parser.parse_args()

    # ── Inject barrier constant into physics engine ──────────────────────────
    import core.physics_laws as _physics
    _physics.BARRIER = args.barrier
    print(f"Livnium BARRIER = {args.barrier}  (default 0.38)")

    # Device — CPU is fastest for dynamic basin training (basin field is CPU-bound;
    # MPS causes constant CPU↔GPU transfers that kill throughput ~20x).
    # Use CUDA only if available. MPS intentionally skipped for this pipeline.
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading SNLI data...")
    train_samples = load_snli_data(Path(args.snli_train), max_samples=args.max_samples)
    print(f"Loaded {len(train_samples)} training samples")
    
    pretrained_encode_fn = None
    vocab = None
    vocab_id_to_token = None

    if args.encoder_type == 'bert':
        # BERT tokenizes internally — no encode_fn needed.
        # dim is fixed at 768 (BERT hidden size).
        print(f"Loading frozen BERT encoder ({args.bert_model}) ...")
        args.dim = 768
        pretrained_encode_fn = lambda text, max_len=args.max_len: [0] * max_len  # dummy, unused
    elif args.encoder_type in ('pretrained', 'quantum'):
        if not args.embed_ckpt:
            raise ValueError("encoder-type=pretrained requires --embed-ckpt pointing to embeddings_final.pt")
        print(f"Loading pretrained encoder vocab from {args.embed_ckpt} ...")
        pretrained_tokenizer = PretrainedTextEncoder(args.embed_ckpt)
        if args.dim != pretrained_tokenizer.dim:
            print(f"Overriding dim {args.dim} -> {pretrained_tokenizer.dim} to match pretrained checkpoint")
            args.dim = pretrained_tokenizer.dim

        def pretrained_encode(text: str, max_len: int = args.max_len):
            tokens = pretrained_tokenizer.tokenize(text)
            ids = [pretrained_tokenizer.word2idx.get(t, pretrained_tokenizer.unk_idx) for t in tokens]
            ids = ids[:max_len]
            if len(ids) < max_len:
                ids.extend([pretrained_tokenizer.pad_idx] * (max_len - len(ids)))
            return ids

        pretrained_encode_fn = pretrained_encode
    else:
        # Build vocabulary from SNLI
        print("Building vocabulary...")
        vocab = build_vocab_from_snli(train_samples, min_count=2)
        print(f"Vocabulary size: {len(vocab)}")
        vocab_id_to_token = vocab.id_to_token_list()
    
    # Create datasets
    train_dataset = SNLIDataset(train_samples, vocab, max_len=args.max_len, encode_fn=pretrained_encode_fn)
    # Optional oversampling of neutral class
    sampler = None
    if args.neutral_oversample > 1.0:
        labels = [s['label'].item() for s in (train_dataset[i] for i in range(len(train_dataset)))]
        # base weights inverse-frequency
        counts = np.bincount(labels, minlength=3)
        base_weights = [1.0 / max(c, 1) for c in counts]
        weights = [base_weights[l] for l in labels]
        # amplify neutral
        weights = [w * (args.neutral_oversample if l == 2 else 1.0) for w, l in zip(weights, labels)]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Dev set
    dev_loader = None
    if args.snli_dev:
        dev_samples = load_snli_data(Path(args.snli_dev))
        dev_dataset = SNLIDataset(dev_samples, vocab, max_len=args.max_len, encode_fn=pretrained_encode_fn)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
        print(f"Loaded {len(dev_samples)} dev samples")
    
    # Create models
    print("Creating models...")
    use_dynamic_basins = not args.disable_dynamic_basins
    basin_field = BasinField(max_basins_per_label=args.basin_max_per_label) if use_dynamic_basins else None
    collapse_engine = VectorCollapseEngine(
        dim=args.dim,
        num_layers=args.num_layers,
        strength_entail=args.strength_entail,
        strength_contra=args.strength_contra,
        strength_neutral=args.strength_neutral,
        strength_neutral_boost=args.strength_neutral_boost,
        basin_tension_threshold=args.basin_tension_threshold,
        basin_align_threshold=args.basin_align_threshold,
        basin_anchor_lr=args.basin_anchor_lr,
        basin_prune_min_count=args.basin_prune_min_count,
        basin_prune_merge_cos=args.basin_merge_cos_threshold,
        rot_rank=args.rot_rank,
        rot_strength=args.rot_strength,
        lock_threshold=args.lock_threshold,
        lock_gain=args.lock_gain,
        lock_temp=args.lock_temp,
        strength_null=args.strength_null,
        adaptive_metric=args.adaptive_metric,
    ).to(device)
    if args.encoder_type == 'bert':
        encoder = BERTSNLIEncoder(
            model_name=args.bert_model,
            freeze=True,
        ).to(device)
    elif args.encoder_type in ('pretrained', 'quantum'):
        encoder = PretrainedSNLIEncoder(
            ckpt_path=args.embed_ckpt,
            alpha_first_token=args.alpha_first_token,
        ).to(device)
    else:
        encoder = SNLIEncoder(
            vocab_size=len(vocab),
            dim=args.dim,
            pad_idx=vocab.pad_idx,
        ).to(device)

    # Head: attractor (full geometry) or linear (baseline probe)
    if args.head_type == 'linear':
        head = LinearSNLIHead(dim=args.dim).to(device)
        print("Head: LinearSNLIHead (baseline)")
    else:
        head = SNLIHead(dim=args.dim).to(device)
        print("Head: SNLIHead (attractor geometry)")

    # Freeze encoder if requested (A/B experiment: isolates encoder effect)
    # encoder.train() is kept intentionally — quantum encoder has no dropout/batchnorm
    # so train/eval mode is identical. Only gradients are disabled.
    if args.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad_(False)

    # Optimizer — excludes encoder params when frozen so Adam holds no stale references
    optimizer = optim.Adam(
        list(collapse_engine.parameters()) +
        (list(encoder.parameters()) if not args.freeze_encoder else []) +
        list(head.parameters()),
        lr=args.lr,
    )

    # Loss with optional class weighting and label smoothing
    class_weights = torch.tensor([1.0, 1.0, args.neutral_weight], device=device, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    best_acc = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_acc, global_step = train_epoch(
            collapse_engine,
            encoder,
            head,
            train_loader,
            optimizer,
            criterion,
            device,
            use_dynamic_basins=use_dynamic_basins,
            basin_field=basin_field,
            spawn_new=use_dynamic_basins,
            prune_every=args.basin_prune_every,
            start_step=global_step,
            lambda_traj=args.lambda_traj,
            lambda_fn=args.lambda_fn,
            lambda_rep=args.lambda_rep,
            margin_rep=args.margin_rep,
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        if use_dynamic_basins and basin_field is not None:
            counts = {l: len(basin_field.anchors[l]) for l in ["E", "N", "C"]}
            print(f"Basin counts (E/N/C): {counts['E']} / {counts['N']} / {counts['C']}")
        
        # Evaluate
        if dev_loader:
            dev_acc, confusion = evaluate(
                collapse_engine,
                encoder,
                head,
                dev_loader,
                device,
                use_dynamic_basins=use_dynamic_basins,
                basin_field=basin_field,
            )
            print(f"Dev Acc: {dev_acc:.4f}")
            # Label mapping: 0=entailment, 1=contradiction, 2=neutral
            print("\nConfusion Matrix (rows=true, cols=pred):")
            print("      E    C    N")
            for i, label in enumerate(['E', 'C', 'N']):
                print(f"{label}  {confusion[i]}")
            e_rec = confusion[0,0] / confusion[0].sum() if confusion[0].sum() > 0 else 0
            c_rec = confusion[1,1] / confusion[1].sum() if confusion[1].sum() > 0 else 0
            n_rec = confusion[2,2] / confusion[2].sum() if confusion[2].sum() > 0 else 0
            print(f"Per-class recall → E: {e_rec*100:.1f}%  C: {c_rec*100:.1f}%  N: {n_rec*100:.1f}%")
            
            # Save best model
            if dev_acc > best_acc:
                best_acc = dev_acc
                torch.save({
                    'collapse_engine': collapse_engine.state_dict(),
                    'encoder': encoder.state_dict(),
                    'head': head.state_dict(),
                    'vocab': vocab,
                    'args': args,
                    'basin_field': basin_field.state_dict() if basin_field is not None else None,
                    'use_dynamic_basins': use_dynamic_basins,
                    'head_type': getattr(args, 'head_type', 'attractor'),
                }, output_dir / 'best_model.pt')
                print(f"✓ Saved best model (acc: {best_acc:.4f})")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best Dev Accuracy: {best_acc:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()

"""
Extract Livnium Basis from a Trained Checkpoint
================================================

Takes a trained BERT+Livnium checkpoint and extracts the attractor geometry:
  - The three anchor vectors (E, N, C) in the original embedding space
  - PCA of collapse trajectories over training data
  - A compact projection matrix from original dim → Livnium basis dim

The Livnium basis is the coordinate system where:
  - Dimension 0: alignment with entailment attractor
  - Dimension 1: alignment with contradiction attractor
  - Dimension 2: alignment with neutral attractor
  - Dimensions 3+: remaining variance from collapse trajectories (PCA)

Saving the basis lets train.py initialise the LivniumNativeEncoder's
collapse engine anchors in the correct positions, rather than random ones.

Usage:
    cd system/snli/model
    python extract_livnium_basis.py \\
        --checkpoint ../../../pretrained/bert-joint/best_model.pt \\
        --snli-train ../../../data/snli/snli_1.0_train.jsonl \\
        --encoder-type bert \\
        --basis-dim 32 \\
        --n-samples 5000 \\
        --output ../../../pretrained/livnium_basis.pt

Output file contains:
    {
        'projection':      Tensor (orig_dim, basis_dim)  — projection matrix
        'anchor_entail':   Tensor (basis_dim,)            — E anchor in basis space
        'anchor_contra':   Tensor (basis_dim,)            — C anchor in basis space
        'anchor_neutral':  Tensor (basis_dim,)            — N anchor in basis space
        'explained_variance': float                       — PCA variance explained
        'orig_dim':        int
        'basis_dim':       int
        'encoder_type':    str
    }
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Path setup
_here = Path(__file__).resolve().parent
_repo = _here.parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_repo))


def load_samples(jsonl_path: Path, n: int) -> List[Dict]:
    """Load up to n valid SNLI samples."""
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            label = data.get('gold_label', '').strip()
            if label not in ('entailment', 'contradiction', 'neutral'):
                continue
            p = data.get('sentence1', '').strip()
            h = data.get('sentence2', '').strip()
            if not p or not h:
                continue
            samples.append({'premise': p, 'hypothesis': h, 'gold_label': label})
            if len(samples) >= n:
                break
    return samples


@torch.no_grad()
def collect_h0_vectors(encoder, samples: List[Dict], batch_size: int, device) -> tuple:
    """
    Run samples through the encoder and collect h0 vectors + labels.

    Returns:
        h0_matrix: (N, dim) numpy array
        labels:    (N,) numpy array — 0=E, 1=C, 2=N
    """
    from tasks.snli import BERTSNLIEncoder, CrossEncoderBERTSNLIEncoder

    label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    encoder.eval()
    all_h0 = []
    all_labels = []

    for i in tqdm(range(0, len(samples), batch_size), desc='Encoding'):
        batch = samples[i:i + batch_size]
        premises    = [s['premise']    for s in batch]
        hypotheses  = [s['hypothesis'] for s in batch]
        batch_labels = [label_map[s['gold_label']] for s in batch]

        if getattr(encoder, 'is_bert', False):
            h0, _, _ = encoder.build_initial_state(premises, hypotheses, add_noise=False, device=device)
        else:
            raise ValueError("collect_h0_vectors only supports BERT-type encoders")

        all_h0.append(h0.cpu().float().numpy())
        all_labels.extend(batch_labels)

    h0_matrix = np.concatenate(all_h0, axis=0)    # (N, dim)
    labels_arr = np.array(all_labels, dtype=np.int64)
    return h0_matrix, labels_arr


def compute_pca_projection(h0_matrix: np.ndarray, basis_dim: int) -> tuple:
    """
    PCA on h0 vectors — find the basis_dim directions of maximum variance.

    Returns:
        components:          (basis_dim, orig_dim) — PCA components
        explained_variance:  float — fraction of total variance captured
        mean:                (orig_dim,) — mean vector (for centering)
    """
    mean = h0_matrix.mean(axis=0)
    centered = h0_matrix - mean

    # Use numpy SVD (works without sklearn)
    print(f"  Running SVD on {h0_matrix.shape} matrix ...")
    # Truncated SVD via covariance: faster for dim >> n_components when n >> dim
    # For 768-dim × 5000 samples: use full SVD then truncate
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)

    components = Vt[:basis_dim]                          # (basis_dim, orig_dim)
    variance_explained = (s[:basis_dim] ** 2).sum() / (s ** 2).sum()

    return components, float(variance_explained), mean


def build_anchor_directions(anchors: np.ndarray) -> np.ndarray:
    """
    Build an orthonormal basis from the 3 anchor directions.

    The 3 anchors define a semantic simplex. Their directions from the
    centroid span the most semantically meaningful subspace.

    Returns: (3, orig_dim) orthonormal basis from anchor geometry
    """
    centroid = anchors.mean(axis=0, keepdims=True)  # (1, orig_dim)
    dirs = anchors - centroid                        # (3, orig_dim)

    # Gram-Schmidt orthonormalisation
    basis = []
    for d in dirs:
        # Remove projection onto already-chosen basis vectors
        for b in basis:
            d = d - np.dot(d, b) * b
        norm = np.linalg.norm(d)
        if norm > 1e-8:
            basis.append(d / norm)

    return np.stack(basis, axis=0)                  # (<=3, orig_dim)


def main():
    parser = argparse.ArgumentParser(
        description='Extract Livnium basis from a trained BERT+Livnium checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Path to best_model.pt or latest_checkpoint.pt')
    parser.add_argument('--snli-train', required=True,
                        help='Path to SNLI training JSONL (for collecting h0 samples)')
    parser.add_argument('--encoder-type', choices=['bert', 'crossbert'], default='bert',
                        help='Encoder type used in the checkpoint')
    parser.add_argument('--bert-model', default='bert-base-uncased',
                        help='HuggingFace BERT model name')
    parser.add_argument('--basis-dim', type=int, default=32,
                        help='Target dimensionality of the Livnium basis (default: 32)')
    parser.add_argument('--n-samples', type=int, default=5000,
                        help='Number of training samples to encode for PCA (default: 5000)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for encoding (default: 64)')
    parser.add_argument('--output', required=True,
                        help='Output path for the basis file (e.g., pretrained/livnium_basis.pt)')
    parser.add_argument('--no-pca', action='store_true',
                        help='Skip PCA — use only anchor geometry directions (fast, 3 basis vectors)')
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Extract anchors
    engine_state = ckpt.get('collapse_engine', ckpt.get('model', {}))
    anchor_keys = {
        'anchor_entail':  'anchor_entail',
        'anchor_contra':  'anchor_contra',
        'anchor_neutral': 'anchor_neutral',
    }
    anchors_dict = {}
    for key, name in anchor_keys.items():
        if key in engine_state:
            anchors_dict[name] = engine_state[key].float().numpy()
            print(f"  ✓ {name}: {anchors_dict[name].shape}")
        else:
            print(f"  ✗ {name}: NOT FOUND in checkpoint — check engine state keys")
            print(f"    Available keys: {[k for k in engine_state.keys() if 'anchor' in k]}")

    if len(anchors_dict) < 3:
        raise ValueError("Could not find all 3 anchor vectors in checkpoint.")

    orig_dim = anchors_dict['anchor_entail'].shape[0]
    print(f"\nOriginal embedding dim: {orig_dim}")
    print(f"Target Livnium basis dim: {args.basis_dim}")

    # ── Build anchor-direction basis ──────────────────────────────────────────
    print("\nBuilding anchor-direction basis ...")
    anchor_matrix = np.stack([
        anchors_dict['anchor_entail'],
        anchors_dict['anchor_contra'],
        anchors_dict['anchor_neutral'],
    ], axis=0)                                           # (3, orig_dim)

    anchor_basis = build_anchor_directions(anchor_matrix)  # (<=3, orig_dim)
    print(f"  Anchor basis: {anchor_basis.shape[0]} orthonormal directions")

    # ── Optionally run PCA for remaining basis dims ───────────────────────────
    if args.no_pca or args.basis_dim <= anchor_basis.shape[0]:
        # Use only anchor directions — fast path
        projection = anchor_basis[:args.basis_dim]            # (basis_dim, orig_dim)
        explained_variance = float('nan')
        print("  PCA skipped (--no-pca or basis_dim <= num_anchor_dirs)")
    else:
        # Load SNLI samples
        print(f"\nLoading {args.n_samples} training samples for PCA ...")
        samples = load_samples(Path(args.snli_train), args.n_samples)
        print(f"  Loaded {len(samples)} samples")

        # Load encoder
        print(f"\nLoading {args.encoder_type} encoder ({args.bert_model}) ...")
        from tasks.snli import BERTSNLIEncoder, CrossEncoderBERTSNLIEncoder
        if args.encoder_type == 'crossbert':
            encoder = CrossEncoderBERTSNLIEncoder(model_name=args.bert_model, freeze=True)
        else:
            encoder = BERTSNLIEncoder(model_name=args.bert_model, freeze=True)
        encoder = encoder.to(device)

        # Collect h0 vectors
        print(f"\nCollecting h0 vectors (batch_size={args.batch_size}) ...")
        h0_matrix, _ = collect_h0_vectors(encoder, samples, args.batch_size, device)
        print(f"  h0 matrix: {h0_matrix.shape}")

        # PCA — find top (basis_dim - n_anchor_dirs) additional components
        n_anchor = anchor_basis.shape[0]
        n_pca = args.basis_dim - n_anchor
        print(f"\nRunning PCA for {n_pca} additional components ...")

        # Remove anchor-subspace variance from h0 before PCA
        # (avoids redundant directions that duplicate anchor geometry)
        h0_proj_out = h0_matrix - (h0_matrix @ anchor_basis.T) @ anchor_basis
        pca_components, explained_variance, _ = compute_pca_projection(h0_proj_out, n_pca)
        print(f"  PCA explained variance: {explained_variance:.1%}")

        # Final projection: anchor directions first, then PCA residual directions
        projection = np.concatenate([anchor_basis, pca_components], axis=0)  # (basis_dim, orig_dim)

    # ── Project anchors into the new basis ───────────────────────────────────
    # projection: (basis_dim, orig_dim)  — maps from orig to basis
    projection_t = torch.tensor(projection.T, dtype=torch.float32)   # (orig_dim, basis_dim)
    proj_np = projection_t.numpy().T                                  # (basis_dim, orig_dim)

    projected = {
        name: (proj_np @ vec).tolist()
        for name, vec in anchors_dict.items()
    }
    print(f"\nProjected anchors into {args.basis_dim}-dim Livnium basis:")
    for name, vec in projected.items():
        nrm = float(np.linalg.norm(vec))
        print(f"  {name}: norm={nrm:.3f}  first3={[round(v, 3) for v in vec[:3]]}")

    # ── Cosine similarities between projected anchors ─────────────────────────
    # We want anchors to be well-separated in the new space
    E = np.array(projected['anchor_entail'])
    C = np.array(projected['anchor_contra'])
    N = np.array(projected['anchor_neutral'])
    cos_EC = float(np.dot(E, C) / (np.linalg.norm(E) * np.linalg.norm(C) + 1e-8))
    cos_EN = float(np.dot(E, N) / (np.linalg.norm(E) * np.linalg.norm(N) + 1e-8))
    cos_CN = float(np.dot(C, N) / (np.linalg.norm(C) * np.linalg.norm(N) + 1e-8))
    print(f"\nAnchor separation in Livnium basis (lower = more separated):")
    print(f"  cos(E, C) = {cos_EC:.3f}   (want < 0.0 ideally)")
    print(f"  cos(E, N) = {cos_EN:.3f}")
    print(f"  cos(C, N) = {cos_CN:.3f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    basis_data = {
        'projection':          projection_t,                                  # (orig_dim, basis_dim)
        'anchor_entail':       torch.tensor(projected['anchor_entail'],  dtype=torch.float32),
        'anchor_contra':       torch.tensor(projected['anchor_contra'],  dtype=torch.float32),
        'anchor_neutral':      torch.tensor(projected['anchor_neutral'], dtype=torch.float32),
        'explained_variance':  explained_variance,
        'orig_dim':            orig_dim,
        'basis_dim':           args.basis_dim,
        'encoder_type':        args.encoder_type,
        'cos_EC':              cos_EC,
        'cos_EN':              cos_EN,
        'cos_CN':              cos_CN,
    }
    torch.save(basis_data, output_path)
    print(f"\n✓ Livnium basis saved to: {output_path}")
    print(f"  projection:    {projection_t.shape}  (orig_dim × basis_dim)")
    print(f"  anchors:       {args.basis_dim}-dim each")
    if not np.isnan(explained_variance):
        print(f"  PCA coverage:  {explained_variance:.1%} of h0 variance")
    print(f"\nNext step:")
    print(f"  python train.py --encoder-type livnium \\")
    print(f"                  --livnium-dim {args.basis_dim} \\")
    print(f"                  --livnium-basis {output_path} \\")
    print(f"                  ...")


if __name__ == '__main__':
    main()

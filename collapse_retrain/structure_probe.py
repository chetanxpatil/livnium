#!/usr/bin/env python3
"""
structure_probe.py — Structure Probe for VectorCollapseEngine.

Analyzes the learned representation space and dynamical warping of the NLI collapse model:
1. Anchor geometry (cosine similarities and Euclidean distances).
2. Layer-wise collapse (trajectories and argmax classification).
3. Basin topology (sizes and boundary regions of E, N, C basins).
4. Jacobian contraction/stability (singular values S at anchors and boundaries).
5. Semantic trajectory categories (Fast Collapse, Correction, Stuck, Failure).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from eval_nli import LABELS, load_checkpoint, resolve_device
from train_collapse_embeddings import (
    NLIDataset,
    _anchor_matrix,
    _meanpool,
    nli_collate,
    read_nli_jsonl,
)


# Helper function to compute step-by-step collapse trajectories
def get_collapse_trajectory(engine, h0):
    """
    Returns a list of tensors [h0, h1, h2, h3, h4] representing intermediate states.
    h0 shape: (B, dim)
    """
    h = h0.clone()
    states = [h.clone()]

    e_dir = F.normalize(engine.anchor_entail, dim=0)
    c_dir = F.normalize(engine.anchor_contra, dim=0)
    n_dir = F.normalize(engine.anchor_neutral, dim=0)

    for _ in range(engine.num_layers):
        h_n = F.normalize(h, dim=-1)
        a_e = (h_n * e_dir).sum(dim=-1)
        a_c = (h_n * c_dir).sum(dim=-1)
        a_n = (h_n * n_dir).sum(dim=-1)

        d_e = 1.0 - a_e
        d_c = 1.0 - a_c
        d_n = 1.0 - a_n

        delta = engine.update(h)

        e_vec = F.normalize(h - e_dir, dim=-1)
        c_vec = F.normalize(h - c_dir, dim=-1)
        n_vec = F.normalize(h - n_dir, dim=-1)

        h = (
            h
            + delta
            - engine.strength_entail * d_e.unsqueeze(-1) * e_vec
            - engine.strength_contra * d_c.unsqueeze(-1) * c_vec
            - engine.strength_neutral * d_n.unsqueeze(-1) * n_vec
        )

        h_norm = h.norm(p=2, dim=-1, keepdim=True)
        h = torch.where(h_norm > 10.0, h * (10.0 / (h_norm + 1e-8)), h)
        states.append(h.clone())

    return states


# Helper function to compute Jacobian at a specific point
def get_jacobian(engine, x_pt):
    """
    x_pt: (dim,) tensor on CPU
    Returns Jacobian matrix (dim, dim) on CPU
    """
    x = x_pt.clone().detach().requires_grad_(True)

    def f_wrapped(x_in):
        h = x_in.unsqueeze(0)
        e_dir = F.normalize(engine.anchor_entail, dim=0)
        c_dir = F.normalize(engine.anchor_contra, dim=0)
        n_dir = F.normalize(engine.anchor_neutral, dim=0)

        for _ in range(engine.num_layers):
            h_n = F.normalize(h, dim=-1)
            a_e = (h_n * e_dir).sum(dim=-1)
            a_c = (h_n * c_dir).sum(dim=-1)
            a_n = (h_n * n_dir).sum(dim=-1)

            d_e = 1.0 - a_e
            d_c = 1.0 - a_c
            d_n = 1.0 - a_n

            delta = engine.update(h)

            e_vec = F.normalize(h - e_dir, dim=-1)
            c_vec = F.normalize(h - c_dir, dim=-1)
            n_vec = F.normalize(h - n_dir, dim=-1)

            h = (
                h
                + delta
                - engine.strength_entail * d_e.unsqueeze(-1) * e_vec
                - engine.strength_contra * d_c.unsqueeze(-1) * c_vec
                - engine.strength_neutral * d_n.unsqueeze(-1) * n_vec
            )

            h_norm = h.norm(p=2, dim=-1, keepdim=True)
            h = torch.where(h_norm > 10.0, h * (10.0 / (h_norm + 1e-8)), h)

        return h.squeeze(0)

    J = torch.autograd.functional.jacobian(f_wrapped, x)
    return J.detach()


def main():
    ap = argparse.ArgumentParser(description="Structure Probe for VectorCollapseEngine.")
    ap.add_argument("--ckpt", type=str, default="collapse_retrain/model_nli_v1/nli_epoch20.pt")
    ap.add_argument(
        "--data", type=str, default="/Users/chetanpatil/Desktop/test/data/snli_1.0_test.jsonl"
    )
    args = ap.parse_args()

    device = resolve_device("auto")
    print(f"🚀 Device resolved: {device}")
    model, engine, vocab = load_checkpoint(args.ckpt, device)

    # --- LEVEL 1: Anchor Geometry ---
    print("\n=== LEVEL 1: Anchor Geometry ===")
    anchors = _anchor_matrix(engine)  # (3, 256)
    AE = anchors[0]
    AN = anchors[1]
    AC = anchors[2]

    cos_en = torch.dot(AE, AN).item()
    cos_ec = torch.dot(AE, AC).item()
    cos_nc = torch.dot(AN, AC).item()

    dist_en = torch.dist(AE, AN).item()
    dist_ec = torch.dist(AE, AC).item()
    dist_nc = torch.dist(AN, AC).item()

    print("Cosine Similarities:")
    print(f"  cos(Entailment, Neutral)      : {cos_en:6.4f}")
    print(f"  cos(Entailment, Contradiction): {cos_ec:6.4f}")
    print(f"  cos(Neutral, Contradiction)   : {cos_nc:6.4f}")
    print("Euclidean Distances:")
    print(f"  dist(Entailment, Neutral)     : {dist_en:6.4f}")
    print(f"  dist(Entailment, Contradiction): {dist_ec:6.4f}")
    print(f"  dist(Neutral, Contradiction)  : {dist_nc:6.4f}")

    # Define Orthonormal Basis on the 2D plane spanned by the anchors
    x_dir = AE - AN
    x_basis = x_dir / torch.norm(x_dir)
    y_dir = AC - AN
    y_orthogonal = y_dir - torch.dot(y_dir, x_basis) * x_basis
    y_basis = y_orthogonal / torch.norm(y_orthogonal)

    # --- LEVEL 2: Layer-by-layer Trajectories ---
    print("\n=== LEVEL 2: Layer-by-layer Trajectories ===")
    examples = read_nli_jsonl(args.data)
    dataset = NLIDataset(examples, vocab)
    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=lambda b: nli_collate(b, vocab.pad_idx),
    )

    pad_idx = vocab.pad_idx
    example_idx = 0

    # We will accumulate intermediate states for the whole test set
    layer_dists_correct = [[] for _ in range(5)]  # 5 steps: x0, x1, x2, x3, x4
    layer_dists_incorrect = [[] for _ in range(5)]
    layer_accs = [0 for _ in range(5)]

    # Store trajectory items for semantic analysis
    all_items = []

    with torch.no_grad():
        for prem, hyp, gold in loader:
            prem, hyp, gold = prem.to(device), hyp.to(device), gold.to(device)
            u = _meanpool(model, prem, pad_idx)
            v = _meanpool(model, hyp, pad_idx)
            h0 = u - v

            states = get_collapse_trajectory(engine, h0)  # list of 5 tensors of shape (B, dim)

            B = h0.size(0)

            for t in range(5):
                ht = states[t]
                ht_n = F.normalize(ht, dim=-1)

                # Cosine similarities to anchors: (B, 3)
                sims = ht_n @ anchors.t()
                preds = sims.argmax(dim=-1)

                # Check accuracy at step t
                layer_accs[t] += (preds == gold).sum().item()

                # Compute Euclidean distances to normalized anchors
                # Note: anchors are unit-normalized. For fair distance comparison, we normalize ht as well.
                # E is index 0, N is index 1, C is index 2
                for i in range(B):
                    g_lbl = int(gold[i].item())
                    pred_lbl = int(preds[i].item())

                    dist_to_E = torch.dist(ht_n[i], AE).item()
                    dist_to_N = torch.dist(ht_n[i], AN).item()
                    dist_to_C = torch.dist(ht_n[i], AC).item()

                    dists = [dist_to_E, dist_to_N, dist_to_C]
                    correct_d = dists[g_lbl]

                    # Incorrect distance = mean of the other two
                    incorrect_d = sum(dists[j] for j in range(3) if j != g_lbl) / 2.0

                    layer_dists_correct[t].append(correct_d)
                    layer_dists_incorrect[t].append(incorrect_d)

                    # Log final trajectory stats for semantic analysis
                    if t == 4:
                        raw_prem, raw_hyp, _ = examples[example_idx]

                        # Compute path length as sum of step distances: sum(||x_{t+1} - x_t||)
                        path_len = 0.0
                        for step_i in range(4):
                            path_len += torch.dist(states[step_i + 1][i], states[step_i][i]).item()

                        all_items.append(
                            {
                                "index": example_idx,
                                "premise": raw_prem,
                                "hypothesis": raw_hyp,
                                "gold_label": LABELS[g_lbl],
                                "predicted_label": LABELS[pred_lbl],
                                "path_length": path_len,
                                "start_cos": float(
                                    (F.normalize(h0[i], dim=-1) @ anchors[g_lbl]).item()
                                ),
                                "final_cos": float((ht_n[i] @ anchors[g_lbl]).item()),
                                "cos_to_all_final": [
                                    float((ht_n[i] @ anchors[j]).item()) for j in range(3)
                                ],
                                "start_dist_correct": torch.dist(
                                    F.normalize(h0[i], dim=-1), anchors[g_lbl]
                                ).item(),
                                "final_dist_correct": correct_d,
                                "confidence": [
                                    float(p) for p in F.softmax(sims[i] * 10.0, dim=-1)
                                ],  # scaling factor matches model logs
                                "trajectory_2d": [
                                    [
                                        torch.dot(states[step_i][i] - anchors[1], x_basis).item(),
                                        torch.dot(states[step_i][i] - anchors[1], y_basis).item(),
                                    ]
                                    for step_i in range(5)
                                ],
                            }
                        )
                        example_idx += 1

    total_examples = len(examples)
    print("Layer-by-Layer Stats:")
    for t in range(5):
        mean_correct = sum(layer_dists_correct[t]) / total_examples
        mean_incorrect = sum(layer_dists_incorrect[t]) / total_examples
        acc = (layer_accs[t] / total_examples) * 100.0
        print(
            f"  Layer {t} | Acc: {acc:5.2f}% | Mean Correct Dist: {mean_correct:.4f} | Mean Incorrect Dist: {mean_incorrect:.4f}"
        )

    # --- LEVEL 3: Basin Structure ---
    print("\n=== LEVEL 3: Basin Structure ===")
    # Project anchors to 2D
    ae_2d = (torch.dot(AE - AN, x_basis).item(), torch.dot(AE - AN, y_basis).item())
    an_2d = (0.0, 0.0)
    ac_2d = (torch.dot(AC - AN, x_basis).item(), torch.dot(AC - AN, y_basis).item())

    us = [ae_2d[0], an_2d[0], ac_2d[0]]
    vs = [ae_2d[1], an_2d[1], ac_2d[1]]

    margin = 1.0
    u_min, u_max = min(us) - margin, max(us) + margin
    v_min, v_max = min(vs) - margin, max(vs) + margin

    grid_size = 100
    u_grid = np.linspace(u_min, u_max, grid_size)
    v_grid = np.linspace(v_min, v_max, grid_size)

    basin_counts = {0: 0, 1: 0, 2: 0}  # 0=E, 1=N, 2=C (matching gold NLI labels)

    grid_points = []
    for u_val in u_grid:
        for v_val in v_grid:
            h = AN + u_val * x_basis + v_val * y_basis
            grid_points.append(h)

    grid_tensor = torch.stack(grid_points).to(device)  # (10000, 256)

    with torch.no_grad():
        final_states = get_collapse_trajectory(engine, grid_tensor)[-1]
        final_states_n = F.normalize(final_states, dim=-1)
        sims = final_states_n @ anchors.t()
        preds = sims.argmax(dim=-1).tolist()

    for p in preds:
        basin_counts[p] += 1

    print("Basin Size Proportions (2D Plane):")
    total_grid = grid_size * grid_size
    print(f"  Entailment Basin      : {(basin_counts[0]/total_grid)*100:5.2f}%")
    print(f"  Neutral Basin         : {(basin_counts[1]/total_grid)*100:5.2f}%")
    print(f"  Contradiction Basin   : {(basin_counts[2]/total_grid)*100:5.2f}%")

    # --- LEVEL 4: Jacobian Contraction Analysis ---
    print("\n=== LEVEL 4: Jacobian Contraction Analysis ===")
    # We compute Jacobian on CPU for stability
    engine_cpu = engine.cpu()

    points_to_check = {
        "Anchor E": AE.cpu(),
        "Anchor N": AN.cpu(),
        "Anchor C": AC.cpu(),
        "Boundary E-C Midpoint": (0.5 * (AE + AC)).cpu(),
        "Boundary N-C Midpoint": (0.5 * (AN + AC)).cpu(),
    }

    for name, pt in points_to_check.items():
        J = get_jacobian(engine_cpu, pt)
        S = torch.linalg.svdvals(J)

        max_s = torch.max(S).item()
        mean_s = torch.mean(S).item()
        num_gt_1 = (S > 1.0).sum().item()
        num_lt_1 = (S < 1.0).sum().item()

        print(
            f"{name:25s} | Max S: {max_s:6.4f} | Mean S: {mean_s:6.4f} | S > 1: {num_gt_1:3d} | S < 1: {num_lt_1:3d}"
        )

    # --- LEVEL 5: Semantic Trajectory Categorization ---
    print("\n=== LEVEL 5: Semantic Trajectory Categorization ===")

    fast_collapse_cases = []
    correction_cases = []
    stuck_cases = []
    failure_cases = []

    for item in all_items:
        is_correct = item["predicted_label"] == item["gold_label"]
        gold_idx = LABELS.index(item["gold_label"])

        if is_correct and item["start_cos"] > 0.6 and item["path_length"] < 0.5:
            fast_collapse_cases.append(item)
        elif is_correct and item["start_cos"] < 0.2 and item["confidence"][gold_idx] > 0.6:
            correction_cases.append(item)
        elif is_correct and item["final_cos"] < 0.4:
            stuck_cases.append(item)
        elif not is_correct and item["start_cos"] > 0.4:
            failure_cases.append(item)

    print("Discovered categories:")
    print(f"  Fast Collapse Cases : {len(fast_collapse_cases)}")
    print(f"  Correction Cases    : {len(correction_cases)}")
    print(f"  Stuck/Ambiguous     : {len(stuck_cases)}")
    print(f"  Failure Cases       : {len(failure_cases)}")

    # Save a detailed JSON report of everything
    output_data = {
        "anchor_geometry": {
            "cos_en": cos_en,
            "cos_ec": cos_ec,
            "cos_nc": cos_nc,
            "dist_en": dist_en,
            "dist_ec": dist_ec,
            "dist_nc": dist_nc,
        },
        "layer_collapse": [
            {
                "layer": t,
                "accuracy": (layer_accs[t] / total_examples) * 100.0,
                "mean_correct_dist": sum(layer_dists_correct[t]) / total_examples,
                "mean_incorrect_dist": sum(layer_dists_incorrect[t]) / total_examples,
            }
            for t in range(5)
        ],
        "basin_sizes": {
            "E": basin_counts[0] / total_grid,
            "N": basin_counts[1] / total_grid,
            "C": basin_counts[2] / total_grid,
        },
        "jacobian_stats": {},
        "examples": {
            "fast_collapse": fast_collapse_cases[:5],
            "correction": correction_cases[:5],
            "stuck": stuck_cases[:5],
            "failure": failure_cases[:5],
        },
    }

    # Add Jacobian stats to JSON
    for name, pt in points_to_check.items():
        J = get_jacobian(engine_cpu, pt)
        S = torch.linalg.svdvals(J).tolist()
        output_data["jacobian_stats"][name] = {
            "max": max(S),
            "mean": sum(S) / len(S),
            "gt_1": sum(1 for s in S if s > 1.0),
            "lt_1": sum(1 for s in S if s < 1.0),
        }

    out_json_path = "docs/structure_probe_data.json"
    os.makedirs("docs", exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSuccessfully saved structural stats to: {out_json_path}")


if __name__ == "__main__":
    main()

"""
visualize_char_collapse.py — the CharCollapse analog of docs/COLLAPSE_VISUALIZATION.md.

Loads the trained typer (char_typer.pt) and renders two figures, in the same
spirit as the VectorCollapseEngine flow-field / warping plots:

  1. char_anchor_map.png    — the 26 LEARNED letter anchors projected to 2D.
                              Shows which letters the model placed near which
                              (the letter "gravity wells").
  2. char_trajectories.png  — real words run through CharCollapse, their
                              per-letter collapse path projected onto the same
                              plane and drawn as curved trajectories into the
                              wells (the analog of the dashed sample paths +
                              gravity wells image).

Projection: PCA (numpy SVD) fit on the 26 unit-normalized letter anchors; the
SAME mean + basis is applied to the trajectory states so wells and paths share
coordinates. This is the 64-dim space honestly flattened to 2D — not a toy
circle layout.

Run:  python3 visualize_char_collapse.py
Saves PNGs into ../docs/images/. Needs torch, numpy, matplotlib.
"""

import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from char_collapse import CharCollapse

TYPER_CKPT = "char_typer.pt"
OUT_DIR = "../docs/images"
WORDS = ["chetan", "collapse", "entail", "neutral", "anchor", "logic", "snowman"]


def fit_pca_2d(anchors_np):
    mean = anchors_np.mean(axis=0)
    X = anchors_np - mean
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    basis = Vt[:2]                      # (2, dim)
    return mean, basis


def proj(v, mean, basis):
    return (np.asarray(v) - mean) @ basis.T


def load_typer():
    ck = torch.load(TYPER_CKPT, map_location="cpu")
    cfg = ck["config"]
    enc = CharCollapse(dim=cfg["dim"], max_len=cfg["max_len"])
    enc.load_state_dict(ck["char_collapse"])
    enc.eval()
    return enc, cfg


def letter_points(enc, mean, basis):
    A = torch.nn.functional.normalize(enc.letter_anchors.detach(), dim=-1).numpy()
    pts = {}
    for i, c in enumerate(enc.vocab.itos):
        if c == enc.vocab.PAD:
            continue
        pts[c] = proj(A[i], mean, basis)
    return pts


def plot_anchor_map(letters_2d, path):
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=130)
    fig.patch.set_facecolor("white")
    xs = [p[0] for p in letters_2d.values()]
    ys = [p[1] for p in letters_2d.values()]
    ax.scatter(xs, ys, s=520, c="#eef2ff", edgecolors="#4f5bd5", linewidths=1.4, zorder=2)
    for c, p in letters_2d.items():
        ax.text(p[0], p[1], c, ha="center", va="center", fontsize=13,
                fontweight="bold", color="#2a2e6e", zorder=3)
    ax.set_title("CharCollapse — learned letter anchor map (64-d → 2D PCA)",
                 fontsize=12, color="#222")
    ax.set_aspect("equal"); ax.grid(True, ls=":", alpha=0.4)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    fig.tight_layout(); fig.savefig(path, facecolor="white"); plt.close(fig)


def plot_trajectories(enc, cfg, letters_2d, mean, basis, path):
    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=130)
    fig.patch.set_facecolor("white")

    # faint wells in the background
    for c, p in letters_2d.items():
        ax.scatter([p[0]], [p[1]], s=360, c="#f2f3f7", edgecolors="#c7cad8",
                   linewidths=1.0, zorder=1)
        ax.text(p[0], p[1], c, ha="center", va="center", fontsize=10,
                color="#9aa0b5", zorder=2)

    cmap = get_cmap("turbo")
    for wi, w in enumerate(WORDS):
        ids = enc.vocab.encode_batch([w], cfg["max_len"])
        with torch.no_grad():
            _, _, _, pth = enc.encode(ids)
        L = len(w)
        states = [np.zeros(cfg["dim"])]  # start point (origin of trajectory)
        states[0] = enc.start.detach().numpy()
        for t in range(L):
            states.append(pth[t][0].detach().numpy())
        P = np.array([proj(s, mean, basis) for s in states])
        col = cmap(wi / max(1, len(WORDS) - 1))
        ax.plot(P[:, 0], P[:, 1], "-", color=col, lw=2.0, alpha=0.9, zorder=3)
        ax.scatter(P[1:, 0], P[1:, 1], s=18, color=col, zorder=4)
        # arrows along the path
        for t in range(len(P) - 1):
            ax.annotate("", xy=P[t + 1], xytext=P[t],
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.3, alpha=0.8),
                        zorder=3)
        ax.text(P[-1, 0], P[-1, 1], "  " + w, color=col, fontsize=11,
                fontweight="bold", va="center", zorder=5)
        ax.scatter([P[0, 0]], [P[0, 1]], s=42, c="black", zorder=5)

    ax.set_title("CharCollapse — word trajectories through the letter wells",
                 fontsize=12, color="#222")
    ax.set_aspect("equal"); ax.grid(True, ls=":", alpha=0.4)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    fig.tight_layout(); fig.savefig(path, facecolor="white"); plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    enc, cfg = load_typer()
    A = torch.nn.functional.normalize(enc.letter_anchors.detach(), dim=-1).numpy()[1:]  # drop PAD
    mean, basis = fit_pca_2d(A)
    letters_2d = letter_points(enc, mean, basis)

    p1 = os.path.join(OUT_DIR, "char_anchor_map.png")
    p2 = os.path.join(OUT_DIR, "char_trajectories.png")
    plot_anchor_map(letters_2d, p1)
    plot_trajectories(enc, cfg, letters_2d, mean, basis, p2)
    print("saved:")
    print(" ", os.path.abspath(p1))
    print(" ", os.path.abspath(p2))


if __name__ == "__main__":
    main()

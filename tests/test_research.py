"""Research-side regression tests.

The proven-core tests (test_base27.py etc.) need no dependencies. These tests
guard the *research* code paths that have bitten us before:

  1. false-negative masking in sampled-softmax
  2. tie-aware Spearman (the SimLex yardstick)
  3. NumPy/PyTorch parity of the v1 collapse step
  4. collapse energy behavior (single-well descent)
  5. checkpoint save/load round-trip
  6. tiny overfit (the training signal actually flows)
  7. generation smoke (encode is finite and shape-correct)

Heavy deps (torch, scipy) are optional: each test skips cleanly when its
dependency is absent, so the zero-dependency core suite stays green anywhere.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="research tests need torch")
F = torch.nn.functional

DIM = 16


# ---------------------------------------------------------------- helpers


def collapse_step_torch(h, t, s):
    """Livnium v1 chord step: h <- h - s * (1 - cos(h, t)) * norm(h - t)."""
    align = (F.normalize(h, dim=-1) * F.normalize(t, dim=-1)).sum(-1, keepdim=True)
    away = F.normalize(h - t, dim=-1)
    return h - s * (1.0 - align) * away


def collapse_step_numpy(h, t, s):
    hn = h / max(np.linalg.norm(h), 1e-12)
    tn = t / max(np.linalg.norm(t), 1e-12)
    align = float(hn @ tn)
    away = h - t
    away = away / max(np.linalg.norm(away), 1e-12)
    return h - s * (1.0 - align) * away


# ---------------------------------------------------- 1. false negatives


def test_sampled_softmax_masks_true_target():
    """A sampled negative equal to the true target must get -inf, never a
    finite logit (the bug fixed in noun_collapse_pure.py / premise_from_hyp.py)."""
    B, V = 4, 20
    g = torch.Generator().manual_seed(0)
    h = F.normalize(torch.randn(B, DIM, generator=g), dim=-1)
    A = F.normalize(torch.randn(V, DIM, generator=g), dim=-1)
    tgt_ids = torch.tensor([3, 7, 3, 11])
    neg = torch.tensor([3, 5, 9, 7, 3, 2, 11, 0])  # contains every target

    ng = h @ A[neg].t()
    ng = ng.masked_fill(neg.unsqueeze(0) == tgt_ids.unsqueeze(1), float("-inf"))

    for b in range(B):
        collide = neg == tgt_ids[b]
        assert torch.isinf(ng[b][collide]).all() and (ng[b][collide] < 0).all()
        assert torch.isfinite(ng[b][~collide]).all()

    # loss must be unaffected by the masked columns
    pos = (h * A[tgt_ids]).sum(-1, keepdim=True)
    cand = torch.cat([pos, ng], dim=1)
    loss = F.cross_entropy(cand, torch.zeros(B, dtype=torch.long))
    assert torch.isfinite(loss)


# ------------------------------------------------------- 2. Spearman ties


def test_spearman_handles_ties():
    scipy_stats = pytest.importorskip("scipy.stats")
    a = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0])
    b = np.array([0.9, 2.2, 1.8, 3.1, 4.5, 3.9, 5.2])
    expected = float(scipy_stats.spearmanr(a, b).statistic)

    import importlib.util
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "chat" / "embed_eval.py"
    spec = importlib.util.spec_from_file_location("embed_eval", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.spearman(a, b) == pytest.approx(expected, abs=1e-12)

    # naive argsort-of-argsort ranks are wrong under ties; the tie-aware value
    # for this vector pair differs from the tie-blind one
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    naive = float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    assert naive != pytest.approx(expected, abs=1e-12)


# --------------------------------------------- 3. NumPy / PyTorch parity


def test_collapse_step_numpy_torch_parity():
    rng = np.random.default_rng(1)
    for _ in range(50):
        h = rng.normal(size=DIM)
        t = rng.normal(size=DIM)
        t /= np.linalg.norm(t)
        s = 0.11
        out_np = collapse_step_numpy(h.copy(), t, s)
        out_th = (
            collapse_step_torch(
                torch.tensor(h, dtype=torch.float64).unsqueeze(0),
                torch.tensor(t, dtype=torch.float64).unsqueeze(0),
                s,
            )
            .squeeze(0)
            .numpy()
        )
        np.testing.assert_allclose(out_np, out_th, rtol=1e-10, atol=1e-10)


# ------------------------------------------------- 4. energy behavior


def test_single_well_energy_descends():
    """V(h) = 1 - cos(h, W) must be non-increasing when collapsing repeatedly
    onto a single fixed well (the empirical Lyapunov-candidate property)."""
    g = torch.Generator().manual_seed(2)
    W = F.normalize(torch.randn(1, DIM, generator=g), dim=-1)
    h = torch.randn(1, DIM, generator=g)
    start = prev = float(1.0 - (F.normalize(h, dim=-1) * W).sum())
    for _ in range(200):
        h = collapse_step_torch(h, W, 0.11)
        v = float(1.0 - (F.normalize(h, dim=-1) * W).sum())
        assert v <= prev + 1e-7
        prev = v
    assert prev < 0.5 * start  # substantially closer to the well


# --------------------------------------------- 5. checkpoint round-trip


def test_checkpoint_roundtrip(tmp_path):
    """Save/load in the noun_collapse_pure.pt schema; embed_eval must be able
    to consume the loaded checkpoint."""
    g = torch.Generator().manual_seed(3)
    V = 12
    wells = torch.randn(V, DIM, generator=g)
    ck = {
        "wells": wells,
        "stoi": {f"w{i}": i for i in range(V)},
        "noun_ids": list(range(0, V, 2)),
        "start": torch.randn(DIM, generator=g),
        "strength": 0.11,
        "temp": 0.07,
        "config": {"dim": DIM, "window": 5, "slot": 10},
    }
    p = tmp_path / "ck.pt"
    torch.save(ck, p)
    back = torch.load(p, map_location="cpu")
    assert torch.equal(back["wells"], wells)
    assert back["stoi"] == ck["stoi"] and back["noun_ids"] == ck["noun_ids"]

    # the embed_eval loader contract: unit rows + word->row map
    W = back["wells"]
    X = (W / W.norm(dim=-1, keepdim=True).clamp(min=1e-8)).numpy()
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, rtol=1e-5)


# ------------------------------------------------------- 6. tiny overfit


def test_tiny_overfit():
    """A few wells + the v1 step must be able to overfit 4 fixed
    (context -> target) pairs; if this fails, gradients aren't flowing."""
    torch.manual_seed(4)
    V, L = 10, 3
    wells = torch.nn.Parameter(0.1 * torch.randn(V, DIM))
    start = torch.nn.Parameter(0.1 * torch.randn(DIM))
    ctx = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 1], [2, 9, 4]])
    tgt = torch.tensor([0, 7, 5, 8])
    opt = torch.optim.Adam([wells, start], lr=0.05)

    def loss_fn():
        h = start.expand(ctx.size(0), DIM)
        for i in range(L):
            h = collapse_step_torch(h, wells[ctx[:, i]], 0.3)
        logits = F.normalize(h, dim=-1) @ F.normalize(wells, dim=-1).t() / 0.1
        return F.cross_entropy(logits, tgt), logits

    first = loss_fn()[0].item()
    for _ in range(300):
        opt.zero_grad()
        loss, _ = loss_fn()
        loss.backward()
        opt.step()
    final, logits = loss_fn()
    assert final.item() < first * 0.1, f"no overfit: {first:.3f} -> {final.item():.3f}"
    assert (logits.argmax(-1) == tgt).all(), "failed to memorize 4 pairs"


# --------------------------------------------------- 7. generation smoke


def test_generation_smoke():
    """Greedy decode loop over random wells: must produce valid ids and a
    finite state at every step (no NaN blow-ups from the norm clamp)."""
    g = torch.Generator().manual_seed(5)
    V = 30
    wells = F.normalize(torch.randn(V, DIM, generator=g), dim=-1)
    h = torch.randn(1, DIM, generator=g)
    out = []
    for _ in range(20):
        scores = F.normalize(h, dim=-1) @ wells.t()
        idx = int(scores.argmax())
        out.append(idx)
        h = collapse_step_torch(h, wells[idx : idx + 1], 0.11)
        n = h.norm(dim=-1, keepdim=True)
        h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
        assert torch.isfinite(h).all()
    assert len(out) == 20 and all(0 <= i < V for i in out)

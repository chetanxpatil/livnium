#!/usr/bin/env python3
"""
Read-only forensic controls for the cube_embed archive.

The probe does not edit cube_embed.  It checks:
  * inventory and source hashes
  * v1 graph topology against its documentation
  * the rank and geometry of the v2 Fourier/QR probe matrix
  * exact sign ambiguity and signature redundancy
  * basis-orientation and character-hash reproducibility
  * sentence-state parameter degeneracies
  * SimLex resubstitution versus pair-held-out counter-fitting
  * whether the chosen cube layout beats random probe-to-position layouts

Run from the archive root:
  PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
      python3 .codex_memory_staging/CUBE_EMBED_AUDIT_PROBE.py

Use --skip-semantic for the fast algebra-only pass.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CUBE = ROOT / "cube_embed"
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from cube_embed import angle_ops
from cube_embed.angle_embed import AngleCubeEmbed
from cube_embed.cooc_ops import build_semantic_vectors
from cube_embed.counter_fit import counter_fit, pairs_from_simlex
from cube_embed.field import CubeField, EDGES, NEIGHBORS
from cube_embed.graph import ADJ, ALL_NODES, CubeGraph
from cube_embed.loops import ALL_LOOPS
from cube_embed.sentence import SentenceField
from cube_embed.simlex_eval import BUILTIN_PAIRS, spearmanr
from cube_embed.word_ops import WordOperator


def heading(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-12 else 0.0


def corr(a: np.ndarray, b: np.ndarray) -> float:
    ac = np.asarray(a, dtype=np.float64) - float(np.mean(a))
    bc = np.asarray(b, dtype=np.float64) - float(np.mean(b))
    return cosine(ac, bc)


def holo_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return (
        0.5 * corr(sig_a[81:], sig_b[81:])
        + 0.3 * corr(sig_a[:54], sig_b[:54])
        + 0.2 * corr(sig_a[54:81], sig_b[54:81])
    )


def field_signature_from_base(base: np.ndarray) -> np.ndarray:
    base = np.asarray(base, dtype=np.float64)
    base /= np.linalg.norm(base)
    probes = angle_ops._get_probes().astype(np.float64)
    acts = probes @ base
    raw = base[None, :] + angle_ops._ALPHA * acts[:, None] * probes
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    field = CubeField()
    field.load(raw.astype(np.float32))
    return field.signature(ALL_LOOPS).astype(np.float64)


def score_pair(
    vecs: Dict[str, np.ndarray], ace: AngleCubeEmbed, w1: str, w2: str
) -> Tuple[float, float]:
    return cosine(vecs[w1], vecs[w2]), ace.similarity(w1, w2)


def scored_triples(
    pairs: Sequence[Tuple[str, str, float]], vecs: Dict[str, np.ndarray]
) -> List[Tuple[str, str, float]]:
    vocab = set(vecs)
    return [(w1, w2, h) for w1, w2, h in pairs if w1 in vocab and w2 in vocab]


def score_triples(
    triples: Sequence[Tuple[str, str, float]],
    vecs: Dict[str, np.ndarray],
    ace: AngleCubeEmbed,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    human, svd, holo = [], [], []
    for w1, w2, h in triples:
        s, g = score_pair(vecs, ace, w1, w2)
        human.append(h)
        svd.append(s)
        holo.append(g)
    return (
        np.asarray(human, dtype=np.float64),
        np.asarray(svd, dtype=np.float64),
        np.asarray(holo, dtype=np.float64),
    )


def inventory_probe() -> None:
    heading("A. INVENTORY AND HASHES")
    files = sorted(p for p in CUBE.rglob("*") if p.is_file())
    live_py = [p for p in files if p.suffix == ".py"]
    pyc = [p for p in files if p.suffix == ".pyc"]
    artifacts = [
        p
        for p in files
        if p.suffix.lower()
        in {".json", ".jsonl", ".log", ".csv", ".tsv", ".png", ".pdf", ".pt", ".pth"}
    ]
    print(
        f"files={len(files)} live_python={len(live_py)} historical_pyc={len(pyc)} "
        f"saved_result_like_files={len(artifacts)}"
    )
    for path in files:
        rel = path.relative_to(ROOT)
        print(f"{sha256(path)}  {path.stat().st_size:>10}  {rel}")


def graph_probe() -> None:
    heading("B. V1 GRAPH TOPOLOGY AND ZERO-STATE OPERATORS")
    degree_by_kind: Dict[str, List[int]] = {"face": [], "edge": [], "corner": []}
    expected = {"face": 8, "edge": 4, "corner": 6}
    for nid, _label, faces in ALL_NODES:
        kind = {1: "face", 2: "edge", 3: "corner"}[len(faces)]
        degree_by_kind[kind].append(len(ADJ[nid]))
    for kind in ("face", "edge", "corner"):
        vals = degree_by_kind[kind]
        print(
            f"{kind:>6}: documented_degree={expected[kind]} "
            f"actual_unique={sorted(set(vals))} values={vals}"
        )

    for word in ("cold", "not", "abstraction", "love", "and"):
        op = WordOperator(word)
        graph = CubeGraph()
        before = np.asarray(graph.act)
        op.apply(graph)
        after = np.asarray(graph.act)
        primary_nodes = [
            nid for nid, _label, faces in ALL_NODES if op.primary in faces
        ]
        print(
            f"{word:>12}: op={op.op_type:<7} energy={op.energy:.4f} "
            f"total_after={after.sum():.4f} "
            f"primary_after={after[primary_nodes].sum():.4f} "
            f"changed_nodes={int(np.count_nonzero(after - before))}"
        )


def raw_fourier_matrix() -> np.ndarray:
    raw = np.empty((27, 27), dtype=np.float64)
    for p in range(27):
        x, y, z = p // 9, (p % 9) // 3, p % 3
        for d in range(27):
            kx, ky, kz = d // 9, (d % 9) // 3, d % 3
            phase = (2.0 * math.pi / 3.0) * (
                x * kx + y * ky + z * kz
            )
            raw[p, d] = math.cos(phase)
    return raw


def algebra_probe() -> None:
    heading("C. V2 PROBE MATRIX, SIGN AMBIGUITY, AND REDUNDANCY")
    raw = raw_fourier_matrix()
    probes = angle_ops._get_probes().astype(np.float64)
    singular = np.linalg.svd(raw, compute_uv=False)
    raw_rank = int(np.linalg.matrix_rank(raw, tol=1e-10))
    ortho_err = float(np.max(np.abs(probes @ probes.T - np.eye(27))))

    edge_set = {tuple(sorted(e)) for e in EDGES}
    neighbor_dots, nonneighbor_dots = [], []
    for p in range(27):
        for q in range(p + 1, 27):
            dot = float(np.dot(probes[p], probes[q]))
            if (p, q) in edge_set:
                neighbor_dots.append(dot)
            else:
                nonneighbor_dots.append(dot)

    print(f"raw_cosine_fourier_rank={raw_rank}/27")
    print(
        "raw_singular_values="
        + np.array2string(singular, precision=6, separator=",")
    )
    print(f"qr_probe_max_orthogonality_error={ortho_err:.3e}")
    print(
        f"final_neighbor_dot mean={np.mean(neighbor_dots):+.3e} "
        f"std={np.std(neighbor_dots):.3e}; "
        f"nonneighbor mean={np.mean(nonneighbor_dots):+.3e} "
        f"std={np.std(nonneighbor_dots):.3e}"
    )

    rng = np.random.RandomState(20260726)
    base = rng.randn(27)
    base /= np.linalg.norm(base)
    coeff = probes @ base
    flipped = probes.T @ (-coeff)
    independent_signs = rng.choice([-1.0, 1.0], size=27)
    sign_flipped = probes.T @ (coeff * independent_signs)
    sig = field_signature_from_base(base)
    sig_neg = field_signature_from_base(flipped)
    sig_many = field_signature_from_base(sign_flipped)
    print(f"signature(base,-base)_max_abs_delta={np.max(np.abs(sig-sig_neg)):.3e}")
    print(
        "signature(independent_probe_sign_flips)_max_abs_delta="
        f"{np.max(np.abs(sig-sig_many)):.3e}"
    )

    alpha = float(angle_ops._ALPHA)
    vecs = base[None, :] + alpha * coeff[:, None] * probes
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    analytic_err = 0.0
    for p, q in EDGES:
        numerator = 1.0 + alpha * (coeff[p] ** 2 + coeff[q] ** 2)
        denominator = math.sqrt(
            (1.0 + (2 * alpha + alpha**2) * coeff[p] ** 2)
            * (1.0 + (2 * alpha + alpha**2) * coeff[q] ** 2)
        )
        analytic_err = max(
            analytic_err,
            abs(float(np.dot(vecs[p], vecs[q])) - numerator / denominator),
        )
    print(f"closed_form_edge_cosine_max_error={analytic_err:.3e}")

    field = CubeField()
    field.load(vecs.astype(np.float32))
    edge = field.edge_phase_vector().astype(np.float64)
    sig_direct = field.signature(ALL_LOOPS).astype(np.float64)
    edge_to_i = {tuple(sorted(e)): i for i, e in enumerate(EDGES)}
    loop_matrix = np.zeros((len(ALL_LOOPS), len(EDGES)), dtype=np.float64)
    for li, loop in enumerate(ALL_LOOPS):
        for i, p in enumerate(loop):
            q = loop[(i + 1) % len(loop)]
            loop_matrix[li, edge_to_i[tuple(sorted((p, q)))]] += 1.0 / len(loop)
    loop_rebuilt = loop_matrix @ edge
    neighbor_rebuilt = np.zeros(27, dtype=np.float64)
    for p in range(27):
        vals = []
        for q in NEIGHBORS[p]:
            vals.append(math.cos(math.pi * edge[edge_to_i[tuple(sorted((p, q)))]]))
        neighbor_rebuilt[p] = float(np.mean(vals))
    sig_rebuilt = np.concatenate([edge, neighbor_rebuilt, loop_rebuilt])
    print(f"loop_mapping_rank={np.linalg.matrix_rank(loop_matrix)}/13")
    print(
        f"loop_block_reconstruction_max_abs_delta="
        f"{np.max(np.abs(sig_direct[81:]-loop_rebuilt)):.3e}"
    )
    print(
        f"neighbor_block_reconstruction_max_abs_delta="
        f"{np.max(np.abs(sig_direct[54:81]-neighbor_rebuilt)):.3e}"
    )
    print(
        f"whole_94D_from_first_54D_max_abs_delta="
        f"{np.max(np.abs(sig_direct-sig_rebuilt)):.3e}"
    )
    loop = ALL_LOOPS[0]
    print(
        "loop_orientation_reversal_delta="
        f"{abs(field.loop_winding(loop)-field.loop_winding(list(reversed(loop)))):.3e}"
    )


def basis_and_hash_probe() -> None:
    heading("D. BASIS ORIENTATION AND CHARACTER-HASH REPRODUCIBILITY")
    rng = np.random.RandomState(314159)
    rotation, _ = np.linalg.qr(rng.randn(27, 27))
    raw_deltas, holo_deltas = [], []
    for _ in range(50):
        a = rng.randn(27)
        b = rng.randn(27)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        raw_deltas.append(abs(cosine(a, b) - cosine(rotation @ a, rotation @ b)))
        before = holo_similarity(
            field_signature_from_base(a), field_signature_from_base(b)
        )
        after = holo_similarity(
            field_signature_from_base(rotation @ a),
            field_signature_from_base(rotation @ b),
        )
        holo_deltas.append(abs(before - after))
    print(
        f"shared_rotation_raw_cosine_delta max={max(raw_deltas):.3e}; "
        f"holonomy_similarity_delta mean={np.mean(holo_deltas):.4f} "
        f"max={max(holo_deltas):.4f}"
    )

    code = (
        "from cube_embed.char_embed import char_vector;"
        "print(','.join(f'{x:.9f}' for x in char_vector('running')))"
    )
    outputs = []
    for seed in ("0", "1", "random"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(ROOT)
        out = subprocess.check_output(
            [sys.executable, "-c", code], cwd=str(ROOT), env=env, text=True
        ).strip()
        outputs.append(out)
    arrays = [np.fromstring(x, sep=",") for x in outputs]
    print(
        f"char_vector_seed0_vs_seed1_cosine={cosine(arrays[0], arrays[1]):+.6f}"
    )
    print(
        f"char_vector_seed0_vs_random_cosine={cosine(arrays[0], arrays[2]):+.6f}"
    )
    print(f"char_vector_outputs_all_equal={len(set(outputs)) == 1}")


def sentence_probe() -> None:
    heading("E. SENTENCE-STATE DEGENERACIES")
    words = ["the", "bank", "river", "flooded"]
    a = SentenceField(alpha=0.7, beta=0.3).embed(words)
    scaled = SentenceField(alpha=0.35, beta=0.15).embed(words)
    last_only = SentenceField(alpha=0.0, beta=1.0).embed(words)
    last_word = SentenceField(alpha=0.0, beta=1.0).embed([words[-1]])
    sf = SentenceField(alpha=0.7, beta=0.3)
    returned = sf.embed(words)
    cumulative = sf.cumulative_holonomy()
    print(
        f"scale_both_alpha_beta_signature_delta={np.max(np.abs(a-scaled)):.3e}"
    )
    print(
        f"alpha_zero_full_sentence_vs_last_word_delta="
        f"{np.max(np.abs(last_only-last_word)):.3e}"
    )
    print(
        f"returned_embedding_dim={len(returned)} "
        f"tracked_cumulative_holonomy_dim={len(cumulative)}"
    )
    print(
        "cumulative_holonomy_is_returned_embedding="
        f"{len(returned) == len(cumulative) and np.array_equal(returned, cumulative)}"
    )


def semantic_controls() -> None:
    heading("F. SIMLEX LEAKAGE AND LAYOUT CONTROLS")
    wiki = CUBE / "wikitext-103" / "wiki.valid.tokens"
    all_words = sorted({w for w1, w2, _ in BUILTIN_PAIRS for w in (w1, w2)})
    print(
        f"built_in_pair_count={len(BUILTIN_PAIRS)} "
        f"(source prose says 150); unique_words={len(all_words)}"
    )
    _vocab, base_vecs = build_semantic_vectors(
        wiki_path=str(wiki),
        vocab_size=4000,
        window=4,
        dim=27,
        min_freq=1,
        target_vocab=all_words,
        verbose=True,
    )
    triples = scored_triples(BUILTIN_PAIRS, base_vecs)
    print(
        f"min_freq_1_scoreable_pairs={len(triples)} "
        f"oov_skipped={len(BUILTIN_PAIRS)-len(triples)}"
    )
    angle_ops.set_semantic_vectors(base_vecs)
    raw_ace = AngleCubeEmbed()
    human, raw_svd, raw_holo = score_triples(triples, base_vecs, raw_ace)
    raw_svd_rho = spearmanr(human, raw_svd)[0]
    raw_holo_rho = spearmanr(human, raw_holo)[0]
    print(f"raw_svd_rho={raw_svd_rho:+.4f}")
    print(f"raw_holonomy_rho={raw_holo_rho:+.4f}")

    all_syn, all_ant = pairs_from_simlex(triples, 8.0, 2.0)
    in_sample_vecs = counter_fit(
        base_vecs,
        all_syn,
        all_ant,
        n_iter=150,
        lr=0.05,
        gamma_syn=0.9,
        gamma_ant=-0.3,
        reg=0.05,
        verbose=False,
    )
    angle_ops.set_semantic_vectors(in_sample_vecs)
    in_sample_ace = AngleCubeEmbed()
    _, in_svd, in_holo = score_triples(triples, in_sample_vecs, in_sample_ace)
    in_svd_rho = spearmanr(human, in_svd)[0]
    in_holo_rho = spearmanr(human, in_holo)[0]
    in_blend_rhos = [
        spearmanr(human, (1 - alpha) * in_svd + alpha * in_holo)[0]
        for alpha in np.linspace(0.0, 1.0, 11)
    ]
    print(
        f"in_sample_constraints={len(all_syn)}_syn+{len(all_ant)}_ant "
        f"svd_cf_rho={in_svd_rho:+.4f} holo_cf_rho={in_holo_rho:+.4f} "
        f"best_same_set_blend_rho={max(in_blend_rhos):+.4f}"
    )

    rng = np.random.RandomState(42)
    order = rng.permutation(len(triples))
    folds = np.array_split(order, 5)
    oof_svd = np.full(len(triples), np.nan)
    oof_holo = np.full(len(triples), np.nan)
    oof_blend = np.full(len(triples), np.nan)
    chosen_alphas: List[float] = []
    all_indices = np.arange(len(triples))
    for fold_i, test_idx in enumerate(folds, 1):
        train_idx = np.setdiff1d(all_indices, test_idx)
        train_triples = [triples[i] for i in train_idx]
        syn, ant = pairs_from_simlex(train_triples, 8.0, 2.0)
        fold_vecs = counter_fit(
            base_vecs,
            syn,
            ant,
            n_iter=150,
            lr=0.05,
            gamma_syn=0.9,
            gamma_ant=-0.3,
            reg=0.05,
            verbose=False,
        )
        angle_ops.set_semantic_vectors(fold_vecs)
        fold_ace = AngleCubeEmbed()
        train_h, train_s, train_g = score_triples(
            train_triples, fold_vecs, fold_ace
        )
        candidates = np.linspace(0.0, 1.0, 11)
        train_rhos = [
            spearmanr(train_h, (1 - alpha) * train_s + alpha * train_g)[0]
            for alpha in candidates
        ]
        best_alpha = float(candidates[int(np.argmax(train_rhos))])
        chosen_alphas.append(best_alpha)
        test_triples = [triples[i] for i in test_idx]
        _test_h, test_s, test_g = score_triples(
            test_triples, fold_vecs, fold_ace
        )
        oof_svd[test_idx] = test_s
        oof_holo[test_idx] = test_g
        oof_blend[test_idx] = (
            (1 - best_alpha) * test_s + best_alpha * test_g
        )
        print(
            f"fold={fold_i} train={len(train_idx)} test={len(test_idx)} "
            f"constraints={len(syn)}+{len(ant)} best_train_alpha={best_alpha:.1f}"
        )
    print(
        f"pair_held_out_oof_svd_cf_rho={spearmanr(human,oof_svd)[0]:+.4f}"
    )
    print(
        f"pair_held_out_oof_holonomy_cf_rho={spearmanr(human,oof_holo)[0]:+.4f}"
    )
    print(
        f"pair_held_out_oof_tuned_blend_rho={spearmanr(human,oof_blend)[0]:+.4f} "
        f"fold_alphas={chosen_alphas}"
    )

    original_probes = angle_ops._get_probes().copy()
    angle_ops.set_semantic_vectors(base_vecs)
    source_layout_rho = raw_holo_rho
    layout_rhos = []
    try:
        for _ in range(20):
            angle_ops._PROBE_MATRIX = original_probes[rng.permutation(27)].copy()
            perm_ace = AngleCubeEmbed()
            _, _, perm_holo = score_triples(triples, base_vecs, perm_ace)
            layout_rhos.append(spearmanr(human, perm_holo)[0])
    finally:
        angle_ops._PROBE_MATRIX = original_probes
        angle_ops.set_semantic_vectors(base_vecs)
    rank = 1 + sum(r > source_layout_rho for r in layout_rhos)
    print(
        f"source_layout_raw_holonomy_rho={source_layout_rho:+.4f}; "
        f"20_random_layouts mean={np.mean(layout_rhos):+.4f} "
        f"std={np.std(layout_rhos):.4f} min={min(layout_rhos):+.4f} "
        f"max={max(layout_rhos):+.4f}; source_rank={rank}/21"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Run only fast inventory/algebra/reproducibility controls.",
    )
    args = parser.parse_args()
    print(f"archive_root={ROOT}")
    print(f"python={sys.version.split()[0]} numpy={np.__version__}")
    print(f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED', '<unset>')}")
    inventory_probe()
    graph_probe()
    algebra_probe()
    basis_and_hash_probe()
    sentence_probe()
    if not args.skip_semantic:
        semantic_controls()
    angle_ops.clear_semantic_vectors()


if __name__ == "__main__":
    main()

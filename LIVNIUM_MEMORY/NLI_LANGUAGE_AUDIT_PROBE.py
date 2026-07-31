#!/usr/bin/env python3
"""
Read-only controls for the recovered organized NLI-Language lineage.

The probe never executes experiment_snli.py's destructive top-level runner and
never saves a source artifact. It checks exact preservation copies, saved
arrays/checkpoints, ideal adaptive-code lengths, character-cluster controls,
word-hash/BoW comparisons, neural n-gram accounting, and saved basin states.

Run:
  PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
      python3 /Users/chetanpatil/Desktop/LIVNIUM_MEMORY/NLI_LANGUAGE_AUDIT_PROBE.py
"""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import types

import numpy as np


ROOT = Path("/Users/chetanpatil/Desktop/test")
ORG = ROOT / "_ORGANIZED" / "02_Experiments" / "NLI-Language"
STATE = ROOT / "state" / "exp_snli"
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def heading(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def inventory_probe() -> None:
    heading("A. PRESERVATION AND DATA IDENTITY")
    files = sorted(p for p in ORG.rglob("*") if p.is_file())
    exact = 0
    for path in files:
        root_copy = ROOT / path.name
        same = (
            root_copy.is_file()
            and root_copy.stat().st_size == path.stat().st_size
            and sha256(root_copy) == sha256(path)
        )
        exact += int(same)
        print(
            f"{path.relative_to(ORG)!s:<64} bytes={path.stat().st_size:<9} "
            f"sha={sha256(path)} exact_root={same}"
        )
    print(f"organized_files={len(files)} exact_same_named_root={exact}")

    datasets = [
        ROOT / "data" / f"snli_1.0_{split}.jsonl"
        for split in ("train", "dev", "test")
    ] + [
        ROOT / "anli_data" / f"anli_{split}_r{rnd}.parquet"
        for rnd in (1, 2, 3)
        for split in ("train", "test")
    ]
    for path in datasets:
        print(
            f"dataset={path.relative_to(ROOT)} bytes={path.stat().st_size} "
            f"sha={sha256(path)}"
        )


def artifact_probe() -> None:
    heading("B. SAVED ARRAY AND CHECKPOINT ARTIFACTS")
    emb_path = ORG / "_rung2_emb.npz"
    vocab_path = ORG / "_rung2_vocab.json"
    matrix = np.load(emb_path)["M"]
    vocab = json.loads(vocab_path.read_text())
    print(
        f"trimmed_glove_shape={matrix.shape} dtype={matrix.dtype} "
        f"finite={np.isfinite(matrix).all()} vocab_entries={len(vocab)} "
        f"index_bijection={set(vocab.values()) == set(range(len(vocab)))}"
    )

    ckpt = np.load(ORG / "rung3_ckpt.npz", allow_pickle=True)
    current_is_best = all(
        np.array_equal(ckpt[f"p{i}"], ckpt[f"b{i}"]) for i in range(5)
    )
    params = sum(ckpt[f"b{i}"].size for i in range(5))
    print(
        f"rung3_epoch={int(ckpt['epoch'])} best_val={float(ckpt['best_val']):.12f} "
        f"optimizer_steps={int(ckpt['t'])} params={params} "
        f"current_equals_best={current_is_best}"
    )


def compression_probe() -> None:
    heading("C. ADAPTIVE COMPRESSION AND SURPRISE ACCOUNTING")
    mod = load_module("rung2_honest_readonly", ROOT / "rung2_honest_compression.py")
    data = mod.load_hypotheses(ROOT / "data" / "snli_1.0_dev.jsonl", 600_000)
    base = mod.classic(data)
    print(f"corpus_bytes={len(data)} distinct_bytes={len(set(data))}")
    print(f"classic={base}")
    for order in (1, 2, 3, 4):
        bits, costs, model = mod.adaptive_bits(data, order)
        print(
            f"adaptive_K={order} ideal_bpc={bits / len(data):.12f} "
            f"plus_64bit_length={(bits + 64) / len(data):.12f}"
        )
        if order == 3:
            arr = np.asarray(costs)
            easy = arr < 1.0
            easy_counts = Counter(
                data[index] for index in np.flatnonzero(easy)
            )
            print(
                f"easy_char_fraction={easy.mean():.12f} "
                f"easy_bit_share={arr[easy].sum() / arr.sum():.12f} "
                f"easy_top_bytes={easy_counts.most_common(8)}"
            )
        if order == 4:
            for context in (b"", b" ", b"the", b" man", b"zzzz"):
                total = sum(model.prob(context, sym) for sym in range(256))
                print(f"probability_sum context={context!r} total={total:.15f}")

    train = mod.load_hypotheses(ROOT / "data" / "snli_1.0_train.jsonl", 3_000_000)
    test = mod.load_hypotheses(ROOT / "data" / "snli_1.0_test.jsonl", 400_000)
    fitted = mod.train_counts(train, 3)
    train_bpc = mod.score_frozen(fitted, train, 3) / len(train)
    test_bpc = mod.score_frozen(fitted, test, 3) / len(test)
    entries = sum(len(fitted.tot[j]) for j in range(4))
    print(
        f"frozen_train_bpc={train_bpc:.12f} "
        f"heldout_test_bpc={test_bpc:.12f} entries={entries}"
    )
    for threshold in (5, 10):
        pruned = mod.train_counts(train, 3)
        kept, removed = mod.prune(pruned, 3, threshold)
        bpc = mod.score_frozen(pruned, test, 3) / len(test)
        print(
            f"prune_lt={threshold} kept={kept} removed={removed} "
            f"heldout_test_bpc={bpc:.12f}"
        )


def character_partition_probe() -> None:
    heading("D. CHARACTER PARTITION CONTROL")
    mod = load_module("decisive_readonly", ROOT / "livnium_decisive_test.py")
    train = mod.load_hyp(ROOT / "data" / "snli_1.0_train.jsonl", 1_500_000)
    test = mod.load_hyp(ROOT / "data" / "snli_1.0_test.jsonl", 300_000)
    class_count = 5
    full = np.arange(256, dtype=np.int64)
    unigram = mod.make_table(
        {char: 1 for char in "abcdefghijklmnopqrstuvwxyz"}, 3
    )
    livnium = mod.eval_bpc(
        train, test, mod.make_table(mod.livnium_map(), class_count), class_count
    )
    random_scores = [
        mod.eval_bpc(
            train,
            test,
            mod.make_table(
                mod.random_map([6, 12, 8], np.random.default_rng(seed)),
                class_count,
            ),
            class_count,
        )
        for seed in range(8)
    ]
    optimized_map = mod.optimized_map(train, 3)
    optimized = mod.eval_bpc(
        train, test, mod.make_table(optimized_map, class_count), class_count
    )
    print(
        f"full_char={mod.eval_bpc(train, test, full, 256):.12f} "
        f"optimized={optimized:.12f} livnium={livnium:.12f} "
        f"random_mean={np.mean(random_scores):.12f} "
        f"random_sd={np.std(random_scores):.12f} "
        f"unigram={mod.eval_bpc(train, test, unigram, 3):.12f}"
    )
    print(f"random_scores={random_scores}")
    saved_rows = np.genfromtxt(
        ORG / "livnium_decisive_test.csv",
        delimiter=",",
        dtype=str,
        skip_header=1,
    )
    print(f"saved_csv_rows={dict(saved_rows)}")


def load_word_snli(mod, split: str):
    premise, hypothesis, labels = [], [], []
    path = ROOT / "livnium-sacred-v2" / "data" / "snli" / f"snli_1.0_{split}.jsonl"
    for line in path.open():
        row = json.loads(line)
        label = row.get("gold_label")
        if label in mod.LABELS:
            premise.append(row["sentence1"])
            hypothesis.append(row["sentence2"])
            labels.append(mod.LABELS[label])
    return premise, hypothesis, np.asarray(labels)


def word_hash_probe() -> None:
    heading("E. WORD-HASH OCCUPANCY VERSUS BAG OF WORDS")
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    mod = load_module("word_decisive_readonly", ROOT / "livnium_word_decisive.py")
    train_p, train_h, train_y = load_word_snli(mod, "train")
    test_p, test_h, test_y = load_word_snli(mod, "test")
    selection = np.random.RandomState(42).permutation(len(train_y))[:50_000]
    train_p = [train_p[i] for i in selection]
    train_h = [train_h[i] for i in selection]
    train_y = train_y[selection]

    words_p = {word for sentence in train_p for word in mod.words(sentence)}
    words_h = {word for sentence in train_h for word in mod.words(sentence)}
    all_words = words_p | words_h
    buckets: dict[int, list[str]] = {}
    for word in all_words:
        buckets.setdefault(mod.cell_of(word), []).append(word)
    print(
        f"unique_words={len(all_words)} occupied_buckets={len(buckets)} "
        f"collision_buckets={sum(len(v) > 1 for v in buckets.values())} "
        f"words_in_collision_buckets="
        f"{sum(len(v) for v in buckets.values() if len(v) > 1)} "
        f"max_bucket={max(map(len, buckets.values()))}"
    )

    cube_train = mod.cube_X(train_p, train_h)
    cube_test = mod.cube_X(test_p, test_h)
    vec_p = CountVectorizer(token_pattern=r"[a-z']+", lowercase=True).fit(train_p)
    vec_h = CountVectorizer(token_pattern=r"[a-z']+", lowercase=True).fit(train_h)
    bow_train = hstack([vec_p.transform(train_p), vec_h.transform(train_h)]).tocsr()
    bow_test = hstack([vec_p.transform(test_p), vec_h.transform(test_h)]).tocsr()
    geom_train = np.hstack([mod.geom(train_p), mod.geom(train_h)])
    geom_test = np.hstack([mod.geom(test_p), mod.geom(test_h)])
    combo_train = hstack([bow_train, csr_matrix(geom_train)]).tocsr()
    combo_test = hstack([bow_test, csr_matrix(geom_test)]).tocsr()

    print(
        f"cube_shape={cube_train.shape} bow_shape={bow_train.shape} "
        f"geometry_shape={geom_train.shape}"
    )
    for name, x_train, x_test, dense in (
        ("cube", cube_train, cube_test, False),
        ("bow", bow_train, bow_test, False),
        ("geometry", geom_train, geom_test, True),
        ("bow_plus_geometry", combo_train, combo_test, False),
    ):
        clf = LogisticRegression(
            max_iter=300 if dense else 200, solver="liblinear"
        ).fit(x_train, train_y)
        result = accuracy_score(test_y, clf.predict(x_test)) * 100
        print(f"{name}_test_accuracy={result:.12f}")


def rung3_probe() -> None:
    heading("F. NEURAL N-GRAM REPLAY AND ACCOUNTING")
    mod = load_module("rung3_readonly", ROOT / "rung3_learned_model.py")
    train = mod.load_hyp(ROOT / "data" / "snli_1.0_train.jsonl", 1_200_000)
    test = mod.load_hyp(ROOT / "data" / "snli_1.0_test.jsonl", 300_000)
    neural_train = train[:-100_000]
    print(
        f"source_train_bytes={len(train)} neural_train_bytes={len(neural_train)} "
        f"validation_bytes=100000 test_bytes={len(test)} "
        f"test_only_bytes={sorted(set(test) - set(train))}"
    )
    for label, corpus in (
        ("source_full_train_including_validation", train),
        ("matched_neural_train", neural_train),
    ):
        for order in (4, 6):
            wb = mod.WB(order)
            wb.train(corpus)
            print(
                f"{label}_K={order} test_bpc={wb.bits_per_char(test):.12f} "
                f"contexts={sum(len(wb.tot[j]) for j in range(order + 1))}"
            )

    vocabulary = sorted(set(train) | set(test))
    lookup = {byte: index for index, byte in enumerate(vocabulary)}
    test_ids = np.asarray([lookup[byte] for byte in test], dtype=np.int64)
    x_test, y_test = mod.make_examples(test_ids, 8, len(vocabulary))
    network = mod.NeuralNGram(len(vocabulary), K=8, d=32, H=256)
    checkpoint = np.load(ROOT / "rung3_ckpt.npz", allow_pickle=True)
    for index, parameter in enumerate(network.params):
        parameter[...] = checkpoint[f"b{index}"]
    predictive_bpc = network.loss_bits(x_test, y_test)
    ideal_float16 = predictive_bpc + network.n_params() * 16 / len(test)
    actual_npz = (
        predictive_bpc
        + (ROOT / "rung3_ckpt.npz").stat().st_size * 8 / len(test)
    )
    print(
        f"neural_predictive_bpc={predictive_bpc:.12f} "
        f"ideal_float16_total_bpc={ideal_float16:.12f} "
        f"actual_saved_npz_total_bpc={actual_npz:.12f}"
    )


def load_experiment_feature_module():
    path = ROOT / "experiment_snli.py"
    source = path.read_text()
    prefix = source.split('DATA_DIR     = "data"')[0]
    module = types.ModuleType("experiment_features_readonly")
    module.__file__ = str(path)
    exec(compile(prefix, str(path), "exec"), module.__dict__)
    return module


def saved_basin_probe() -> None:
    heading("G. SAVED SNLI BASINS AND RECEIPTS")
    from demo_karma import KarmicBasinField
    from nova_basin_store import NovaBasinStore
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    for json_path in sorted(STATE.glob("*/basin.json")):
        state = json.loads(json_path.read_text())
        archive = Path(str(json_path) + ".ledger_archive.jsonl")
        previous = None
        archive_lines = 0
        breaks = 0
        if archive.exists():
            with archive.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    archive_lines += 1
                    if (
                        previous is not None
                        and row.get("state_hash_before") != previous
                    ):
                        breaks += 1
                    previous = row.get("state_hash_after")
        for row in state.get("ledger", []):
            if (
                previous is not None
                and row.get("state_hash_before") != previous
            ):
                breaks += 1
            previous = row.get("state_hash_after")
        canonical = json.dumps(
            state, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        dimensions = sorted(
            {
                len(anchor.get("center", []))
                for anchors in state.get("anchors", {}).values()
                for anchor in anchors
            }
        )
        print(
            f"{json_path.parent.name}: step={state.get('step')} dims={dimensions} "
            f"anchors={sum(len(x) for x in state.get('anchors', {}).values())} "
            f"archive={archive_lines} live={len(state.get('ledger', []))} "
            f"chain_breaks={breaks} "
            f"stored_hash_matches_current="
            f"{state.get('state_hash') == hashlib.sha256(canonical).hexdigest()}"
        )

    features = load_experiment_feature_module()
    train = features.load_snli(
        str(ROOT / "data" / "snli_1.0_train.jsonl"), 5_000, 42
    )
    dev = features.load_snli(
        str(ROOT / "data" / "snli_1.0_dev.jsonl"), 500, 99
    )
    labels = features.LABELS
    x_train = np.asarray(
        [features.encode_pair(p, h) for p, h, _ in train]
    )
    y_train = np.asarray([labels.index(label) for _, _, label in train])
    x_dev = np.asarray([features.encode_pair(p, h) for p, h, _ in dev])
    y_dev = np.asarray([labels.index(label) for _, _, label in dev])
    logistic = LogisticRegression(max_iter=1_000, C=1.0).fit(x_train, y_train)
    print(
        f"same_features_logistic_dev_accuracy="
        f"{accuracy_score(y_dev, logistic.predict(x_dev)):.12f}"
    )

    def predict(vector, store, mode):
        best_score = -math.inf
        best_label = labels[0]
        for label in labels:
            others = [candidate for candidate in labels if candidate != label]
            if mode == "naive_pull":
                score = store.score(
                    vector, label_good=label, label_bad=None, mode=mode
                )[0]
            else:
                score = sum(
                    store.score(
                        vector, label_good=label, label_bad=other, mode=mode
                    )[2]
                    for other in others
                ) / 2
            if score > best_score:
                best_score, best_label = score, label
        return best_label

    for cheat in (False, True):
        for mode in ("naive_pull", "naive_both", "karmic"):
            field = KarmicBasinField(
                rng=random.Random(42),
                spawn_distance=0.30,
                decay_eta=0.08,
                max_decay_budget=0.10,
            )
            path = STATE / f"{mode}_cheat{cheat}" / "basin.json"
            store = NovaBasinStore(field, store_path=str(path))
            correct = 0
            for premise, hypothesis, true_label in dev:
                vector = features.encode_pair(
                    premise,
                    hypothesis,
                    cheat_label=true_label if cheat else None,
                )
                correct += predict(vector, store, mode) == true_label
            print(
                f"saved_{mode}_cheat{cheat}_dev_accuracy="
                f"{correct / len(dev):.12f} step={store.summary()['step']}"
            )


def v8_static_probe() -> None:
    heading("H. NLI-V8/NOVA STATIC FEATURE BOUNDARIES")
    mod = load_module("nli_v8_readonly", ROOT / "nli_v8_nova.py")
    vocab, embedding = mod.load_nova_embeddings(mod.NOVA_EMB)
    dev = mod.load_snli(mod.SNLI_DEV, max_n=500)
    fire = 0
    mean_errors = []
    max_errors = []
    oov = 0
    total = 0
    for row in dev:
        first = mod.tokenize(row["s1"])
        second = mod.tokenize(row["s2"])
        oov += sum(vocab.get(word) is None for word in first + second)
        total += len(first) + len(second)
        p_vecs = mod.get_word_vecs(first, vocab, embedding)
        h_vecs = mod.get_word_vecs(second, vocab, embedding)
        alignment = mod._warp.align(p_vecs, h_vecs)
        warp = mod.feat_warp(p_vecs, h_vecs)
        fracture = mod.feat_fracture(p_vecs, h_vecs, alignment)
        fire += fracture[0] > 0.5
        mean_errors.append(abs(fracture[2] - (1 - warp[0])))
        max_errors.append(abs(fracture[3] - (1 - warp[1])))
    print(
        f"dev500_oov_rate={oov / total:.12f} "
        f"fracture_fire_rate={fire / len(dev):.12f} "
        f"mean_energy_redundancy_maxerr={max(mean_errors):.12g} "
        f"max_energy_redundancy_maxerr={max(max_errors):.12g}"
    )
    print(
        "Exact full replay command: PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 "
        "python3 /Users/chetanpatil/Desktop/test/nli_v8_nova.py"
    )


def main() -> None:
    print(f"archive_root={ROOT}")
    inventory_probe()
    artifact_probe()
    compression_probe()
    character_partition_probe()
    word_hash_probe()
    rung3_probe()
    saved_basin_probe()
    v8_static_probe()


if __name__ == "__main__":
    main()

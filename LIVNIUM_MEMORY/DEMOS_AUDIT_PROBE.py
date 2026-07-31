#!/usr/bin/env python3
"""Read-only controls for the organized Livnium Demos family.

The historical demo scripts are never edited. Any persistence replay writes
only inside Python TemporaryDirectory instances. In particular, this probe
does not execute demo_nova_bridge.py's destructive top-level cleanup against
the archive's state/basin_memory.json.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import random
import statistics
import sys
import tempfile
from collections import Counter, deque
from datetime import datetime
from pathlib import Path


ARCHIVE = Path(
    os.environ.get("LIVNIUM_ARCHIVE_ROOT", "/Users/chetanpatil/Desktop/test")
).resolve()
ORGANIZED = ARCHIVE / "_ORGANIZED/02_Experiments/Demos"
STATE_PATH = ARCHIVE / "state/basin_memory.json"

sys.path.insert(0, str(ARCHIVE))

import demo_base27 as base27  # noqa: E402
import demo_karma as karma  # noqa: E402
from livnium.basins import BasinField  # noqa: E402
from livnium.canonical import CanonicalLattice, CanonicalState  # noqa: E402
from livnium.transforms import SwapSymbol  # noqa: E402
from nova_basin_store import NovaBasinStore  # noqa: E402


DEMO_FILES = (
    "demo_base27.py",
    "demo_feedback.py",
    "demo_inside.py",
    "demo_karma.py",
    "demo_learning.py",
    "demo_nova_bridge.py",
)


def heading(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(obj: object) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def execute_prefix(path: Path, marker: str) -> dict:
    source = path.read_text(encoding="utf-8")
    prefix = source.split(marker, 1)[0]
    namespace = {
        "__file__": str(path),
        "__name__": "livnium_demos_readonly_probe",
    }
    exec(compile(prefix, str(path), "exec"), namespace)
    return namespace


def preservation_probe() -> None:
    heading("A. PRESERVATION, MIRRORS, AND CHRONOLOGY")
    for name in DEMO_FILES:
        organized = ORGANIZED / name
        root = ARCHIVE / name
        exact = root.exists() and organized.read_bytes() == root.read_bytes()
        timestamp = datetime.fromtimestamp(organized.stat().st_mtime).isoformat(
            sep=" ", timespec="seconds"
        )
        print(
            f"{name:<22} bytes={organized.stat().st_size:<6} "
            f"sha={sha256(organized)} exact_root={exact} mtime={timestamp}"
        )

    for path in (
        ARCHIVE / "demo_nova_bridge.py",
        STATE_PATH,
        ARCHIVE / "demo_karma.py",
        ARCHIVE / "nova_basin_store.py",
    ):
        print(
            f"chronology={path.name:<42} "
            f"{datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=' ', timespec='seconds')}"
        )


def base27_probe() -> None:
    heading("B. BASE-27 NUMERAL AND ROTATION BOUNDARIES")
    examples = ("", "0", "00", "0a", "a", "00livnium", "livnium")
    rows = [
        (text, base27.string_to_int(text), base27.int_to_string(base27.string_to_int(text)))
        for text in examples
    ]
    print(f"integer_roundtrips={rows}")

    lattice = CanonicalLattice(3)
    order = base27.get_canonical_coordinate_order(lattice)
    identity = CanonicalState(lattice, dict(zip(order, base27.ALPHABET, strict=True)))
    identity_text = base27.state_to_base27_string(identity, order)
    recovered = base27.int_to_string(base27.string_to_int(identity_text))
    rotation = base27.CanonicalRotation.generate_group()[3]
    rotated_text = base27.state_to_base27_string(
        identity.apply_rotation(rotation), order
    )
    print(
        f"rotation_group_size={len(base27.CanonicalRotation.generate_group())} "
        f"identity_len={len(identity_text)} rotated_len={len(rotated_text)} "
        f"core_fixed={rotated_text[0] == '0'} "
        f"symbol_multiset_preserved={sorted(identity_text) == sorted(rotated_text)}"
    )
    print(
        f"identity_integer_roundtrip={recovered!r} "
        f"leading_core_digit_lost={recovered != identity_text} "
        f"negative_int_encoding={base27.int_to_string(-1)!r}"
    )


def learning_probe() -> None:
    heading("C. TOY BASIN LEARNING AND INITIALIZATION CONTROL")
    ns = execute_prefix(ARCHIVE / "demo_learning.py", "# RUN A")
    field_class = ns["BasinField"]
    class_a = ns["class_a"]
    class_b = ns["class_b"]
    examples = ns["examples"]

    swapped = field_class(rng=random.Random(0))
    swapped.spawn_anchor((-0.6, -0.6), "A")
    swapped.spawn_anchor((0.6, 0.6), "B")
    swapped_initial = ns["accuracy"](swapped, class_a, class_b)
    for epoch in range(8):
        ns["train_epoch"](swapped, examples, epoch)
    swapped_final = ns["accuracy"](swapped, class_a, class_b)

    source_rng = ns["rng"]
    chosen = field_class(rng=random.Random(0))
    chosen.spawn_anchor(
        (source_rng.uniform(-0.3, 0.3), source_rng.uniform(-0.3, 0.3)), "A"
    )
    chosen.spawn_anchor(
        (source_rng.uniform(-0.3, 0.3), source_rng.uniform(-0.3, 0.3)), "B"
    )
    chosen_initial = ns["accuracy"](chosen, class_a, class_b)
    for epoch in range(8):
        ns["train_epoch"](chosen, examples, epoch)
    chosen_final = ns["accuracy"](chosen, class_a, class_b)

    rows = []
    for seed in range(100):
        anchor_rng = random.Random(seed)
        field = field_class(rng=random.Random(seed))
        field.spawn_anchor(
            (anchor_rng.uniform(-0.3, 0.3), anchor_rng.uniform(-0.3, 0.3)), "A"
        )
        field.spawn_anchor(
            (anchor_rng.uniform(-0.3, 0.3), anchor_rng.uniform(-0.3, 0.3)), "B"
        )
        initial = ns["accuracy"](field, class_a, class_b)
        for epoch in range(8):
            ns["train_epoch"](field, examples, epoch)
        final = ns["accuracy"](field, class_a, class_b)
        rows.append((initial, final))

    direct_rule = (
        sum((x + y > 0) for x, y in class_a)
        + sum((x + y < 0) for x, y in class_b)
    ) / (len(class_a) + len(class_b))
    print(
        f"source_swapped_initial={swapped_initial:.3f} "
        f"source_swapped_final={swapped_final:.3f}"
    )
    print(
        f"source_chosen_random_initial={chosen_initial:.3f} "
        f"source_chosen_random_final={chosen_final:.3f}"
    )
    print(
        f"hundred_init_initial_mean={statistics.mean(x for x, _ in rows):.4f} "
        f"min={min(x for x, _ in rows):.3f} max={max(x for x, _ in rows):.3f} "
        f"perfect={sum(x == 1.0 for x, _ in rows)}"
    )
    print(
        f"hundred_init_final_mean={statistics.mean(y for _, y in rows):.4f} "
        f"min={min(y for _, y in rows):.3f} max={max(y for _, y in rows):.3f} "
        f"perfect={sum(y == 1.0 for _, y in rows)} "
        f"below_90={sum(y < 0.9 for _, y in rows)} "
        f"direct_sign_rule={direct_rule:.3f}"
    )


def tuple_neighbors(state: tuple[int, ...]) -> list[tuple[int, ...]]:
    blank = state.index(0)
    row, col = divmod(blank, 3)
    neighbors = []
    for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = row + drow, col + dcol
        if not (0 <= nr < 3 and 0 <= nc < 3):
            continue
        other = nr * 3 + nc
        candidate = list(state)
        candidate[blank], candidate[other] = candidate[other], candidate[blank]
        neighbors.append(tuple(candidate))
    return neighbors


def exact_depth(start: tuple[int, ...], target: tuple[int, ...]) -> int:
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        state, depth = queue.popleft()
        if state == target:
            return depth
        for candidate in tuple_neighbors(state):
            if candidate not in seen:
                seen.add(candidate)
                queue.append((candidate, depth + 1))
    raise AssertionError("unreachable puzzle")


def tuple_manhattan(state: tuple[int, ...]) -> int:
    total = 0
    for index, tile in enumerate(state):
        if tile == 0:
            continue
        target = tile - 1
        row, col = divmod(index, 3)
        trow, tcol = divmod(target, 3)
        total += abs(row - trow) + abs(col - tcol)
    return total


def inside_probe() -> None:
    heading("D. INSIDE-THE-ENGINE PUZZLE CONTROL")
    ns = execute_prefix(ARCHIVE / "demo_inside.py", 'print("=" * 70)')
    current = ns["puzzled"]
    board_cells = ns["board_cells"]
    start_symbols = tuple(current.cell_to_symbol[c] for c in board_cells)
    to_int = lambda symbol: 0 if symbol == "blank" else int(symbol[1:])
    start = tuple(to_int(symbol) for symbol in start_symbols)
    target = (1, 2, 3, 4, 5, 6, 7, 8, 0)

    temperature = 4.0
    rng = random.Random(42)
    accepted = rejected = 0
    solved_step = None
    for step in range(5001):
        if int(ns["manhattan"](current)) == 0:
            solved_step = step
            break
        blank = current.symbol_to_cell[ns["BLANK"]]
        neighbors = [
            cell
            for cell in ns["lattice"].neighbors(blank)
            if cell in set(board_cells)
        ]
        swap_with = rng.choice(neighbors)
        candidate = SwapSymbol(blank, swap_with).apply(current)
        before = ns["model"].total_energy(current)
        after = ns["model"].total_energy(candidate)
        delta = after - before
        if delta < 0:
            accept = True
        elif temperature <= 0:
            accept = False
        else:
            accept = rng.random() < math.exp(-delta / temperature)
        if accept:
            current = candidate
            accepted += 1
        else:
            rejected += 1
        if step % 500 == 0 and step > 0:
            temperature *= 0.8

    greedy = start
    greedy_steps = 0
    while greedy != target and greedy_steps < 100:
        greedy = min(tuple_neighbors(greedy), key=tuple_manhattan)
        greedy_steps += 1
    print(
        f"nominal_shuffle_steps=25 start={start} "
        f"initial_manhattan={tuple_manhattan(start)} "
        f"exact_shortest_depth={exact_depth(start, target)}"
    )
    print(
        f"source_solved_step={solved_step} source_steps={accepted + rejected} "
        f"accepted={accepted} rejected={rejected} "
        f"final_manhattan={int(ns['manhattan'](current))} "
        f"greedy_steps={greedy_steps}"
    )


def frozen_policy_game(
    field: karma.KarmicBasinField,
    mode: str,
    rng: random.Random,
    scorer=None,
) -> str:
    state = karma.fresh_state()
    turn = "X"
    result = "Ongoing"
    controller = karma.LawController(field) if mode == "karmic" else None
    while result == "Ongoing":
        grid = karma.to_grid(state)
        if turn == "X":
            empty = karma.empty_cells(grid)
            if mode == "off" or not field.anchors:
                target = rng.choice(empty)
            else:
                best_score = -math.inf
                target = empty[0]
                for cell in empty:
                    simulated = SwapSymbol(
                        cell, karma.reserve(state, "X_")
                    ).apply(state)
                    vec = karma.feat(karma.to_grid(simulated))
                    if scorer is not None:
                        _good, _bad, score = scorer(vec)
                    elif mode == "karmic":
                        _good, _bad, score = controller.score(vec)
                    elif mode == "naive_pull":
                        distance = field.distance_to_label(vec, "X_Win")
                        score = 1.0 / (distance + 0.01)
                    else:
                        dx = field.distance_to_label(vec, "X_Win")
                        good = 1.0 / (dx + 0.01)
                        bad = 0.0
                        if field.anchors.get("O_Win"):
                            do = field.distance_to_label(vec, "O_Win")
                            bad = 1.0 / (do + 0.01)
                        score = good - 0.5 * bad
                    score += rng.uniform(-0.05, 0.05)
                    if score > best_score:
                        best_score = score
                        target = cell
            state = SwapSymbol(karma.reserve(state, "X_"), target).apply(state)
            turn = "O"
        else:
            target = karma.heuristic_o(grid, rng)
            state = SwapSymbol(karma.reserve(state, "O_"), target).apply(state)
            turn = "X"
        result = karma.check_win(karma.to_grid(state))
    return result


def choose_symbolic(grid: dict, player: str, rng: random.Random):
    empty = karma.empty_cells(grid)
    other = "O" if player == "X" else "X"
    for mark in (player, other):
        for cell in empty:
            simulated = grid.copy()
            simulated[cell] = mark
            if karma.check_win(simulated) == mark:
                return cell
    center = (0, 0, 1)
    if center in empty:
        return center
    corners = [
        cell
        for cell in empty
        if cell in ((-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, 1))
    ]
    return rng.choice(corners or empty)


def symbolic_game(rng: random.Random) -> str:
    grid = {cell: "-" for cell in karma.board_cells}
    turn = "X"
    result = "Ongoing"
    while result == "Ongoing":
        grid[choose_symbolic(grid, turn, rng)] = turn
        result = karma.check_win(grid)
        turn = "O" if turn == "X" else "X"
    return result


def feedback_probe() -> None:
    heading("E. FEEDBACK/KARMA ONLINE, FROZEN, AND DIRECT CONTROLS")
    feedback = execute_prefix(ARCHIVE / "demo_feedback.py", "# ── Main")
    feedback_rows = []
    for mode, beta in (
        ("off", 0.0),
        ("pull", 0.0),
        ("pull_push", 0.1),
        ("pull_push", 0.25),
        ("pull_push", 0.5),
        ("pull_push", 1.0),
        ("pull_push", 2.0),
    ):
        result = feedback["run_proper"](mode, beta=beta, games=500, seed=42)
        feedback_rows.append(
            (
                mode,
                beta,
                result["wins"],
                result["losses"],
                result["draws"],
                result["path_diversity"],
            )
        )
    print(f"feedback_seed42_mode_beta_WLD_diversity={feedback_rows}")

    seeds = (1, 7, 42, 99, 2026)
    modes = ("off", "naive_pull", "naive_both", "karmic")
    original = karma.KarmicBasinField

    class CapturingField(original):
        created: list["CapturingField"] = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__class__.created.append(self)

    karma.KarmicBasinField = CapturingField
    try:
        for mode in modes:
            online = Counter()
            frozen = Counter()
            source_seed_42 = None
            global_steps = []
            o_bad_karma = []
            for seed in seeds:
                CapturingField.created.clear()
                result = karma.run(mode, games=500, seed=seed)
                field = CapturingField.created[-1]
                wins, losses, draws = (
                    int(value) for value in result["W/L/D"].split("/")
                )
                online.update({"X": wins, "O": losses, "Draw": draws})
                if seed == 42:
                    source_seed_42 = result["W/L/D"]
                before = {
                    label: [(anchor.center, anchor.count) for anchor in anchors]
                    for label, anchors in field.anchors.items()
                }
                local = Counter(
                    frozen_policy_game(
                        field, mode, random.Random(10_000 + seed)
                    )
                    for _ in range(500)
                )
                after = {
                    label: [(anchor.center, anchor.count) for anchor in anchors]
                    for label, anchors in field.anchors.items()
                }
                assert before == after
                frozen.update(local)
                global_steps.append(field._global_step)
                o_bad_karma.append(field.mean_bad_karma("O_Win"))
            print(
                f"mode={mode:<11} source_seed42={source_seed_42} "
                f"five_seed_online={dict(online)} "
                f"five_seed_frozen={dict(frozen)}"
            )
            if mode == "karmic":
                print(
                    f"karmic_global_steps={global_steps} "
                    f"karmic_O_Win_bad_karma={o_bad_karma}"
                )
    finally:
        karma.KarmicBasinField = original

    symbolic = Counter()
    for seed in seeds:
        rng = random.Random(seed)
        symbolic.update(symbolic_game(rng) for _ in range(500))
    print(f"symbolic_heuristic_X_vs_heuristic_O={dict(symbolic)}")
    print(
        "source_metric_boundary="
        "karma energy_jump compares each score with the first move, not consecutive moves"
    )


def saved_state_probe() -> None:
    heading("F. SAVED NOVA BRIDGE STATE AND RECEIPT CONTRACT")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    archive_path = Path(str(STATE_PATH) + ".ledger_archive.jsonl")
    archived = [
        json.loads(line)
        for line in archive_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    receipts = archived + state["ledger"]
    operations = Counter(receipt["op"] for receipt in receipts)
    unchanged = Counter(
        receipt["op"]
        for receipt in receipts
        if receipt["state_hash_before"] == receipt["state_hash_after"]
    )
    chain_breaks = sum(
        receipts[index]["state_hash_after"]
        != receipts[index + 1]["state_hash_before"]
        for index in range(len(receipts) - 1)
    )
    anchors = [
        (label, anchor)
        for label, values in state["anchors"].items()
        for anchor in values
    ]
    current_hash = hashlib.sha256(canonical(state)).hexdigest()
    print(
        f"live={len(state['ledger'])} archived={len(archived)} "
        f"physical_receipts={len(receipts)} "
        f"stored_ledger_total_count={state['ledger_total_count']} "
        f"operation_counts={dict(operations)}"
    )
    print(
        f"receipt_chain_breaks={chain_breaks} unchanged_hash_by_op={dict(unchanged)} "
        f"stored_self_hash_matches_current={state['state_hash'] == current_hash}"
    )
    print(
        f"anchors={len(anchors)} labels={dict(Counter(label for label, _ in anchors))} "
        f"statuses={dict(Counter(anchor['status'] for _, anchor in anchors))} "
        f"support_total={sum(anchor['support_count'] for _, anchor in anchors)} "
        f"harm_total={sum(anchor['harm_count'] for _, anchor in anchors)}"
    )
    print(
        "promoted_X_Win="
        + repr(
            [
                (
                    anchor["id"],
                    anchor["support_count"],
                    anchor["harm_count"],
                    anchor["bad_karma"],
                )
                for label, anchor in anchors
                if label == "X_Win" and anchor["status"] == "promoted"
            ]
        )
    )


def court_contract_probe() -> None:
    heading("G. COURT, HARM ATTRIBUTION, AND PERSISTENCE CONTROLS")
    with tempfile.TemporaryDirectory(prefix="livnium-demos-court-") as temp:
        field = karma.KarmicBasinField(
            rng=random.Random(1), spawn_distance=0.4, decay_eta=0.1
        )
        store = NovaBasinStore(field, store_path=str(Path(temp) / "court.json"))
        vec = (1.0, 0.0)
        for _ in range(6):
            anchor = store.reinforce(vec, "X_Win")
        aid = store._anchor_id(anchor)
        status_after_support = store._status[aid]
        for _ in range(30):
            store.decay(vec, "X_Win")
        print(
            f"post_promotion_harm_test status_after_support={status_after_support} "
            f"final_status={store._status[aid]} support={store._support_count[aid]} "
            f"harm={store._harm_count[aid]} "
            f"bad_karma={field.bad_karma_of(anchor):.3f}"
        )

    with tempfile.TemporaryDirectory(prefix="livnium-demos-attribution-") as temp:
        field = karma.KarmicBasinField(
            rng=random.Random(2), spawn_distance=0.4, decay_eta=0.1
        )
        store = NovaBasinStore(field, store_path=str(Path(temp) / "harm.json"))
        store.reinforce((1.0, 0.0), "X_Win")
        store.reinforce((0.0, 1.0), "X_Win")
        ids = [store._anchor_id(anchor) for anchor in field.anchors["X_Win"]]
        before = [store._harm_count[aid] for aid in ids]
        store.decay((1.0, 0.0), "X_Win")
        after = [store._harm_count[aid] for aid in ids]
        decay_receipts = [
            receipt
            for receipt in store._receipts
            if receipt.op == "decay"
        ]
        print(
            f"single_nearest_decay_two_anchor_harm before={before} after={after} "
            f"decay_receipts={len(decay_receipts)} "
            f"all_decay_hashes_unchanged={all(r.state_hash_before == r.state_hash_after for r in decay_receipts)}"
        )

    with tempfile.TemporaryDirectory(prefix="livnium-demos-status-") as temp:
        field = karma.KarmicBasinField(
            rng=random.Random(3), spawn_distance=0.4, decay_eta=0.1
        )
        store = NovaBasinStore(field, store_path=str(Path(temp) / "status.json"))
        anchor = store.reinforce((1.0, 0.0), "X_Win")
        aid = store._anchor_id(anchor)
        provisional = store.score((1.0, 0.0))
        store._status[aid] = "promoted"
        promoted = store.score((1.0, 0.0))
        print(
            f"provisional_score={provisional} promoted_score={promoted} "
            f"promotion_changes_score={provisional != promoted}"
        )


def bridge_prefix() -> dict:
    return execute_prefix(ARCHIVE / "demo_nova_bridge.py", "# ── Main")


def bridge_pair_probe() -> None:
    heading("H. WARM-START VERSUS MATCHED COLD-START BRIDGE")
    ns = bridge_prefix()
    run_session = ns["run_session"]

    with tempfile.TemporaryDirectory(prefix="livnium-demos-headline-") as temp:
        warm_path = str(Path(temp) / "warm.json")
        cold_path = str(Path(temp) / "cold.json")
        with contextlib.redirect_stdout(io.StringIO()):
            first = run_session("source cold 42", 300, 42, warm_path)
            warm = run_session("source warm 99", 300, 99, warm_path)
            cold = run_session("matched cold 99", 300, 99, cold_path)
        print(
            f"source_headline_session1={first['wins']}/{first['losses']}/{first['draws']} "
            f"warm_seed99={warm['wins']}/{warm['losses']}/{warm['draws']} "
            f"matched_cold_seed99={cold['wins']}/{cold['losses']}/{cold['draws']}"
        )

    seeds = (1, 7, 42, 99, 2026)
    warm_total = Counter()
    cold_total = Counter()
    pairs = []
    for seed in seeds:
        with tempfile.TemporaryDirectory(prefix="livnium-demos-pair-") as temp:
            warm_path = str(Path(temp) / "warm.json")
            cold_path = str(Path(temp) / "cold.json")
            second_seed = 10_000 + seed
            with contextlib.redirect_stdout(io.StringIO()):
                run_session("stage 1", 300, seed, warm_path)
                warm = run_session("warm", 300, second_seed, warm_path)
                cold = run_session("cold", 300, second_seed, cold_path)
            warm_total.update(
                {"X": warm["wins"], "O": warm["losses"], "Draw": warm["draws"]}
            )
            cold_total.update(
                {"X": cold["wins"], "O": cold["losses"], "Draw": cold["draws"]}
            )
            pairs.append(
                (
                    seed,
                    warm["wins"],
                    warm["losses"],
                    warm["draws"],
                    cold["wins"],
                    cold["losses"],
                    cold["draws"],
                )
            )
    print(f"paired_rows_seed_warmWLD_coldWLD={pairs}")
    print(
        f"five_pair_warm_total={dict(warm_total)} "
        f"five_pair_cold_total={dict(cold_total)}"
    )
    print(
        "protocol_boundary=both warm and cold runs continue learning on the "
        "reported 300-game stream; this is a prequential continuation test"
    )


def saved_frozen_probe() -> None:
    heading("I. SAVED BRIDGE POLICY FROZEN CONTROL")
    totals = Counter()
    for seed in (1, 7, 42, 99, 2026):
        field = karma.KarmicBasinField(
            rng=random.Random(seed),
            spawn_distance=0.4,
            decay_eta=0.1,
            max_decay_budget=0.12,
        )
        store = NovaBasinStore(field, store_path=str(STATE_PATH))
        rng = random.Random(20_000 + seed)
        totals.update(
            frozen_policy_game(
                field,
                "karmic",
                rng,
                scorer=lambda vec, saved=store: saved.score(vec, mode="karmic"),
            )
            for _ in range(500)
        )
    print(f"saved_state_five_seed_frozen_vs_heuristic={dict(totals)}")
    print(
        "comparison_symbolic_heuristic="
        "the direct heuristic control in section E loses zero of 2,500 games"
    )


def main() -> None:
    print(f"archive_root={ARCHIVE}")
    preservation_probe()
    base27_probe()
    learning_probe()
    inside_probe()
    feedback_probe()
    saved_state_probe()
    court_contract_probe()
    bridge_pair_probe()
    saved_frozen_probe()


if __name__ == "__main__":
    main()

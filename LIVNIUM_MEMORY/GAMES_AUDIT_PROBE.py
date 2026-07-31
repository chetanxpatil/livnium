#!/usr/bin/env python3
"""
Read-only controls for the recovered Livnium Games lineage.

The probe never edits the historical game sources or saved basin memories.  It
checks chess transport/ranking, tic-tac-toe online versus frozen behavior,
sliding-puzzle depth/feature/memory boundaries, and the sorting baseline.

Run from the archive root:
  PYTHONDONTWRITEBYTECODE=1 \
      python3 .codex_memory_staging/GAMES_AUDIT_PROBE.py
"""

from __future__ import annotations

from collections import Counter, deque
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import chess
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from evaluate_chess.basin_features import extract_basin_features
from evaluate_chess.basin_ranker import basin_score
from evaluate_chess.decoding import decode_board
from evaluate_chess.encoding import encode_board
from evaluate_chess.move_apply import apply_livnium_move
from evaluate_chess.puzzle_dataset import (
    MatePuzzle,
    generate_random_mate_in_one_puzzles,
    generate_random_non_mate_check_positions,
    load_handcrafted_puzzles,
)
from evaluate_chess.run_level15 import (
    run_adversarial_harness,
    run_random_harness,
)
from evaluate_chess.run_phase2 import (
    evaluate_mate_in_one,
    evaluate_non_mate_checks,
)
from evaluate_chess.verification import (
    full_state_tuple,
    normalized_state_tuple,
)
from livnium.basins import BasinField, _dist_cosine
import evaluate_tictactoe as ttt


GAME_FILES = (
    "LIVNIUM_CHESS_M1.md",
    "demo_sliding_puzzle.py",
    "demo_sorting.py",
    "evaluate_chess_legacy.py",
    "evaluate_tictactoe.py",
    "experiment_sliding.py",
)


def heading(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_probe() -> None:
    heading("A. PRESERVATION INVENTORY")
    organized = ROOT / "_ORGANIZED" / "02_Experiments" / "Games"
    for name in GAME_FILES:
        root_path = ROOT / name
        org_path = organized / name
        print(
            f"{name:<29} bytes={root_path.stat().st_size:<6} "
            f"root_sha={sha256(root_path)} exact_pair="
            f"{root_path.read_bytes() == org_path.read_bytes()}"
        )

    chess_files = sorted(
        p
        for p in (ROOT / "evaluate_chess").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
    print(f"evaluate_chess_live_files={len(chess_files)}")
    for path in chess_files:
        print(f"  {sha256(path)} {path.stat().st_size:>6} {path.relative_to(ROOT)}")
    test_path = ROOT / "tests" / "test_evaluate_chess.py"
    print(f"  {sha256(test_path)} {test_path.stat().st_size:>6} {test_path.relative_to(ROOT)}")

    state_files = sorted(p for p in (ROOT / "state" / "exp_sliding").rglob("*") if p.is_file())
    print(f"saved_sliding_state_files={len(state_files)}")
    for path in state_files:
        print(f"  {sha256(path)} {path.stat().st_size:>10} {path.relative_to(ROOT)}")


def random_move_coverage(epochs: int = 1_000, seed: int = 42) -> Counter:
    board = chess.Board()
    rng = random.Random(seed)
    counts: Counter = Counter()
    for _ in range(epochs):
        if board.is_game_over():
            board.reset()
            counts["game_resets"] += 1
        move = rng.choice(list(board.legal_moves))
        counts["moves"] += 1
        counts["capture"] += int(board.is_capture(move))
        counts["en_passant"] += int(board.is_en_passant(move))
        counts["castling"] += int(board.is_castling(move))
        counts["promotion"] += int(move.promotion is not None)
        board.push(move)
    return counts


def continuous_chess_harness(epochs: int = 1_000, seed: int = 42) -> dict:
    board = chess.Board()
    state = encode_board(board)
    initial_symbols = Counter(state.cell_to_symbol.values())
    rng = random.Random(seed)
    failures = 0
    conservation_failures = 0
    resets = 0
    for _ in range(epochs):
        if board.is_game_over():
            board = chess.Board()
            state = encode_board(board)
            initial_symbols = Counter(state.cell_to_symbol.values())
            resets += 1
        move = rng.choice(list(board.legal_moves))
        state = apply_livnium_move(state, board, move)
        board.push(move)
        decoded = decode_board(state)
        if normalized_state_tuple(decoded) != normalized_state_tuple(board):
            failures += 1
            break
        if Counter(state.cell_to_symbol.values()) != initial_symbols:
            conservation_failures += 1
            break
    return {
        "moves": epochs,
        "resets": resets,
        "state_failures": failures,
        "symbol_multiset_failures": conservation_failures,
    }


ADVERSARIAL_SCENARIOS = (
    (
        chess.STARTING_FEN,
        ("e2e4", "e7e5", "e1e2", "b8c6", "e2e1"),
    ),
    (
        chess.STARTING_FEN,
        ("h2h3", "a7a6", "h1h2", "a6a5", "h2h1"),
    ),
    (
        chess.STARTING_FEN,
        ("e2e4", "a7a6", "e4e5", "d7d5", "g1f3"),
    ),
    (
        chess.STARTING_FEN,
        ("e2e4", "a7a6", "e4e5", "d7d5", "e5d6"),
    ),
    (
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        ("a7a8q",),
    ),
    (
        "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        ("a7b8q",),
    ),
)


def continuous_adversarial_harness() -> dict:
    moves_checked = 0
    failures = 0
    for fen, moves in ADVERSARIAL_SCENARIOS:
        board = chess.Board(fen)
        state = encode_board(board)
        initial_symbols = Counter(state.cell_to_symbol.values())
        for uci in moves:
            move = chess.Move.from_uci(uci)
            state = apply_livnium_move(state, board, move)
            board.push(move)
            moves_checked += 1
            if normalized_state_tuple(decode_board(state)) != normalized_state_tuple(board):
                failures += 1
            if Counter(state.cell_to_symbol.values()) != initial_symbols:
                failures += 1
    return {"moves": moves_checked, "failures": failures}


def basic_mate_rank(board: chess.Board) -> Tuple[bool, bool]:
    """Return top1 success for check-only and check-plus-reply-count baselines."""
    mate_moves = set()
    rows = []
    for order, move in enumerate(board.legal_moves):
        after = board.copy(stack=False)
        after.push(move)
        is_mate = after.is_checkmate()
        if is_mate:
            mate_moves.add(move.uci())
        rows.append(
            {
                "move": move.uci(),
                "check": int(after.is_check()),
                "replies": len(list(after.legal_moves)),
                "order": order,
            }
        )
    check_top = max(rows, key=lambda row: (row["check"], -row["order"]))
    reply_top = max(
        rows,
        key=lambda row: (row["check"], -row["replies"], -row["order"]),
    )
    return check_top["move"] in mate_moves, reply_top["move"] in mate_moves


def source_top_is_any_mate(summary: dict) -> int:
    return sum(
        int(bool(result["ranked_moves"]) and result["ranked_moves"][0]["is_mate"])
        for result in summary["results"]
    )


def chess_probe() -> None:
    heading("B. CHESS TRANSPORT AND MATE-RANKING CONTROLS")
    random_stats = run_random_harness(1_000, 42)
    adversarial_stats = run_adversarial_harness()
    coverage = random_move_coverage(1_000, 42)
    print(f"fresh_reencode_random_harness={random_stats}")
    print(f"fresh_reencode_adversarial_harness={adversarial_stats}")
    print(f"random_harness_move_type_coverage={dict(coverage)}")
    print(f"continuous_random_harness={continuous_chess_harness()}")
    print(f"continuous_adversarial_harness={continuous_adversarial_harness()}")

    board = chess.Board()
    state = encode_board(board, include_clocks=True)
    for uci in ("e2e4", "e7e5"):
        move = chess.Move.from_uci(uci)
        state = apply_livnium_move(state, board, move)
        board.push(move)
    decoded = decode_board(state)
    print(
        f"full_clock_state_equal_after_two_plies="
        f"{full_state_tuple(decoded) == full_state_tuple(board)} "
        f"decoded_clocks={decoded.halfmove_clock,decoded.fullmove_number} "
        f"expected_clocks={board.halfmove_clock,board.fullmove_number}"
    )

    handcrafted_all = load_handcrafted_puzzles()
    print(
        f"handcrafted_candidates_in_source=20 retained_valid={len(handcrafted_all)}"
    )
    hand_decoded = evaluate_mate_in_one(handcrafted_all, feature_mode="decoded")
    hand_hybrid = evaluate_mate_in_one(handcrafted_all, feature_mode="hybrid")
    print(
        f"handcrafted_target_top1={hand_decoded['top1']}/{hand_decoded['total']} "
        f"any_mate_top1={source_top_is_any_mate(hand_decoded)}/{hand_decoded['total']} "
        f"decoded_hybrid_rankings_equal="
        f"{[r['mate_rank'] for r in hand_decoded['results']] == [r['mate_rank'] for r in hand_hybrid['results']]}"
    )

    generated = generate_random_mate_in_one_puzzles(
        100, seed=42, max_games=50_000, max_plies_per_game=200
    )
    decoded_summary = evaluate_mate_in_one(generated, feature_mode="decoded")
    hybrid_summary = evaluate_mate_in_one(generated, feature_mode="hybrid")
    source_any = source_top_is_any_mate(decoded_summary)
    check_hits = 0
    reply_hits = 0
    for puzzle in generated:
        check_hit, reply_hit = basic_mate_rank(chess.Board(puzzle.fen))
        check_hits += int(check_hit)
        reply_hits += int(reply_hit)
    ranking_equal = all(
        [row["move"] for row in dec["ranked_moves"]]
        == [row["move"] for row in hyb["ranked_moves"]]
        for dec, hyb in zip(
            decoded_summary["results"], hybrid_summary["results"], strict=True
        )
    )
    print(
        f"generated100_source_target_top1={decoded_summary['top1']} "
        f"source_any_mate_top1={source_any} check_only_top1={check_hits} "
        f"check_plus_fewest_legal_replies_top1={reply_hits}"
    )
    print(
        f"generated100_source_top3={decoded_summary['top3']} "
        f"mean_rank={decoded_summary['mean_rank']:.3f} "
        f"decoded_hybrid_full_rankings_equal={ranking_equal}"
    )

    adversarial_fens = generate_random_non_mate_check_positions(100, seed=43)
    nonmate = evaluate_non_mate_checks(adversarial_fens)
    mate_top_scores = [
        max(row["score"] for row in result["ranked_moves"] if row["is_mate"])
        for result in decoded_summary["results"]
    ]
    nonmate_top_scores = [row["top"]["score"] for row in nonmate["results"]]
    print(
        f"nonmate_check_positions={nonmate['total']} "
        f"top_is_check_rate={nonmate['top_is_check_rate']:.3f} "
        f"mate_score_range=({min(mate_top_scores):.2f},{max(mate_top_scores):.2f}) "
        f"nonmate_top_score_range=({min(nonmate_top_scores):.2f},{max(nonmate_top_scores):.2f})"
    )


def update_ttt_field(field: BasinField, winner: str, history: list) -> None:
    for vec, _prediction in history:
        if winner == "X":
            field.update_correct(vec, "X_Win")
        elif winner == "O":
            field.decay_incorrect(vec, "X_Win")
            field.update_correct(vec, "O_Win")


TTT_COORDS = tuple(sorted(ttt.initialize_game_state()[1]))


@lru_cache(maxsize=None)
def cached_minimax_score(
    values: Tuple[str, ...], is_maximizing: bool, depth: int
) -> int:
    grid = dict(zip(TTT_COORDS, values))
    winner = ttt.check_winner(grid)
    if winner == "X":
        return 10 - depth
    if winner == "O":
        return -10 + depth
    if winner == "Draw":
        return 0

    scores = []
    mark = "X" if is_maximizing else "O"
    for index, value in enumerate(values):
        if value != "-":
            continue
        next_values = list(values)
        next_values[index] = mark
        scores.append(
            cached_minimax_score(
                tuple(next_values), not is_maximizing, depth + 1
            )
        )
    return max(scores) if is_maximizing else min(scores)


def cached_minimax_move(
    grid: Dict[Tuple[int, int, int], str], player: str
) -> Tuple[int, int, int]:
    values = tuple(grid[coord] for coord in TTT_COORDS)
    scored = []
    for index, value in enumerate(values):
        if value != "-":
            continue
        next_values = list(values)
        next_values[index] = player
        score = cached_minimax_score(
            tuple(next_values), player == "O", 0
        )
        scored.append((score, TTT_COORDS[index]))
    if player == "X":
        return max(scored, key=lambda item: item[0])[1]
    return min(scored, key=lambda item: item[0])[1]


def train_ttt_field(seed: int, games: int = 500) -> Tuple[BasinField, Counter]:
    rng = random.Random(seed)
    field = BasinField(
        rng=rng,
        spawn_distance=0.4,
        decay_eta=0.1,
    )
    outcomes: Counter = Counter()
    for _ in range(games):
        winner, history = ttt.play_match(
            "Heuristic", field, rng, use_basins=True, use_o_repulsion=True
        )
        outcomes[winner] += 1
        update_ttt_field(field, winner, history)
    return field, outcomes


def frozen_ttt_eval(
    field: BasinField, opponent: str, seed: int, games: int = 500
) -> Counter:
    rng = random.Random(seed)
    outcomes: Counter = Counter()
    before = {
        label: [(anchor.center, anchor.count) for anchor in anchors]
        for label, anchors in field.anchors.items()
    }
    for _ in range(games):
        winner, _history = ttt.play_match(
            opponent, field, rng, use_basins=True, use_o_repulsion=True
        )
        outcomes[winner] += 1
    after = {
        label: [(anchor.center, anchor.count) for anchor in anchors]
        for label, anchors in field.anchors.items()
    }
    assert before == after
    return outcomes


def play_symbolic_ttt(
    x_policy: str, o_policy: str, seed: int, games: int = 500
) -> Counter:
    rng = random.Random(seed)
    outcomes: Counter = Counter()
    for _ in range(games):
        grid = {coord: "-" for coord in ttt.initialize_game_state()[1]}
        player = "X"
        winner = "Ongoing"
        while winner == "Ongoing":
            if player == "X":
                if x_policy == "heuristic":
                    move = ttt.heuristic_move(grid, "X", rng)
                elif x_policy == "minimax":
                    move = ttt.minimax_move(grid, "X")
                else:
                    move = ttt.random_move(grid, rng)
            else:
                if o_policy == "heuristic":
                    move = ttt.heuristic_move(grid, "O", rng)
                elif o_policy == "minimax":
                    move = ttt.minimax_move(grid, "O")
                else:
                    move = ttt.random_move(grid, rng)
            grid[move] = player
            winner = ttt.check_winner(grid)
            player = "O" if player == "X" else "X"
        outcomes[winner] += 1
    return outcomes


def tictactoe_probe() -> None:
    heading("C. TIC-TAC-TOE ONLINE VERSUS FROZEN CONTROLS")
    source_minimax_move = ttt.minimax_move
    ttt.minimax_move = cached_minimax_move
    seeds = (1, 7, 42, 99, 2026)
    frozen_by_opponent = {
        "Random": Counter(),
        "Heuristic": Counter(),
        "Minimax": Counter(),
    }
    training_totals: Counter = Counter()
    anchor_counts = []
    for seed in seeds:
        field, training = train_ttt_field(seed, games=500)
        training_totals.update(training)
        anchor_counts.append(
            {
                label: len(anchors)
                for label, anchors in field.anchors.items()
            }
        )
        for opponent in frozen_by_opponent:
            frozen_by_opponent[opponent].update(
                frozen_ttt_eval(
                    field, opponent, seed=10_000 + seed, games=500
                )
            )
    print(f"five_seed_online_training_vs_heuristic={dict(training_totals)}")
    print(f"anchor_counts_after_training={anchor_counts}")
    for opponent, outcomes in frozen_by_opponent.items():
        print(f"five_seed_frozen_vs_{opponent.lower()}={dict(outcomes)}")

    for x_policy in ("random", "heuristic", "minimax"):
        for o_policy in ("Heuristic", "Minimax"):
            totals: Counter = Counter()
            for seed in seeds:
                totals.update(
                    play_symbolic_ttt(
                        x_policy, o_policy.lower(), seed, games=500
                    )
                )
            print(
                f"symbolic_{x_policy}_X_vs_{o_policy.lower()}_O={dict(totals)}"
            )
    ttt.minimax_move = source_minimax_move
    print(f"cached_minimax_states={cached_minimax_score.cache_info().currsize}")


Puzzle = Tuple[int, ...]
PUZZLE_TARGET: Puzzle = (1, 2, 3, 4, 5, 6, 7, 8, 0)
INDEX_COORDS = tuple((i // 3, i % 3) for i in range(9))
BLANK_NEIGHBORS = {
    i: tuple(
        j
        for j in range(9)
        if abs(INDEX_COORDS[i][0] - INDEX_COORDS[j][0])
        + abs(INDEX_COORDS[i][1] - INDEX_COORDS[j][1])
        == 1
    )
    for i in range(9)
}


def puzzle_neighbors(state: Puzzle) -> Iterable[Puzzle]:
    blank = state.index(0)
    for other in BLANK_NEIGHBORS[blank]:
        out = list(state)
        out[blank], out[other] = out[other], out[blank]
        yield tuple(out)


def puzzle_distance_map() -> Dict[Puzzle, int]:
    distance = {PUZZLE_TARGET: 0}
    queue = deque([PUZZLE_TARGET])
    while queue:
        state = queue.popleft()
        next_distance = distance[state] + 1
        for nxt in puzzle_neighbors(state):
            if nxt not in distance:
                distance[nxt] = next_distance
                queue.append(nxt)
    return distance


def shuffled_puzzle(n_moves: int, seed: int) -> Puzzle:
    rng = random.Random(seed)
    state = PUZZLE_TARGET
    for _ in range(n_moves):
        state = rng.choice(list(puzzle_neighbors(state)))
    return state


def puzzle_feature(state: Puzzle) -> Tuple[int, ...]:
    values = []
    for position, tile in enumerate(state):
        if tile == 0:
            values.append(0)
            continue
        target_position = tile - 1
        values.append(
            abs(INDEX_COORDS[position][0] - INDEX_COORDS[target_position][0])
            + abs(INDEX_COORDS[position][1] - INDEX_COORDS[target_position][1])
        )
    return tuple(values)


def primitive_ray(feature: Tuple[int, ...]) -> Tuple[int, ...]:
    divisor = 0
    for value in feature:
        divisor = math.gcd(divisor, value)
    if divisor == 0:
        return feature
    return tuple(value // divisor for value in feature)


def validate_saved_sliding_state() -> None:
    root = ROOT / "state" / "exp_sliding"
    for json_path in sorted(root.glob("*/basin.json")):
        state = json.loads(json_path.read_text(encoding="utf-8"))
        archive_path = Path(str(json_path) + ".ledger_archive.jsonl")
        live = state.get("ledger", [])
        archive_lines = 0
        chain_breaks = 0
        previous_after = None
        if archive_path.exists():
            with archive_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    archive_lines += 1
                    if (
                        previous_after is not None
                        and row.get("state_hash_before") != previous_after
                    ):
                        chain_breaks += 1
                    previous_after = row.get("state_hash_after")
        for row in live:
            if (
                previous_after is not None
                and row.get("state_hash_before") != previous_after
            ):
                chain_breaks += 1
            previous_after = row.get("state_hash_after")

        canonical = json.dumps(
            state, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        direct_hash = hashlib.sha256(canonical).hexdigest()
        anchor_count = sum(len(rows) for rows in state.get("anchors", {}).values())
        print(
            f"{json_path.parent.name}: anchors={anchor_count} step={state.get('step')} "
            f"archive={archive_lines} live={len(live)} "
            f"ledger_total_count_field={state.get('ledger_total_count')} "
            f"adjacent_receipt_chain_breaks={chain_breaks} "
            f"stored_hash_matches_current_canonical={state.get('state_hash') == direct_hash}"
        )


def sliding_probe() -> None:
    heading("D. SLIDING-PUZZLE DEPTH, FEATURES, AND SAVED MEMORY")
    distances = puzzle_distance_map()
    print(
        f"reachable_states={len(distances)} diameter={max(distances.values())}"
    )
    demo_state = shuffled_puzzle(40, 99)
    print(
        f"demo_claimed_shuffle_moves=40 initial_manhattan="
        f"{sum(puzzle_feature(demo_state))} exact_optimal_depth={distances[demo_state]}"
    )

    source_starts = []
    for session in range(5):
        session_seed = 77 + session * 37
        for attempt in range(30):
            source_starts.append(
                shuffled_puzzle(25, session_seed * 1000 + attempt)
            )
    start_depths = [distances[state] for state in source_starts]
    print(
        f"experiment_starts=150 claimed_shuffle_moves=25 unique={len(set(source_starts))} "
        f"optimal_depth_mean={np.mean(start_depths):.3f} "
        f"median={np.median(start_depths):.1f} min={min(start_depths)} max={max(start_depths)}"
    )
    print(
        f"exact_solver_with_300_step_budget_solve_rate="
        f"{sum(depth <= 300 for depth in start_depths)}/{len(start_depths)} "
        f"mean_optimal_steps={np.mean(start_depths):.3f}"
    )

    exact_counts: Counter = Counter()
    ray_counts: Counter = Counter()
    for state in distances:
        feature = puzzle_feature(state)
        exact_counts[feature] += 1
        ray_counts[primitive_ray(feature)] += 1
    print(
        f"feature_vectors_unique={len(exact_counts)}/{len(distances)} "
        f"max_exact_collision={max(exact_counts.values())} "
        f"cosine_rays_unique={len(ray_counts)}/{len(distances)} "
        f"max_cosine_ray_collision={max(ray_counts.values())}"
    )
    solved_feature = puzzle_feature(PUZZLE_TARGET)
    sample_feature = puzzle_feature(demo_state)
    print(
        f"solved_feature={solved_feature} "
        f"cosine_distance_solved_to_nonzero={_dist_cosine(solved_feature,sample_feature):.1f}"
    )
    validate_saved_sliding_state()


def minimal_arbitrary_swaps(order: Sequence[int]) -> int:
    visited = [False] * len(order)
    cycles = 0
    for start in range(len(order)):
        if visited[start]:
            continue
        cycles += 1
        current = start
        while not visited[current]:
            visited[current] = True
            current = order[current]
    return len(order) - cycles


def direct_place_steps(order: Sequence[int]) -> int:
    current = list(order)
    steps = 0
    for index in range(len(current)):
        if current[index] == index:
            continue
        other = current.index(index)
        current[index], current[other] = current[other], current[index]
        steps += 1
    assert current == list(range(len(current)))
    return steps


def sorting_probe() -> None:
    heading("E. SORTING BASELINE")
    values = list(range(10))
    rng = random.Random(7)
    rng.shuffle(values)
    print(
        f"source_initial_permutation={values} "
        f"minimum_arbitrary_swaps={minimal_arbitrary_swaps(values)} "
        f"direct_place_steps={direct_place_steps(values)} "
        f"source_annealing_steps=1345"
    )
    mins = []
    for seed in range(1_000):
        order = list(range(10))
        random.Random(seed).shuffle(order)
        mins.append(minimal_arbitrary_swaps(order))
    print(
        f"1000_random_permutations_minimum_swaps_mean={np.mean(mins):.3f} "
        f"min={min(mins)} max={max(mins)} direct_method_solve_rate=1000/1000"
    )


def main() -> None:
    print(f"archive_root={ROOT}")
    print(f"python={sys.version.split()[0]} chess={chess.__version__}")
    inventory_probe()
    chess_probe()
    tictactoe_probe()
    sliding_probe()
    sorting_probe()


if __name__ == "__main__":
    main()

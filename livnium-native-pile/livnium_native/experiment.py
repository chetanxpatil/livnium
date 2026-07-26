"""Train and evaluate the Livnium-native memory prototype."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .data import TOKEN_TO_ACTION, NavigationBatch, sample_navigation_batch
from .geometry import (
    ACTION_NAMES,
    INVERSE_ACTION,
    action_index_maps,
    validate_geometry,
)
from .model import PAD_TOKEN, LivniumNativeModel
from .pile import HierarchicalLivniumPile


@dataclass
class ExperimentConfig:
    seed: int = 7
    steps: int = 500
    batch_size: int = 256
    learning_rate: float = 0.01
    train_min_path: int = 1
    train_max_path: int = 3
    test_min_path: int = 4
    test_max_path: int = 6
    eval_examples: int = 4096


def _accuracy(model: LivniumNativeModel, batch: NavigationBatch, *, hard: bool) -> float:
    model.eval()
    device_batch = batch.to(model.device)
    with torch.no_grad():
        output = model(
            device_batch.macro_start,
            device_batch.macro_tokens,
            device_batch.micro_start,
            device_batch.micro_tokens,
            hard=hard,
        )
        predicted_ranks = output.payload_probabilities.argmax(dim=-1)
        predictions = model.pile.values_for_ranks(predicted_ranks)
    return float((predictions == device_batch.target_payloads).float().mean().item())


def _predictions(
    model: LivniumNativeModel,
    batch: NavigationBatch,
) -> torch.Tensor:
    model.eval()
    device_batch = batch.to(model.device)
    with torch.no_grad():
        output = model(
            device_batch.macro_start,
            device_batch.macro_tokens,
            device_batch.micro_start,
            device_batch.micro_tokens,
            hard=True,
        )
    predicted_ranks = output.payload_probabilities.argmax(dim=-1)
    return model.pile.values_for_ranks(predicted_ranks).cpu()


def _no_flow_accuracy(model: LivniumNativeModel, batch: NavigationBatch) -> float:
    start_payloads = model.pile.payload_at(
        batch.macro_start.to(model.device),
        batch.micro_start.to(model.device),
    )
    return float(
        (start_payloads == batch.target_payloads.to(model.device)).float().mean().item()
    )


def _slice_batch(batch: NavigationBatch, start: int, end: int) -> NavigationBatch:
    return NavigationBatch(
        **{
            name: value[start:end]
            for name, value in batch.__dict__.items()
        }
    )


def _causal_intervention_tests(
    model: LivniumNativeModel,
    batch: NavigationBatch,
    *,
    examples: int = 64,
) -> tuple[float, float, bool]:
    """Test target causality, non-target isolation, and pile immutability."""
    original_payloads = model.pile.payload_values.detach().clone()
    original_hash = model.pile.inventory_hash()
    followed = 0
    tested = min(examples, batch.macro_start.shape[0])

    # Change the addressed payload while holding the query fixed. The answer
    # should follow the new content at that exact address.
    for index in range(tested):
        one = _slice_batch(batch, index, index + 1)
        leaf = int(one.target_macro.item() * 27 + one.target_micro.item())
        other = (leaf + 1) % model.pile.leaf_count
        changed = original_payloads.clone()
        changed[leaf], changed[other] = changed[other].clone(), changed[leaf].clone()
        model.pile.load_payloads(changed)
        expected = int(changed[leaf])
        predicted = int(_predictions(model, one).item())
        followed += int(predicted == expected)

    model.pile.load_payloads(original_payloads)

    # Change two leaves that are not addressed by a query subset. Predictions
    # should be completely unchanged.
    subset = _slice_batch(batch, 0, tested)
    before = _predictions(model, subset)
    targeted = set(
        (
            subset.target_macro * 27 + subset.target_micro
        ).tolist()
    )
    non_targets = [
        leaf
        for leaf in range(model.pile.leaf_count)
        if leaf not in targeted
    ]
    changed = original_payloads.clone()
    a, b = non_targets[:2]
    changed[a], changed[b] = changed[b].clone(), changed[a].clone()
    model.pile.load_payloads(changed)
    after = _predictions(model, subset)
    non_target_stability = float((before == after).float().mean().item())

    model.pile.load_payloads(original_payloads)
    hash_restored = model.pile.inventory_hash() == original_hash
    return followed / tested, non_target_stability, hash_restored


def _token_shuffle_control(batch: NavigationBatch, seed: int) -> NavigationBatch:
    del seed
    # Guaranteed derangement: no active token retains its original meaning.
    token_permutation = (
        torch.arange(len(ACTION_NAMES)) + 1
    ) % len(ACTION_NAMES)

    def replace(tokens: torch.Tensor) -> torch.Tensor:
        active = tokens != PAD_TOKEN
        shuffled = tokens.clone()
        shuffled[active] = token_permutation[tokens[active]]
        return shuffled

    return NavigationBatch(
        macro_start=batch.macro_start,
        macro_tokens=replace(batch.macro_tokens),
        micro_start=batch.micro_start,
        micro_tokens=replace(batch.micro_tokens),
        target_macro=batch.target_macro,
        target_micro=batch.target_micro,
        target_payloads=batch.target_payloads,
        target_payload_ranks=batch.target_payload_ranks,
    )


def _reverse_instruction_order(batch: NavigationBatch) -> NavigationBatch:
    """Reverse each active path while keeping the original expected answer."""

    def reverse(tokens: torch.Tensor) -> torch.Tensor:
        reversed_tokens = torch.full_like(tokens, PAD_TOKEN)
        for row in range(tokens.shape[0]):
            active = tokens[row][tokens[row] != PAD_TOKEN]
            reversed_tokens[row, : active.numel()] = active.flip(0)
        return reversed_tokens

    return NavigationBatch(
        macro_start=batch.macro_start,
        macro_tokens=reverse(batch.macro_tokens),
        micro_start=batch.micro_start,
        micro_tokens=reverse(batch.micro_tokens),
        target_macro=batch.target_macro,
        target_micro=batch.target_micro,
        target_payloads=batch.target_payloads,
        target_payload_ranks=batch.target_payload_ranks,
    )


def _reference_endpoint(starts: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    maps = action_index_maps()
    positions = starts.clone()
    for step in range(tokens.shape[1]):
        active = tokens[:, step] != PAD_TOKEN
        safe_tokens = torch.where(active, tokens[:, step], torch.zeros_like(tokens[:, step]))
        actions = TOKEN_TO_ACTION[safe_tokens]
        proposed = maps[actions, positions]
        positions = torch.where(active, proposed, positions)
    return positions


def _order_reversal_metrics(
    model: LivniumNativeModel,
    batch: NavigationBatch,
) -> tuple[float, float, float]:
    reversed_batch = _reverse_instruction_order(batch)
    predictions = _predictions(model, reversed_batch)
    raw_accuracy = float(
        (predictions == batch.target_payloads).float().mean().item()
    )
    reversed_macro = _reference_endpoint(
        batch.macro_start,
        reversed_batch.macro_tokens,
    )
    reversed_micro = _reference_endpoint(
        batch.micro_start,
        reversed_batch.micro_tokens,
    )
    original_leaf = batch.target_macro * 27 + batch.target_micro
    reversed_leaf = reversed_macro * 27 + reversed_micro
    changed = reversed_leaf != original_leaf
    change_rate = float(changed.float().mean().item())
    conditional_accuracy = float(
        (predictions[changed] == batch.target_payloads[changed]).float().mean().item()
    )
    return raw_accuracy, change_rate, conditional_accuracy


def _round_trip_rate(examples: int, max_steps: int, seed: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    maps = action_index_maps()
    starts = torch.randint(0, 27, (examples,), generator=generator)
    actions = torch.randint(
        0,
        len(ACTION_NAMES),
        (examples, max_steps),
        generator=generator,
    )
    positions = starts.clone()
    for step in range(max_steps):
        positions = maps[actions[:, step], positions]
    for step in reversed(range(max_steps)):
        inverse = torch.tensor(INVERSE_ACTION)[actions[:, step]]
        positions = maps[inverse, positions]
    return float((positions == starts).float().mean().item())


def run_experiment(config: ExperimentConfig) -> tuple[dict, LivniumNativeModel]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    validate_geometry()

    train_pile = HierarchicalLivniumPile.random(config.seed + 10)
    model = LivniumNativeModel(train_pile)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_generator = torch.Generator().manual_seed(config.seed + 20)

    loss_curve: list[float] = []
    for step in range(1, config.steps + 1):
        model.train()
        batch = sample_navigation_batch(
            model.pile,
            batch_size=config.batch_size,
            min_steps=config.train_min_path,
            max_steps=config.train_max_path,
            generator=train_generator,
        ).to(model.device)
        output = model(
            batch.macro_start,
            batch.macro_tokens,
            batch.micro_start,
            batch.micro_tokens,
            hard=False,
        )
        loss = model.answer_loss(output, batch.target_payload_ranks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_curve.append(float(loss.item()))

    seen_pile_hash = model.pile.inventory_hash()
    training_payload_values = model.pile.payload_values.detach().clone()
    eval_generator = torch.Generator().manual_seed(config.seed + 30)
    seen_batch = sample_navigation_batch(
        model.pile,
        batch_size=config.eval_examples,
        min_steps=config.train_min_path,
        max_steps=config.train_max_path,
        generator=eval_generator,
    )
    seen_accuracy = _accuracy(model, seen_batch, hard=True)

    # Generalization test: replace every payload with values from a disjoint range.
    unseen_pile = HierarchicalLivniumPile.random(config.seed + 40)
    model.pile.load_payloads(unseen_pile.payload_values)
    unseen_pile_hash = model.pile.inventory_hash()

    unseen_seen_length_batch = sample_navigation_batch(
        model.pile,
        batch_size=config.eval_examples,
        min_steps=config.train_min_path,
        max_steps=config.train_max_path,
        generator=torch.Generator().manual_seed(config.seed + 50),
    )
    unseen_seen_length_accuracy = _accuracy(
        model,
        unseen_seen_length_batch,
        hard=True,
    )

    longer_batch = sample_navigation_batch(
        model.pile,
        batch_size=config.eval_examples,
        min_steps=config.test_min_path,
        max_steps=config.test_max_path,
        generator=torch.Generator().manual_seed(config.seed + 60),
    )
    longer_accuracy = _accuracy(model, longer_batch, hard=True)
    no_flow_accuracy = _no_flow_accuracy(model, longer_batch)
    token_shuffle_accuracy = _accuracy(
        model,
        _token_shuffle_control(longer_batch, config.seed + 70),
        hard=True,
    )
    (
        reversed_order_accuracy,
        reversed_order_endpoint_change_rate,
        reversed_order_changed_path_accuracy,
    ) = _order_reversal_metrics(
        model,
        longer_batch,
    )

    # Causal pile-use test: keep answers from the current pile, silently replace
    # the pile, and require accuracy to collapse.
    wrong_pile = HierarchicalLivniumPile.random(config.seed + 80)
    model.pile.load_payloads(wrong_pile.payload_values)
    wrong_pile_accuracy = _accuracy(model, longer_batch, hard=True)
    model.pile.load_payloads(unseen_pile.payload_values)

    target_follow_rate, non_target_stability, intervention_hash_restored = (
        _causal_intervention_tests(model, longer_batch)
    )

    # An untrained router has the same pile interface but no learned token
    # meanings. It should remain near chance.
    random.seed(config.seed + 100)
    torch.manual_seed(config.seed + 100)
    random_router = LivniumNativeModel(
        HierarchicalLivniumPile(unseen_pile.payload_values)
    )
    random_router_accuracy = _accuracy(random_router, longer_batch, hard=True)

    accuracy_by_path_length: dict[str, float] = {}
    for path_length in range(
        config.train_min_path,
        config.test_max_path + 1,
    ):
        path_batch = sample_navigation_batch(
            model.pile,
            batch_size=config.eval_examples,
            min_steps=path_length,
            max_steps=path_length,
            generator=torch.Generator().manual_seed(
                config.seed + 200 + path_length
            ),
        )
        accuracy_by_path_length[str(path_length)] = _accuracy(
            model,
            path_batch,
            hard=True,
        )

    learned_map = model.learned_action_map().detach().cpu()
    action_mapping_accuracy = float(
        (learned_map == TOKEN_TO_ACTION).float().mean().item()
    )

    metrics = {
        "config": asdict(config),
        "chance_accuracy": 1.0 / HierarchicalLivniumPile.leaf_count,
        "train": {
            "initial_loss": loss_curve[0],
            "final_loss": loss_curve[-1],
            "minimum_loss": min(loss_curve),
        },
        "tests": {
            "seen_pile_seen_paths_hard_accuracy": seen_accuracy,
            "unseen_pile_seen_paths_hard_accuracy": unseen_seen_length_accuracy,
            "unseen_pile_longer_paths_hard_accuracy": longer_accuracy,
            "wrong_pile_control_accuracy": wrong_pile_accuracy,
            "shuffled_instruction_control_accuracy": token_shuffle_accuracy,
            "reversed_instruction_order_control_accuracy": reversed_order_accuracy,
            "reversed_order_endpoint_change_rate": reversed_order_endpoint_change_rate,
            "reversed_order_changed_path_accuracy": reversed_order_changed_path_accuracy,
            "no_flow_control_accuracy": no_flow_accuracy,
            "random_router_control_accuracy": random_router_accuracy,
            "target_payload_intervention_follow_rate": target_follow_rate,
            "non_target_intervention_stability": non_target_stability,
            "pile_hash_restored_after_interventions": intervention_hash_restored,
            "learned_action_mapping_accuracy": action_mapping_accuracy,
            "exact_observer_round_trip_rate": _round_trip_rate(
                config.eval_examples,
                config.test_max_path,
                config.seed + 90,
            ),
        },
        "hard_accuracy_by_path_length": accuracy_by_path_length,
        "learned_token_meanings": {
            str(token): ACTION_NAMES[int(action)]
            for token, action in enumerate(learned_map.tolist())
        },
        "hidden_true_token_meanings": {
            str(token): ACTION_NAMES[int(action)]
            for token, action in enumerate(TOKEN_TO_ACTION.tolist())
        },
        "pile_hashes": {
            "training_pile": seen_pile_hash,
            "unseen_test_pile": unseen_pile_hash,
            "checkpoint_pile": seen_pile_hash,
        },
        "scope": (
            "This demonstrates an embedded pile interface whose neural action head "
            "learned an obfuscated instruction-to-operation dictionary and composed "
            "those operations on unseen payload values. The memory read path itself "
            "is fixed, not autonomously learned. This does not prove language "
            "understanding or an advantage over flat memory."
        ),
    }
    # The returned/saved model owns the pile it was trained with. Evaluation
    # piles are temporary causal tests, not silently persisted deployment state.
    model.pile.load_payloads(training_payload_values)
    return metrics, model


def _verdict(metrics: dict) -> str:
    tests = metrics["tests"]
    passed = (
        tests["unseen_pile_longer_paths_hard_accuracy"] >= 0.95
        and tests["wrong_pile_control_accuracy"] <= 0.02
        and tests["shuffled_instruction_control_accuracy"] <= 0.05
        and tests["reversed_order_endpoint_change_rate"] >= 0.50
        and tests["reversed_order_changed_path_accuracy"] <= 0.05
        and tests["no_flow_control_accuracy"] <= 0.02
        and tests["random_router_control_accuracy"] <= 0.02
        and tests["target_payload_intervention_follow_rate"] >= 0.95
        and tests["non_target_intervention_stability"] >= 0.95
        and tests["pile_hash_restored_after_interventions"]
        and tests["learned_action_mapping_accuracy"] == 1.0
        and tests["exact_observer_round_trip_rate"] == 1.0
    )
    return "PASS" if passed else "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=ExperimentConfig.steps)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--eval-examples", type=int, default=ExperimentConfig.eval_examples)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--output", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/model.pt"))
    args = parser.parse_args()

    config = ExperimentConfig(
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_examples=args.eval_examples,
    )
    metrics, model = run_experiment(config)
    metrics["verdict"] = _verdict(metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "metrics": metrics,
        },
        args.checkpoint,
    )

    print(json.dumps(metrics, indent=2))
    print(f"\nVERDICT: {metrics['verdict']}")
    print(f"Metrics: {args.output}")
    print(f"Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()

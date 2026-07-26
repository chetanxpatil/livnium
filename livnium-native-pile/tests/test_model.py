import torch

from livnium_native.data import TOKEN_TO_ACTION, sample_navigation_batch
from livnium_native.experiment import ExperimentConfig, run_experiment
from livnium_native.model import LivniumNativeModel
from livnium_native.pile import HierarchicalLivniumPile


def test_forward_preserves_probability_and_payload_inventory():
    pile = HierarchicalLivniumPile.random(seed=4)
    model = LivniumNativeModel(pile)
    before = pile.inventory_hash()
    batch = sample_navigation_batch(
        pile,
        batch_size=8,
        min_steps=1,
        max_steps=3,
        generator=torch.Generator().manual_seed(5),
    )
    output = model(
        batch.macro_start,
        batch.macro_tokens,
        batch.micro_start,
        batch.micro_tokens,
    )
    assert torch.allclose(output.macro_distribution.sum(dim=1), torch.ones(8))
    assert torch.allclose(output.micro_distribution.sum(dim=1), torch.ones(8))
    assert torch.allclose(output.payload_probabilities.sum(dim=1), torch.ones(8))
    assert pile.inventory_hash() == before


def test_short_training_learns_native_action_meanings():
    metrics, model = run_experiment(
        ExperimentConfig(
            seed=11,
            steps=250,
            batch_size=192,
            eval_examples=512,
        )
    )
    assert torch.equal(model.learned_action_map().cpu(), TOKEN_TO_ACTION)
    assert metrics["tests"]["unseen_pile_seen_paths_hard_accuracy"] > 0.95
    assert metrics["tests"]["exact_observer_round_trip_rate"] == 1.0


def test_full_model_state_round_trip_keeps_pile_and_controller():
    original = LivniumNativeModel(HierarchicalLivniumPile.random(seed=41))
    restored = LivniumNativeModel()
    restored.load_state_dict(original.state_dict())
    assert restored.pile.inventory_hash() == original.pile.inventory_hash()
    for left, right in zip(original.parameters(), restored.parameters(), strict=True):
        assert torch.equal(left, right)

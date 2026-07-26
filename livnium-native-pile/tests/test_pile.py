import torch

from livnium_native.pile import HierarchicalLivniumPile


def test_hierarchy_has_729_exact_payloads():
    pile = HierarchicalLivniumPile.random(seed=1)
    assert pile.payload_values.shape == (729,)
    assert pile.payload_values.unique().numel() == 729
    assert int(pile.payload_values.min()) >= 1_000_000


def test_hard_observer_read_returns_exact_payload():
    pile = HierarchicalLivniumPile.random(seed=2)
    macro = torch.nn.functional.one_hot(torch.tensor([3, 12]), 27).float()
    micro = torch.nn.functional.one_hot(torch.tensor([8, 26]), 27).float()
    distribution = pile.read(macro, micro)
    expected = pile.payload_at(torch.tensor([3, 12]), torch.tensor([8, 26]))
    predicted = pile.values_for_ranks(distribution.argmax(dim=1))
    assert torch.equal(predicted, expected)
    assert torch.equal(distribution.sum(dim=1), torch.ones(2))


def test_pile_is_saved_inside_model_state():
    pile = HierarchicalLivniumPile.random(seed=3)
    state = pile.state_dict()
    assert "payload_values" in state
    assert "payload_ranks" in state
    assert "rank_to_value" in state
    assert torch.equal(state["payload_values"], pile.payload_values)

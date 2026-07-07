"""Configuration for the vector collapse engine.

Single source of truth for every tunable. Load from YAML for experiments,
or construct in code with overrides:

    cfg = CollapseConfig.from_yaml("config.yaml")
    cfg = CollapseConfig(dim=512, strengths={"E": 0.2, "C": 0.2, "N": 0.1})
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class BasinConfig:
    """Dynamic basin field (per-label micro-basins)."""

    max_basins_per_label: int = 64
    tension_threshold: float = 0.15   # spawn if tension exceeds this...
    align_threshold: float = 0.6      # ...and alignment is below this
    anchor_lr: float = 0.05           # moving-average rate for center updates
    prune_min_count: int = 10         # basins used fewer times get pruned
    prune_merge_cos: float = 0.97     # basins more similar than this get merged


@dataclass
class CollapseConfig:
    """Vector collapse dynamics.

    Label order defines the integer encoding: labels[i] <-> label index i.
    Default matches the trained NLI models: 0=E, 1=C, 2=N.
    """

    dim: int = 256
    num_layers: int = 4
    max_norm: float = 10.0
    labels: Tuple[str, ...] = ("E", "C", "N")
    strengths: Dict[str, float] = field(
        default_factory=lambda: {"E": 0.1, "C": 0.1, "N": 0.05}
    )
    mode: str = "gradient_descent"  # choices: "gradient_descent", "attention_projection", "mlp_legacy"
    beta: float = 20.0             # Boltzmann sharpness parameter
    alpha: float = 0.2             # gradient descent step size
    basin: BasinConfig = field(default_factory=BasinConfig)

    def __post_init__(self):
        missing = [l for l in self.labels if l not in self.strengths]
        if missing:
            raise ValueError(f"strengths missing for labels: {missing}")

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    def strength_tensor(self):
        import torch

        return torch.tensor([self.strengths[l] for l in self.labels])

    # ---- serialization ----

    @classmethod
    def from_dict(cls, d: dict) -> "CollapseConfig":
        d = dict(d)
        if "labels" in d:
            d["labels"] = tuple(d["labels"])
        if isinstance(d.get("basin"), dict):
            d["basin"] = BasinConfig(**d["basin"])
        return cls(**d)

    @classmethod
    def from_yaml(cls, path) -> "CollapseConfig":
        import yaml

        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> dict:
        d = asdict(self)
        d["labels"] = list(self.labels)
        return d

    def save_yaml(self, path) -> None:
        import yaml

        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

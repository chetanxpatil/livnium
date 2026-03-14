"""
Bridge for quantum embeddings inside nova_v3.

Exposes QuantumTextEncoder by importing the implementation from
`nova/quantum_embed/text_encoder_quantum.py` and ensuring that path is on sys.path.
"""

import sys
from pathlib import Path

# Add quantum_embed directory to sys.path so we can import the implementation.
quantum_dir = (Path(__file__).resolve().parents[2] / "quantum_embed").resolve()
if str(quantum_dir) not in sys.path:
    sys.path.insert(0, str(quantum_dir))

try:
    from text_encoder_quantum import QuantumTextEncoder  # type: ignore
except ImportError as e:  # pragma: no cover - import path guard
    raise ImportError(
        "Could not import QuantumTextEncoder. Ensure nova/quantum_embed is present "
        "and contains text_encoder_quantum.py."
    ) from e


__all__ = ["QuantumTextEncoder"]

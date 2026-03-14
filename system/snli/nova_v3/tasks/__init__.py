"""
Task Heads

Task-specific heads that attach to the core physics engine.
"""

from .snli.head_snli import SNLIHead
from .snli.encoding_snli import SNLIEncoder, QuantumSNLIEncoder

__all__ = ['SNLIHead', 'SNLIEncoder', 'QuantumSNLIEncoder']

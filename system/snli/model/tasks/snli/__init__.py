"""
SNLI Task Head

SNLI-specific encoding and classification head.
"""

from .head_snli import SNLIHead
from .encoding_snli import SNLIEncoder, PretrainedSNLIEncoder, QuantumSNLIEncoder

__all__ = ['SNLIHead', 'SNLIEncoder', 'PretrainedSNLIEncoder', 'QuantumSNLIEncoder']

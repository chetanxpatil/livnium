"""
SNLI Task Head

SNLI-specific encoding and classification head.
"""

from .head_snli import SNLIHead, LinearSNLIHead
from .encoding_snli import SNLIEncoder, PretrainedSNLIEncoder, QuantumSNLIEncoder, BERTSNLIEncoder

__all__ = [
    'SNLIHead', 'LinearSNLIHead',
    'SNLIEncoder', 'PretrainedSNLIEncoder', 'QuantumSNLIEncoder', 'BERTSNLIEncoder',
]

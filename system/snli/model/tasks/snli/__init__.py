"""
SNLI Task Head

SNLI-specific encoding and classification head.
"""

from .head_snli import SNLIHead, BinaryHead, LinearSNLIHead
from .encoding_snli import (
    SNLIEncoder,
    PretrainedSNLIEncoder,
    QuantumSNLIEncoder,
    BERTSNLIEncoder,
    CrossEncoderBERTSNLIEncoder,
    LlamaCppBERTSNLIEncoder,
    LivniumNativeEncoder,
)

__all__ = [
    # Heads
    'SNLIHead', 'BinaryHead', 'LinearSNLIHead',
    # Encoders — legacy / pretrained BoW
    'SNLIEncoder', 'PretrainedSNLIEncoder', 'QuantumSNLIEncoder',
    # Encoders — BERT-based (bootstrap)
    'BERTSNLIEncoder', 'CrossEncoderBERTSNLIEncoder', 'LlamaCppBERTSNLIEncoder',
    # Encoder — Livnium-native (Stage 4: no BERT)
    'LivniumNativeEncoder',
]

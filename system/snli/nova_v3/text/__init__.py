"""
Text Encoding Layer

Task-agnostic text encoding.
"""

from .encoder import TextEncoder
from .quantum_text_encoder import QuantumTextEncoder

__all__ = ['TextEncoder', 'QuantumTextEncoder']

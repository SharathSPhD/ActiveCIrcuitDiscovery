# YorK_RP Circuit Analysis Module
"""Real circuit discovery using circuit-tracer + Gemma-2-2B transcoders."""

from .real_tracer import RealCircuitTracer

__all__ = [
    'RealCircuitTracer'
]
from .model_manager import ModelManager, model_manager

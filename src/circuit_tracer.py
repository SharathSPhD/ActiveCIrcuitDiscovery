"""Compatibility wrapper for tests.

This module exposes ``RealCircuitTracer`` and related data structures
at the top level so that ``from circuit_tracer import ...`` works when
``src`` is added to ``sys.path``.
"""

from circuit_analysis.tracer import CircuitTracer
from core.data_structures import SAEFeature, InterventionResult, AttributionGraph
from config.experiment_config import get_config, DeviceType


class RealCircuitTracer(CircuitTracer):
    """Thin wrapper allowing a ``device`` argument for tests."""

    def __init__(self, device: str | None = None):
        config = get_config()
        if device:
            config.model.device = DeviceType(device)
        super().__init__(config)

__all__ = [
    "RealCircuitTracer",
    "SAEFeature",
    "InterventionResult",
    "AttributionGraph",
]

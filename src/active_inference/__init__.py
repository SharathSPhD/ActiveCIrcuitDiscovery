# YorK_RP Active Inference Module
"""Active Inference agent implementation with pymdp integration."""

"""Active Inference package public API."""

from .agent import ActiveInferenceAgent
from core.data_structures import BeliefState
from config.experiment_config import get_config
from circuit_analysis.tracer import CircuitTracer


class ActiveInferenceGuide:
    """Convenience wrapper used in the tests.

    It wraps :class:`ActiveInferenceAgent` and exposes a slightly simpler
    interface where beliefs are initialised directly from an input string.
    """

    def __init__(self, tracer: CircuitTracer):
        self.tracer = tracer
        self.agent = ActiveInferenceAgent(get_config(), tracer)
        self.belief_state: BeliefState | None = None

    def initialize_beliefs(self, text: str) -> BeliefState:
        """Discover features for ``text`` and initialise agent beliefs."""
        features = self.tracer.find_active_features(
            text, self.agent.config.sae.activation_threshold
        )
        self.belief_state = self.agent.initialize_beliefs(features)
        return self.belief_state


def compare_intervention_strategies() -> dict:
    """Placeholder used in unit tests."""
    return {"comparison": "not_implemented"}


__all__ = [
    "ActiveInferenceAgent",
    "ActiveInferenceGuide",
    "BeliefState",
    "compare_intervention_strategies",
]

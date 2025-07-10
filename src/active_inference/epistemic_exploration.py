"""
Active Inference epistemic exploration for strategy learning.
Uses expected free energy and epistemic value instead of RL-based exploration.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class EpistemicExplorer:
    """Active Inference-based exploration using epistemic value and expected free energy."""
    
    def __init__(self, num_strategies: int = 144):
        self.num_strategies = num_strategies
        
        # Strategy uncertainty tracking (Active Inference principle)
        self.strategy_uncertainty = {}  # Dict[str, float]
        self.strategy_trial_counts = {}  # Dict[str, int]  
        self.strategy_outcome_history = {}  # Dict[str, List[float]]
        self.strategy_precision = {}  # Dict[str, float]
        
        # Initialize with high uncertainty (epistemic drive)
        for strategy_idx in range(num_strategies):
            strategy_key = f"strategy_{strategy_idx}"
            self.strategy_uncertainty[strategy_key] = 1.0  # High initial uncertainty
            self.strategy_trial_counts[strategy_key] = 0
            self.strategy_outcome_history[strategy_key] = []
            self.strategy_precision[strategy_key] = 0.1  # Low initial precision
    
    def calculate_epistemic_value(self, strategy_key: str) -> float:
        """Calculate epistemic value (expected information gain) for a strategy."""
        uncertainty = self.strategy_uncertainty.get(strategy_key, 1.0)
        trial_count = self.strategy_trial_counts.get(strategy_key, 0)
        
        # Epistemic value = potential uncertainty reduction
        # Higher uncertainty = higher epistemic value
        # More trials = lower epistemic value (diminishing returns)
        epistemic_value = uncertainty / (1.0 + 0.1 * trial_count)
        
        return epistemic_value
    
    def calculate_pragmatic_value(self, strategy_key: str, strategy_beliefs: Dict) -> float:
        """Calculate pragmatic value (expected effectiveness) for a strategy."""
        # Get current belief about strategy effectiveness
        if strategy_key in strategy_beliefs:
            effectiveness = strategy_beliefs[strategy_key]
        else:
            effectiveness = 0.5  # Neutral prior
        
        # Pragmatic value = expected utility from strategy
        return effectiveness
    
    def calculate_expected_free_energy(self, strategy_key: str, strategy_beliefs: Dict) -> float:
        """Calculate expected free energy for strategy selection (Active Inference)."""
        # EFE = -pragmatic_value + epistemic_value
        pragmatic_value = self.calculate_pragmatic_value(strategy_key, strategy_beliefs)
        epistemic_value = self.calculate_epistemic_value(strategy_key)
        
        # Expected free energy (to be minimized)
        efe = -pragmatic_value + epistemic_value
        
        return efe
    
    def select_strategy_by_efe(self, available_strategies: List[str], 
                             strategy_beliefs: Dict) -> str:
        """Select strategy by minimizing expected free energy (Active Inference principle)."""
        if not available_strategies:
            return None
        
        # Calculate EFE for each available strategy
        efe_scores = {}
        for strategy_key in available_strategies:
            efe = self.calculate_expected_free_energy(strategy_key, strategy_beliefs)
            efe_scores[strategy_key] = efe
        
        # Select strategy with minimum expected free energy
        selected_strategy = min(efe_scores.keys(), key=lambda k: efe_scores[k])
        
        logger.info(f"Strategy selection by EFE: {selected_strategy} (EFE: {efe_scores[selected_strategy]:.3f})")
        
        return selected_strategy
    
    def update_strategy_beliefs(self, strategy_key: str, outcome: float):
        """Update strategy beliefs and uncertainty after intervention (Active Inference learning)."""
        # Update trial count
        self.strategy_trial_counts[strategy_key] = self.strategy_trial_counts.get(strategy_key, 0) + 1
        
        # Update outcome history
        if strategy_key not in self.strategy_outcome_history:
            self.strategy_outcome_history[strategy_key] = []
        self.strategy_outcome_history[strategy_key].append(outcome)
        
        # Calculate outcome variance (measure of uncertainty)
        outcomes = self.strategy_outcome_history[strategy_key]
        if len(outcomes) > 1:
            outcome_variance = np.var(outcomes)
            # Higher variance = higher uncertainty
            self.strategy_uncertainty[strategy_key] = min(1.0, outcome_variance + 0.1)
        else:
            # Single trial - still uncertain
            self.strategy_uncertainty[strategy_key] = 0.8
        
        # Update precision (inverse of uncertainty)
        trial_count = self.strategy_trial_counts[strategy_key]
        base_precision = 1.0 / (1.0 + self.strategy_uncertainty[strategy_key])
        # Precision increases with more trials (confidence)
        self.strategy_precision[strategy_key] = base_precision * (1.0 + 0.1 * trial_count)
        
        logger.info(f"Updated strategy {strategy_key}: uncertainty={self.strategy_uncertainty[strategy_key]:.3f}, "
                   f"precision={self.strategy_precision[strategy_key]:.3f}")
    
    def get_strategy_uncertainty_scores(self) -> Dict[str, float]:
        """Get current uncertainty scores for all strategies."""
        return self.strategy_uncertainty.copy()
    
    def get_strategy_precision_weights(self) -> Dict[str, float]:
        """Get precision weights for belief updating."""
        return self.strategy_precision.copy()
    
    def calculate_temporal_variance(self, window_size: int = 5) -> float:
        """Calculate temporal variance in strategy selection for statistical validation."""
        # Get recent strategy uncertainties
        recent_uncertainties = []
        for strategy_key in sorted(self.strategy_uncertainty.keys()):
            if self.strategy_trial_counts[strategy_key] > 0:  # Only strategies that have been tried
                recent_uncertainties.append(self.strategy_uncertainty[strategy_key])
        
        if len(recent_uncertainties) >= 2:
            return float(np.var(recent_uncertainties))
        else:
            return 0.1  # Small non-zero variance for initialization
    
    def get_exploration_statistics(self) -> Dict:
        """Get statistics about exploration behavior for analysis."""
        total_trials = sum(self.strategy_trial_counts.values())
        strategies_tried = sum(1 for count in self.strategy_trial_counts.values() if count > 0)
        avg_uncertainty = np.mean(list(self.strategy_uncertainty.values()))
        avg_precision = np.mean(list(self.strategy_precision.values()))
        
        return {
            "total_trials": total_trials,
            "strategies_tried": strategies_tried,
            "avg_uncertainty": avg_uncertainty,
            "avg_precision": avg_precision,
            "temporal_variance": self.calculate_temporal_variance()
        }

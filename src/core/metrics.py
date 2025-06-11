# Correspondence Metrics Calculator with Proper Calibration
# Fixes the >100% correspondence issue identified in status analysis

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import logging
from dataclasses import dataclass

from .data_structures import (
    BeliefState, InterventionResult, CorrespondenceMetrics
)

logger = logging.getLogger(__name__)

@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    correlation: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    significance: bool
    sample_size: int

class CorrespondenceCalculator:
    """
    Properly calibrated correspondence metrics calculator.
    Ensures all metrics are bounded in [0, 100%] range.
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        
    def calculate_correspondence(self, ai_beliefs: BeliefState, 
                               circuit_behavior: List[InterventionResult]) -> CorrespondenceMetrics:
        """Calculate properly calibrated correspondence metrics."""
        logger.info("Calculating correspondence metrics with proper bounds")
        
        if not circuit_behavior:
            logger.warning("No circuit behavior data provided")
            return self._create_empty_correspondence()
        
        # Extract behavioral patterns
        circuit_patterns = self._extract_circuit_patterns(circuit_behavior)
        ai_patterns = self._extract_ai_patterns(ai_beliefs)
        
        # Calculate individual correspondences
        belief_correspondence = self._calculate_belief_updating_correspondence(
            ai_patterns['belief_updates'], circuit_patterns['activation_changes']
        )
        
        precision_correspondence = self._calculate_precision_weighting_correspondence(
            ai_patterns['precision_weights'], circuit_patterns['attention_patterns']
        )
        
        prediction_correspondence = self._calculate_prediction_error_correspondence(
            ai_patterns['prediction_errors'], circuit_patterns['intervention_effects']
        )
        
        # Calculate overall correspondence (weighted average)
        overall_correspondence = self._calculate_overall_correspondence([
            belief_correspondence.correlation,
            precision_correspondence.correlation, 
            prediction_correspondence.correlation
        ])
        
        return CorrespondenceMetrics(
            belief_updating_correspondence=belief_correspondence.correlation,
            precision_weighting_correspondence=precision_correspondence.correlation,
            prediction_error_correspondence=prediction_correspondence.correlation,
            overall_correspondence=overall_correspondence,
            circuit_attention_patterns=circuit_patterns['attention_patterns'],
            ai_precision_patterns=ai_patterns['precision_weights'],
            circuit_prediction_errors=circuit_patterns['intervention_effects'],
            ai_prediction_errors=ai_patterns['prediction_errors']
        )
    
    def _extract_circuit_patterns(self, interventions: List[InterventionResult]) -> Dict[str, List[float]]:
        """Extract behavioral patterns from circuit interventions."""
        patterns = {
            'activation_changes': [],
            'attention_patterns': [],
            'intervention_effects': []
        }
        
        for intervention in interventions:
            # Activation changes (normalized effect sizes)
            patterns['activation_changes'].append(
                min(1.0, abs(intervention.effect_size))
            )
            
            # Attention patterns (from logit differences)
            attention_strength = min(1.0, intervention.logit_diff_l2 / 10.0)
            patterns['attention_patterns'].append(attention_strength)
            
            # Intervention effects (KL divergence normalized)
            effect = min(1.0, intervention.kl_divergence / 5.0)
            patterns['intervention_effects'].append(effect)
        
        return patterns
    
    def _extract_ai_patterns(self, beliefs: BeliefState) -> Dict[str, List[float]]:
        """Extract patterns from Active Inference belief state."""
        patterns = {
            'belief_updates': [],
            'precision_weights': [],
            'prediction_errors': []
        }
        
        # Belief updates (from feature importance changes)
        for importance in beliefs.feature_importances.values():
            patterns['belief_updates'].append(min(1.0, max(0.0, importance)))
        
        # Precision weights (from uncertainty inverse)
        for uncertainty in beliefs.uncertainty.values():
            precision = max(0.0, min(1.0, 1.0 - uncertainty))
            patterns['precision_weights'].append(precision)
        
        # Prediction errors (from entropy)
        entropy = beliefs.get_entropy()
        normalized_entropy = min(1.0, entropy / 5.0)  # Normalize by max expected entropy
        patterns['prediction_errors'] = [normalized_entropy] * len(beliefs.feature_importances)
        
        return patterns
    
    def _calculate_belief_updating_correspondence(self, ai_beliefs: List[float], 
                                                circuit_changes: List[float]) -> StatisticalResult:
        """Calculate belief updating correspondence with proper bounds."""
        return self._calculate_bounded_correspondence(ai_beliefs, circuit_changes, "belief_updating")
    
    def _calculate_precision_weighting_correspondence(self, ai_precision: List[float],
                                                    circuit_attention: List[float]) -> StatisticalResult:
        """Calculate precision weighting correspondence with proper bounds."""
        return self._calculate_bounded_correspondence(ai_precision, circuit_attention, "precision_weighting")
    
    def _calculate_prediction_error_correspondence(self, ai_errors: List[float],
                                                 circuit_effects: List[float]) -> StatisticalResult:
        """Calculate prediction error correspondence with proper bounds.""" 
        return self._calculate_bounded_correspondence(ai_errors, circuit_effects, "prediction_error")
    
    def _calculate_bounded_correspondence(self, x: List[float], y: List[float], 
                                        metric_name: str) -> StatisticalResult:
        """Calculate correspondence ensuring proper [0, 100%] bounds."""
        if len(x) == 0 or len(y) == 0:
            logger.warning(f"Empty data for {metric_name} correspondence")
            return StatisticalResult(0.0, 1.0, (0.0, 0.0), 0.0, False, 0)
        
        # Ensure equal lengths by truncating to shorter
        min_len = min(len(x), len(y))
        x_norm = np.array(x[:min_len])
        y_norm = np.array(y[:min_len])
        
        # Ensure values are in [0, 1] range
        x_norm = np.clip(x_norm, 0.0, 1.0)
        y_norm = np.clip(y_norm, 0.0, 1.0)
        
        if np.var(x_norm) == 0 or np.var(y_norm) == 0:
            logger.warning(f"Zero variance in {metric_name} data")
            return StatisticalResult(0.0, 1.0, (0.0, 0.0), 0.0, False, min_len)
        
        # Calculate Pearson correlation
        correlation, p_value = pearsonr(x_norm, y_norm)
        
        # Handle NaN correlations
        if np.isnan(correlation):
            correlation = 0.0
            p_value = 1.0
        
        # Calculate confidence interval using Fisher transformation
        ci_lower, ci_upper = self._calculate_correlation_ci(correlation, min_len)
        
        # Convert correlation [-1, 1] to correspondence percentage [0, 100%]
        # Using: correspondence = (correlation + 1) / 2 * 100
        correspondence_pct = (correlation + 1) / 2 * 100
        
        # Ensure bounds [0, 100]
        correspondence_pct = max(0.0, min(100.0, correspondence_pct))
        
        # Effect size (Cohen's conventions for correlation)
        effect_size = abs(correlation)
        
        # Significance test
        significant = p_value < self.significance_level
        
        logger.debug(f"{metric_name} correspondence: {correspondence_pct:.1f}% "
                    f"(r={correlation:.3f}, p={p_value:.3f})")
        
        return StatisticalResult(
            correlation=correspondence_pct,
            p_value=p_value,
            confidence_interval=(
                max(0.0, (ci_lower + 1) / 2 * 100),
                min(100.0, (ci_upper + 1) / 2 * 100)
            ),
            effect_size=effect_size,
            significance=significant,
            sample_size=min_len
        )
    
    def _calculate_correlation_ci(self, r: float, n: int, 
                                confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for correlation using Fisher transformation."""
        if n < 3:
            return (r, r)  # Cannot calculate CI with too few samples
        
        # Fisher transformation
        z = 0.5 * np.log((1 + r) / (1 - r)) if abs(r) < 0.999 else np.sign(r) * 3
        
        # Standard error
        se = 1 / np.sqrt(n - 3)
        
        # Critical value for confidence interval
        alpha = 1 - confidence
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        # Confidence interval in Fisher z space
        z_lower = z - z_crit * se
        z_upper = z + z_crit * se
        
        # Transform back to correlation space
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return (r_lower, r_upper)
    
    def _calculate_overall_correspondence(self, individual_correspondences: List[float]) -> float:
        """Calculate overall correspondence from individual metrics."""
        valid_correspondences = [c for c in individual_correspondences if not np.isnan(c)]
        
        if not valid_correspondences:
            return 0.0
        
        # Simple average (could use weighted if desired)
        overall = np.mean(valid_correspondences)
        
        # Ensure bounds [0, 100]
        return max(0.0, min(100.0, overall))
    
    def _create_empty_correspondence(self) -> CorrespondenceMetrics:
        """Create empty correspondence metrics for edge cases."""
        return CorrespondenceMetrics(
            belief_updating_correspondence=0.0,
            precision_weighting_correspondence=0.0,
            prediction_error_correspondence=0.0,
            overall_correspondence=0.0,
            circuit_attention_patterns=[],
            ai_precision_patterns=[],
            circuit_prediction_errors=[],
            ai_prediction_errors=[]
        )

class EfficiencyCalculator:
    """Calculator for RQ2 efficiency metrics."""
    
    def __init__(self):
        self.baseline_methods = ['random', 'exhaustive', 'gradient_based']
    
    def calculate_efficiency_improvement(self, ai_interventions: int,
                                       baseline_results: Dict[str, int]) -> Dict[str, float]:
        """Calculate efficiency improvement over baseline methods."""
        improvements = {}
        
        for method, baseline_count in baseline_results.items():
            if baseline_count > 0:
                # Improvement = (baseline - ai) / baseline * 100
                improvement = (baseline_count - ai_interventions) / baseline_count * 100
                improvements[f"{method}_improvement"] = max(0.0, improvement)
            else:
                improvements[f"{method}_improvement"] = 0.0
        
        # Overall efficiency improvement (average)
        if improvements:
            improvements['overall_efficiency'] = np.mean(list(improvements.values()))
        else:
            improvements['overall_efficiency'] = 0.0
        
        return improvements

class ValidationCalculator:
    """Calculator for research question validation."""
    
    def __init__(self, rq1_target: float = 70.0, rq2_target: float = 30.0, 
                 rq3_target: int = 3):
        self.rq1_target = rq1_target
        self.rq2_target = rq2_target
        self.rq3_target = rq3_target
    
    def validate_research_questions(self, correspondence: CorrespondenceMetrics,
                                  efficiency: Dict[str, float],
                                  predictions: List['NovelPrediction']) -> Dict[str, bool]:
        """Validate all research questions against targets."""
        
        # RQ1: Correspondence target
        rq1_passed = correspondence.overall_correspondence >= self.rq1_target
        
        # RQ2: Efficiency target  
        rq2_passed = efficiency.get('overall_efficiency', 0.0) >= self.rq2_target
        
        # RQ3: Novel predictions target
        validated_predictions = sum(1 for p in predictions 
                                  if p.validation_status == 'validated')
        rq3_passed = validated_predictions >= self.rq3_target
        
        return {
            'rq1_passed': rq1_passed,
            'rq2_passed': rq2_passed, 
            'rq3_passed': rq3_passed,
            'overall_success': rq1_passed and rq2_passed and rq3_passed
        }
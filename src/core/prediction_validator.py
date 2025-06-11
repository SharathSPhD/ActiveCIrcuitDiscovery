# Empirical Prediction Validation Framework
# Critical for RQ3: Validates novel predictions with statistical rigor

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu
from scipy import stats
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .data_structures import NovelPrediction, InterventionResult, AttributionGraph
from .prediction_system import ValidationResult, PredictionEvidence

logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for prediction validation."""
    significance_level: float = 0.05
    min_sample_size: int = 10
    bootstrap_samples: int = 1000
    effect_size_threshold: float = 0.3
    correlation_threshold: float = 0.6

class ValidationMethod(ABC):
    """Abstract base for validation methods."""
    
    @abstractmethod
    def validate(self, prediction: NovelPrediction, test_data: Dict[str, Any],
                config: ValidationConfig) -> ValidationResult:
        pass

class AttentionPatternValidator(ValidationMethod):
    """Validates attention pattern predictions."""
    
    def validate(self, prediction: NovelPrediction, test_data: Dict[str, Any],
                config: ValidationConfig) -> ValidationResult:
        """Validate attention pattern predictions."""
        logger.info(f"Validating attention pattern prediction: {prediction.description[:50]}...")
        
        if "uncertainty_attention_correlation" in prediction.testable_hypothesis.lower():
            return self._validate_uncertainty_attention_correlation(prediction, test_data, config)
        elif "dynamic" in prediction.testable_hypothesis.lower():
            return self._validate_dynamic_attention_reweighting(prediction, test_data, config)
        else:
            return self._validate_generic_attention_pattern(prediction, test_data, config)
    
    def _validate_uncertainty_attention_correlation(self, prediction: NovelPrediction,
                                                  test_data: Dict[str, Any],
                                                  config: ValidationConfig) -> ValidationResult:
        """Validate uncertainty-attention correlation prediction."""
        
        # Extract uncertainty and attention data
        uncertainties = test_data.get('feature_uncertainties', [])
        attention_weights = test_data.get('attention_weights', [])
        
        if len(uncertainties) < config.min_sample_size or len(attention_weights) < config.min_sample_size:
            return ValidationResult(
                prediction_id=getattr(prediction, 'prediction_id', 'unknown'),
                validation_status='inconclusive',
                evidence={},
                overall_confidence=0.0,
                statistical_summary={'error': 'insufficient_data'}
            )
        
        # Ensure equal lengths
        min_len = min(len(uncertainties), len(attention_weights))
        uncertainties = np.array(uncertainties[:min_len])
        attention_weights = np.array(attention_weights[:min_len])
        
        # Calculate correlation
        correlation, p_value = pearsonr(uncertainties, attention_weights)
        
        # Bootstrap confidence interval
        bootstrap_correlations = []
        for _ in range(config.bootstrap_samples):
            indices = np.random.choice(min_len, min_len, replace=True)
            boot_r, _ = pearsonr(uncertainties[indices], attention_weights[indices])
            if not np.isnan(boot_r):
                bootstrap_correlations.append(boot_r)
        
        ci_lower, ci_upper = np.percentile(bootstrap_correlations, [2.5, 97.5])
        
        # Determine validation status
        expected_correlation = 0.7  # From prediction
        validation_status = 'validated' if (correlation >= expected_correlation and 
                                          p_value < config.significance_level) else 'falsified'
        
        evidence = {
            'correlation': PredictionEvidence(
                measurement_type='pearson_correlation',
                expected_value=expected_correlation,
                observed_value=correlation,
                statistical_significance=p_value,
                effect_size=abs(correlation),
                confidence_interval=(ci_lower, ci_upper)
            )
        }
        
        overall_confidence = self._calculate_overall_confidence(evidence, config)
        
        return ValidationResult(
            prediction_id=getattr(prediction, 'prediction_id', 'unknown'),
            validation_status=validation_status,
            evidence=evidence,
            overall_confidence=overall_confidence,
            statistical_summary={
                'correlation': correlation,
                'p_value': p_value,
                'sample_size': min_len,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
        )
    
    def _validate_dynamic_attention_reweighting(self, prediction: NovelPrediction,
                                              test_data: Dict[str, Any],
                                              config: ValidationConfig) -> ValidationResult:
        """Validate dynamic attention reweighting prediction."""
        
        # Extract before/after attention data
        attention_before = test_data.get('attention_before_intervention', [])
        attention_after = test_data.get('attention_after_intervention', [])
        belief_changes = test_data.get('belief_changes', [])
        
        if (len(attention_before) < config.min_sample_size or 
            len(attention_after) < config.min_sample_size):
            return self._insufficient_data_result(prediction)
        
        # Calculate attention changes
        attention_changes = np.array(attention_after) - np.array(attention_before)
        significant_belief_changes = np.array(belief_changes) > 0.5
        
        # Test effect size for significant belief changes
        if np.sum(significant_belief_changes) < 3:
            return self._insufficient_data_result(prediction)
        
        significant_attention_changes = attention_changes[significant_belief_changes]
        baseline_attention_changes = attention_changes[~significant_belief_changes]
        
        # Statistical test
        if len(baseline_attention_changes) > 0:
            statistic, p_value = ttest_ind(significant_attention_changes, baseline_attention_changes)
            effect_size = self._calculate_cohens_d(significant_attention_changes, baseline_attention_changes)
        else:
            # One-sample test against zero
            statistic, p_value = stats.ttest_1samp(significant_attention_changes, 0)
            effect_size = np.mean(significant_attention_changes) / np.std(significant_attention_changes)
        
        # Expected effect size from prediction
        expected_effect_size = 0.5
        validation_status = 'validated' if (abs(effect_size) >= expected_effect_size and 
                                          p_value < config.significance_level) else 'falsified'
        
        evidence = {
            'effect_size': PredictionEvidence(
                measurement_type='cohens_d',
                expected_value=expected_effect_size,
                observed_value=effect_size,
                statistical_significance=p_value,
                effect_size=abs(effect_size),
                confidence_interval=self._effect_size_ci(significant_attention_changes, baseline_attention_changes)
            )
        }
        
        return ValidationResult(
            prediction_id=getattr(prediction, 'prediction_id', 'unknown'),
            validation_status=validation_status,
            evidence=evidence,
            overall_confidence=self._calculate_overall_confidence(evidence, config),
            statistical_summary={
                'effect_size': effect_size,
                'p_value': p_value,
                'significant_changes': len(significant_attention_changes),
                'baseline_changes': len(baseline_attention_changes)
            }
        )
    
    def _validate_generic_attention_pattern(self, prediction: NovelPrediction,
                                          test_data: Dict[str, Any],
                                          config: ValidationConfig) -> ValidationResult:
        """Validate generic attention pattern prediction."""
        # Placeholder for other attention pattern validations
        return ValidationResult(
            prediction_id=getattr(prediction, 'prediction_id', 'unknown'),
            validation_status='inconclusive',
            evidence={},
            overall_confidence=0.5,
            statistical_summary={'method': 'generic_attention_validation'}
        )

class FeatureInteractionValidator(ValidationMethod):
    """Validates feature interaction predictions."""
    
    def validate(self, prediction: NovelPrediction, test_data: Dict[str, Any],
                config: ValidationConfig) -> ValidationResult:
        """Validate feature interaction predictions."""
        logger.info(f"Validating feature interaction prediction: {prediction.description[:50]}...")
        
        if "causal" in prediction.testable_hypothesis.lower():
            return self._validate_causal_connection_strength(prediction, test_data, config)
        elif "hierarchical" in prediction.testable_hypothesis.lower():
            return self._validate_hierarchical_information_flow(prediction, test_data, config)
        else:
            return self._validate_generic_feature_interaction(prediction, test_data, config)
    
    def _validate_causal_connection_strength(self, prediction: NovelPrediction,
                                           test_data: Dict[str, Any],
                                           config: ValidationConfig) -> ValidationResult:
        """Validate causal connection strength prediction."""
        
        # Extract ablation effect data
        high_belief_effects = test_data.get('high_belief_connection_effects', [])
        low_belief_effects = test_data.get('low_belief_connection_effects', [])
        
        if len(high_belief_effects) < config.min_sample_size:
            return self._insufficient_data_result(prediction)
        
        # Expected effect size from prediction (> 0.4)
        expected_effect = 0.4
        observed_mean_effect = np.mean(high_belief_effects)
        
        # One-sample t-test against expected threshold
        statistic, p_value = stats.ttest_1samp(high_belief_effects, expected_effect)
        
        # Effect size (standardized mean difference)
        effect_size = (observed_mean_effect - expected_effect) / np.std(high_belief_effects)
        
        validation_status = 'validated' if (observed_mean_effect >= expected_effect and 
                                          p_value < config.significance_level) else 'falsified'
        
        evidence = {
            'mean_effect': PredictionEvidence(
                measurement_type='mean_intervention_effect',
                expected_value=expected_effect,
                observed_value=observed_mean_effect,
                statistical_significance=p_value,
                effect_size=abs(effect_size),
                confidence_interval=stats.t.interval(0.95, len(high_belief_effects)-1,
                                                   loc=observed_mean_effect,
                                                   scale=stats.sem(high_belief_effects))
            )
        }
        
        return ValidationResult(
            prediction_id=getattr(prediction, 'prediction_id', 'unknown'),
            validation_status=validation_status,
            evidence=evidence,
            overall_confidence=self._calculate_overall_confidence(evidence, config),
            statistical_summary={
                'mean_effect': observed_mean_effect,
                'expected_effect': expected_effect,
                'p_value': p_value,
                'sample_size': len(high_belief_effects)
            }
        )
    
    def _validate_hierarchical_information_flow(self, prediction: NovelPrediction,
                                              test_data: Dict[str, Any],
                                              config: ValidationConfig) -> ValidationResult:
        """Validate hierarchical information flow prediction."""
        
        # Extract layer connectivity data
        layer_depths = test_data.get('layer_depths', [])
        in_out_ratios = test_data.get('in_out_degree_ratios', [])
        
        if len(layer_depths) < config.min_sample_size or len(in_out_ratios) < config.min_sample_size:
            return self._insufficient_data_result(prediction)
        
        # Calculate correlation between layer depth and connectivity ratio
        correlation, p_value = pearsonr(layer_depths, in_out_ratios)
        
        expected_correlation = 0.6  # From prediction
        validation_status = 'validated' if (correlation >= expected_correlation and 
                                          p_value < config.significance_level) else 'falsified'
        
        evidence = {
            'hierarchy_correlation': PredictionEvidence(
                measurement_type='pearson_correlation',
                expected_value=expected_correlation,
                observed_value=correlation,
                statistical_significance=p_value,
                effect_size=abs(correlation),
                confidence_interval=self._correlation_ci(correlation, len(layer_depths))
            )
        }
        
        return ValidationResult(
            prediction_id=getattr(prediction, 'prediction_id', 'unknown'),
            validation_status=validation_status,
            evidence=evidence,
            overall_confidence=self._calculate_overall_confidence(evidence, config),
            statistical_summary={
                'correlation': correlation,
                'p_value': p_value,
                'sample_size': len(layer_depths)
            }
        )
    
    def _validate_generic_feature_interaction(self, prediction: NovelPrediction,
                                            test_data: Dict[str, Any],
                                            config: ValidationConfig) -> ValidationResult:
        """Validate generic feature interaction prediction."""
        return ValidationResult(
            prediction_id=getattr(prediction, 'prediction_id', 'unknown'),
            validation_status='inconclusive',
            evidence={},
            overall_confidence=0.5,
            statistical_summary={'method': 'generic_interaction_validation'}
        )

class FailureModeValidator(ValidationMethod):
    """Validates failure mode predictions."""
    
    def validate(self, prediction: NovelPrediction, test_data: Dict[str, Any],
                config: ValidationConfig) -> ValidationResult:
        """Validate failure mode predictions."""
        logger.info(f"Validating failure mode prediction: {prediction.description[:50]}...")
        
        if "uncertainty" in prediction.testable_hypothesis.lower():
            return self._validate_uncertainty_degradation(prediction, test_data, config)
        elif "cascade" in prediction.testable_hypothesis.lower():
            return self._validate_error_cascades(prediction, test_data, config)
        else:
            return self._validate_generic_failure_mode(prediction, test_data, config)
    
    def _validate_uncertainty_degradation(self, prediction: NovelPrediction,
                                        test_data: Dict[str, Any],
                                        config: ValidationConfig) -> ValidationResult:
        """Validate uncertainty degradation prediction."""
        
        # Extract uncertainty and performance data
        uncertainties = test_data.get('feature_uncertainties', [])
        performance_scores = test_data.get('performance_under_noise', [])
        
        if len(uncertainties) < config.min_sample_size or len(performance_scores) < config.min_sample_size:
            return self._insufficient_data_result(prediction)
        
        # Calculate correlation (should be negative)
        correlation, p_value = pearsonr(uncertainties, performance_scores)
        
        expected_correlation = -0.6  # From prediction
        validation_status = 'validated' if (correlation <= expected_correlation and 
                                          p_value < config.significance_level) else 'falsified'
        
        evidence = {
            'uncertainty_performance_correlation': PredictionEvidence(
                measurement_type='pearson_correlation',
                expected_value=expected_correlation,
                observed_value=correlation,
                statistical_significance=p_value,
                effect_size=abs(correlation),
                confidence_interval=self._correlation_ci(correlation, len(uncertainties))
            )
        }
        
        return ValidationResult(
            prediction_id=getattr(prediction, 'prediction_id', 'unknown'),
            validation_status=validation_status,
            evidence=evidence,
            overall_confidence=self._calculate_overall_confidence(evidence, config),
            statistical_summary={
                'correlation': correlation,
                'p_value': p_value,
                'sample_size': len(uncertainties)
            }
        )
    
    def _validate_error_cascades(self, prediction: NovelPrediction,
                               test_data: Dict[str, Any],
                               config: ValidationConfig) -> ValidationResult:
        """Validate error cascade prediction."""
        
        # Extract error propagation data
        upstream_errors = test_data.get('upstream_errors', [])
        downstream_effects = test_data.get('downstream_effects', [])
        
        if len(upstream_errors) < config.min_sample_size or len(downstream_effects) < config.min_sample_size:
            return self._insufficient_data_result(prediction)
        
        # Calculate effect size for error propagation
        effect_size = np.mean(downstream_effects) / np.std(downstream_effects) if np.std(downstream_effects) > 0 else 0
        
        # Test against threshold
        expected_effect = 0.3  # From prediction
        t_stat, p_value = stats.ttest_1samp(downstream_effects, expected_effect)
        
        validation_status = 'validated' if (effect_size >= expected_effect and 
                                          p_value < config.significance_level) else 'falsified'
        
        evidence = {
            'error_propagation_effect': PredictionEvidence(
                measurement_type='effect_size',
                expected_value=expected_effect,
                observed_value=effect_size,
                statistical_significance=p_value,
                effect_size=abs(effect_size),
                confidence_interval=stats.t.interval(0.95, len(downstream_effects)-1,
                                                   loc=np.mean(downstream_effects),
                                                   scale=stats.sem(downstream_effects))
            )
        }
        
        return ValidationResult(
            prediction_id=getattr(prediction, 'prediction_id', 'unknown'),
            validation_status=validation_status,
            evidence=evidence,
            overall_confidence=self._calculate_overall_confidence(evidence, config),
            statistical_summary={
                'effect_size': effect_size,
                'p_value': p_value,
                'sample_size': len(downstream_effects)
            }
        )
    
    def _validate_generic_failure_mode(self, prediction: NovelPrediction,
                                     test_data: Dict[str, Any],
                                     config: ValidationConfig) -> ValidationResult:
        """Validate generic failure mode prediction."""
        return ValidationResult(
            prediction_id=getattr(prediction, 'prediction_id', 'unknown'),
            validation_status='inconclusive',
            evidence={},
            overall_confidence=0.5,
            statistical_summary={'method': 'generic_failure_validation'}
        )
    
    # Helper methods
    def _insufficient_data_result(self, prediction: NovelPrediction) -> ValidationResult:
        """Return result for insufficient data."""
        return ValidationResult(
            prediction_id=getattr(prediction, 'prediction_id', 'unknown'),
            validation_status='inconclusive',
            evidence={},
            overall_confidence=0.0,
            statistical_summary={'error': 'insufficient_data'}
        )
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1-1)*np.var(group1) + (n2-1)*np.var(group2)) / (n1+n2-2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
    
    def _effect_size_ci(self, group1: np.ndarray, group2: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Calculate confidence interval for effect size."""
        if group2 is not None:
            # Two-sample case
            effect = self._calculate_cohens_d(group1, group2)
            se = np.sqrt((len(group1) + len(group2)) / (len(group1) * len(group2)) + 
                        effect**2 / (2 * (len(group1) + len(group2))))
        else:
            # One-sample case
            effect = np.mean(group1) / np.std(group1)
            se = np.sqrt(1/len(group1) + effect**2/(2*len(group1)))
        
        return (effect - 1.96*se, effect + 1.96*se)
    
    def _correlation_ci(self, r: float, n: int) -> Tuple[float, float]:
        """Calculate confidence interval for correlation."""
        if n < 3:
            return (r, r)
        
        # Fisher transformation
        z = 0.5 * np.log((1 + r) / (1 - r)) if abs(r) < 0.999 else np.sign(r) * 3
        se = 1 / np.sqrt(n - 3)
        
        z_lower = z - 1.96 * se
        z_upper = z + 1.96 * se
        
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return (r_lower, r_upper)
    
    def _calculate_overall_confidence(self, evidence: Dict[str, PredictionEvidence],
                                    config: ValidationConfig) -> float:
        """Calculate overall confidence from evidence."""
        if not evidence:
            return 0.0
        
        confidences = []
        for ev in evidence.values():
            # Base confidence on statistical significance and effect size
            sig_confidence = 1 - ev.statistical_significance
            effect_confidence = min(1.0, ev.effect_size / config.effect_size_threshold)
            confidences.append(np.mean([sig_confidence, effect_confidence]))
        
        return np.mean(confidences)

class PredictionValidator:
    """
    Main prediction validation system.
    Empirically tests novel predictions with statistical rigor.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.validators = {
            'attention_pattern': AttentionPatternValidator(),
            'feature_interaction': FeatureInteractionValidator(),
            'failure_mode': FailureModeValidator()
        }
        self.validation_history = []
    
    def validate_prediction(self, prediction: NovelPrediction, 
                          test_data: Dict[str, Any]) -> ValidationResult:
        """Validate a single prediction with appropriate method."""
        logger.info(f"Validating prediction: {prediction.prediction_type}")
        
        validator = self.validators.get(prediction.prediction_type)
        if not validator:
            logger.warning(f"No validator found for prediction type: {prediction.prediction_type}")
            return ValidationResult(
                prediction_id=getattr(prediction, 'prediction_id', 'unknown'),
                validation_status='inconclusive',
                evidence={},
                overall_confidence=0.0,
                statistical_summary={'error': 'no_validator'}
            )
        
        try:
            result = validator.validate(prediction, test_data, self.config)
            self.validation_history.append({
                'prediction': prediction,
                'result': result,
                'timestamp': np.datetime64('now')
            })
            return result
            
        except Exception as e:
            logger.error(f"Error validating prediction: {e}")
            return ValidationResult(
                prediction_id=getattr(prediction, 'prediction_id', 'unknown'),
                validation_status='inconclusive',
                evidence={},
                overall_confidence=0.0,
                statistical_summary={'error': str(e)}
            )
    
    def validate_predictions(self, predictions: List[NovelPrediction],
                           test_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate multiple predictions."""
        results = []
        for prediction in predictions:
            result = self.validate_prediction(prediction, test_data)
            results.append(result)
        
        # Log summary
        validated_count = sum(1 for r in results if r.validation_status == 'validated')
        logger.info(f"Validation complete: {validated_count}/{len(results)} predictions validated")
        
        return results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        if not self.validation_history:
            return {'total_validations': 0}
        
        results = [entry['result'] for entry in self.validation_history]
        
        return {
            'total_validations': len(results),
            'validated_count': sum(1 for r in results if r.validation_status == 'validated'),
            'falsified_count': sum(1 for r in results if r.validation_status == 'falsified'),
            'inconclusive_count': sum(1 for r in results if r.validation_status == 'inconclusive'),
            'average_confidence': np.mean([r.overall_confidence for r in results]),
            'validation_rate': sum(1 for r in results if r.validation_status == 'validated') / len(results)
        }
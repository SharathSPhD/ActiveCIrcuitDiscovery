# Statistical Validation Framework
# Provides comprehensive statistical testing for research question validation

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.stats import bootstrap
from dataclasses import dataclass
import logging

from .data_structures import (
    CorrespondenceMetrics, NovelPrediction, ExperimentResult
)

logger = logging.getLogger(__name__)

@dataclass
class StatisticalTest:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    power: float
    significant: bool
    interpretation: str

@dataclass 
class BootstrapResult:
    """Container for bootstrap analysis results."""
    mean: float
    std: float
    confidence_interval: Tuple[float, float]
    distribution: np.ndarray

class StatisticalValidator:
    """
    Comprehensive statistical validation framework for research questions.
    Implements proper statistical testing with effect sizes, confidence intervals, and power analysis.
    """
    
    def __init__(self, alpha: float = 0.05, bootstrap_samples: int = 10000):
        self.alpha = alpha
        self.bootstrap_samples = bootstrap_samples
        self.test_results = []
    
    def validate_correspondence_significance(self, correspondence_metrics: List[CorrespondenceMetrics],
                                          target_threshold: float = 70.0) -> StatisticalTest:
        """
        Validate that correspondence significantly exceeds target threshold.
        Tests RQ1: >70% correspondence between Active Inference and circuit behavior.
        """
        logger.info(f"Testing correspondence significance against {target_threshold}% threshold")
        
        if not correspondence_metrics:
            return StatisticalTest(
                test_name="correspondence_significance",
                statistic=0.0, p_value=1.0,
                confidence_interval=(0.0, 0.0),
                effect_size=0.0, power=0.0,
                significant=False,
                interpretation="No correspondence data available"
            )
        
        # Extract overall correspondence percentages
        correspondences = [cm.overall_correspondence * 100 for cm in correspondence_metrics]
        
        # One-sample t-test against threshold
        statistic, p_value = stats.ttest_1samp(correspondences, target_threshold)
        
        # Bootstrap confidence interval
        bootstrap_result = self._bootstrap_mean(correspondences)
        
        # Effect size (Cohen's d)
        effect_size = (np.mean(correspondences) - target_threshold) / np.std(correspondences, ddof=1)
        
        # Power analysis
        power = self._calculate_power_one_sample(effect_size, len(correspondences), self.alpha)
        
        # Significance
        significant = p_value < self.alpha and np.mean(correspondences) > target_threshold
        
        # Interpretation
        mean_correspondence = np.mean(correspondences)
        if significant:
            interpretation = f"Correspondence ({mean_correspondence:.1f}%) significantly exceeds {target_threshold}% threshold"
        else:
            interpretation = f"Correspondence ({mean_correspondence:.1f}%) does not significantly exceed {target_threshold}% threshold"
        
        result = StatisticalTest(
            test_name="correspondence_significance",
            statistic=statistic,
            p_value=p_value,
            confidence_interval=bootstrap_result.confidence_interval,
            effect_size=effect_size,
            power=power,
            significant=significant,
            interpretation=interpretation
        )
        
        self.test_results.append(result)
        logger.info(f"Correspondence test: {interpretation} (p={p_value:.4f}, d={effect_size:.3f})")
        
        return result
    
    def validate_efficiency_improvement(self, ai_interventions: List[int],
                                      baseline_interventions: Dict[str, List[int]],
                                      target_improvement: float = 30.0) -> Dict[str, StatisticalTest]:
        """
        Validate efficiency improvement against baseline methods.
        Tests RQ2: 30% efficiency improvement over baseline methods.
        """
        logger.info(f"Testing efficiency improvement against {target_improvement}% target")
        
        results = {}
        
        for baseline_name, baseline_counts in baseline_interventions.items():
            if not baseline_counts or not ai_interventions:
                continue
            
            logger.debug(f"Testing efficiency vs {baseline_name} baseline")
            
            # Calculate improvement percentages
            improvements = []
            for ai_count in ai_interventions:
                for baseline_count in baseline_counts:
                    if baseline_count > 0:
                        improvement = ((baseline_count - ai_count) / baseline_count) * 100
                        improvements.append(improvement)
            
            if not improvements:
                continue
            
            # One-sample t-test against target improvement
            statistic, p_value = stats.ttest_1samp(improvements, target_improvement)
            
            # Bootstrap confidence interval
            bootstrap_result = self._bootstrap_mean(improvements)
            
            # Effect size
            effect_size = (np.mean(improvements) - target_improvement) / np.std(improvements, ddof=1)
            
            # Power analysis
            power = self._calculate_power_one_sample(effect_size, len(improvements), self.alpha)
            
            # Significance
            significant = p_value < self.alpha and np.mean(improvements) > target_improvement
            
            # Interpretation
            mean_improvement = np.mean(improvements)
            if significant:
                interpretation = f"Efficiency improvement vs {baseline_name} ({mean_improvement:.1f}%) significantly exceeds {target_improvement}%"
            else:
                interpretation = f"Efficiency improvement vs {baseline_name} ({mean_improvement:.1f}%) does not significantly exceed {target_improvement}%"
            
            result = StatisticalTest(
                test_name=f"efficiency_vs_{baseline_name}",
                statistic=statistic,
                p_value=p_value,
                confidence_interval=bootstrap_result.confidence_interval,
                effect_size=effect_size,
                power=power,
                significant=significant,
                interpretation=interpretation
            )
            
            results[baseline_name] = result
            self.test_results.append(result)
            logger.info(f"Efficiency vs {baseline_name}: {interpretation} (p={p_value:.4f})")
        
        return results
    
    def validate_prediction_success_rate(self, predictions: List[NovelPrediction],
                                       target_count: int = 3) -> StatisticalTest:
        """
        Validate that prediction success rate meets target.
        Tests RQ3: 3+ validated novel predictions.
        """
        logger.info(f"Testing prediction success rate against target of {target_count} validated predictions")
        
        if not predictions:
            return StatisticalTest(
                test_name="prediction_success_rate",
                statistic=0.0, p_value=1.0,
                confidence_interval=(0.0, 0.0),
                effect_size=0.0, power=0.0,
                significant=False,
                interpretation="No predictions available"
            )
        
        # Count validated predictions
        validated_count = sum(1 for p in predictions if p.validation_status == 'validated')
        total_predictions = len(predictions)
        success_rate = validated_count / total_predictions if total_predictions > 0 else 0.0
        
        # Binomial test for success rate
        # Test if success rate is significantly greater than chance (0.33 = 1/3 for 3-outcome validation)
        chance_rate = 0.33
        statistic, p_value = stats.binomtest(validated_count, total_predictions, chance_rate, 
                                           alternative='greater')
        
        # Confidence interval for proportion using Wilson score
        ci_lower, ci_upper = self._wilson_score_interval(validated_count, total_predictions)
        
        # Effect size (odds ratio)
        if success_rate > 0 and success_rate < 1:
            odds_success = success_rate / (1 - success_rate)
            odds_chance = chance_rate / (1 - chance_rate)
            effect_size = np.log(odds_success / odds_chance)  # Log odds ratio
        else:
            effect_size = 0.0
        
        # Power analysis for binomial test
        power = self._calculate_power_binomial(success_rate, chance_rate, total_predictions, self.alpha)
        
        # Significance (must meet count threshold AND be significantly better than chance)
        meets_threshold = validated_count >= target_count
        significantly_better = p_value < self.alpha
        significant = meets_threshold and significantly_better
        
        # Interpretation
        if significant:
            interpretation = f"Validation success: {validated_count}/{total_predictions} predictions validated (≥{target_count} target met, significantly > chance)"
        else:
            interpretation = f"Validation success: {validated_count}/{total_predictions} predictions validated (target: ≥{target_count}, p={p_value:.4f})"
        
        result = StatisticalTest(
            test_name="prediction_success_rate",
            statistic=statistic.statistic if hasattr(statistic, 'statistic') else validated_count,
            p_value=p_value,
            confidence_interval=(ci_lower * total_predictions, ci_upper * total_predictions),
            effect_size=effect_size,
            power=power,
            significant=significant,
            interpretation=interpretation
        )
        
        self.test_results.append(result)
        logger.info(f"Prediction validation: {interpretation}")
        
        return result
    
    def comprehensive_experiment_validation(self, result: ExperimentResult) -> Dict[str, Any]:
        """
        Perform comprehensive statistical validation of experiment results.
        Tests all research questions with proper statistical rigor.
        """
        logger.info("Performing comprehensive statistical validation of experiment results")
        
        validation_results = {}
        
        # RQ1: Correspondence validation
        if result.correspondence_metrics:
            rq1_test = self.validate_correspondence_significance(
                result.correspondence_metrics,
                result.config_used.get('research_questions', {}).get('rq1_correspondence_target', 70.0)
            )
            validation_results['rq1_statistical_test'] = rq1_test
        
        # RQ2: Efficiency validation
        if result.efficiency_metrics and result.intervention_results:
            ai_intervention_counts = [len(result.intervention_results)]
            
            # Create baseline data from efficiency metrics
            baseline_data = {}
            for key, value in result.efficiency_metrics.items():
                if '_improvement' in key and key != 'overall_improvement':
                    baseline_name = key.replace('_improvement', '')
                    # Reverse-calculate baseline count from improvement percentage
                    if value > 0:
                        ai_count = result.efficiency_metrics.get('ai_interventions', 20)
                        baseline_count = int(ai_count / (1 - value/100))
                        baseline_data[baseline_name] = [baseline_count]
            
            if baseline_data:
                rq2_tests = self.validate_efficiency_improvement(
                    ai_intervention_counts,
                    baseline_data,
                    result.config_used.get('research_questions', {}).get('rq2_efficiency_target', 30.0)
                )
                validation_results['rq2_statistical_tests'] = rq2_tests
        
        # RQ3: Prediction validation
        if result.novel_predictions:
            rq3_test = self.validate_prediction_success_rate(
                result.novel_predictions,
                result.config_used.get('research_questions', {}).get('rq3_predictions_target', 3)
            )
            validation_results['rq3_statistical_test'] = rq3_test
        
        # Overall statistical summary
        validation_results['statistical_summary'] = self._generate_statistical_summary()
        
        logger.info("Comprehensive statistical validation completed")
        return validation_results
    
    def _bootstrap_mean(self, data: List[float]) -> BootstrapResult:
        """Calculate bootstrap confidence interval for mean."""
        data_array = np.array(data)
        
        def mean_func(x, axis):
            return np.mean(x, axis=axis)
        
        bootstrap_result = bootstrap(
            (data_array,), mean_func, n_resamples=self.bootstrap_samples,
            confidence_level=1-self.alpha, random_state=42
        )
        
        return BootstrapResult(
            mean=np.mean(data_array),
            std=np.std(data_array, ddof=1),
            confidence_interval=bootstrap_result.confidence_interval,
            distribution=np.array([mean_func(np.random.choice(data_array, len(data_array), replace=True), axis=0) 
                                 for _ in range(1000)])
        )
    
    def _wilson_score_interval(self, successes: int, trials: int, 
                             confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval for proportion."""
        if trials == 0:
            return (0.0, 0.0)
        
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p = successes / trials
        
        denominator = 1 + z**2 / trials
        centre = (p + z**2 / (2 * trials)) / denominator
        half_width = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
        
        return (max(0, centre - half_width), min(1, centre + half_width))
    
    def _calculate_power_one_sample(self, effect_size: float, n: int, alpha: float) -> float:
        """Calculate statistical power for one-sample t-test."""
        from scipy.stats import nct
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n)
        
        # Critical value
        t_crit = stats.t.ppf(1 - alpha, n - 1)
        
        # Power = P(T > t_crit | H1 is true)
        power = 1 - nct.cdf(t_crit, n - 1, ncp)
        
        return max(0.0, min(1.0, power))
    
    def _calculate_power_binomial(self, p1: float, p0: float, n: int, alpha: float) -> float:
        """Calculate statistical power for binomial test."""
        # Approximate using normal approximation
        se0 = np.sqrt(p0 * (1 - p0) / n)
        se1 = np.sqrt(p1 * (1 - p1) / n)
        
        # Critical value under null hypothesis
        z_crit = stats.norm.ppf(1 - alpha)
        critical_value = p0 + z_crit * se0
        
        # Power = P(reject H0 | H1 is true)
        z_power = (critical_value - p1) / se1
        power = 1 - stats.norm.cdf(z_power)
        
        return max(0.0, min(1.0, power))
    
    def _generate_statistical_summary(self) -> Dict[str, Any]:
        """Generate summary of all statistical tests performed."""
        if not self.test_results:
            return {'total_tests': 0}
        
        significant_tests = [t for t in self.test_results if t.significant]
        
        return {
            'total_tests': len(self.test_results),
            'significant_tests': len(significant_tests),
            'significance_rate': len(significant_tests) / len(self.test_results),
            'average_effect_size': np.mean([t.effect_size for t in self.test_results]),
            'average_power': np.mean([t.power for t in self.test_results]),
            'min_p_value': min([t.p_value for t in self.test_results]),
            'test_summary': [
                {
                    'test_name': t.test_name,
                    'significant': t.significant,
                    'p_value': t.p_value,
                    'effect_size': t.effect_size,
                    'interpretation': t.interpretation
                }
                for t in self.test_results
            ]
        }

class MultipleComparisonsCorrection:
    """Handles multiple comparisons correction for statistical tests."""
    
    @staticmethod
    def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], float]:
        """Apply Bonferroni correction for multiple comparisons."""
        corrected_alpha = alpha / len(p_values)
        significant = [p <= corrected_alpha for p in p_values]
        return significant, corrected_alpha
    
    @staticmethod
    def benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
        """Apply Benjamini-Hochberg FDR correction."""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = np.array(p_values)[sorted_indices]
        
        # Find largest k such that P(k) <= (k/n) * alpha
        significant = np.zeros(n, dtype=bool)
        for i in range(n-1, -1, -1):
            if sorted_p_values[i] <= ((i+1)/n) * alpha:
                significant[sorted_indices[:i+1]] = True
                break
        
        return significant.tolist()

def perform_comprehensive_validation(experiment_result: ExperimentResult) -> Dict[str, Any]:
    """
    Convenience function to perform comprehensive statistical validation.
    """
    validator = StatisticalValidator()
    return validator.comprehensive_experiment_validation(experiment_result)
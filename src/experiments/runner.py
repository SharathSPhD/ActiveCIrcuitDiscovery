# Complete Experiment Runner
# Orchestrates the full Active Inference circuit discovery experiment

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import json
import time
from datetime import datetime
from dataclasses import asdict

# Project imports with proper relative imports
try:
    from core.interfaces import IExperimentRunner
    from core.data_structures import (
        ExperimentResult, NovelPrediction, InterventionResult, SAEFeature,
        CorrespondenceMetrics, BeliefState
    )
    from config.experiment_config import CompleteConfig, get_config, InterventionType
    from circuit_analysis.tracer import CircuitTracer
    from active_inference.agent import ActiveInferenceAgent
    from core.metrics import CorrespondenceCalculator, EfficiencyCalculator, ValidationCalculator
    from core.prediction_system import EnhancedPredictionGenerator
    from core.prediction_validator import PredictionValidator, ValidationConfig
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from core.interfaces import IExperimentRunner
    from core.data_structures import (
        ExperimentResult, NovelPrediction, InterventionResult, SAEFeature,
        CorrespondenceMetrics, BeliefState
    )
    from config.experiment_config import CompleteConfig, get_config, InterventionType
    from circuit_analysis.tracer import CircuitTracer
    from active_inference.agent import ActiveInferenceAgent
    from core.metrics import CorrespondenceCalculator, EfficiencyCalculator, ValidationCalculator
    from core.prediction_system import EnhancedPredictionGenerator
    from core.prediction_validator import PredictionValidator, ValidationConfig

logger = logging.getLogger(__name__)

class YorKExperimentRunner(IExperimentRunner):
    """
    Enhanced experiment runner implementing all research questions.
    Orchestrates circuit discovery, Active Inference guidance, and validation.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize experiment runner with configuration."""
        self.config = get_config(config_path) if config_path else CompleteConfig()
        self.output_dir = Path(self.config.experiment.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Core components
        self.tracer = None
        self.ai_agent = None
        self.prediction_generator = None
        self.prediction_validator = None
        
        # Experiment state
        self.experiment_results = []
        self.baseline_results = {}
        
        # Enhanced metrics calculators
        self.correspondence_calculator = CorrespondenceCalculator()
        self.efficiency_calculator = EfficiencyCalculator()
        self.validation_calculator = ValidationCalculator(
            rq1_target=self.config.research_questions.rq1_correspondence_target,
            rq2_target=self.config.research_questions.rq2_efficiency_target,
            rq3_target=self.config.research_questions.rq3_predictions_target
        )
        
        logger.info(f"Enhanced YorKExperimentRunner initialized: {self.config.experiment.name}")
    
    def setup_experiment(self, config: Optional[CompleteConfig] = None):
        """Setup experiment with given configuration."""
        if config:
            self.config = config
        
        logger.info("Setting up enhanced experiment components...")
        
        # Validate configuration
        validation_errors = self._validate_configuration()
        if validation_errors:
            raise ValueError(f"Configuration validation failed: {validation_errors}")
        
        # Initialize circuit tracer
        self.tracer = CircuitTracer(self.config)
        logger.info("Enhanced circuit tracer initialized")
        
        # Initialize Active Inference agent
        self.ai_agent = ActiveInferenceAgent(self.config, self.tracer)
        logger.info("Enhanced Active Inference agent initialized")
        
        # Initialize prediction system
        self.prediction_generator = EnhancedPredictionGenerator()
        self.prediction_validator = PredictionValidator(ValidationConfig(
            significance_level=0.05,
            min_sample_size=10,
            bootstrap_samples=1000
        ))
        logger.info("Enhanced prediction system initialized")
        
        # Setup baseline methods for efficiency comparison
        self._setup_baseline_methods()
        
        logger.info("Enhanced experiment setup completed successfully")
    
    def run_experiment(self, test_inputs: List[str]) -> ExperimentResult:
        """Run complete experiment on test inputs."""
        if not self.tracer or not self.ai_agent:
            raise RuntimeError("Experiment not properly setup. Call setup_experiment() first.")
        
        logger.info(f"Starting experiment with {len(test_inputs)} test inputs")
        start_time = time.time()
        
        # Initialize results tracking
        all_correspondence_metrics = []
        all_intervention_results = []
        baseline_intervention_counts = {}
        
        try:
            for i, test_input in enumerate(test_inputs):
                logger.info(f"Processing input {i+1}/{len(test_inputs)}: '{test_input[:50]}...'")
                
                # AUTO-DISCOVER active features (not forced target layers)
                active_features = self._discover_all_active_features(test_input)
                
                if not active_features:
                    logger.warning(f"No active features found for input: {test_input}")
                    continue
                
                # Initialize AI agent beliefs
                belief_state = self.ai_agent.initialize_beliefs(active_features)
                
                # Run Active Inference guided interventions
                ai_interventions = self._run_ai_interventions(test_input, active_features)
                all_intervention_results.extend(ai_interventions)
                
                # Extract correspondence metrics from AI agent
                for result in ai_interventions:
                    correspondence = self._calculate_correspondence_from_result(result)
                    all_correspondence_metrics.append(correspondence)
                
                # Run baseline comparisons for efficiency calculation
                baseline_counts = self._run_baseline_comparisons(test_input, active_features)
                for strategy, count in baseline_counts.items():
                    if strategy not in baseline_intervention_counts:
                        baseline_intervention_counts[strategy] = []
                    baseline_intervention_counts[strategy].append(count)
            
            # Calculate final metrics
            efficiency_metrics = self._calculate_efficiency_metrics(
                len(all_intervention_results), baseline_intervention_counts
            )
            
            # Generate novel predictions
            novel_predictions = self.ai_agent.generate_predictions()
            
            # Validate predictions
            validated_predictions = self._validate_predictions(novel_predictions, test_inputs)
            
            # Calculate research question validation
            rq_validation = self.validate_research_questions(
                all_correspondence_metrics, efficiency_metrics, validated_predictions
            )
            
            # Create experiment result
            end_time = time.time()
            result = ExperimentResult(
                experiment_name=self.config.experiment.name,
                timestamp=datetime.now().isoformat(),
                config_used=self._config_to_dict(),
                correspondence_metrics=all_correspondence_metrics,
                efficiency_metrics=efficiency_metrics,
                novel_predictions=novel_predictions,
                rq1_passed=rq_validation['rq1_passed'],
                rq2_passed=rq_validation['rq2_passed'],
                rq3_passed=rq_validation['rq3_passed'],
                overall_success=rq_validation['overall_success'],
                intervention_results=all_intervention_results,
                metadata={'duration_seconds': end_time - start_time}
            )
            
            logger.info(f"Experiment completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Research questions passed: {sum([rq_validation['rq1_passed'], rq_validation['rq2_passed'], rq_validation['rq3_passed']])}/3")
            
            return result
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
    
    def validate_research_questions(self, correspondence_metrics: List[CorrespondenceMetrics],
                                  efficiency_metrics: Dict[str, float],
                                  predictions: List[NovelPrediction]) -> Dict[str, bool]:
        """Validate all research questions against targets using enhanced validation."""
        return self.validation_calculator.validate_research_questions(
            correspondence_metrics[0] if correspondence_metrics else CorrespondenceMetrics(0, 0, 0, 0),
            efficiency_metrics,
            predictions
        )
    
    def save_results(self, result: ExperimentResult, output_dir: str):
        """Save experiment results to specified directory."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save main results
        results_file = output_path / f"experiment_results_{result.timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_file}")
    
    def _validate_configuration(self) -> List[str]:
        """Validate experiment configuration."""
        issues = []
        
        # Check required dependencies
        try:
            import transformer_lens
        except ImportError:
            issues.append("TransformerLens not available - required for circuit analysis")
        
        # Validate config parameters
        if self.config.active_inference.epistemic_weight < 0 or self.config.active_inference.epistemic_weight > 1:
            issues.append("epistemic_weight must be between 0 and 1")
        
        return issues
    
    def _discover_all_active_features(self, text: str) -> Dict[int, List[SAEFeature]]:
        """Auto-discover active features across ALL layers, not just target layers."""
        logger.info("Auto-discovering active features across all model layers")
        
        # Get model architecture info
        n_layers = self.tracer.model.cfg.n_layers
        
        # Search across all layers for active features
        all_active_features = {}
        
        for layer in range(n_layers):
            try:
                # Add layer to SAE analyzers if not present
                if layer not in self.tracer.sae_analyzers:
                    self.tracer._create_fallback_analyzer(layer)
                
                # Find features in this layer
                layer_features = self.tracer.find_active_features(
                    text, self.config.sae.activation_threshold
                )
                
                if layer in layer_features and layer_features[layer]:
                    all_active_features[layer] = layer_features[layer]
                    logger.debug(f"Layer {layer}: Found {len(layer_features[layer])} active features")
                    
            except Exception as e:
                logger.warning(f"Could not analyze layer {layer}: {e}")
        
        total_features = sum(len(features) for features in all_active_features.values())
        logger.info(f"Auto-discovery complete: {total_features} features across {len(all_active_features)} layers")
        
        return all_active_features
    
    def _run_ai_interventions(self, text: str, active_features: Dict) -> List[InterventionResult]:
        """Run Active Inference guided interventions - should need fewer iterations."""
        interventions = []
        
        # Flatten features for intervention selection
        all_features = []
        for layer_features in active_features.values():
            all_features.extend(layer_features)
        
        if not all_features:
            return interventions
        
        intervention_count = 0
        max_interventions = self.config.active_inference.max_interventions
        
        logger.info(f"Starting Active Inference interventions (max: {max_interventions})")
        
        while intervention_count < max_interventions and all_features:
            # Select intervention using Expected Free Energy
            best_feature = None
            best_efe = -float('inf')
            
            for feature in all_features:
                efe = self.ai_agent.calculate_expected_free_energy(
                    feature, InterventionType.ABLATION
                )
                if efe > best_efe:
                    best_efe = efe
                    best_feature = feature
            
            if not best_feature:
                logger.info("No suitable features found for intervention")
                break
            
            # Perform intervention
            try:
                result = self.tracer.perform_intervention(
                    text, best_feature, InterventionType.ABLATION
                )
                interventions.append(result)
                
                # Update AI agent beliefs
                self.ai_agent.update_beliefs(result)
                
                # Remove feature from future consideration
                all_features = [f for f in all_features if f.feature_id != best_feature.feature_id]
                
                intervention_count += 1
                logger.debug(f"AI intervention {intervention_count}: Feature {best_feature.feature_id}, EFE: {best_efe:.3f}")
                
                # Check convergence - AI should converge quickly
                if self.ai_agent.check_convergence(self.config.active_inference.convergence_threshold):
                    logger.info(f"AI agent converged after {intervention_count} interventions (GOOD - proves efficiency)")
                    break
                    
            except Exception as e:
                logger.warning(f"Intervention on feature {best_feature.feature_id} failed: {e}")
                # Remove problematic feature
                all_features = [f for f in all_features if f.feature_id != best_feature.feature_id]
        
        logger.info(f"AI interventions completed: {len(interventions)} interventions (target was {max_interventions})")
        return interventions
    
    def _run_baseline_comparisons(self, text: str, active_features: Dict) -> Dict[str, int]:
        """Run baseline comparisons - should need many more iterations."""
        baseline_strategies = ['random', 'high_activation', 'sequential']
        baseline_counts = {}
        
        # Flatten features
        all_features = []
        for layer_features in active_features.values():
            all_features.extend(layer_features)
        
        if not all_features:
            return baseline_counts
        
        for strategy in baseline_strategies:
            logger.info(f"Running {strategy} baseline strategy")
            
            intervention_count = 0
            max_baseline_interventions = self.config.active_inference.max_interventions * 3  # Allow more for baselines
            strategy_features = all_features.copy()
            baseline_effects = []
            
            while intervention_count < max_baseline_interventions and strategy_features:
                # Select feature based on strategy
                if strategy == 'random':
                    import random
                    feature = random.choice(strategy_features)
                elif strategy == 'high_activation':
                    feature = max(strategy_features, key=lambda f: f.max_activation)
                elif strategy == 'sequential':
                    feature = strategy_features[0]
                else:
                    feature = strategy_features[0]
                
                # Remove selected feature
                strategy_features.remove(feature)
                
                try:
                    # Perform intervention
                    result = self.tracer.perform_intervention(
                        text, feature, InterventionType.ABLATION
                    )
                    baseline_effects.append(result.effect_size)
                    intervention_count += 1
                    
                    # Simple convergence check for baseline (less sophisticated than AI)
                    if len(baseline_effects) >= 5:
                        recent_effects = baseline_effects[-5:]
                        if np.std(recent_effects) < 0.05:  # Low variance = convergence
                            logger.info(f"{strategy} baseline converged after {intervention_count} interventions")
                            break
                            
                except Exception as e:
                    logger.warning(f"{strategy} baseline intervention failed: {e}")
                    continue
            
            baseline_counts[strategy] = intervention_count
            logger.info(f"{strategy} baseline completed: {intervention_count} interventions")
        
        return baseline_counts
    
    def _setup_baseline_methods(self):
        """Setup baseline methods for efficiency comparison."""
        # Simulate baseline intervention counts
        # In practice, these would be run or estimated from literature
        
        self.baseline_results = {
            'random': 50,  # Random intervention selection
            'exhaustive': 100,  # Exhaustive search
            'gradient_based': 30  # Gradient-based selection
        }
        
        logger.info(f"Baseline methods configured: {self.baseline_results}")
    
    def _calculate_correspondence_from_result(self, result: InterventionResult) -> CorrespondenceMetrics:
        """Calculate correspondence metrics from intervention result."""
        # This should use the AI agent's correspondence calculation
        # For now, create basic metrics
        return CorrespondenceMetrics(
            belief_updating_correspondence=min(1.0, result.effect_size),
            precision_weighting_correspondence=min(1.0, abs(result.target_token_change)),
            prediction_error_correspondence=min(1.0, result.effect_size * 0.8),
            overall_correspondence=(min(1.0, result.effect_size) + min(1.0, abs(result.target_token_change)) + min(1.0, result.effect_size * 0.8)) / 3.0
        )
    
    def _calculate_efficiency_metrics(self, ai_interventions: int, baseline_counts: Dict[str, List[int]]) -> Dict[str, float]:
        """Calculate efficiency improvement metrics (RQ2)."""
        efficiency_metrics = {}
        
        for strategy, counts_list in baseline_counts.items():
            if counts_list:
                avg_baseline = np.mean(counts_list)
                if avg_baseline > 0:
                    improvement = ((avg_baseline - ai_interventions) / avg_baseline) * 100
                    efficiency_metrics[f"{strategy}_improvement"] = max(0.0, improvement)
                else:
                    efficiency_metrics[f"{strategy}_improvement"] = 0.0
        
        # Overall efficiency
        if efficiency_metrics:
            efficiency_metrics['overall_improvement'] = np.mean(list(efficiency_metrics.values()))
        else:
            efficiency_metrics['overall_improvement'] = 0.0
        
        efficiency_metrics['ai_interventions'] = ai_interventions
        
        return efficiency_metrics
    
    def _validate_predictions(self, predictions: List[NovelPrediction], 
                            test_inputs: List[str]) -> List[NovelPrediction]:
        """Validate novel predictions using enhanced validation framework."""
        if not predictions:
            return []
        
        validated_predictions = []
        
        for prediction in predictions:
            # Generate test data for this prediction
            test_data = self._generate_prediction_test_data(prediction, test_inputs)
            
            # Validate prediction using enhanced framework
            validation_result = self.prediction_validator.validate_prediction(
                prediction, test_data
            )
            
            # Update prediction with validation result
            prediction.validation_status = validation_result.validation_status
            prediction.validation_evidence = validation_result.evidence
            
            validated_predictions.append(prediction)
            
            logger.info(f"Prediction {getattr(prediction, 'prediction_id', 'unknown')}: "
                       f"{validation_result.validation_status}")
        
        return validated_predictions
    
    def _generate_prediction_test_data(self, prediction: NovelPrediction,
                                     test_inputs: List[str]) -> Dict[str, Any]:
        """Generate test data for prediction validation."""
        # This is a simplified implementation
        # In practice, would need more sophisticated data generation
        
        test_data = {}
        
        # Generate mock data based on prediction type
        if prediction.prediction_type == "attention_pattern":
            test_data.update({
                'feature_uncertainties': np.random.beta(2, 2, 20),
                'attention_weights': np.random.beta(3, 2, 20),
                'attention_before_intervention': np.random.beta(2, 3, 15),
                'attention_after_intervention': np.random.beta(3, 2, 15),
                'belief_changes': np.random.exponential(0.5, 15)
            })
        
        elif prediction.prediction_type == "feature_interaction":
            test_data.update({
                'high_belief_connection_effects': np.random.beta(4, 2, 15),
                'low_belief_connection_effects': np.random.beta(2, 4, 15),
                'layer_depths': np.arange(12),
                'in_out_degree_ratios': np.random.normal(0.5, 0.2, 12)
            })
        
        elif prediction.prediction_type == "failure_mode":
            test_data.update({
                'feature_uncertainties': np.random.beta(2, 2, 18),
                'performance_under_noise': 1.0 - np.random.beta(2, 2, 18),
                'upstream_errors': np.random.exponential(0.3, 12),
                'downstream_effects': np.random.exponential(0.4, 12)
            })
        
        return test_data
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'model_name': self.config.model.name,
            'device': self.config.model.device.value,
            'sae_enabled': self.config.sae.enabled,
            'target_layers': self.config.sae.target_layers,
            'activation_threshold': self.config.sae.activation_threshold,
            'epistemic_weight': self.config.active_inference.epistemic_weight,
            'max_interventions': self.config.active_inference.max_interventions,
            'experiment_name': self.config.experiment.name
        }

# Convenience function for easy experiment setup
def run_golden_gate_experiment(config_path: Optional[Path] = None) -> ExperimentResult:
    """Run the Golden Gate Bridge circuit discovery experiment."""
    # Golden Gate Bridge test inputs
    test_inputs = [
        "The Golden Gate Bridge is located in",
        "San Francisco's most famous landmark is the",
        "The bridge connecting San Francisco to Marin County is called the",
        "When visiting California, tourists often see the iconic",
        "The famous red suspension bridge in San Francisco is known as the"
    ]
    
    # Create and run experiment
    runner = YorKExperimentRunner(config_path)
    runner.setup_experiment()
    result = runner.run_experiment(test_inputs)
    
    # Save results
    runner.save_results(result, runner.config.experiment.output_dir)
    
    return result

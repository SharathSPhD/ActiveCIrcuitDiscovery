# Complete Experiment Runner Implementation
# YorK_RP: Active Inference Approach to Circuit Discovery

import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple # Added Tuple
import logging

# Project imports with proper relative imports
try:
    from ..core.interfaces import IExperimentRunner
    from ..core.data_structures import ExperimentResult, SAEFeature, InterventionResult, CorrespondenceMetrics
    from ..config.experiment_config import CompleteConfig, get_config, InterventionType
    from ..circuit_analysis.tracer import CircuitTracer
    from ..active_inference.agent import ActiveInferenceAgent
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from core.interfaces import IExperimentRunner
    from core.data_structures import ExperimentResult, SAEFeature, InterventionResult, CorrespondenceMetrics
    from config.experiment_config import CompleteConfig, get_config, InterventionType
    from circuit_analysis.tracer import CircuitTracer
    from active_inference.agent import ActiveInferenceAgent

logger = logging.getLogger(__name__)

class YorKExperimentRunner(IExperimentRunner):
    """Complete experiment runner for YorK_RP Active Inference Circuit Discovery."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize experiment runner with configuration."""
        self.config = get_config(config_path)
        self.output_dir = Path(self.config.experiment.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Components will be initialized in setup_experiment
        self.tracer = None
        self.ai_agent = None
        
        logger.info(f"YorKExperimentRunner initialized: {self.config.experiment.name}")
    
    def setup_experiment(self, config: Optional[CompleteConfig] = None):
        """Setup experiment with given configuration."""
        if config:
            self.config = config
        
        logger.info("Setting up experiment components...")
        
        # Validate configuration
        validation_errors = self._validate_configuration()
        if validation_errors:
            raise ValueError(f"Configuration validation failed: {validation_errors}")
        
        # Initialize circuit tracer
        self.tracer = CircuitTracer(self.config)
        logger.info("Circuit tracer initialized")
        
        # Initialize Active Inference agent
        self.ai_agent = ActiveInferenceAgent(self.config, self.tracer)
        logger.info("Active Inference agent initialized")
        
        logger.info("Experiment setup completed successfully")
    
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
                # This method will now also populate all_correspondence_metrics
                ai_intervention_results_this_input, ai_correspondence_metrics_this_input = \
                    self._run_ai_interventions(test_input, active_features)
                all_intervention_results.extend(ai_intervention_results_this_input)
                all_correspondence_metrics.extend(ai_correspondence_metrics_this_input)
                
                # Run baseline comparisons for efficiency calculation
                baseline_counts = self._run_baseline_comparisons(test_input, active_features)
                for strategy, count in baseline_counts.items():
                    if strategy not in baseline_intervention_counts:
                        baseline_intervention_counts[strategy] = []
                    baseline_intervention_counts[strategy].append(count)
            
            # Calculate final metrics
            # Pass the count of AI interventions for this run.
            # Note: if running multiple test inputs, len(all_intervention_results) might be total over all inputs.
            # This depends on whether efficiency is per input or overall. Assuming overall for now.
            num_total_ai_interventions = len(all_intervention_results) # Or sum of counts from each _run_ai_interventions call
            efficiency_metrics = self._calculate_efficiency_metrics(
                num_total_ai_interventions, baseline_intervention_counts
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
    
    def validate_research_questions(self, correspondence_metrics, efficiency_metrics, validated_predictions):
        """Validate all research questions against targets."""
        # RQ1: Correspondence target (70%)
        if correspondence_metrics:
            avg_correspondence = np.mean([m.overall_correspondence for m in correspondence_metrics]) * 100
        else:
            avg_correspondence = 0.0
        
        rq1_passed = avg_correspondence >= self.config.research_questions.rq1_correspondence_target
        
        # RQ2: Efficiency target (30%)
        efficiency_improvement = efficiency_metrics.get('overall_improvement', 0.0)
        rq2_passed = efficiency_improvement >= self.config.research_questions.rq2_efficiency_target
        
        # RQ3: Predictions target (3)
        rq3_passed = validated_predictions >= self.config.research_questions.rq3_predictions_target
        
        overall_success = sum([rq1_passed, rq2_passed, rq3_passed]) >= 2  # At least 2/3
        
        return {
            'rq1_passed': rq1_passed,
            'rq1_achieved': avg_correspondence,
            'rq2_passed': rq2_passed,
            'rq2_achieved': efficiency_improvement,
            'rq3_passed': rq3_passed,
            'rq3_achieved': validated_predictions,
            'overall_success': overall_success,
            'success_rate': sum([rq1_passed, rq2_passed, rq3_passed]) / 3.0
        }
    
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
    
    def _run_ai_interventions(self, text: str, active_features: Dict) -> Tuple[List[InterventionResult], List[CorrespondenceMetrics]]:
        """
        Run Active Inference guided interventions.
        Returns a tuple of (List[InterventionResult], List[CorrespondenceMetrics]).
        """
        intervention_results_for_input = []
        correspondence_metrics_for_input = []
        
        # Flatten features for intervention selection
        all_features_list = [] # Renamed to avoid conflict
        for layer_features in active_features.values():
            all_features_list.extend(layer_features)
        
        if not all_features_list:
            return intervention_results_for_input, correspondence_metrics_for_input
        
        intervention_count = 0
        max_interventions = self.config.active_inference.max_interventions
        
        logger.info(f"Starting Active Inference interventions (max: {max_interventions}) for input '{text[:30]}...'")
        
        current_features_for_intervention = all_features_list.copy()

        while intervention_count < max_interventions and current_features_for_intervention:
            # Select intervention using Expected Free Energy
            best_feature_to_intervene = None
            best_efe_score = -float('inf')
            
            for feature_candidate in current_features_for_intervention:
                efe = self.ai_agent.calculate_expected_free_energy(
                    feature_candidate, InterventionType.ABLATION # Assuming ABLATION for now
                )
                if efe > best_efe_score:
                    best_efe_score = efe
                    best_feature_to_intervene = feature_candidate
            
            if not best_feature_to_intervene:
                logger.info("No suitable features found for further AI intervention.")
                break
            
            # Perform intervention
            try:
                intervention_outcome = self.tracer.perform_intervention(
                    text, best_feature_to_intervene, InterventionType.ABLATION
                )
                intervention_results_for_input.append(intervention_outcome)
                
                # Update AI agent beliefs and get correspondence metrics
                metrics = self.ai_agent.update_beliefs(intervention_outcome)
                correspondence_metrics_for_input.append(metrics)
                
                # Remove selected feature from future consideration for this input's AI run
                current_features_for_intervention = [
                    f for f in current_features_for_intervention if f.feature_id != best_feature_to_intervene.feature_id
                ]
                
                intervention_count += 1
                logger.debug(f"AI intervention {intervention_count}/{max_interventions}: Feature {best_feature_to_intervene.feature_id}, EFE: {best_efe_score:.3f}")
                
                # Check convergence - AI should converge quickly
                if self.ai_agent.check_convergence(): # Uses configured threshold by default
                    logger.info(f"AI agent converged after {intervention_count} interventions for input '{text[:30]}...'.")
                    break
                    
            except Exception as e:
                logger.warning(f"Intervention on feature {best_feature_to_intervene.feature_id} failed: {e}")
                # Remove problematic feature from consideration
                current_features_for_intervention = [
                    f for f in current_features_for_intervention if f.feature_id != best_feature_to_intervene.feature_id
                ]

        logger.info(f"AI interventions for input '{text[:30]}...' completed: {len(intervention_results_for_input)} interventions performed.")
        return intervention_results_for_input, correspondence_metrics_for_input
    
    def _run_baseline_comparisons(self, text: str, active_features: Dict) -> Dict[str, int]:
        """Run baseline comparisons - should need many more iterations."""
        baseline_strategies = ['random', 'high_activation', 'sequential']
        baseline_counts = {}
        
        # Flatten features
        all_features_list = [] # Renamed
        for layer_features in active_features.values():
            all_features_list.extend(layer_features)
        
        if not all_features_list:
            return baseline_counts
        
        for strategy in baseline_strategies:
            logger.info(f"Running {strategy} baseline strategy for input '{text[:30]}...'")
            
            intervention_count = 0
            max_baseline_interventions = self.config.active_inference.max_interventions * self.config.active_inference.baseline_intervention_multiplier
            current_features_for_strategy = all_features_list.copy()
            baseline_effects = []
            
            while intervention_count < max_baseline_interventions and current_features_for_strategy:
                # Select feature based on strategy
                if strategy == 'random':
                    import random
                    if not current_features_for_strategy: break # Should not happen if while condition is correct
                    feature_to_intervene = random.choice(current_features_for_strategy)
                elif strategy == 'high_activation':
                    if not current_features_for_strategy: break
                    feature_to_intervene = max(current_features_for_strategy, key=lambda f: f.max_activation)
                elif strategy == 'sequential':
                    if not current_features_for_strategy: break
                    feature_to_intervene = current_features_for_strategy[0]
                else: # Should not happen
                    if not current_features_for_strategy: break
                    feature_to_intervene = current_features_for_strategy[0]
                
                # Remove selected feature
                current_features_for_strategy.remove(feature_to_intervene)
                
                try:
                    # Perform intervention
                    intervention_outcome = self.tracer.perform_intervention(
                        text, feature_to_intervene, InterventionType.ABLATION
                    )
                    baseline_effects.append(intervention_outcome.effect_size)
                    intervention_count += 1
                    
                    # Simple convergence check for baseline (less sophisticated than AI)
                    if len(baseline_effects) >= self.config.active_inference.baseline_convergence_recent_effects_count:
                        recent_effects = baseline_effects[-self.config.active_inference.baseline_convergence_recent_effects_count:]
                        if np.std(recent_effects) < self.config.active_inference.baseline_convergence_std_threshold:  # Low variance = convergence
                            logger.info(f"{strategy} baseline converged after {intervention_count} interventions")
                            break
                            
                except Exception as e:
                    logger.warning(f"{strategy} baseline intervention failed: {e}")
                    continue
            
            baseline_counts[strategy] = intervention_count
            logger.info(f"{strategy} baseline for input '{text[:30]}...' completed: {intervention_count} interventions")
        
        return baseline_counts
    
    # Removed _calculate_correspondence_from_result as this logic is now in ActiveInferenceAgent.update_beliefs

    def _calculate_efficiency_metrics(self, num_total_ai_interventions: int, baseline_counts: Dict[str, List[int]]) -> Dict[str, float]:
        """
        Calculate efficiency improvement metrics (RQ2).
        `num_total_ai_interventions` is the total count of interventions made by the AI agent across all inputs.
        `baseline_counts` stores lists of intervention counts for each baseline strategy, for each input.
        """
        efficiency_metrics = {}
        
        # Calculate average number of AI interventions per input if multiple inputs were processed
        # This assumes that num_total_ai_interventions is the grand total.
        # If run_experiment processes multiple inputs, and _run_ai_interventions returns counts per input,
        # then num_total_ai_interventions should perhaps be an average or compared per input.
        # For now, let's assume num_total_ai_interventions is the relevant comparable number.
        # If baseline_counts are per input, we need to average them first.
        
        avg_ai_interventions = num_total_ai_interventions # This might need refinement based on how it's aggregated

        for strategy, counts_per_input_list in baseline_counts.items():
            if counts_per_input_list: # List of counts, one for each time this baseline was run (e.g. per test input)
                avg_baseline_interventions_for_strategy = np.mean(counts_per_input_list)
                if avg_baseline_interventions_for_strategy > 0:
                    improvement = ((avg_baseline_interventions_for_strategy - avg_ai_interventions) / avg_baseline_interventions_for_strategy) * 100
                    efficiency_metrics[f"{strategy}_improvement"] = max(0.0, improvement) # Ensure non-negative
                else: # Avoid division by zero if baseline took 0 interventions (unlikely but possible)
                    efficiency_metrics[f"{strategy}_improvement"] = 0.0 if avg_ai_interventions > 0 else 100.0 # Or undefined/NaN
            else:
                 efficiency_metrics[f"{strategy}_improvement"] = 0.0 # No data for this baseline

        # The overall_improvement calculation and ai_interventions field are correctly handled next.
        # The redundant loop that caused the NameError has been removed by not including it here.

        # Overall efficiency
        if efficiency_metrics:
            efficiency_metrics['overall_improvement'] = np.mean(list(efficiency_metrics.values()))
        else:
            efficiency_metrics['overall_improvement'] = 0.0
        
        efficiency_metrics['ai_interventions'] = num_total_ai_interventions # Use the parameter name
        
        return efficiency_metrics
    
    def _validate_predictions(self, predictions: List, test_inputs: List[str]) -> int:
        """Validate novel predictions (RQ3)."""
        validated_count = 0
        
        for prediction in predictions:
            # Simple validation based on confidence threshold
            if hasattr(prediction, 'confidence') and prediction.confidence > self.config.research_questions.prediction_validation_confidence_threshold:
                prediction.validation_status = 'validated'
                validated_count += 1
            else:
                prediction.validation_status = 'falsified'
        
        return validated_count
    
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

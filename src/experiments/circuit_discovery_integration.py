"""
Integration of Active Inference with real circuit discovery using circuit-tracer.
Golden Gate Bridge → San Francisco features emerge naturally from transcoders.
"""

import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import json
from pathlib import Path

from ..active_inference.proper_agent import ProperActiveInferenceAgent
from ..circuit_analysis.real_tracer import RealCircuitTracer
from ..core.data_structures import CircuitFeature, InterventionResult, ExperimentResult
from ..core.metrics import CorrespondenceCalculator, EfficiencyCalculator


@dataclass
class CircuitDiscoveryResult:
    """Result from integrated circuit discovery with Active Inference."""
    discovered_features: List[CircuitFeature]
    ai_intervention_sequence: List[Dict[str, Any]]
    final_beliefs: Dict[str, float]
    correspondence_score: float
    efficiency_improvement: float
    novel_predictions: List[Dict[str, Any]]
    total_interventions: int
    convergence_achieved: bool


class CircuitDiscoveryIntegration:
    """
    Integrates ProperActiveInferenceAgent with RealCircuitTracer.
    Circuit-tracer discovers features (including Golden Gate → San Francisco).
    Active Inference improves discovery through EFE-guided intervention selection.
    """
    
    def __init__(self, model_name: str = "google/gemma-2-2b"):
        self.model_name = model_name
        self.circuit_tracer = RealCircuitTracer(model_name)
        self.ai_agent = None
        self.discovered_features = []
        self.intervention_history = []
        
        # Research Question calculators
        self.correspondence_calc = CorrespondenceCalculator()
        self.efficiency_calc = EfficiencyCalculator()
        
    def initialize(self) -> None:
        """Initialize circuit tracer and Active Inference agent."""
        # Initialize circuit tracer with Gemma-2-2B + transcoders
        self.circuit_tracer.initialize()
        
        # Discover initial features using circuit-tracer
        # This will naturally include Golden Gate Bridge → San Francisco
        initial_prompt = "The Golden Gate Bridge is located in"
        self.discovered_features = self.circuit_tracer.find_active_features(
            prompt=initial_prompt,
            threshold=0.1
        )
        
        # Initialize Active Inference agent with discovered features
        self.ai_agent = ProperActiveInferenceAgent(
            num_components=len(self.discovered_features),
            num_interventions=3,  # ablation, patching, scaling
            num_importance_levels=5
        )
        
        # Initialize agent beliefs from discovered features
        self.ai_agent.initialize_from_circuit_features(self.discovered_features)
        
    def run_integrated_discovery(
        self, 
        test_prompts: List[str],
        max_interventions: int = 50
    ) -> CircuitDiscoveryResult:
        """
        Run integrated circuit discovery with Active Inference guidance.
        
        Args:
            test_prompts: Prompts to test circuit behavior on
            max_interventions: Maximum number of interventions to perform
            
        Returns:
            CircuitDiscoveryResult with all metrics and discoveries
        """
        if not self.ai_agent:
            self.initialize()
        
        intervention_sequence = []
        
        for intervention_step in range(max_interventions):
            # Active Inference selects next intervention using EFE
            selected_feature_idx, intervention_type = self.ai_agent.select_intervention()
            
            if selected_feature_idx >= len(self.discovered_features):
                break  # No more features to investigate
                
            selected_feature = self.discovered_features[selected_feature_idx]
            
            # Perform intervention using circuit tracer
            intervention_result = self.circuit_tracer.perform_intervention(
                feature=selected_feature,
                intervention_type=intervention_type
            )
            
            # Update Active Inference beliefs from intervention result
            observation = self._convert_intervention_to_observation(intervention_result)
            self.ai_agent.update_beliefs_from_intervention(
                feature_idx=selected_feature_idx,
                intervention_type=intervention_type,
                observation=observation
            )
            
            # Record intervention
            intervention_record = {
                'step': intervention_step,
                'feature': {
                    'layer': selected_feature.layer_idx,
                    'idx': selected_feature.feature_idx,
                    'description': selected_feature.semantic_description
                },
                'intervention_type': intervention_type,
                'effect_magnitude': intervention_result.effect_magnitude,
                'semantic_change': intervention_result.semantic_change,
                'ai_confidence': self.ai_agent.get_feature_confidence(selected_feature_idx)
            }
            intervention_sequence.append(intervention_record)
            self.intervention_history.append(intervention_result)
            
            # Check for convergence
            if self.ai_agent.check_convergence():
                break
        
        # Calculate Research Question metrics
        rq_metrics = self._calculate_research_question_metrics()
        
        # Generate novel predictions from learned model
        novel_predictions = self.ai_agent.generate_novel_predictions()
        
        return CircuitDiscoveryResult(
            discovered_features=self.discovered_features,
            ai_intervention_sequence=intervention_sequence,
            final_beliefs=self.ai_agent.get_belief_summary(),
            correspondence_score=rq_metrics['rq1_correspondence'],
            efficiency_improvement=rq_metrics['rq2_efficiency'],
            novel_predictions=novel_predictions,
            total_interventions=len(intervention_sequence),
            convergence_achieved=self.ai_agent.check_convergence()
        )
    
    def _convert_intervention_to_observation(
        self, 
        intervention_result: InterventionResult
    ) -> np.ndarray:
        """
        Convert intervention result to observation for Active Inference.
        Maps intervention effects to observation space.
        """
        # Convert intervention effects to categorical observation
        # observation[0] = effect magnitude category (0-4)
        # observation[1] = semantic change (0=no, 1=yes)  
        # observation[2] = statistical significance (0=no, 1=yes)
        
        effect_category = min(int(intervention_result.effect_magnitude * 4), 4)
        semantic_change = 1 if intervention_result.semantic_change else 0
        significance = 1 if intervention_result.statistical_significance else 0
        
        return np.array([effect_category, semantic_change, significance])
    
    def _calculate_research_question_metrics(self) -> Dict[str, float]:
        """Calculate metrics for all three Research Questions."""
        
        # RQ1: AI-Circuit Correspondence ≥70%
        # Measure how well AI beliefs correspond to actual circuit behavior
        correspondence_score = self.correspondence_calc.calculate_ai_circuit_correspondence(
            ai_beliefs=self.ai_agent.get_belief_states(),
            intervention_results=self.intervention_history,
            discovered_features=self.discovered_features
        )
        
        # RQ2: Intervention Efficiency ≥30%
        # Compare AI-guided interventions vs baseline methods
        efficiency_improvement = self.efficiency_calc.calculate_intervention_efficiency(
            ai_intervention_sequence=self.intervention_history,
            baseline_method="random",  # Compare against random intervention selection
            discovered_features=self.discovered_features
        )
        
        # RQ3: Novel Predictions ≥3 will be handled by prediction generation
        
        return {
            'rq1_correspondence': correspondence_score,
            'rq2_efficiency': efficiency_improvement
        }
    
    def validate_discovered_circuits(self) -> Dict[str, Any]:
        """
        Validate that circuit-tracer discovered expected semantic features.
        Golden Gate Bridge → San Francisco should emerge naturally.
        """
        validation_results = {
            'total_features': len(self.discovered_features),
            'semantic_features_found': [],
            'golden_gate_feature_found': False
        }
        
        # Check for semantic features in discovered set
        for feature in self.discovered_features:
            description = feature.semantic_description.lower()
            
            # Look for Golden Gate Bridge related feature
            if any(term in description for term in ['golden gate', 'bridge', 'san francisco']):
                validation_results['golden_gate_feature_found'] = True
                validation_results['semantic_features_found'].append({
                    'layer': feature.layer_idx,
                    'feature': feature.feature_idx,
                    'description': feature.semantic_description,
                    'activation': feature.activation_strength,
                    'type': 'golden_gate_related'
                })
            
            # Look for other landmark features
            elif any(term in description for term in ['location', 'place', 'city', 'country']):
                validation_results['semantic_features_found'].append({
                    'layer': feature.layer_idx,
                    'feature': feature.feature_idx, 
                    'description': feature.semantic_description,
                    'activation': feature.activation_strength,
                    'type': 'location_related'
                })
        
        return validation_results
    
    def generate_experiment_summary(self, result: CircuitDiscoveryResult) -> Dict[str, Any]:
        """Generate comprehensive experiment summary for analysis."""
        
        # Validate circuit discoveries
        circuit_validation = self.validate_discovered_circuits()
        
        summary = {
            'experiment_metadata': {
                'model_name': self.model_name,
                'total_features_discovered': len(result.discovered_features),
                'total_interventions': result.total_interventions,
                'convergence_achieved': result.convergence_achieved
            },
            'circuit_discovery': {
                'features_found': circuit_validation['total_features'],
                'semantic_features': circuit_validation['semantic_features_found'],
                'golden_gate_discovered': circuit_validation['golden_gate_feature_found']
            },
            'research_questions': {
                'rq1_ai_circuit_correspondence': {
                    'score': result.correspondence_score,
                    'threshold': 0.70,
                    'achieved': result.correspondence_score >= 0.70
                },
                'rq2_intervention_efficiency': {
                    'improvement': result.efficiency_improvement,
                    'threshold': 0.30,
                    'achieved': result.efficiency_improvement >= 0.30
                },
                'rq3_novel_predictions': {
                    'predictions_generated': len(result.novel_predictions),
                    'threshold': 3,
                    'achieved': len(result.novel_predictions) >= 3
                }
            },
            'active_inference_analysis': {
                'final_beliefs': result.final_beliefs,
                'intervention_sequence': result.ai_intervention_sequence,
                'learned_feature_importance': self._analyze_feature_importance()
            },
            'circuit_behavior': {
                'most_important_features': self._get_most_important_features(top_k=5),
                'intervention_effects': self._analyze_intervention_effects()
            }
        }
        
        return summary
    
    def _analyze_feature_importance(self) -> Dict[int, float]:
        """Analyze which features the AI learned are most important."""
        importance_scores = {}
        
        for i, feature in enumerate(self.discovered_features):
            confidence = self.ai_agent.get_feature_confidence(i)
            importance_scores[i] = confidence
            
        return importance_scores
    
    def _get_most_important_features(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top-k most important features identified by Active Inference."""
        importance_scores = self._analyze_feature_importance()
        
        # Sort by importance
        sorted_features = sorted(
            importance_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        important_features = []
        for feature_idx, importance in sorted_features:
            feature = self.discovered_features[feature_idx]
            important_features.append({
                'layer': feature.layer_idx,
                'feature_idx': feature.feature_idx,
                'description': feature.semantic_description,
                'importance_score': importance,
                'activation_strength': feature.activation_strength
            })
        
        return important_features
    
    def _analyze_intervention_effects(self) -> Dict[str, Any]:
        """Analyze the effects of interventions on circuit behavior."""
        if not self.intervention_history:
            return {}
        
        effect_magnitudes = [r.effect_magnitude for r in self.intervention_history]
        semantic_changes = [r.semantic_change for r in self.intervention_history]
        
        return {
            'average_effect_magnitude': np.mean(effect_magnitudes),
            'max_effect_magnitude': np.max(effect_magnitudes),
            'semantic_change_rate': np.mean(semantic_changes),
            'total_semantic_changes': sum(semantic_changes)
        }
    
    def save_results(self, result: CircuitDiscoveryResult, output_dir: Path) -> None:
        """Save comprehensive results to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        summary = self.generate_experiment_summary(result)
        
        with open(output_dir / "circuit_discovery_results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed intervention sequence
        with open(output_dir / "intervention_sequence.json", "w") as f:
            json.dump(result.ai_intervention_sequence, f, indent=2, default=str)
        
        # Save discovered features
        features_data = [
            {
                'layer': f.layer_idx,
                'feature': f.feature_idx,
                'component_type': f.component_type,
                'activation': f.activation_strength,
                'description': f.semantic_description,
                'intervention_sites': f.intervention_sites
            }
            for f in result.discovered_features
        ]
        
        with open(output_dir / "discovered_features.json", "w") as f:
            json.dump(features_data, f, indent=2)
        
        # Save novel predictions
        with open(output_dir / "novel_predictions.json", "w") as f:
            json.dump(result.novel_predictions, f, indent=2, default=str)
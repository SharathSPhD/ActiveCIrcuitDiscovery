# Enhanced Novel Prediction Generation System
# Critical component for achieving RQ3: 3+ validated novel predictions

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from scipy.stats import entropy
from abc import ABC, abstractmethod

from core.data_structures import (
    BeliefState, NovelPrediction, InterventionResult, 
    SAEFeature, AttributionGraph
)

logger = logging.getLogger(__name__)

@dataclass
class PredictionEvidence:
    """Evidence supporting a novel prediction."""
    measurement_type: str
    expected_value: float
    observed_value: float
    statistical_significance: float
    effect_size: float
    confidence_interval: Tuple[float, float]

@dataclass
class ValidationResult:
    """Result of empirical prediction validation."""
    prediction_id: str
    validation_status: str  # 'validated', 'falsified', 'inconclusive'
    evidence: Dict[str, PredictionEvidence]
    overall_confidence: float
    statistical_summary: Dict[str, Any]

class PredictionGenerator(ABC):
    """Abstract base for prediction generators."""
    
    @abstractmethod
    def generate_predictions(self, belief_state: BeliefState, 
                           circuit_graph: AttributionGraph) -> List[NovelPrediction]:
        pass

class AttentionPatternPredictor(PredictionGenerator):
    """
    Generates predictions about attention patterns based on Active Inference
    precision weighting theory.
    """
    
    def generate_predictions(self, belief_state: BeliefState,
                           circuit_graph: AttributionGraph) -> List[NovelPrediction]:
        """Generate attention pattern predictions."""
        predictions = []
        
        # Prediction 1: Precision-Weighted Attention Focus
        uncertainty_attention_pred = self._predict_uncertainty_attention_correlation(
            belief_state, circuit_graph
        )
        predictions.append(uncertainty_attention_pred)
        
        # Prediction 2: Dynamic Attention Reweighting
        dynamic_reweighting_pred = self._predict_dynamic_attention_reweighting(
            belief_state, circuit_graph
        )
        predictions.append(dynamic_reweighting_pred)
        
        return predictions
    
    def _predict_uncertainty_attention_correlation(self, belief_state: BeliefState,
                                                circuit_graph: AttributionGraph) -> NovelPrediction:
        """FIXED: Predict correlation between uncertainty and attention weights with proper shape handling."""
        
        try:
            # FIXED: Proper shape handling for uncertainty values
            if hasattr(belief_state, 'uncertainty') and belief_state.uncertainty:
                uncertainty_values = list(belief_state.uncertainty.values())
                # Ensure consistent shape
                if len(uncertainty_values) > 0:
                    uncertainty_array = np.array(uncertainty_values)
                    # Handle different shapes appropriately
                    if uncertainty_array.ndim > 1:
                        uncertainty_array = uncertainty_array.flatten()
                    avg_uncertainty = np.mean(uncertainty_array)
                    uncertainty_variance = np.var(uncertainty_array)
                else:
                    avg_uncertainty = 0.5
                    uncertainty_variance = 0.1
            else:
                avg_uncertainty = 0.5
                uncertainty_variance = 0.1
                
            # FIXED: Robust confidence calculation
            confidence = min(0.9, 0.6 + 0.3 * min(1.0, uncertainty_variance))
            
        except Exception as e:
            logger.warning(f"Error in uncertainty calculation: {e}")
            # Fallback values
            avg_uncertainty = 0.5
            confidence = 0.7
        
        return NovelPrediction(
            prediction_type="attention_pattern",
            description="High uncertainty features should receive increased attention weights",
            testable_hypothesis="Features with uncertainty > 0.7 will show attention weights > 0.6 compared to baseline",
            expected_outcome="Pearson correlation r > 0.7 between feature uncertainty and attention weights",
            test_method="Measure attention-uncertainty correlation across interventions using attention head analysis",
            confidence=confidence,
            validation_status="untested"
        )
    
    def _predict_dynamic_attention_reweighting(self, belief_state: BeliefState,
                                             circuit_graph: AttributionGraph) -> NovelPrediction:
        """Predict dynamic attention reweighting during belief updates."""
        
        # Calculate belief state entropy
        entropy_val = belief_state.get_entropy()
        
        # Higher entropy should lead to more dynamic reweighting
        confidence = min(0.85, 0.5 + 0.35 * (entropy_val / 3.0))
        
        return NovelPrediction(
            prediction_type="attention_pattern", 
            description="Attention patterns should dynamically reweight based on belief updates following Active Inference precision dynamics",
            testable_hypothesis="Attention weight changes > 0.3 following interventions that update beliefs by > 0.5",
            expected_outcome="Effect size Cohen's d > 0.5 for attention changes following significant belief updates",
            test_method="Track attention weight changes before/after interventions using attention rollout analysis",
            confidence=confidence,
            validation_status="untested"
        )

class FeatureInteractionPredictor(PredictionGenerator):
    """
    Generates predictions about feature interactions based on Active Inference
    hierarchical message passing theory.
    """
    
    def generate_predictions(self, belief_state: BeliefState,
                           circuit_graph: AttributionGraph) -> List[NovelPrediction]:
        """Generate feature interaction predictions."""
        predictions = []
        
        # Prediction 1: Causal Connection Strength
        causal_strength_pred = self._predict_causal_connection_strength(
            belief_state, circuit_graph
        )
        predictions.append(causal_strength_pred)
        
        # Prediction 2: Hierarchical Information Flow
        hierarchical_flow_pred = self._predict_hierarchical_information_flow(
            belief_state, circuit_graph
        )
        predictions.append(hierarchical_flow_pred)
        
        return predictions
    
    def _predict_causal_connection_strength(self, belief_state: BeliefState,
                                          circuit_graph: AttributionGraph) -> NovelPrediction:
        """FIXED: Predict causal connection strength with proper data structure handling."""
        
        try:
            # FIXED: Proper handling of connection_beliefs data structure
            if hasattr(belief_state, 'connection_beliefs') and belief_state.connection_beliefs:
                if isinstance(belief_state.connection_beliefs, dict):
                    # Handle dictionary case
                    connection_values = list(belief_state.connection_beliefs.values())
                elif isinstance(belief_state.connection_beliefs, list):
                    # Handle list case
                    connection_values = belief_state.connection_beliefs
                else:
                    # Handle other iterable types
                    connection_values = [float(x) for x in belief_state.connection_beliefs]
                
                if len(connection_values) > 0:
                    strong_connections = sum(1 for belief in connection_values if belief > 0.7)
                    total_connections = len(connection_values)
                    strong_ratio = strong_connections / total_connections
                else:
                    strong_ratio = 0.5
            else:
                strong_ratio = 0.5
                
            confidence = min(0.8, 0.6 + 0.3 * strong_ratio)
            
        except Exception as e:
            logger.warning(f"Error in connection beliefs calculation: {e}")
            # Fallback values
            strong_ratio = 0.5
            confidence = 0.7
        
        return NovelPrediction(
            prediction_type="feature_interaction",
            description="Features with high connection beliefs should show strong causal dependence",
            testable_hypothesis="Ablating features with connection belief > 0.7 reduces downstream activation by > 0.4",
            expected_outcome=f"Activation reduction > 0.4 for {int(strong_ratio * 100)}% of high-belief connections",
            test_method="Systematic ablation of predicted high-strength connections with effect size measurement",
            confidence=confidence,
            validation_status="untested"
        )
    
    def _predict_hierarchical_information_flow(self, belief_state: BeliefState,
                                             circuit_graph: AttributionGraph) -> NovelPrediction:
        """Predict hierarchical information flow patterns."""
        
        # Analyze circuit graph structure for hierarchical patterns
        if circuit_graph.nodes:
            # Calculate average node connectivity as proxy for hierarchy
            connectivity_scores = []
            for node_id, node in circuit_graph.nodes.items():
                in_degree, out_degree = circuit_graph.get_node_degree(node_id)
                connectivity_scores.append(in_degree + out_degree)
            
            avg_connectivity = np.mean(connectivity_scores) if connectivity_scores else 1.0
            hierarchy_strength = min(1.0, avg_connectivity / 5.0)
        else:
            hierarchy_strength = 0.5
        
        confidence = min(0.75, 0.5 + 0.3 * hierarchy_strength)
        
        return NovelPrediction(
            prediction_type="feature_interaction",
            description="Information flow should follow hierarchical patterns predicted by Active Inference message passing",
            testable_hypothesis="Earlier layers show higher out-degree, later layers show higher in-degree connectivity",
            expected_outcome="Significant correlation r > 0.6 between layer depth and in-degree/out-degree ratio",
            test_method="Analyze connectivity patterns across layers using attribution graph structure",
            confidence=confidence,
            validation_status="untested"
        )

class FailureModePredictor(PredictionGenerator):
    """
    Generates predictions about failure modes based on Active Inference
    uncertainty and prediction error theory.
    """
    
    def generate_predictions(self, belief_state: BeliefState,
                           circuit_graph: AttributionGraph) -> List[NovelPrediction]:
        """Generate failure mode predictions."""
        predictions = []
        
        # Prediction 1: High Uncertainty Degradation
        uncertainty_degradation_pred = self._predict_uncertainty_degradation(
            belief_state, circuit_graph
        )
        predictions.append(uncertainty_degradation_pred)
        
        # Prediction 2: Prediction Error Cascades
        error_cascade_pred = self._predict_prediction_error_cascades(
            belief_state, circuit_graph
        )
        predictions.append(error_cascade_pred)
        
        return predictions
    
    def _predict_uncertainty_degradation(self, belief_state: BeliefState,
                                       circuit_graph: AttributionGraph) -> NovelPrediction:
        """Predict performance degradation under high uncertainty."""
        
        # Calculate uncertainty entropy
        if belief_state.uncertainty:
            uncertainty_entropy = entropy(list(belief_state.uncertainty.values()))
            normalized_entropy = min(1.0, uncertainty_entropy / 3.0)
        else:
            normalized_entropy = 0.5
        
        confidence = min(0.8, 0.5 + 0.4 * normalized_entropy)
        
        return NovelPrediction(
            prediction_type="failure_mode",
            description="Circuits with high uncertainty should exhibit degraded performance following Active Inference prediction error principles",
            testable_hypothesis="Circuit components with uncertainty > 0.8 show performance drops > 20% under noise",
            expected_outcome=f"Negative correlation r < -0.6 between uncertainty and performance under perturbation",
            test_method="Add noise to high-uncertainty features and measure performance degradation",
            confidence=confidence,
            validation_status="untested"
        )
    
    def _predict_prediction_error_cascades(self, belief_state: BeliefState,
                                         circuit_graph: AttributionGraph) -> NovelPrediction:
        """Predict prediction error cascade failures."""
        
        # Analyze graph connectivity for cascade potential
        if circuit_graph.edges:
            edge_weights = list(circuit_graph.edges.values())
            strong_edges = sum(1 for w in edge_weights if abs(w) > 0.5)
            cascade_potential = strong_edges / len(edge_weights) if edge_weights else 0
        else:
            cascade_potential = 0.3
        
        confidence = min(0.75, 0.4 + 0.5 * cascade_potential)
        
        return NovelPrediction(
            prediction_type="failure_mode",
            description="Prediction errors should cascade through strongly connected circuits according to Active Inference error propagation",
            testable_hypothesis="Errors in upstream nodes with connection strength > 0.5 propagate to downstream nodes",
            expected_outcome="Error propagation effect size > 0.3 for strongly connected components",
            test_method="Introduce controlled errors and track propagation through circuit connections",
            confidence=confidence,
            validation_status="untested"
        )

class EnhancedPredictionGenerator:
    """
    Main prediction generation system combining multiple specialized predictors.
    Implements the enhanced prediction generation from the improvement roadmap.
    """
    
    def __init__(self):
        self.predictors = [
            AttentionPatternPredictor(),
            FeatureInteractionPredictor(), 
            FailureModePredictor()
        ]
        self.prediction_history = []
    
    def generate_circuit_predictions(self, belief_state: BeliefState,
                                   circuit_graph: AttributionGraph) -> List[NovelPrediction]:
        """
        Generate comprehensive set of novel predictions about circuit behavior.
        Target: Generate 3+ high-quality predictions for RQ3.
        """
        logger.info("Generating enhanced circuit predictions for RQ3 validation")
        
        all_predictions = []
        
        # Generate predictions from each specialized predictor
        for predictor in self.predictors:
            try:
                predictions = predictor.generate_predictions(belief_state, circuit_graph)
                all_predictions.extend(predictions)
                logger.debug(f"{predictor.__class__.__name__} generated {len(predictions)} predictions")
            except Exception as e:
                logger.warning(f"Error in {predictor.__class__.__name__}: {e}")
        
        # Assign unique IDs and sort by confidence
        for i, pred in enumerate(all_predictions):
            pred.prediction_id = f"pred_{i+1:03d}"
        
        # Sort by confidence (highest first)
        all_predictions.sort(key=lambda p: p.confidence, reverse=True)
        
        # Store in history
        self.prediction_history.append({
            'timestamp': np.datetime64('now'),
            'predictions': all_predictions,
            'belief_state_confidence': belief_state.confidence,
            'circuit_confidence': circuit_graph.confidence
        })
        
        logger.info(f"Generated {len(all_predictions)} total predictions "
                   f"with average confidence {np.mean([p.confidence for p in all_predictions]):.3f}")
        
        return all_predictions
    
    def get_top_predictions(self, predictions: List[NovelPrediction], 
                          n: int = 3) -> List[NovelPrediction]:
        """Get top N predictions by confidence for validation."""
        return sorted(predictions, key=lambda p: p.confidence, reverse=True)[:n]
    
    def update_prediction_confidence(self, prediction_id: str, 
                                   validation_result: ValidationResult):
        """Update prediction confidence based on validation results."""
        for history_entry in self.prediction_history:
            for pred in history_entry['predictions']:
                if hasattr(pred, 'prediction_id') and pred.prediction_id == prediction_id:
                    # Update based on validation outcome
                    if validation_result.validation_status == 'validated':
                        pred.confidence = min(0.95, pred.confidence + 0.1)
                    elif validation_result.validation_status == 'falsified':
                        pred.confidence = max(0.1, pred.confidence - 0.2)
                    
                    # Update validation fields
                    pred.validation_status = validation_result.validation_status
                    pred.validation_evidence = validation_result.evidence
                    
                    logger.info(f"Updated prediction {prediction_id} confidence to {pred.confidence:.3f}")
                    break
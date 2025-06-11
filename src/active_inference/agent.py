# Active Inference Agent Implementation  
# Following patterns from https://pymdp-rtd.readthedocs.io/en/latest/

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

# pymdp integration
try:
    import pymdp
    from pymdp import Agent
    from pymdp.maths import softmax, kl_div, dot
    from pymdp.utils import obj_array, random_A_matrix, random_B_matrix
    PYMDP_AVAILABLE = True
except ImportError:
    PYMDP_AVAILABLE = False

# Project imports with proper relative imports
try:
    from ..core.interfaces import IActiveInferenceAgent
    from ..core.data_structures import (
        SAEFeature, InterventionResult, BeliefState, 
        CorrespondenceMetrics, NovelPrediction
    )
    from ..config.experiment_config import CompleteConfig, InterventionType
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from core.interfaces import IActiveInferenceAgent
    from core.data_structures import (
        SAEFeature, InterventionResult, BeliefState, 
        CorrespondenceMetrics, NovelPrediction
    )
    from config.experiment_config import CompleteConfig, InterventionType

logger = logging.getLogger(__name__)

class ActiveInferenceAgent(IActiveInferenceAgent):
    """
    Active Inference agent for circuit discovery.
    Implements proper pymdp patterns for belief updating and action selection.
    """
    
    def __init__(self, config: CompleteConfig, tracer):
        """Initialize Active Inference agent with configuration."""
        self.config = config
        self.tracer = tracer
        self.use_pymdp = config.active_inference.use_pymdp and PYMDP_AVAILABLE
        
        # Active Inference parameters
        self.epistemic_weight = config.active_inference.epistemic_weight
        self.exploration_weight = config.active_inference.exploration_weight
        self.convergence_threshold = config.active_inference.convergence_threshold
        
        # State tracking
        self.belief_state = None
        self.intervention_history = []
        self.correspondence_history = []
        self.novel_predictions = []
        
        # pymdp agent if available
        self.pymdp_agent = None
        
        logger.info(f"ActiveInferenceAgent initialized with pymdp={self.use_pymdp}")
    
    def initialize_beliefs(self, features: Dict[int, List[SAEFeature]]) -> BeliefState:
        """Initialize belief state from discovered features."""
        logger.info("Initializing Active Inference beliefs from discovered features")
        
        # Flatten features
        all_features = []
        for layer_features in features.values():
            all_features.extend(layer_features)
        
        if not all_features:
            logger.warning("No features found for belief initialization")
            return self._create_empty_belief_state()
        
        # Initialize beliefs about feature importance
        feature_importances = {}
        uncertainty = {}
        
        for feature in all_features:
            feature_importances[feature.feature_id] = feature.max_activation
            uncertainty[feature.feature_id] = max(0.1, 1.0 - feature.max_activation)
        
        # Initialize connection beliefs
        connection_beliefs = self._initialize_connection_beliefs(all_features)
        
        # Initialize pymdp components if available
        generative_model = None
        posterior_beliefs = None
        precision_matrix = None
        
        if self.use_pymdp:
            try:
                (generative_model, posterior_beliefs, 
                 precision_matrix) = self._initialize_pymdp_components(all_features)
                logger.info("PyMDP components initialized successfully")
            except Exception as e:
                logger.warning(f"PyMDP initialization failed: {e}")
                self.use_pymdp = False
        
        # Create belief state
        n_features = len(all_features)
        qs = np.ones(n_features) / n_features if n_features > 0 else np.array([1.0])
        
        confidence = self._calculate_initial_confidence(feature_importances, uncertainty)
        
        self.belief_state = BeliefState(
            qs=qs,
            feature_importances=feature_importances,
            connection_beliefs=connection_beliefs,
            uncertainty=uncertainty,
            confidence=confidence,
            generative_model=generative_model,
            posterior_beliefs=posterior_beliefs,
            precision_matrix=precision_matrix
        )
        
        logger.info(f"Initialized beliefs for {len(all_features)} features with confidence {confidence:.3f}")
        return self.belief_state
    
    def calculate_expected_free_energy(self, feature: SAEFeature, 
                                     intervention_type: InterventionType) -> float:
        """Calculate expected free energy for potential intervention."""
        if self.belief_state is None:
            return 0.0
        
        epistemic_value = self._calculate_epistemic_value(feature, intervention_type)
        pragmatic_value = self._calculate_pragmatic_value(feature, intervention_type)
        
        # Enhanced EFE with pymdp integration
        if self.use_pymdp and self.belief_state.generative_model is not None:
            model_uncertainty_reduction = self._calculate_model_uncertainty_reduction(feature)
            causal_info_gain = self._calculate_causal_information_gain(feature)
            
            expected_free_energy = (
                self.epistemic_weight * (epistemic_value + model_uncertainty_reduction) +
                (1 - self.epistemic_weight) * (pragmatic_value + causal_info_gain)
            )
        else:
            expected_free_energy = (
                self.epistemic_weight * epistemic_value + 
                (1 - self.epistemic_weight) * pragmatic_value
            )
        
        return expected_free_energy
    
    def update_beliefs(self, intervention_result: InterventionResult) -> CorrespondenceMetrics:
        """Update beliefs based on intervention result and calculate correspondence."""
        logger.debug(f"Updating beliefs based on intervention on feature {intervention_result.target_feature.feature_id}")
        
        if self.belief_state is None:
            logger.warning("No belief state to update")
            return self._create_empty_correspondence()
        
        # Update feature importance based on intervention effect
        feature_id = intervention_result.target_feature.feature_id
        effect_size = intervention_result.effect_size
        
        # Belief update following Active Inference prediction error minimization
        prediction_error = abs(effect_size - self.belief_state.feature_importances.get(feature_id, 0.5))
        learning_rate = 0.1
        
        if feature_id in self.belief_state.feature_importances:
            # Update importance based on prediction error
            old_importance = self.belief_state.feature_importances[feature_id]
            new_importance = old_importance + learning_rate * prediction_error * np.sign(effect_size - old_importance)
            self.belief_state.feature_importances[feature_id] = max(0.0, min(1.0, new_importance))
            
            # Update uncertainty (reduce with evidence)
            if feature_id in self.belief_state.uncertainty:
                old_uncertainty = self.belief_state.uncertainty[feature_id]
                uncertainty_reduction = min(0.1, effect_size * 0.2)
                self.belief_state.uncertainty[feature_id] = max(0.1, old_uncertainty - uncertainty_reduction)
        
        # Update connection beliefs for related features
        self._update_connection_beliefs(intervention_result)
        
        # Update pymdp beliefs if available
        if self.use_pymdp and self.belief_state.posterior_beliefs is not None:
            self._update_pymdp_beliefs(intervention_result)
        
        # Recalculate overall confidence
        self.belief_state.confidence = self._recalculate_confidence()
        
        # Calculate correspondence metrics
        correspondence = self._calculate_correspondence_metrics(intervention_result)
        
        # Store in history
        self.correspondence_history.append(correspondence)
        self.intervention_history.append(intervention_result)
        
        logger.info(f"Updated beliefs: confidence={self.belief_state.confidence:.3f}, "
                   f"correspondence={correspondence.overall_correspondence:.1f}%")
        
        return correspondence
    
    def generate_predictions(self) -> List[NovelPrediction]:
        """Generate novel predictions from current beliefs."""
        logger.info("Generating novel predictions from Active Inference beliefs")
        
        if self.belief_state is None:
            logger.warning("No belief state available for prediction generation")
            return []
        
        # Import here to avoid circular imports
        from ..core.prediction_system import EnhancedPredictionGenerator
        from ..core.data_structures import AttributionGraph
        
        # Create a mock attribution graph from current beliefs
        mock_graph = self._create_mock_attribution_graph()
        
        # Generate predictions using the enhanced system
        prediction_generator = EnhancedPredictionGenerator()
        predictions = prediction_generator.generate_circuit_predictions(
            self.belief_state, mock_graph
        )
        
        # Store predictions
        self.novel_predictions.extend(predictions)
        
        logger.info(f"Generated {len(predictions)} novel predictions")
        return predictions
    
    def check_convergence(self, threshold: float = 0.15) -> bool:
        """Check if beliefs have converged."""
        if len(self.intervention_history) < 3:
            return False
        
        # Check if recent belief changes are below threshold
        recent_changes = []
        for i in range(len(self.intervention_history) - 2):
            old_intervention = self.intervention_history[i]
            new_intervention = self.intervention_history[i + 1]
            
            old_feature_id = old_intervention.target_feature.feature_id
            new_feature_id = new_intervention.target_feature.feature_id
            
            if (old_feature_id in self.belief_state.feature_importances and 
                new_feature_id in self.belief_state.feature_importances):
                
                old_importance = self.belief_state.feature_importances[old_feature_id]
                new_importance = self.belief_state.feature_importances[new_feature_id]
                change = abs(new_importance - old_importance)
                recent_changes.append(change)
        
        if not recent_changes:
            return False
        
        avg_change = np.mean(recent_changes)
        converged = avg_change < threshold
        
        if converged:
            logger.info(f"Beliefs converged: average change {avg_change:.3f} < threshold {threshold}")
        
        return converged
    
    def _update_connection_beliefs(self, intervention_result: InterventionResult):
        """Update beliefs about feature connections based on intervention."""
        feature_id = intervention_result.target_feature.feature_id
        layer = intervention_result.target_feature.layer
        effect_size = intervention_result.effect_size
        
        # Update connections to features in adjacent layers
        for other_feature_id, importance in self.belief_state.feature_importances.items():
            if other_feature_id != feature_id:
                # Create connection key
                connection_key = (feature_id, other_feature_id)
                
                # Update connection belief based on effect correlation
                if connection_key not in self.belief_state.connection_beliefs:
                    self.belief_state.connection_beliefs[connection_key] = 0.5
                
                # Simple heuristic: if both features have high importance and intervention
                # had significant effect, strengthen connection belief
                if importance > 0.6 and effect_size > 0.3:
                    self.belief_state.connection_beliefs[connection_key] = min(
                        1.0, 
                        self.belief_state.connection_beliefs[connection_key] + 0.1
                    )
                elif effect_size < 0.1:
                    self.belief_state.connection_beliefs[connection_key] = max(
                        0.0,
                        self.belief_state.connection_beliefs[connection_key] - 0.05
                    )
    
    def _update_pymdp_beliefs(self, intervention_result: InterventionResult):
        """Update pymdp belief state based on intervention."""
        if not self.use_pymdp or self.pymdp_agent is None:
            return
        
        try:
            # Create observation from intervention result
            observation = self._intervention_to_observation(intervention_result)
            
            # Update beliefs using pymdp
            if hasattr(self.pymdp_agent, 'infer_states'):
                self.pymdp_agent.infer_states(observation)
                
                # Update our belief state with pymdp posteriors
                if hasattr(self.pymdp_agent, 'qs'):
                    self.belief_state.qs = self.pymdp_agent.qs[0]  # First factor
                    
        except Exception as e:
            logger.warning(f"PyMDP belief update failed: {e}")
    
    def _intervention_to_observation(self, intervention_result: InterventionResult) -> np.ndarray:
        """Convert intervention result to pymdp observation."""
        # Simple encoding: effect size discretized
        effect_size = intervention_result.effect_size
        
        if effect_size > 0.7:
            obs_idx = 2  # High effect
        elif effect_size > 0.3:
            obs_idx = 1  # Medium effect
        else:
            obs_idx = 0  # Low effect
        
        observation = np.zeros(3)
        observation[obs_idx] = 1.0
        return observation
    
    def _calculate_correspondence_metrics(self, intervention_result: InterventionResult) -> CorrespondenceMetrics:
        """Calculate correspondence between AI beliefs and circuit behavior."""
        from ..core.metrics import CorrespondenceCalculator
        
        calculator = CorrespondenceCalculator()
        
        # Create single-item lists for the calculator
        circuit_behavior = [intervention_result]
        
        return calculator.calculate_correspondence(self.belief_state, circuit_behavior)
    
    def _recalculate_confidence(self) -> float:
        """Recalculate overall belief confidence."""
        if not self.belief_state.feature_importances:
            return 0.0
        
        # Base confidence on uncertainty inverse
        uncertainties = list(self.belief_state.uncertainty.values())
        avg_uncertainty = np.mean(uncertainties) if uncertainties else 1.0
        uncertainty_confidence = 1.0 - avg_uncertainty
        
        # Add entropy-based confidence
        entropy_confidence = 1.0 - (self.belief_state.get_entropy() / 5.0)
        
        # Combine with intervention history
        intervention_confidence = min(1.0, len(self.intervention_history) / 10.0)
        
        overall_confidence = (0.5 * uncertainty_confidence + 
                            0.3 * entropy_confidence + 
                            0.2 * intervention_confidence)
        
        return max(0.0, min(1.0, overall_confidence))
    
    def _create_mock_attribution_graph(self) -> 'AttributionGraph':
        """Create mock attribution graph from beliefs for prediction generation."""
        from ..core.data_structures import AttributionGraph, CircuitNode
        
        # Create nodes from feature beliefs
        nodes = {}
        for i, (feature_id, importance) in enumerate(self.belief_state.feature_importances.items()):
            # Create mock SAE feature
            mock_feature = type('MockFeature', (), {
                'feature_id': feature_id,
                'layer': feature_id // 100,  # Simple layer assignment
                'max_activation': importance,
                'activation_threshold': 0.1,
                'description': f"Feature {feature_id}",
                'examples': []
            })()
            
            nodes[i] = CircuitNode(
                feature=mock_feature,
                activation_value=importance,
                causal_influence=importance
            )
        
        # Create edges from connection beliefs
        edges = {}
        for (source_id, target_id), belief in self.belief_state.connection_beliefs.items():
            if belief > 0.5:  # Only strong connections
                # Find node indices
                source_idx = None
                target_idx = None
                for idx, node in nodes.items():
                    if node.feature.feature_id == source_id:
                        source_idx = idx
                    elif node.feature.feature_id == target_id:
                        target_idx = idx
                
                if source_idx is not None and target_idx is not None:
                    edges[(source_idx, target_idx)] = belief
        
        return AttributionGraph(
            input_text="mock_input",
            nodes=nodes,
            edges=edges,
            target_output="mock_output",
            confidence=self.belief_state.confidence,
            metadata={'mock': True}
        )
    
    def _create_empty_correspondence(self) -> CorrespondenceMetrics:
        """Create empty correspondence metrics."""
        from ..core.data_structures import CorrespondenceMetrics
        
        return CorrespondenceMetrics(
            belief_updating_correspondence=0.0,
            precision_weighting_correspondence=0.0,
            prediction_error_correspondence=0.0,
            overall_correspondence=0.0
        )
    
    def _calculate_epistemic_value(self, feature: 'SAEFeature', 
                                 intervention_type: 'InterventionType') -> float:
        """Calculate epistemic value (information gain) of intervention."""
        if self.belief_state is None:
            return 0.0
        
        feature_id = feature.feature_id
        
        # Information gain based on current uncertainty
        current_uncertainty = self.belief_state.uncertainty.get(feature_id, 1.0)
        
        # Higher uncertainty features provide more information
        epistemic_value = current_uncertainty
        
        # Bonus for unexplored features
        if feature_id not in [ir.target_feature.feature_id for ir in self.intervention_history]:
            epistemic_value += 0.2
        
        return min(1.0, epistemic_value)
    
    def _calculate_pragmatic_value(self, feature: 'SAEFeature',
                                 intervention_type: 'InterventionType') -> float:
        """Calculate pragmatic value (goal achievement) of intervention."""
        if self.belief_state is None:
            return 0.0
        
        feature_id = feature.feature_id
        
        # Pragmatic value based on expected importance
        expected_importance = self.belief_state.feature_importances.get(feature_id, 0.5)
        
        # Higher importance features have more pragmatic value
        pragmatic_value = expected_importance
        
        # Bonus for features with strong connections
        connection_bonus = 0.0
        for (source_id, target_id), belief in self.belief_state.connection_beliefs.items():
            if source_id == feature_id or target_id == feature_id:
                connection_bonus += belief * 0.1
        
        return min(1.0, pragmatic_value + connection_bonus)
    
    def _calculate_model_uncertainty_reduction(self, feature: 'SAEFeature') -> float:
        """Calculate expected model uncertainty reduction (pymdp enhancement)."""
        if not self.use_pymdp or self.belief_state.generative_model is None:
            return 0.0
        
        # Simplified calculation based on belief entropy
        current_entropy = self.belief_state.get_entropy()
        max_entropy = 5.0  # Approximate maximum
        
        # Features with high activation should reduce uncertainty more
        uncertainty_reduction = feature.max_activation * (current_entropy / max_entropy)
        
        return min(0.5, uncertainty_reduction)
    
    def _calculate_causal_information_gain(self, feature: 'SAEFeature') -> float:
        """Calculate expected causal information gain."""
        feature_id = feature.feature_id
        
        # Information gain based on how many connections this feature has
        connection_count = sum(1 for (source_id, target_id) in self.belief_state.connection_beliefs.keys()
                             if source_id == feature_id or target_id == feature_id)
        
        # More connected features provide more causal information
        causal_info = min(0.3, connection_count * 0.05)
        
        return causal_info
    
    def _initialize_connection_beliefs(self, features: List['SAEFeature']) -> Dict[Tuple[int, int], float]:
        """Initialize connection beliefs between features."""
        connections = {}
        
        # Initialize weak connections between features in adjacent layers
        for i, feature_a in enumerate(features):
            for j, feature_b in enumerate(features):
                if (i != j and 
                    abs(feature_a.layer - feature_b.layer) == 1 and
                    feature_a.layer < feature_b.layer):
                    
                    # Initialize with weak prior belief
                    connection_key = (feature_a.feature_id, feature_b.feature_id)
                    connections[connection_key] = 0.3  # Weak prior
        
        return connections
    
    def _calculate_initial_confidence(self, feature_importances: Dict[int, float],
                                    uncertainty: Dict[int, float]) -> float:
        """Calculate initial confidence from feature distributions."""
        if not feature_importances:
            return 0.0
        
        # Base confidence on inverse of average uncertainty
        avg_uncertainty = np.mean(list(uncertainty.values())) if uncertainty else 1.0
        uncertainty_confidence = 1.0 - avg_uncertainty
        
        # Add variance-based confidence (more spread = less confident)
        importance_variance = np.var(list(feature_importances.values()))
        variance_confidence = 1.0 / (1.0 + importance_variance)
        
        return (0.7 * uncertainty_confidence + 0.3 * variance_confidence)
    
    def _create_empty_belief_state(self) -> BeliefState:
        """Create empty belief state for edge cases."""
        return BeliefState(
            qs=np.array([1.0]),
            feature_importances={},
            connection_beliefs={},
            uncertainty={},
            confidence=0.0
        )
    
    def _initialize_pymdp_components(self, features: List['SAEFeature']) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """Initialize pymdp generative model components."""
        if not PYMDP_AVAILABLE:
            return None, None, None
        
        try:
            n_features = len(features)
            n_obs_levels = 3  # Low, medium, high effect
            
            # Create A matrix (observation model)
            # Maps hidden states (feature importance) to observations (intervention effects)
            A = random_A_matrix([n_obs_levels], [n_features])
            
            # Create B matrix (transition model)
            # Feature importance can change based on interventions
            n_actions = 3  # Different intervention types
            B = random_B_matrix([n_features], [n_actions])
            
            # Create C matrix (preferences)
            # Prefer observations that provide information
            C = np.log(softmax(np.array([0.1, 0.5, 1.0])))  # Prefer high-effect observations
            
            # Initial posterior beliefs (uniform)
            qs = np.ones(n_features) / n_features
            
            # Create precision matrix
            precision_matrix = np.eye(n_features) * 2.0  # Moderate precision
            
            generative_model = {
                'A': A,
                'B': B, 
                'C': C,
                'n_states': [n_features],
                'n_obs': [n_obs_levels],
                'n_actions': [n_actions]
            }
            
            return generative_model, qs, precision_matrix
            
        except Exception as e:
            logger.warning(f"PyMDP initialization failed: {e}")
            return None, None, None
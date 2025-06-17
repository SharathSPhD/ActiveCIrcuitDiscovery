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
    from core.interfaces import IActiveInferenceAgent
    from core.data_structures import (
        SAEFeature, InterventionResult, BeliefState, 
        CorrespondenceMetrics, NovelPrediction
    )
    from config.experiment_config import CompleteConfig, InterventionType
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
        
        # Enhanced pymdp agent integration
        self.pymdp_agent = None
        self.generative_model = None
        self.action_space_size = 3  # ablation, patching, mean_ablation
        
        # Proper pymdp components
        self.A_matrices = None  # Observation model
        self.B_matrices = None  # Transition model  
        self.C_vectors = None   # Preference vectors
        self.D_vectors = None   # Prior beliefs
        self.state_space_dims = None
        self.obs_space_dims = None
        
        logger.info(f"ActiveInferenceAgent initialized with pymdp={self.use_pymdp}")
    
    def get_current_beliefs(self) -> BeliefState:
        """Get the current belief state."""
        if self.belief_state is None:
            logger.warning("No belief state initialized, returning empty state")
            return self._create_empty_belief_state()
        return self.belief_state
    
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
            # Normalize activation to [0,1] range for importance
            normalized_activation = min(1.0, max(0.0, feature.max_activation))
            feature_importances[feature.feature_id] = normalized_activation
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
                self._create_pymdp_agent(all_features)
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
        """Calculate Expected Free Energy using proper Active Inference decomposition."""
        if self.belief_state is None:
            return 0.0
        
        # Map intervention type to action index
        action_mapping = {
            InterventionType.ABLATION: 0,
            InterventionType.ACTIVATION_PATCHING: 1, 
            InterventionType.MEAN_ABLATION: 2
        }
        action = action_mapping.get(intervention_type, 0)
        
        # Use pymdp for proper EFE calculation if available
        if self.use_pymdp and self.pymdp_agent is not None:
            try:
                # Get feature state index
                feature_state = self._feature_to_state_index(feature)
                
                # Calculate proper EFE using pymdp utilities
                # EFE = Epistemic Value + Pragmatic Value
                
                # 1. Epistemic Value: Expected information gain about states
                epistemic_value = self._calculate_epistemic_value_pymdp(feature_state, action)
                
                # 2. Pragmatic Value: Expected utility/preference satisfaction
                pragmatic_value = self._calculate_pragmatic_value_pymdp(feature_state, action)
                
                # 3. Parameter information gain (model learning)
                param_info_gain = self._calculate_parameter_info_gain(feature_state, action)
                
                efe = epistemic_value + pragmatic_value + param_info_gain
                
                # Store components in belief state for tracking
                self.belief_state.epistemic_value = epistemic_value
                self.belief_state.pragmatic_value = pragmatic_value
                
                return efe
                
            except Exception as e:
                logger.warning(f"PyMDP EFE calculation failed: {e}")
                
        # Fallback to simplified calculation
        epistemic_value = self._calculate_epistemic_value(feature, intervention_type)
        pragmatic_value = self._calculate_pragmatic_value(feature, intervention_type)
        
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
        from core.prediction_system import EnhancedPredictionGenerator
        from core.data_structures import AttributionGraph
        
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
    
    def _create_pymdp_agent(self, features: List['SAEFeature']):
        """Create proper pymdp Agent instance."""
        if not PYMDP_AVAILABLE or self.A_matrices is None:
            return
        
        try:
            # Create pymdp Agent with proper components
            self.pymdp_agent = Agent(
                A=self.A_matrices,
                B=self.B_matrices, 
                C=self.C_vectors,
                D=self.D_vectors,
                use_utility=True,
                use_states_info_gain=True,
                use_param_info_gain=True,
                action_precision=16.0,
                policies=self._generate_policies()
            )
            
            logger.info("PyMDP Agent created successfully")
            
        except Exception as e:
            logger.warning(f"PyMDP Agent creation failed: {e}")
            self.pymdp_agent = None
    
    def _generate_policies(self) -> List[List[int]]:
        """Generate policy space for intervention selection."""
        # Simple policies: single-step actions
        policies = []
        for action in range(self.action_space_size):
            policies.append([action])  # Single-step policy
        
        # Add some multi-step policies for exploration
        policies.append([0, 1])  # Ablate then patch
        policies.append([1, 0])  # Patch then ablate
        policies.append([2, 0])  # Mean ablate then full ablate
        
        return policies
    
    def _update_pymdp_beliefs(self, intervention_result: InterventionResult):
        """Update pymdp belief state using proper variational message passing."""
        if not self.use_pymdp or self.pymdp_agent is None:
            return
        
        try:
            # Create observation from intervention result
            observation = self._intervention_to_observation(intervention_result)
            
            # Perform full Active Inference update cycle
            # 1. Infer states (posterior over hidden states)
            self.pymdp_agent.infer_states(observation)
            
            # 2. Infer policies (expected free energy calculation) 
            self.pymdp_agent.infer_policies()
            
            # 3. Update model parameters (learning)
            self.pymdp_agent.update_A(observation)
            
            # 4. Update our belief state with pymdp posteriors
            if hasattr(self.pymdp_agent, 'qs') and len(self.pymdp_agent.qs) > 0:
                self.belief_state.qs = self.pymdp_agent.qs[0].copy()  # First factor
                
                # Update feature importances based on posterior
                for i, feature_id in enumerate(self.belief_state.feature_importances.keys()):
                    if i < len(self.belief_state.qs):
                        # Map posterior belief to importance
                        importance = self.belief_state.qs[i]
                        self.belief_state.feature_importances[feature_id] = importance
            
            # Extract precision information for uncertainty
            if hasattr(self.pymdp_agent, 'gamma') and self.pymdp_agent.gamma is not None:
                avg_precision = np.mean(self.pymdp_agent.gamma)
                # Update uncertainty inversely to precision
                for feature_id in self.belief_state.uncertainty.keys():
                    self.belief_state.uncertainty[feature_id] = max(0.1, 1.0 / (1.0 + avg_precision))
                    
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
        from core.metrics import CorrespondenceCalculator
        
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
        from core.data_structures import AttributionGraph, GraphNode, GraphEdge
        
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
            
            nodes[i] = GraphNode(
                node_id=f"feature_{feature_id}_layer_{feature_id // 100}",
                layer=feature_id // 100,
                feature_id=feature_id,
                importance=importance,
                description=f"Feature {feature_id}",
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
                    edges[(source_idx, target_idx)] = GraphEdge(
                    source_id=f"feature_{source_id}_layer_{source_id // 100}",
                    target_id=f"feature_{target_id}_layer_{target_id // 100}",
                    weight=belief,
                    confidence=belief,
                    edge_type="predicted"
                )
        
        return AttributionGraph(
            input_text="mock_input",
            nodes=list(nodes.values()),
            edges=list(edges.values()),
            target_output="mock_output",
            confidence=self.belief_state.confidence,
            metadata={'mock': True}
        )
    
    def _create_empty_correspondence(self) -> CorrespondenceMetrics:
        """Create empty correspondence metrics."""
        from core.data_structures import CorrespondenceMetrics
        
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
    
    def _feature_to_state_index(self, feature: SAEFeature) -> int:
        """Map feature to state space index."""
        # Simple mapping: normalize feature importance to state index
        if not self.state_space_dims:
            return 0
        
        max_states = self.state_space_dims[0]
        # Map feature activation to discrete state
        state_idx = int(feature.max_activation * (max_states - 1))
        return min(state_idx, max_states - 1)
    
    def _calculate_epistemic_value_pymdp(self, state_idx: int, action: int) -> float:
        """Calculate epistemic value using pymdp information theory."""
        if not self.use_pymdp or self.A_matrices is None:
            return 0.0
        
        try:
            # Calculate expected information gain about hidden states
            # This is the mutual information between observations and states
            current_qs = self.belief_state.qs.copy()
            
            # Predict observation distribution under this action
            if len(self.B_matrices) > 0 and action < self.B_matrices[0].shape[2]:
                predicted_qs = dot(self.B_matrices[0][:, :, action], current_qs)
            else:
                predicted_qs = current_qs
            
            # Calculate expected observation under predicted beliefs
            expected_obs = dot(self.A_matrices[0], predicted_qs)
            
            # Calculate entropy of observation distribution
            obs_entropy = -np.sum(expected_obs * np.log(expected_obs + 1e-16))
            
            # Information gain is reduction in entropy
            max_entropy = np.log(len(expected_obs))
            info_gain = max_entropy - obs_entropy
            
            return float(info_gain)
            
        except Exception as e:
            logger.warning(f"Epistemic value calculation failed: {e}")
            return 0.0
    
    def _calculate_pragmatic_value_pymdp(self, state_idx: int, action: int) -> float:
        """Calculate pragmatic value using pymdp preference satisfaction."""
        if not self.use_pymdp or self.C_vectors is None:
            return 0.0
        
        try:
            # Calculate expected utility under this action
            current_qs = self.belief_state.qs.copy()
            
            # Predict next state distribution
            if len(self.B_matrices) > 0 and action < self.B_matrices[0].shape[2]:
                predicted_qs = dot(self.B_matrices[0][:, :, action], current_qs)
            else:
                predicted_qs = current_qs
            
            # Calculate expected observation
            expected_obs = dot(self.A_matrices[0], predicted_qs)
            
            # Calculate expected utility (preference satisfaction)
            expected_utility = dot(expected_obs, self.C_vectors[0])
            
            return float(expected_utility)
            
        except Exception as e:
            logger.warning(f"Pragmatic value calculation failed: {e}")
            return 0.0
    
    def _calculate_parameter_info_gain(self, state_idx: int, action: int) -> float:
        """Calculate expected information gain about model parameters."""
        if not self.use_pymdp:
            return 0.0
        
        try:
            # Simplified parameter information gain
            # Higher for less certain model parameters
            uncertainty = self.belief_state.get_average_uncertainty()
            return uncertainty * 0.1  # Weight parameter learning
            
        except Exception as e:
            logger.warning(f"Parameter info gain calculation failed: {e}")
            return 0.0
    
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
        """Initialize pymdp generative model components with proper structure."""
        if not PYMDP_AVAILABLE:
            return None, None, None
        
        try:
            n_features = max(16, len(features))  # Minimum state space size
            n_obs_levels = 3  # Low, medium, high intervention effect
            n_actions = 3    # Different intervention types
            
            # Store dimensions
            self.state_space_dims = [n_features]
            self.obs_space_dims = [n_obs_levels]
            
            # Create A matrix (observation model): P(obs|states)
            # Features with higher importance more likely to show high intervention effects
            A = obj_array([np.zeros((n_obs_levels, n_features))])
            for s in range(n_features):
                # Higher state indices = higher feature importance
                importance = s / n_features
                # High importance features more likely to show high effects
                A[0][0, s] = 0.7 - 0.5 * importance  # Low effect
                A[0][1, s] = 0.2 + 0.2 * importance  # Medium effect  
                A[0][2, s] = 0.1 + 0.3 * importance  # High effect
            
            # Normalize to proper probabilities
            for s in range(n_features):
                A[0][:, s] = A[0][:, s] / A[0][:, s].sum()
            
            # Create B matrices (transition model): P(states_t+1|states_t, actions)
            B = obj_array([np.zeros((n_features, n_features, n_actions))])
            
            for action in range(n_actions):
                for s_prev in range(n_features):
                    for s_next in range(n_features):
                        if action == 0:  # Ablation - can decrease importance
                            if s_next <= s_prev:
                                B[0][s_next, s_prev, action] = 0.7 if s_next == s_prev else 0.3 / s_prev if s_prev > 0 else 0
                        elif action == 1:  # Patching - maintains importance
                            B[0][s_next, s_prev, action] = 0.9 if s_next == s_prev else 0.1 / (n_features - 1)
                        else:  # Mean ablation - moderate decrease
                            if s_next <= s_prev:
                                B[0][s_next, s_prev, action] = 0.8 if s_next == s_prev else 0.2 / s_prev if s_prev > 0 else 0
                    
                    # Normalize
                    B[0][:, s_prev, action] = B[0][:, s_prev, action] / B[0][:, s_prev, action].sum()
            
            # Create C vectors (preferences): prefer informative observations
            C = obj_array([np.log(softmax(np.array([0.1, 0.3, 0.6])))])  # Prefer high-effect observations
            
            # Create D vectors (prior beliefs): uniform prior
            D = obj_array([np.ones(n_features) / n_features])
            
            # Store matrices
            self.A_matrices = A
            self.B_matrices = B
            self.C_vectors = C
            self.D_vectors = D
            
            # Initial posterior beliefs
            qs = np.ones(n_features) / n_features
            
            # Create precision matrix for uncertainty tracking
            precision_matrix = np.eye(n_features) * 2.0
            
            generative_model = {
                'A': A,
                'B': B, 
                'C': C,
                'D': D,
                'n_states': self.state_space_dims,
                'n_obs': self.obs_space_dims,
                'n_actions': [n_actions]
            }
            
            return generative_model, qs, precision_matrix
            
        except Exception as e:
            logger.warning(f"PyMDP initialization failed: {e}")
            return None, None, None
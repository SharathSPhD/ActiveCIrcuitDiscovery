#!/usr/bin/env python3
"""
Circuit-Specific Generative Model Components for Active Inference

This module provides utilities for creating proper A, B, C, D matrices
for circuit discovery using Active Inference. No shortcuts or approximations.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from pymdp.utils import obj_array, norm_dist_obj_arr
from pymdp.maths import softmax
from core.data_structures import SAEFeature, InterventionType

logger = logging.getLogger(__name__)

class CircuitGenerativeModelBuilder:
    """Build generative model components specifically for circuit discovery."""
    
    def __init__(self, max_components: int = 64, importance_levels: int = 4, 
                 intervention_types: int = 3, effect_magnitudes: int = 5, 
                 confidence_levels: int = 3):
        """Initialize model builder with circuit discovery parameters."""
        self.max_components = max_components
        self.importance_levels = importance_levels
        self.intervention_types = intervention_types
        self.effect_magnitudes = effect_magnitudes
        self.confidence_levels = confidence_levels
        
        # State space design
        self.num_states = [max_components, importance_levels, intervention_types]
        self.num_obs = [effect_magnitudes, confidence_levels]
        
        logger.info(f"Generative model builder initialized: states={self.num_states}, obs={self.num_obs}")
    
    def build_observation_model(self) -> List[np.ndarray]:
        """
        Build A matrices: P(observations | states)
        
        Returns proper observation likelihood matrices based on circuit discovery domain knowledge:
        - Higher component importance leads to higher intervention effects
        - Different intervention types have different effect patterns
        - Confidence correlates with importance and effect clarity
        """
        
        # A[0]: P(effect_magnitude | component, importance, intervention)
        A_effect = np.zeros((self.effect_magnitudes, self.max_components, 
                           self.importance_levels, self.intervention_types))
        
        # A[1]: P(confidence | component, importance, intervention)  
        A_confidence = np.zeros((self.confidence_levels, self.max_components,
                               self.importance_levels, self.intervention_types))
        
        for component in range(self.max_components):
            for importance in range(self.importance_levels):
                for intervention in range(self.intervention_types):
                    
                    # Effect magnitude distribution based on importance and intervention
                    if intervention == 0:  # Ablation - strong effects for important components
                        # Higher importance → higher probability of large effects
                        effect_weights = np.array([2.0, 1.5, 1.0, 0.8, 0.3]) 
                        effect_weights = effect_weights * (1.0 + importance * 0.8)
                        
                    elif intervention == 1:  # Activation patching - variable effects
                        # More moderate effects, still correlated with importance
                        effect_weights = np.array([1.5, 2.0, 1.8, 1.0, 0.5])
                        effect_weights = effect_weights * (1.0 + importance * 0.5)
                        
                    else:  # Mean ablation - generally moderate effects
                        # Conservative effects regardless of importance
                        effect_weights = np.array([1.0, 2.5, 2.0, 1.2, 0.3])
                        effect_weights = effect_weights * (1.0 + importance * 0.3)
                    
                    # Normalize to proper probability distribution
                    A_effect[:, component, importance, intervention] = softmax(effect_weights)
                    
                    # Confidence distribution
                    # Higher importance and clearer effects → higher confidence
                    if importance >= 2:  # High importance components
                        if intervention == 0:  # Ablation gives clearest results
                            conf_weights = np.array([0.2, 0.3, 2.5])  # High confidence
                        else:
                            conf_weights = np.array([0.5, 1.5, 1.0])  # Medium confidence
                    else:  # Lower importance components
                        conf_weights = np.array([2.0, 1.0, 0.3])  # Lower confidence
                    
                    A_confidence[:, component, importance, intervention] = softmax(conf_weights)
        
        return obj_array([A_effect, A_confidence])
    
    def build_transition_model(self) -> List[np.ndarray]:
        """
        Build B matrices: P(next_state | current_state, action)
        
        Returns proper state transition matrices for circuit discovery:
        - Component IDs mostly stable (don't change during experiment)
        - Importance can be refined based on intervention evidence
        - Intervention type controlled deterministically by agent actions
        """
        
        # B[0]: Component transitions (mostly identity)
        B_component = np.zeros((self.max_components, self.max_components, self.intervention_types))
        for action in range(self.intervention_types):
            B_component[:, :, action] = np.eye(self.max_components)
        
        # B[1]: Importance transitions (can change based on evidence)
        B_importance = np.zeros((self.importance_levels, self.importance_levels, self.intervention_types))
        
        for action in range(self.intervention_types):
            for current_imp in range(self.importance_levels):
                if action == 0:  # Ablation - can reveal true importance
                    # Allow evidence-based importance updates
                    # More likely to stay same, but can change based on results
                    transition_probs = np.ones(self.importance_levels) * 0.1
                    transition_probs[current_imp] = 0.7  # Mostly stay same
                    
                    # Slight bias toward revealing higher importance if ablation has strong effect
                    if current_imp < self.importance_levels - 1:
                        transition_probs[current_imp + 1] *= 2.0
                    
                elif action == 1:  # Patching - moderate evidence about importance
                    transition_probs = np.ones(self.importance_levels) * 0.05
                    transition_probs[current_imp] = 0.85  # Mostly stable
                    
                else:  # Mean ablation - conservative updates
                    transition_probs = np.ones(self.importance_levels) * 0.02
                    transition_probs[current_imp] = 0.94  # Very stable
                
                B_importance[:, current_imp, action] = softmax(transition_probs)
        
        # B[2]: Intervention type transitions (deterministic control)
        B_intervention = np.zeros((self.intervention_types, self.intervention_types, self.intervention_types))
        for action in range(self.intervention_types):
            B_intervention[action, :, action] = 1.0  # Deterministic transition to chosen intervention
        
        return obj_array([B_component, B_importance, B_intervention])
    
    def build_preferences(self, epistemic_weight: float = 0.7) -> List[np.ndarray]:
        """
        Build C vectors: Preferences over observations
        
        Args:
            epistemic_weight: Weight for information-seeking vs outcome preferences
            
        Returns proper preference vectors that encourage:
        - Information gain (prefer observations that reduce uncertainty)
        - Clear, confident results over ambiguous ones
        """
        
        # C[0]: Preferences over effect magnitudes
        # Prefer informative effects (either strong effects or clear no-effects)
        effect_preferences = np.array([0.3, 0.1, 0.2, 0.5, 0.7])  # Prefer larger effects
        
        # Add epistemic bonus for extreme values (very clear signals)
        epistemic_bonus = np.array([0.4, 0.0, 0.0, 0.2, 0.4])  # Bonus for clear results
        effect_prefs_final = effect_preferences + epistemic_weight * epistemic_bonus
        
        # C[1]: Preferences over confidence levels
        confidence_preferences = np.array([0.0, 0.3, 0.8])  # Strongly prefer high confidence
        
        return obj_array([effect_prefs_final, confidence_preferences])
    
    def build_priors(self, feature_importances: Optional[Dict[int, float]] = None) -> List[np.ndarray]:
        """
        Build D vectors: Prior beliefs over initial states
        
        Args:
            feature_importances: Optional mapping of feature IDs to importance scores
            
        Returns proper prior distributions based on:
        - Uniform priors over components (or informed by feature analysis)
        - Uniform priors over importance levels initially
        - Preference for starting with ablation interventions
        """
        
        # D[0]: Prior over components
        if feature_importances:
            # Use feature importance to set component priors
            component_priors = np.ones(self.max_components) * 0.01  # Small uniform baseline
            
            for feature_id, importance in feature_importances.items():
                if feature_id < self.max_components:
                    component_priors[feature_id] = importance
            
            # Normalize
            component_priors = component_priors / component_priors.sum()
        else:
            # Uniform prior over components
            component_priors = np.ones(self.max_components) / self.max_components
        
        # D[1]: Prior over importance levels (slightly favor medium importance)
        importance_priors = softmax(np.array([0.8, 1.2, 1.0, 0.6]))
        
        # D[2]: Prior over intervention types (prefer starting with ablation)
        intervention_priors = softmax(np.array([2.0, 1.0, 0.8]))  # Ablation, patching, mean_ablation
        
        return obj_array([component_priors, importance_priors, intervention_priors])
    
    def validate_generative_model(self, A: List[np.ndarray], B: List[np.ndarray], 
                                C: List[np.ndarray], D: List[np.ndarray]) -> bool:
        """
        Validate that generative model components are properly formed.
        
        Returns True if all matrices satisfy pymdp requirements, False otherwise.
        """
        
        try:
            # Validate A matrices (should sum to 1 over first dimension)
            for m, A_m in enumerate(A):
                for state_combo in np.ndindex(A_m.shape[1:]):
                    obs_dist = A_m[:, state_combo]
                    if not np.isclose(obs_dist.sum(), 1.0, atol=1e-6):
                        logger.error(f"A[{m}] not normalized at state {state_combo}: sum={obs_dist.sum()}")
                        return False
            
            # Validate B matrices (should sum to 1 over first dimension)
            for f, B_f in enumerate(B):
                for state in range(B_f.shape[1]):
                    for action in range(B_f.shape[2]):
                        trans_dist = B_f[:, state, action]
                        if not np.isclose(trans_dist.sum(), 1.0, atol=1e-6):
                            logger.error(f"B[{f}] not normalized at state {state}, action {action}: sum={trans_dist.sum()}")
                            return False
            
            # Validate D vectors (should sum to 1)
            for f, D_f in enumerate(D):
                if not np.isclose(D_f.sum(), 1.0, atol=1e-6):
                    logger.error(f"D[{f}] not normalized: sum={D_f.sum()}")
                    return False
            
            # Check dimensions consistency
            if len(A) != len(self.num_obs):
                logger.error(f"A matrices count {len(A)} != observation modalities {len(self.num_obs)}")
                return False
            
            if len(B) != len(self.num_states):
                logger.error(f"B matrices count {len(B)} != state factors {len(self.num_states)}")
                return False
            
            logger.info("Generative model validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Generative model validation failed: {e}")
            return False
    
    def create_complete_model(self, feature_importances: Optional[Dict[int, float]] = None,
                            epistemic_weight: float = 0.7) -> Tuple[List[np.ndarray], List[np.ndarray], 
                                                                   List[np.ndarray], List[np.ndarray]]:
        """
        Create complete generative model for circuit discovery.
        
        Returns validated A, B, C, D matrices ready for pymdp Agent.
        """
        
        logger.info("Building complete generative model for circuit discovery")
        
        # Build all components
        A = self.build_observation_model()
        B = self.build_transition_model()
        C = self.build_preferences(epistemic_weight)
        D = self.build_priors(feature_importances)
        
        # Validate model
        if not self.validate_generative_model(A, B, C, D):
            raise ValueError("Generated model failed validation")
        
        logger.info("Complete generative model created and validated")
        return A, B, C, D

def create_circuit_discovery_model(features: List[SAEFeature], 
                                 max_components: int = 64,
                                 epistemic_weight: float = 0.7) -> Tuple[List[np.ndarray], List[np.ndarray], 
                                                                        List[np.ndarray], List[np.ndarray]]:
    """
    Convenience function to create generative model from discovered features.
    
    Args:
        features: List of discovered SAE features
        max_components: Maximum number of components to track
        epistemic_weight: Weight for epistemic vs pragmatic preferences
        
    Returns:
        Tuple of (A, B, C, D) matrices ready for pymdp Agent
    """
    
    # Extract feature importances
    feature_importances = {}
    for i, feature in enumerate(features[:max_components]):
        feature_importances[i] = min(feature.max_activation, 1.0)
    
    # Build model
    builder = CircuitGenerativeModelBuilder(max_components=max_components)
    return builder.create_complete_model(feature_importances, epistemic_weight)
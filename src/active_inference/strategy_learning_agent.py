#!/usr/bin/env python3
"""
Strategy Learning Active Inference Agent

This agent learns intervention strategy effectiveness patterns rather than
tracking individual circuit features. It addresses the fundamental scale
mismatch identified in the CORRECTION_STRATEGY.md analysis.

Key Innovation: Agent learns which intervention strategies are most effective
for circuit discovery, not individual feature importance.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from scipy.stats import pearsonr

# Real pymdp imports
import pymdp
from pymdp.agent import Agent
from pymdp.utils import obj_array, random_A_matrix, random_B_matrix, norm_dist, norm_dist_obj_arr
from pymdp.maths import softmax, kl_div, entropy, spm_dot
from pymdp import control, inference, learning

# Project imports
from core.interfaces import IActiveInferenceAgent
from core.data_structures import (
    CircuitFeature, InterventionResult, BeliefState, 
    CorrespondenceMetrics, NovelPrediction
)
from config.experiment_config import CompleteConfig, InterventionType

logger = logging.getLogger(__name__)

class StrategyLearningAgent(IActiveInferenceAgent):
    """
    Strategy Learning Active Inference Agent
    
    Learns intervention strategy effectiveness patterns:
    - State space: [12 layers × 3 interventions × 4 contexts] = 144 strategies
    - Observations: [5 informativeness × 3 semantic coherence] 
    - Focus: Strategy effectiveness, not individual features
    """
    
    def __init__(self, config: CompleteConfig):
        """Initialize strategy learning agent."""
        self.config = config
        
        # Strategy-focused parameters
        self.num_layers = 12  # Model layers
        self.num_interventions = 3  # ablation, patching, mean_ablation
        self.num_contexts = 4  # early, middle, late, semantic
        self.num_informativeness = 5  # none, low, medium, high, exceptional
        self.num_semantic_coherence = 3  # low, medium, high
        
        # Design strategy-focused state and observation spaces
        self._design_strategy_spaces()
        
        # Create generative model for strategy learning
        self._create_strategy_generative_model()
        
        # Initialize pymdp agent
        self._initialize_strategy_agent()
        
        # Strategy tracking
        self.strategy_beliefs = {}
        self.strategy_success_rates = {}
        self.intervention_history = []
        self.belief_history = []
        
        logger.info(f"StrategyLearningAgent initialized for {self.num_strategies} strategies")
    
    def _design_strategy_spaces(self):
        """Design state and observation spaces for strategy learning."""
        # State factors for strategy learning:
        # Factor 0: Layer type (0-11 for 12 layers)
        # Factor 1: Intervention type (0: ablation, 1: patching, 2: mean_ablation)
        # Factor 2: Discovery context (0: early, 1: middle, 2: late, 3: semantic)
        
        self.num_states = [self.num_layers, self.num_interventions, self.num_contexts]
        
        # Observations for strategy effectiveness:
        # Modality 0: Informativeness (0: none, 1: low, 2: medium, 3: high, 4: exceptional)
        # Modality 1: Semantic coherence (0: low, 1: medium, 2: high)
        
        self.num_obs = [self.num_informativeness, self.num_semantic_coherence]
        
        # Total number of strategy combinations
        self.num_strategies = self.num_layers * self.num_interventions * self.num_contexts
        
        logger.info(f"Strategy space: {self.num_states}, Observation space: {self.num_obs}")
        logger.info(f"Total strategies: {self.num_strategies}")
    
    def _create_strategy_generative_model(self):
        """Create generative model for strategy learning."""
        
        # A matrices: P(observations | states)
        # A[0]: P(informativeness | layer, intervention, context)
        # A[1]: P(semantic_coherence | layer, intervention, context)
        
        self.A = obj_array(2)
        self.A[0] = np.zeros((self.num_obs[0], self.num_states[0], self.num_states[1], self.num_states[2]))
        self.A[1] = np.zeros((self.num_obs[1], self.num_states[0], self.num_states[1], self.num_states[2]))
        
        # Initialize A[0]: Informativeness depends on layer, intervention, context
        for layer in range(self.num_states[0]):
            for intervention in range(self.num_states[1]):
                for context in range(self.num_states[2]):
                    
                    # Layer-specific informativeness patterns
                    if layer < 4:  # Early layers
                        base_info = np.array([0.4, 0.3, 0.2, 0.1, 0.0])  # Lower informativeness
                    elif layer < 8:  # Middle layers  
                        base_info = np.array([0.2, 0.3, 0.3, 0.2, 0.0])  # Medium informativeness
                    else:  # Late layers
                        base_info = np.array([0.1, 0.2, 0.3, 0.3, 0.1])  # Higher informativeness
                    
                    # Intervention-specific adjustments
                    if intervention == 0:  # Ablation - generally more informative
                        base_info = base_info * np.array([0.5, 0.8, 1.2, 1.5, 2.0])
                    elif intervention == 1:  # Patching - variable informativeness
                        base_info = base_info * np.array([0.8, 1.0, 1.2, 1.0, 0.8])
                    else:  # Mean ablation - moderate informativeness
                        base_info = base_info * np.array([0.7, 1.0, 1.1, 1.0, 0.7])
                    
                    # Context-specific adjustments
                    if context == 3:  # Semantic context - potentially more informative
                        base_info = base_info * np.array([0.6, 0.8, 1.0, 1.3, 1.8])
                    
                    self.A[0][:, layer, intervention, context] = softmax(base_info)
        
        # Initialize A[1]: Semantic coherence depends on intervention and context
        for layer in range(self.num_states[0]):
            for intervention in range(self.num_states[1]):
                for context in range(self.num_states[2]):
                    
                    # Base semantic coherence
                    if context == 3:  # Semantic context
                        coherence_probs = softmax(np.array([0.1, 0.3, 0.6]))  # Higher coherence
                    else:
                        coherence_probs = softmax(np.array([0.4, 0.4, 0.2]))  # Lower coherence
                    
                    self.A[1][:, layer, intervention, context] = coherence_probs
        
        # B matrices: P(next_state | current_state, action)
        # For strategy learning, transitions are mostly deterministic
        self.B = obj_array(3)
        self.B[0] = np.zeros((self.num_states[0], self.num_states[0], self.num_interventions))
        self.B[1] = np.zeros((self.num_states[1], self.num_states[1], self.num_interventions))
        self.B[2] = np.zeros((self.num_states[2], self.num_states[2], self.num_interventions))
        
        # Layer transitions (agent chooses layer based on action)
        for action in range(self.num_interventions):
            self.B[0][:, :, action] = np.eye(self.num_states[0])  # Layers don't transition
        
        # Intervention transitions (deterministic based on action)
        for action in range(self.num_interventions):
            self.B[1][action, :, action] = 1.0  # Transition to selected intervention
        
        # Context transitions (evolve based on discovery progress)
        for action in range(self.num_interventions):
            for context in range(self.num_states[2]):
                if context < 3:  # Can advance to next context
                    self.B[2][context, context, action] = 0.7
                    self.B[2][context + 1, context, action] = 0.3
                else:  # Semantic context - stay
                    self.B[2][context, context, action] = 1.0
        
        # C vectors: Preferences over observations (prefer informative, coherent results)
        self.C = obj_array(2)
        self.C[0] = np.array([0.0, 0.1, 0.3, 0.6, 1.0])  # Prefer higher informativeness
        self.C[1] = np.array([0.0, 0.5, 1.0])             # Prefer higher coherence
        
        # D vectors: Prior beliefs over initial states
        self.D = obj_array(3)
        self.D[0] = np.ones(self.num_states[0]) / self.num_states[0]  # Uniform over layers
        self.D[1] = np.array([0.6, 0.3, 0.1])                        # Prefer ablation initially
        self.D[2] = np.array([0.4, 0.3, 0.2, 0.1])                   # Start with early context
        
        logger.info("Strategy generative model created")
    
    def _initialize_strategy_agent(self):
        """Initialize pymdp agent for strategy learning."""
        
        # Generate policies for strategy selection
        self.policies = control.construct_policies(
            num_states=self.num_states,
            num_controls=[1, self.num_interventions, 1],  # Control intervention type
            policy_len=2,
            control_fac_idx=[1]  # Control factor 1 (intervention type)
        )
        
        # Create pymdp agent
        self.agent = Agent(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            policies=self.policies,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=True,
            policy_len=2,
            inference_horizon=1,
            action_selection="deterministic",
            gamma=16.0,
            alpha=16.0
        )
        
        logger.info(f"Strategy agent initialized with {len(self.policies)} policies")
    
    def encode_intervention_outcome(self, intervention_result: InterventionResult) -> List[int]:
        """Encode intervention result as strategy effectiveness observation."""
        
        # Informativeness based on effect size and significance
        if intervention_result.effect_magnitude > 0.8 and intervention_result.statistical_significance:
            informativeness = 4  # exceptional
        elif intervention_result.effect_magnitude > 0.6:
            informativeness = 3  # high
        elif intervention_result.effect_magnitude > 0.4:
            informativeness = 2  # medium
        elif intervention_result.effect_magnitude > 0.2:
            informativeness = 1  # low
        else:
            informativeness = 0  # none
        
        # Semantic coherence based on semantic change
        if intervention_result.semantic_change and intervention_result.statistical_significance:
            semantic_coherence = 2  # high
        elif intervention_result.semantic_change:
            semantic_coherence = 1  # medium
        else:
            semantic_coherence = 0  # low
        
        return [informativeness, semantic_coherence]
    
    def track_strategy_beliefs(self) -> Dict[str, float]:
        """Track agent's beliefs about strategy effectiveness."""
        if not hasattr(self.agent, 'qs'):
            return {}
        
        strategy_beliefs = {}
        
        for layer in range(self.num_layers):
            for intervention in range(self.num_interventions):
                for context in range(self.num_contexts):
                    # Calculate joint belief about this strategy
                    belief = (self.agent.qs[0][layer] * 
                             self.agent.qs[1][intervention] * 
                             self.agent.qs[2][context])
                    
                    strategy_key = f"L{layer}_I{intervention}_C{context}"
                    strategy_beliefs[strategy_key] = float(belief)
        
        return strategy_beliefs
    
    def track_discovery_success(self, intervention_results: List[InterventionResult]) -> Dict[str, float]:
        """Track actual success rates for different intervention strategies."""
        
        strategy_success = {}
        
        for result in intervention_results:
            # Map intervention result to strategy key
            layer = result.intervention_layer_idx
            intervention = self._map_intervention_type(result.intervention_type)
            context = self._infer_context(result)
            
            strategy_key = f"L{layer}_I{intervention}_C{context}"
            
            if strategy_key not in strategy_success:
                strategy_success[strategy_key] = []
            
            # Success = semantic change detected
            success = 1.0 if result.semantic_change else 0.0
            strategy_success[strategy_key].append(success)
        
        # Calculate average success rate for each strategy
        for strategy in strategy_success:
            strategy_success[strategy] = np.mean(strategy_success[strategy])
        
        return strategy_success
    
    def _map_intervention_type(self, intervention_type: InterventionType) -> int:
        """Map InterventionType enum to integer index."""
        mapping = {
            InterventionType.ABLATION: 0,
            InterventionType.ACTIVATION_PATCHING: 1,
            InterventionType.MEAN_ABLATION: 2
        }
        return mapping.get(intervention_type, 0)
    
    def _infer_context(self, result: InterventionResult) -> int:
        """Infer discovery context from intervention result."""
        # Simple heuristic based on result characteristics
        if result.semantic_change and result.statistical_significance:
            return 3  # semantic context
        elif result.effect_magnitude > 0.5:
            return 2  # late context
        elif result.effect_magnitude > 0.2:
            return 1  # middle context
        else:
            return 0  # early context
    
    def calculate_strategy_correspondence(self, 
                                        strategy_beliefs: Dict[str, float],
                                        actual_success_rates: Dict[str, float]) -> float:
        """Calculate correspondence between strategy beliefs and actual success."""
        
        # Extract matching strategy keys
        common_strategies = set(strategy_beliefs.keys()) & set(actual_success_rates.keys())
        
        if len(common_strategies) < 2:
            return 0.0
        
        # Calculate correlation between beliefs and actual success
        beliefs = [strategy_beliefs[s] for s in common_strategies]
        success_rates = [actual_success_rates[s] for s in common_strategies]
        
        try:
            correlation, p_value = pearsonr(beliefs, success_rates)
            
            # Convert to percentage, handle NaN
            if np.isnan(correlation):
                return 0.0
            
            return abs(correlation) * 100.0
            
        except Exception as e:
            logger.warning(f"Correlation calculation failed: {e}")
            return 0.0
    
    def select_intervention_strategy(self, available_features: List[CircuitFeature]) -> Tuple[CircuitFeature, InterventionType]:
        """Select intervention based on strategy learning."""
        
        if not available_features:
            raise ValueError("No features available for intervention selection")
        
        # Use agent's policy inference to select strategy
        if hasattr(self.agent, 'qs'):
            q_pi, G = self.agent.infer_policies()
            action = self.agent.sample_action()
            
            # Map action to intervention type
            intervention_type_map = {
                0: InterventionType.ABLATION,
                1: InterventionType.ACTIVATION_PATCHING,
                2: InterventionType.MEAN_ABLATION
            }
            
            selected_intervention = intervention_type_map.get(action[1], InterventionType.ABLATION)
            
            # Select feature based on current layer belief
            layer_beliefs = self.agent.qs[0]
            preferred_layer = np.argmax(layer_beliefs)
            
            # Find feature in preferred layer
            layer_features = [f for f in available_features if f.layer_idx == preferred_layer]
            if layer_features:
                selected_feature = layer_features[0]
            else:
                selected_feature = available_features[0]
            
            logger.info(f"Selected strategy: {selected_intervention} on layer {preferred_layer}")
            return selected_feature, selected_intervention
        
        else:
            # Fallback
            return available_features[0], InterventionType.ABLATION
    
    def update_beliefs_from_intervention(self, 
                                       feature_idx: int,
                                       intervention_type: InterventionType,
                                       observation: np.ndarray,
                                       intervention_result: InterventionResult = None) -> None:
        """Update strategy beliefs from intervention results."""
        
        # Encode intervention result as strategy effectiveness observation
        if intervention_result is not None:
            obs = self.encode_intervention_outcome(intervention_result)
        else:
            obs = [int(observation[0]), int(observation[1])]
        
        # Update agent beliefs
        try:
            self.agent.infer_states(obs)
            
            # Store beliefs for tracking
            self.belief_history.append(self.agent.qs.copy())
            self.intervention_history.append(intervention_result)
            
            # Update strategy tracking
            self.strategy_beliefs = self.track_strategy_beliefs()
            
            logger.info(f"Updated strategy beliefs from intervention")
            
        except Exception as e:
            logger.warning(f"Strategy belief update failed: {e}")
    
    def get_current_beliefs(self) -> BeliefState:
        """Get current strategy beliefs."""
        if not hasattr(self.agent, 'qs'):
            return BeliefState(
                qs=[d.copy() for d in self.agent.D],
                feature_importances={},
                connection_beliefs={},
                uncertainty={},
                confidence=0.0
            )
        
        # Calculate confidence from belief entropy
        try:
            entropies = []
            for qs in self.agent.qs:
                if len(qs) > 0:
                    # Simple entropy calculation for safety
                    qs_safe = np.array(qs) + 1e-10  # Add small epsilon
                    h = -np.sum(qs_safe * np.log(qs_safe))
                    entropies.append(h)
                else:
                    entropies.append(0.0)
            
            if entropies:
                avg_entropy = np.mean(entropies)
                max_entropy = np.log(max([len(qs) for qs in self.agent.qs if len(qs) > 0] + [2]))
                confidence = 1.0 - (avg_entropy / max_entropy) if max_entropy > 0 else 0.5
            else:
                confidence = 0.5
        except Exception as e:
            logger.warning(f'Entropy calculation failed: {e}')
            confidence = 0.5
        
        return BeliefState(
            qs=self.agent.qs.copy(),
            feature_importances=self.strategy_beliefs,
            connection_beliefs={},
            uncertainty={i: entropies[i] for i in range(len(entropies))},
            confidence=confidence
        )
    
    def generate_predictions(self) -> List[NovelPrediction]:
        """Generate predictions about effective intervention strategies."""
        predictions = []
        
        if not hasattr(self.agent, 'qs'):
            return predictions
        
        # Generate predictions based on learned strategy effectiveness
        strategy_beliefs = self.track_strategy_beliefs()
        
        # Find top strategies
        top_strategies = sorted(strategy_beliefs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for i, (strategy_key, belief) in enumerate(top_strategies):
            # Parse strategy key
            parts = strategy_key.split('_')
            layer = int(parts[0][1:])
            intervention = int(parts[1][1:])
            context = int(parts[2][1:])
            
            intervention_names = ['ablation', 'patching', 'mean_ablation']
            context_names = ['early', 'middle', 'late', 'semantic']
            
            prediction = NovelPrediction(
                prediction_type="feature_interaction",
                description=f"Layer {layer} with {intervention_names[intervention]} in {context_names[context]} context will be highly effective",
                testable_hypothesis=f"Interventions on layer {layer} using {intervention_names[intervention]} will show >60% success rate",
                expected_outcome=f"High informativeness and semantic coherence for L{layer}_I{intervention}_C{context}",
                test_method="strategy_validation",
                confidence=float(belief),
                validation_status="untested"
            )
            predictions.append(prediction)
        
        logger.info(f"Generated {len(predictions)} strategy predictions")
        return predictions
    
    def check_convergence(self) -> bool:
        """Check if strategy learning has converged."""
        if len(self.belief_history) < 5:
            return False
        
        # Check if strategy beliefs are stabilizing
        recent_beliefs = self.belief_history[-5:]
        
        # Calculate change in beliefs
        belief_changes = []
        for i in range(1, len(recent_beliefs)):
            change = sum(kl_div(recent_beliefs[i][j], recent_beliefs[i-1][j]) 
                        for j in range(len(recent_beliefs[i])))
            belief_changes.append(change)
        
        # Check if changes are small
        avg_change = np.mean(belief_changes)
        converged = avg_change < 0.01
        
        if converged:
            logger.info(f"Strategy learning converged: avg change {avg_change:.4f}")
        
        return converged
    
    # Interface compatibility methods
    def initialize_beliefs(self, features: Dict[int, List[CircuitFeature]]) -> BeliefState:
        """Initialize strategy beliefs (compatibility method)."""
        return self.get_current_beliefs()
    
    def calculate_expected_free_energy(self, feature: CircuitFeature, 
                                     intervention_type: InterventionType) -> float:
        """Calculate EFE for strategy selection."""
        try:
            # Use agent's policy inference
            if hasattr(self.agent, 'qs'):
                q_pi, G = self.agent.infer_policies()
                return float(np.mean(G))
            return 0.0
        except:
            return 0.0
    
    def update_beliefs(self, intervention_result: InterventionResult) -> CorrespondenceMetrics:
        """Update strategy beliefs and calculate correspondence."""
        
        # Encode and update beliefs
        obs = self.encode_intervention_outcome(intervention_result)
        self.agent.infer_states(obs)
        
        # Update strategy tracking
        self.strategy_beliefs = self.track_strategy_beliefs()
        
        # Calculate strategy correspondence
        if hasattr(self, 'strategy_success_rates'):
            correspondence = self.calculate_strategy_correspondence(
                self.strategy_beliefs, self.strategy_success_rates
            )
        else:
            correspondence = 0.0
        
        return CorrespondenceMetrics(
            belief_updating_correspondence=correspondence,
            precision_weighting_correspondence=correspondence,
            prediction_error_correspondence=correspondence,
            overall_correspondence=correspondence,
            circuit_attention_patterns=[],
            ai_precision_patterns=[],
            circuit_prediction_errors=[],
            ai_prediction_errors=[]
        )
    
    def select_intervention(self, available_features: List[CircuitFeature]) -> Tuple[CircuitFeature, InterventionType]:
        """Select intervention using strategy learning."""
        return self.select_intervention_strategy(available_features)
    


    def get_feature_confidence(self, feature_id: int) -> float:
        """Get confidence score for a specific feature based on strategy learning."""
        try:
            # Strategy learning focuses on overall confidence rather than individual features
            if hasattr(self.agent, 'qs'):
                # Calculate confidence from belief entropy
                entropies = [entropy(qs) for qs in self.agent.qs]
                avg_entropy = np.mean(entropies)
                max_entropy = np.log(max([len(qs) for qs in self.agent.qs]))
                confidence = 1.0 - (avg_entropy / max_entropy)
                return float(confidence)
            else:
                return 0.5  # Default confidence
        except Exception as e:
            logger.warning(f'Feature confidence calculation failed: {e}')
            return 0.5
    def initialize_from_circuit_features(self, features) -> None:
        """Initialize strategy learning from discovered circuit features."""
        if isinstance(features, dict):
            total_features = sum(len(f_list) for f_list in features.values())
            self.discovered_features = features
        else:
            total_features = len(features)
            self.discovered_features = features
        
        logger.info(f'Initializing strategy learning from {total_features} features')
        
        # Initialize strategy success tracking
        self.strategy_success_rates = {}
        
        logger.info('Strategy learning initialized from circuit features')

    def get_belief_summary(self) -> Dict[str, Any]:
        """Get summary of current strategy beliefs."""
        try:
            current_beliefs = self.get_current_beliefs()
            strategy_beliefs = self.track_strategy_beliefs()
            
            return {
                'total_strategies': self.num_strategies,
                'strategy_beliefs': strategy_beliefs,
                'confidence': current_beliefs.confidence,
                'uncertainty': current_beliefs.uncertainty,
                'belief_history_length': len(self.belief_history),
                'intervention_history_length': len(self.intervention_history),
                'converged': self.check_convergence()
            }
        except Exception as e:
            logger.warning(f'Belief summary failed: {e}')
            return {
                'total_strategies': self.num_strategies,
                'strategy_beliefs': {},
                'confidence': 0.5,
                'uncertainty': {},
                'belief_history_length': 0,
                'intervention_history_length': 0,
                'converged': False
            }
    def generate_novel_predictions(self) -> List[NovelPrediction]:
        """Generate novel predictions about strategies."""
        return self.generate_predictions()
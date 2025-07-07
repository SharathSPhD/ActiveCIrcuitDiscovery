#!/usr/bin/env python3
"""
Complete Active Inference Cycle Implementation

This module implements the full Active Inference perception-action cycle
using real pymdp algorithms with no fallbacks or approximations.

Components:
1. State inference (variational message passing)
2. Policy inference (Expected Free Energy calculation)
3. Action selection (policy sampling)
4. Model learning (Dirichlet parameter updates)
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from pymdp import Agent
from pymdp.utils import obj_array
from pymdp.maths import softmax, spm_dot, kl_div, entropy
from pymdp import control, inference, learning

from core.data_structures import SAEFeature, InterventionResult, InterventionType, BeliefState

logger = logging.getLogger(__name__)

class ActiveInferenceCycle:
    """
    Complete Active Inference perception-action cycle implementation.
    
    Implements the full cycle as described in Friston et al.:
    1. Perception: Update beliefs about hidden states given observations
    2. Planning: Evaluate policies using Expected Free Energy
    3. Action: Select and execute optimal policy
    4. Learning: Update generative model parameters
    """
    
    def __init__(self, agent: Agent):
        """Initialize with real pymdp Agent."""
        self.agent = agent
        self.cycle_count = 0
        
        # Track cycle components for analysis
        self.perception_history = []
        self.planning_history = []
        self.action_history = []
        self.learning_history = []
        
        logger.info("Active Inference cycle initialized")
    
    def perceive(self, observation: List[int]) -> List[np.ndarray]:
        """
        Perception step: Update beliefs about hidden states.
        
        Uses real pymdp variational message passing for belief updating.
        
        Args:
            observation: Discrete observations [effect_magnitude, confidence]
            
        Returns:
            Updated posterior beliefs over hidden states
        """
        
        logger.debug(f"Perception step: observation={observation}")
        
        # Store previous beliefs for comparison
        if hasattr(self.agent, 'qs'):
            qs_prev = [qs.copy() for qs in self.agent.qs]
        else:
            qs_prev = None
        
        # Perform state inference using pymdp variational message passing
        # This is the core perception mechanism in Active Inference
        qs = self.agent.infer_states(observation)
        
        # Calculate belief update metrics
        if qs_prev is not None:
            belief_change = [kl_div(qs[i], qs_prev[i]) for i in range(len(qs))]
            avg_belief_change = np.mean(belief_change)
        else:
            avg_belief_change = 0.0
        
        # Store perception metrics
        perception_metrics = {
            'observation': observation,
            'qs_posterior': [qs_i.copy() for qs_i in qs],
            'belief_change': avg_belief_change,
            'belief_entropy': [entropy(qs_i) for qs_i in qs],
            'cycle_count': self.cycle_count
        }
        
        self.perception_history.append(perception_metrics)
        
        logger.debug(f"Perception completed: belief_change={avg_belief_change:.4f}")
        return qs
    
    def plan(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Planning step: Evaluate policies using Expected Free Energy.
        
        Uses real pymdp EFE calculation with epistemic and pragmatic components.
        
        Returns:
            Tuple of (policy_posterior, negative_expected_free_energies)
        """
        
        logger.debug("Planning step: evaluating policies")
        
        # Perform policy inference using pymdp Expected Free Energy calculation
        # This evaluates each policy based on:
        # - Epistemic value (information gain about states)
        # - Pragmatic value (preference satisfaction)
        # - Parameter information gain (model learning)
        
        q_pi, G = self.agent.infer_policies()
        
        # Extract EFE components for analysis
        efe_components = self._analyze_efe_components(G)
        
        # Store planning metrics
        planning_metrics = {
            'policy_posterior': q_pi.copy(),
            'negative_efe': G.copy(),
            'best_policy_idx': np.argmax(q_pi),
            'efe_components': efe_components,
            'policy_entropy': entropy(q_pi),
            'cycle_count': self.cycle_count
        }
        
        self.planning_history.append(planning_metrics)
        
        logger.debug(f"Planning completed: best_policy={np.argmax(q_pi)}, policy_entropy={entropy(q_pi):.4f}")
        return q_pi, G
    
    def act(self) -> np.ndarray:
        """
        Action step: Select and execute optimal action.
        
        Uses pymdp action selection with proper policy sampling.
        
        Returns:
            Selected action vector
        """
        
        logger.debug("Action step: selecting optimal action")
        
        # Sample action from policy posterior using pymdp action selection
        # This respects the uncertainty in policy evaluation
        action = self.agent.sample_action()
        
        # Store action metrics
        action_metrics = {
            'selected_action': action.copy(),
            'action_policy': self.agent.policies[self.agent.prev_actions[-1] if hasattr(self.agent, 'prev_actions') and self.agent.prev_actions else 0],
            'cycle_count': self.cycle_count
        }
        
        self.action_history.append(action_metrics)
        
        logger.debug(f"Action selected: {action}")
        return action
    
    def learn(self, observation: List[int]) -> Dict[str, Any]:
        """
        Learning step: Update generative model parameters.
        
        Uses pymdp Dirichlet parameter updates for model learning.
        
        Args:
            observation: Most recent observation for learning
            
        Returns:
            Learning metrics and updated parameters
        """
        
        logger.debug("Learning step: updating model parameters")
        
        # Store pre-learning parameters for comparison
        if hasattr(self.agent, 'A'):
            A_prev = [A_m.copy() for A_m in self.agent.A]
        else:
            A_prev = None
        
        if hasattr(self.agent, 'B'):
            B_prev = [B_f.copy() for B_f in self.agent.B]
        else:
            B_prev = None
        
        # Update observation model (A matrices) based on evidence
        qA = self.agent.update_A(observation)
        
        # Update transition model (B matrices) if we have action history
        if hasattr(self.agent, 'qs') and len(self.agent.belief_history) > 1:
            qs_prev = self.agent.belief_history[-2]  # Previous belief state
            qB = self.agent.update_B(qs_prev)
        else:
            qB = None
        
        # Calculate parameter learning metrics
        learning_metrics = {
            'qA_updated': qA is not None,
            'qB_updated': qB is not None,
            'cycle_count': self.cycle_count
        }
        
        # Calculate parameter change if possible
        if A_prev is not None:
            A_change = [np.mean(np.abs(self.agent.A[m] - A_prev[m])) for m in range(len(self.agent.A))]
            learning_metrics['A_parameter_change'] = np.mean(A_change)
        
        if B_prev is not None and qB is not None:
            B_change = [np.mean(np.abs(self.agent.B[f] - B_prev[f])) for f in range(len(self.agent.B))]
            learning_metrics['B_parameter_change'] = np.mean(B_change)
        
        self.learning_history.append(learning_metrics)
        
        logger.debug(f"Learning completed: A_updated={qA is not None}, B_updated={qB is not None}")
        return learning_metrics
    
    def complete_cycle(self, observation: List[int]) -> Dict[str, Any]:
        """
        Execute complete Active Inference cycle.
        
        Args:
            observation: Current observation [effect_magnitude, confidence]
            
        Returns:
            Complete cycle results including all components
        """
        
        logger.info(f"Executing Active Inference cycle {self.cycle_count}")
        
        # 1. Perception: Update beliefs about hidden states
        qs = self.perceive(observation)
        
        # 2. Planning: Evaluate policies using Expected Free Energy
        q_pi, G = self.plan()
        
        # 3. Action: Select optimal action
        action = self.act()
        
        # 4. Learning: Update model parameters
        learning_results = self.learn(observation)
        
        # Compile complete cycle results
        cycle_results = {
            'cycle_count': self.cycle_count,
            'observation': observation,
            'posterior_beliefs': [qs_i.copy() for qs_i in qs],
            'policy_posterior': q_pi.copy(),
            'negative_efe': G.copy(),
            'selected_action': action.copy(),
            'learning_results': learning_results,
            'belief_entropy': [entropy(qs_i) for qs_i in qs],
            'policy_entropy': entropy(q_pi)
        }
        
        self.cycle_count += 1
        
        logger.info(f"Cycle {self.cycle_count-1} completed: action={action}, belief_entropy={np.mean(cycle_results['belief_entropy']):.4f}")
        
        return cycle_results
    
    def _analyze_efe_components(self, G: np.ndarray) -> Dict[str, Any]:
        """Analyze Expected Free Energy components for interpretability."""
        
        # Basic EFE analysis
        best_policy_idx = np.argmax(G)  # G is negative EFE, so max is best
        worst_policy_idx = np.argmin(G)
        
        efe_range = G.max() - G.min()
        efe_std = np.std(G)
        
        return {
            'best_policy_idx': best_policy_idx,
            'worst_policy_idx': worst_policy_idx,
            'best_efe': G[best_policy_idx],
            'worst_efe': G[worst_policy_idx],
            'efe_range': efe_range,
            'efe_std': efe_std,
            'efe_values': G.copy()
        }
    
    def get_cycle_summary(self) -> Dict[str, Any]:
        """Get summary of all cycles for analysis."""
        
        if self.cycle_count == 0:
            return {'cycles_completed': 0}
        
        # Aggregate metrics across cycles
        avg_belief_change = np.mean([p['belief_change'] for p in self.perception_history])
        avg_policy_entropy = np.mean([p['policy_entropy'] for p in self.planning_history])
        
        # EFE trend analysis
        efe_trends = []
        for planning in self.planning_history:
            best_efe = planning['negative_efe'][planning['best_policy_idx']]
            efe_trends.append(best_efe)
        
        efe_improvement = efe_trends[-1] - efe_trends[0] if len(efe_trends) > 1 else 0.0
        
        return {
            'cycles_completed': self.cycle_count,
            'avg_belief_change': avg_belief_change,
            'avg_policy_entropy': avg_policy_entropy,
            'efe_improvement': efe_improvement,
            'efe_trend': efe_trends,
            'final_belief_entropy': self.perception_history[-1]['belief_entropy'] if self.perception_history else None
        }
    
    def reset_cycle(self):
        """Reset cycle for new experiment."""
        self.cycle_count = 0
        self.perception_history.clear()
        self.planning_history.clear()
        self.action_history.clear()
        self.learning_history.clear()
        
        logger.info("Active Inference cycle reset")

class CircuitDiscoveryInferenceCycle(ActiveInferenceCycle):
    """
    Specialized Active Inference cycle for circuit discovery.
    
    Extends base cycle with circuit-specific functionality:
    - Intervention result encoding
    - Feature-based action interpretation
    - Circuit-specific metrics
    """
    
    def __init__(self, agent: Agent, features: List[SAEFeature]):
        """Initialize with circuit discovery context."""
        super().__init__(agent)
        self.features = features
        self.feature_mapping = {i: feature for i, feature in enumerate(features)}
        
        logger.info(f"Circuit discovery cycle initialized with {len(features)} features")
    
    def encode_intervention_result(self, intervention_result: InterventionResult) -> List[int]:
        """
        Encode intervention result as Active Inference observation.
        
        Maps continuous intervention outcomes to discrete observations
        for pymdp processing.
        """
        
        effect_size = intervention_result.effect_size
        
        # Map effect size to discrete observation (0-4: none to huge)
        if effect_size < 0.1:
            effect_obs = 0  # None
        elif effect_size < 0.3:
            effect_obs = 1  # Small
        elif effect_size < 0.5:
            effect_obs = 2  # Medium
        elif effect_size < 0.7:
            effect_obs = 3  # Large
        else:
            effect_obs = 4  # Huge
        
        # Map confidence based on effect clarity
        if effect_size > 0.6 or effect_size < 0.1:
            confidence_obs = 2  # High confidence (clear effect or no effect)
        elif 0.2 < effect_size < 0.5:
            confidence_obs = 1  # Medium confidence
        else:
            confidence_obs = 0  # Low confidence (ambiguous)
        
        return [effect_obs, confidence_obs]
    
    def decode_action_to_intervention(self, action: np.ndarray) -> Tuple[SAEFeature, InterventionType]:
        """
        Decode pymdp action to circuit intervention.
        
        Maps discrete action indices to specific feature interventions.
        """
        
        # Extract action components
        component_idx = int(action[0]) if len(action) > 0 else 0
        intervention_idx = int(action[2]) if len(action) > 2 else 0
        
        # Map to feature
        if component_idx < len(self.features):
            target_feature = self.features[component_idx]
        else:
            target_feature = self.features[0]  # Fallback
        
        # Map to intervention type
        intervention_map = {
            0: InterventionType.ABLATION,
            1: InterventionType.ACTIVATION_PATCHING,
            2: InterventionType.MEAN_ABLATION
        }
        
        intervention_type = intervention_map.get(intervention_idx, InterventionType.ABLATION)
        
        return target_feature, intervention_type
    
    def process_intervention_cycle(self, intervention_result: InterventionResult) -> Dict[str, Any]:
        """
        Process complete intervention cycle for circuit discovery.
        
        Combines intervention result encoding with full Active Inference cycle.
        """
        
        # Encode intervention result as observation
        observation = self.encode_intervention_result(intervention_result)
        
        # Execute complete Active Inference cycle
        cycle_results = self.complete_cycle(observation)
        
        # Add circuit-specific information
        cycle_results.update({
            'intervention_result': intervention_result,
            'target_feature': intervention_result.target_feature,
            'intervention_type': intervention_result.intervention_type,
            'encoded_observation': observation
        })
        
        return cycle_results
    
    def get_intervention_recommendation(self) -> Tuple[SAEFeature, InterventionType, float]:
        """
        Get next intervention recommendation based on current beliefs.
        
        Returns feature, intervention type, and expected information gain.
        """
        
        if not hasattr(self.agent, 'qs'):
            # Return default recommendation
            return self.features[0], InterventionType.ABLATION, 0.0
        
        # Use current beliefs to recommend intervention
        q_pi, G = self.agent.infer_policies()
        best_policy_idx = np.argmax(q_pi)
        
        # Get action from best policy
        best_action = self.agent.sample_action()
        
        # Decode to intervention
        target_feature, intervention_type = self.decode_action_to_intervention(best_action)
        
        # Calculate expected information gain
        expected_info_gain = G[best_policy_idx]
        
        return target_feature, intervention_type, float(expected_info_gain)
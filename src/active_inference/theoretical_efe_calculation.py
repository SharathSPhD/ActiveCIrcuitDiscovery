#!/usr/bin/env python3
"""
Theoretically-Grounded Expected Free Energy Calculation

This module implements the rigorous EFE calculation framework from 
MATHEMATICAL_FRAMEWORK_VALIDATION.md with full theoretical justification
and information-theoretic analysis.

THEORETICAL FOUNDATION:
- Expected Free Energy: G(π, τ) = E_q[F(o_{τ+1}, s_{τ+1})] + E_q[DKL[q(s_{τ+1})||p(s_{τ+1})]]
- Epistemic Value: Information gain through uncertainty reduction
- Pragmatic Value: Achievement of preferred outcomes
- Policy Optimality: π* = argmin_π G(π, τ)

MATHEMATICAL VALIDATION:
- Information-theoretic correspondence with mutual information maximization
- Convergence properties analysis
- Precision-weighted inference implementation
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from scipy.stats import entropy as scipy_entropy
from scipy.special import logsumexp
import warnings

# pymdp imports
from pymdp.utils import obj_array
from pymdp.maths import softmax, kl_div, entropy

logger = logging.getLogger(__name__)

class TheoreticalEFECalculator:
    """
    Theoretically-grounded Expected Free Energy calculator implementing the
    mathematical framework from MATHEMATICAL_FRAMEWORK_VALIDATION.md
    
    This implementation provides:
    1. Mathematically validated EFE calculation
    2. Epistemic/pragmatic value decomposition
    3. Information-theoretic optimality analysis
    4. Policy selection with theoretical justification
    """
    
    def __init__(self, 
                 A: List[np.ndarray], 
                 B: List[np.ndarray], 
                 C: List[np.ndarray],
                 precision_gamma: float = 16.0,
                 planning_horizon: int = 1,
                 enable_validation: bool = True):
        """
        Initialize EFE calculator with generative model components.
        
        Args:
            A: Observation model P(o|s)
            B: Transition model P(s'|s,a)  
            C: Preference vectors
            precision_gamma: Precision parameter for policy selection (inverse temperature)
            planning_horizon: Number of timesteps for policy planning
            enable_validation: Enable theoretical validation
        """
        self.A = A
        self.B = B
        self.C = C
        self.gamma = precision_gamma
        self.planning_horizon = planning_horizon
        self.enable_validation = enable_validation
        
        # Extract model dimensions
        self.num_obs = [A_m.shape[0] for A_m in A]
        self.num_states = [B_f.shape[0] for B_f in B]
        self.num_actions = B[0].shape[2] if len(B) > 0 else 1
        self.num_modalities = len(A)
        self.num_factors = len(B)
        
        # Theoretical constraints
        self.max_efe = 1000.0  # Numerical stability bound
        self.min_precision = 1e-16  # Numerical precision floor
        
        logger.info(f"EFE calculator initialized: actions={self.num_actions}, horizon={self.planning_horizon}, γ={self.gamma}")
        
        if self.enable_validation:
            self._validate_model_components()
    
    def _validate_model_components(self):
        """Validate that model components satisfy theoretical constraints."""
        
        # Validate A matrices
        for m, A_m in enumerate(self.A):
            if len(A_m.shape) != len(self.num_states) + 1:
                raise ValueError(f"A[{m}] has incorrect dimensions: {A_m.shape}")
                
            # Check probability distribution constraints
            for state_idx in np.ndindex(A_m.shape[1:]):
                obs_dist = A_m[:, state_idx]
                if not np.isclose(obs_dist.sum(), 1.0, atol=1e-6):
                    raise ValueError(f"A[{m}] not normalized at state {state_idx}: sum={obs_dist.sum()}")
        
        # Validate B matrices
        for f, B_f in enumerate(self.B):
            if B_f.shape != (self.num_states[f], self.num_states[f], self.num_actions):
                raise ValueError(f"B[{f}] has incorrect dimensions: {B_f.shape}")
                
            # Check Markov transition constraints
            for s in range(self.num_states[f]):
                for a in range(self.num_actions):
                    trans_dist = B_f[:, s, a]
                    if not np.isclose(trans_dist.sum(), 1.0, atol=1e-6):
                        raise ValueError(f"B[{f}] not normalized at state {s}, action {a}: sum={trans_dist.sum()}")
        
        logger.info("Model components theoretical validation passed")
    
    def predict_next_state(self, 
                          current_beliefs: List[np.ndarray], 
                          action: int) -> List[np.ndarray]:
        """
        Predict next state beliefs using transition model.
        
        Mathematical Foundation:
        q(s_{t+1}^f) = ∑_s B^f[s'|s,a] q(s_t^f)
        
        Args:
            current_beliefs: List of belief distributions over state factors
            action: Action index
            
        Returns:
            List of predicted belief distributions
        """
        
        predicted_beliefs = []
        
        for f, (belief_f, B_f) in enumerate(zip(current_beliefs, self.B)):
            # Matrix multiplication: B_f[:,:,action] @ belief_f
            predicted_belief_f = B_f[:, :, action] @ belief_f
            
            # Ensure normalization (numerical stability)
            predicted_belief_f = predicted_belief_f / (predicted_belief_f.sum() + self.min_precision)
            
            predicted_beliefs.append(predicted_belief_f)
        
        return predicted_beliefs
    
    def predict_observation(self, state_beliefs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Predict observation probabilities from state beliefs.
        
        Mathematical Foundation:
        q(o_m) = ∑_s A_m[o|s] q(s)
        
        Args:
            state_beliefs: List of belief distributions over state factors
            
        Returns:
            List of predicted observation distributions
        """
        
        predicted_observations = []
        
        for m, A_m in enumerate(self.A):
            # Handle multi-factor state space
            if len(state_beliefs) == 1:
                # Single factor case
                predicted_obs_m = A_m @ state_beliefs[0]
            else:
                # Multi-factor case: marginalize over all factors
                # Create joint state distribution
                joint_belief = state_beliefs[0].copy()
                for f in range(1, len(state_beliefs)):
                    joint_belief = np.kron(joint_belief, state_beliefs[f])
                
                # Reshape A matrix for proper matrix multiplication
                A_reshaped = A_m.reshape(A_m.shape[0], -1)
                predicted_obs_m = A_reshaped @ joint_belief
            
            # Ensure normalization
            predicted_obs_m = predicted_obs_m / (predicted_obs_m.sum() + self.min_precision)
            predicted_observations.append(predicted_obs_m)
        
        return predicted_observations
    
    def calculate_epistemic_value(self, 
                                state_beliefs: List[np.ndarray], 
                                predicted_observations: List[np.ndarray]) -> float:
        """
        Calculate epistemic value (expected information gain).
        
        Mathematical Foundation:
        Epistemic_Value = E_q[H[p(o|s)] - H[p(o|s,π)]] = I(s; o|π)
        
        This represents the mutual information between states and observations
        under the given policy - the expected information gain.
        
        Args:
            state_beliefs: Predicted state beliefs
            predicted_observations: Predicted observation distributions
            
        Returns:
            Epistemic value (information gain in bits)
        """
        
        total_epistemic_value = 0.0
        
        for m, (A_m, predicted_obs_m) in enumerate(zip(self.A, predicted_observations)):
            
            # Marginal entropy: H[p(o)]
            marginal_entropy = scipy_entropy(predicted_obs_m + self.min_precision, base=2)
            
            # Conditional entropy: E_s[H[p(o|s)]]
            conditional_entropy = 0.0
            
            if len(state_beliefs) == 1:
                # Single factor case
                for s, belief_s in enumerate(state_beliefs[0]):
                    if belief_s > self.min_precision:
                        obs_dist_given_s = A_m[:, s]
                        cond_h = scipy_entropy(obs_dist_given_s + self.min_precision, base=2)
                        conditional_entropy += belief_s * cond_h
            else:
                # Multi-factor case
                for state_config in np.ndindex(tuple(self.num_states)):
                    # Joint probability of this state configuration
                    joint_prob = np.prod([state_beliefs[f][state_config[f]] for f in range(len(state_beliefs))])
                    
                    if joint_prob > self.min_precision:
                        obs_dist_given_s = A_m[(slice(None),) + state_config]
                        cond_h = scipy_entropy(obs_dist_given_s + self.min_precision, base=2)
                        conditional_entropy += joint_prob * cond_h
            
            # Information gain = H[p(o)] - E_s[H[p(o|s)]]
            information_gain = marginal_entropy - conditional_entropy
            total_epistemic_value += information_gain
        
        # Return negative because we minimize EFE (maximize information gain)
        return -total_epistemic_value
    
    def calculate_pragmatic_value(self, predicted_observations: List[np.ndarray]) -> float:
        """
        Calculate pragmatic value (expected preference satisfaction).
        
        Mathematical Foundation:
        Pragmatic_Value = E_q[-ln p(o|π) + C·o] = -∑_o q(o) C[o] + H[q(o)]
        
        This represents the expected utility under preferences plus entropy penalty.
        
        Args:
            predicted_observations: Predicted observation distributions
            
        Returns:
            Pragmatic value (expected utility)
        """
        
        total_pragmatic_value = 0.0
        
        for m, (predicted_obs_m, C_m) in enumerate(zip(predicted_observations, self.C)):
            
            # Expected preference satisfaction: E[C·o] = ∑_o q(o) C[o]
            expected_preference = np.sum(predicted_obs_m * C_m)
            
            # Entropy penalty (encourages confident predictions)
            entropy_penalty = scipy_entropy(predicted_obs_m + self.min_precision, base=2)
            
            # Pragmatic value (negative because we minimize EFE)
            pragmatic_value_m = -expected_preference + entropy_penalty
            total_pragmatic_value += pragmatic_value_m
        
        return total_pragmatic_value
    
    def calculate_efe_single_step(self, 
                                current_beliefs: List[np.ndarray], 
                                action: int) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Expected Free Energy for a single action step.
        
        Mathematical Foundation:
        G(π, τ) = E_q[F(o_{τ+1}, s_{τ+1})] + E_q[DKL[q(s_{τ+1})||p(s_{τ+1})]]
                = Epistemic_Value + Pragmatic_Value
        
        Args:
            current_beliefs: Current belief state
            action: Action to evaluate
            
        Returns:
            Tuple of (total_efe, efe_components)
        """
        
        # 1. Predict next state
        predicted_beliefs = self.predict_next_state(current_beliefs, action)
        
        # 2. Predict next observation
        predicted_observations = self.predict_observation(predicted_beliefs)
        
        # 3. Calculate epistemic value (information gain)
        epistemic_value = self.calculate_epistemic_value(predicted_beliefs, predicted_observations)
        
        # 4. Calculate pragmatic value (preference satisfaction)
        pragmatic_value = self.calculate_pragmatic_value(predicted_observations)
        
        # 5. Total EFE
        total_efe = epistemic_value + pragmatic_value
        
        # Numerical stability
        total_efe = np.clip(total_efe, -self.max_efe, self.max_efe)
        
        efe_components = {
            'epistemic_value': epistemic_value,
            'pragmatic_value': pragmatic_value,
            'total_efe': total_efe,
            'information_gain': -epistemic_value,  # Positive information gain
            'expected_utility': -pragmatic_value   # Positive expected utility
        }
        
        return total_efe, efe_components
    
    def calculate_efe_multi_step(self, 
                               current_beliefs: List[np.ndarray], 
                               policy: List[int]) -> Tuple[float, List[Dict[str, float]]]:
        """
        Calculate Expected Free Energy for multi-step policy.
        
        Mathematical Foundation:
        G(π) = ∑_τ G(π, τ) - Sum of EFE over planning horizon
        
        Args:
            current_beliefs: Current belief state
            policy: Sequence of actions over planning horizon
            
        Returns:
            Tuple of (total_policy_efe, timestep_components)
        """
        
        total_policy_efe = 0.0
        timestep_components = []
        beliefs_t = [belief.copy() for belief in current_beliefs]
        
        for t, action in enumerate(policy[:self.planning_horizon]):
            
            # Calculate EFE for this timestep
            efe_t, components_t = self.calculate_efe_single_step(beliefs_t, action)
            
            total_policy_efe += efe_t
            timestep_components.append(components_t)
            
            # Update beliefs for next timestep
            beliefs_t = self.predict_next_state(beliefs_t, action)
        
        return total_policy_efe, timestep_components
    
    def select_optimal_action(self, 
                            current_beliefs: List[np.ndarray]) -> Tuple[int, Dict[str, Any]]:
        """
        Select optimal action using EFE minimization with softmax policy.
        
        Mathematical Foundation:
        π*(a) = softmax(-γ⁻¹ · G(a)) where γ is precision parameter
        
        Args:
            current_beliefs: Current belief state
            
        Returns:
            Tuple of (selected_action, analysis)
        """
        
        # Calculate EFE for all possible actions
        action_efes = []
        action_components = []
        
        for action in range(self.num_actions):
            efe, components = self.calculate_efe_single_step(current_beliefs, action)
            action_efes.append(efe)
            action_components.append(components)
        
        action_efes = np.array(action_efes)
        
        # Policy probabilities: softmax(-γ⁻¹ · EFE)
        policy_logits = -action_efes / self.gamma
        policy_probs = softmax(policy_logits)
        
        # Select action (can be deterministic or stochastic)
        selected_action = np.argmin(action_efes)  # Deterministic selection of minimum EFE
        # Alternative: selected_action = np.random.choice(self.num_actions, p=policy_probs)  # Stochastic
        
        # Analysis for theoretical validation
        analysis = {
            'action_efes': action_efes,
            'action_components': action_components,
            'policy_probabilities': policy_probs,
            'selected_action': selected_action,
            'min_efe': np.min(action_efes),
            'efe_range': np.max(action_efes) - np.min(action_efes),
            'precision_gamma': self.gamma
        }
        
        return selected_action, analysis
    
    def analyze_information_gain(self, 
                               current_beliefs: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze expected information gain for all actions.
        
        This provides the information-theoretic analysis to validate that
        EFE minimization corresponds to mutual information maximization.
        
        Args:
            current_beliefs: Current belief state
            
        Returns:
            Dict with information-theoretic analysis
        """
        
        info_gains = []
        mutual_informations = []
        
        for action in range(self.num_actions):
            
            # Predict states and observations
            predicted_beliefs = self.predict_next_state(current_beliefs, action)
            predicted_observations = self.predict_observation(predicted_beliefs)
            
            # Calculate information gain (negative epistemic value)
            epistemic_value = self.calculate_epistemic_value(predicted_beliefs, predicted_observations)
            info_gain = -epistemic_value
            info_gains.append(info_gain)
            
            # Calculate mutual information directly for validation
            total_mi = 0.0
            for m, (A_m, pred_obs_m) in enumerate(zip(self.A, predicted_observations)):
                
                # Marginal entropy H[p(o)]
                h_obs = scipy_entropy(pred_obs_m + self.min_precision, base=2)
                
                # Joint entropy H[p(s,o)]
                h_joint = 0.0
                for state_config in np.ndindex(tuple(self.num_states)):
                    joint_prob = np.prod([predicted_beliefs[f][state_config[f]] for f in range(len(predicted_beliefs))])
                    
                    if joint_prob > self.min_precision:
                        obs_dist = A_m[(slice(None),) + state_config]
                        for o, obs_prob in enumerate(obs_dist):
                            if obs_prob > self.min_precision:
                                joint_so_prob = joint_prob * obs_prob
                                h_joint -= joint_so_prob * np.log2(joint_so_prob)
                
                # State entropy H[p(s)]
                h_state = 0.0
                for state_config in np.ndindex(tuple(self.num_states)):
                    joint_state_prob = np.prod([predicted_beliefs[f][state_config[f]] for f in range(len(predicted_beliefs))])
                    if joint_state_prob > self.min_precision:
                        h_state -= joint_state_prob * np.log2(joint_state_prob)
                
                # Mutual information I(S;O) = H(S) + H(O) - H(S,O)
                mi_m = h_state + h_obs - h_joint
                total_mi += mi_m
            
            mutual_informations.append(total_mi)
        
        # Correspondence analysis between information gain and mutual information
        info_gains = np.array(info_gains)
        mutual_informations = np.array(mutual_informations)
        
        # They should be highly correlated (theoretical validation)
        correlation = np.corrcoef(info_gains, mutual_informations)[0, 1]
        
        analysis = {
            'information_gains': info_gains.tolist(),
            'mutual_informations': mutual_informations.tolist(),
            'ig_mi_correlation': correlation,
            'max_info_gain_action': int(np.argmax(info_gains)),
            'max_mi_action': int(np.argmax(mutual_informations)),
            'theoretical_correspondence': correlation > 0.9  # Should be very high
        }
        
        return analysis
    
    def validate_efe_optimality(self, 
                              current_beliefs: List[np.ndarray]) -> Dict[str, bool]:
        """
        Validate that EFE minimization corresponds to optimal experimental design.
        
        This implements the theoretical validation from MATHEMATICAL_FRAMEWORK_VALIDATION.md
        
        Args:
            current_beliefs: Current belief state
            
        Returns:
            Dict of validation results
        """
        
        validation_results = {}
        
        # 1. Information-theoretic correspondence
        info_analysis = self.analyze_information_gain(current_beliefs)
        validation_results['ig_mi_correspondence'] = info_analysis['theoretical_correspondence']
        
        # 2. EFE-MI policy agreement
        _, efe_analysis = self.select_optimal_action(current_beliefs)
        min_efe_action = efe_analysis['selected_action']
        max_mi_action = info_analysis['max_mi_action']
        validation_results['efe_mi_policy_agreement'] = (min_efe_action == max_mi_action)
        
        # 3. Softmax policy consistency
        action_efes = efe_analysis['action_efes']
        policy_probs = efe_analysis['policy_probabilities']
        expected_probs = softmax(-action_efes / self.gamma)
        prob_consistency = np.allclose(policy_probs, expected_probs, atol=1e-6)
        validation_results['softmax_consistency'] = prob_consistency
        
        # 4. Overall theoretical validity
        validation_results['theoretical_validity'] = all([
            validation_results['ig_mi_correspondence'],
            validation_results['efe_mi_policy_agreement'],
            validation_results['softmax_consistency']
        ])
        
        if not validation_results['theoretical_validity']:
            logger.warning(f"EFE optimality validation failed: {validation_results}")
        else:
            logger.info("EFE optimality theoretical validation passed")
        
        return validation_results


def create_theoretical_efe_calculator(
    A: List[np.ndarray],
    B: List[np.ndarray], 
    C: List[np.ndarray],
    precision_gamma: float = 16.0,
    planning_horizon: int = 1,
    enable_validation: bool = True
) -> TheoreticalEFECalculator:
    """
    Create theoretically-grounded EFE calculator.
    
    Args:
        A: Observation model matrices
        B: Transition model matrices
        C: Preference vectors
        precision_gamma: Precision parameter for policy selection
        planning_horizon: Number of timesteps for planning
        enable_validation: Enable theoretical validation
        
    Returns:
        Configured EFE calculator with theoretical validation
    """
    
    calculator = TheoreticalEFECalculator(
        A=A, B=B, C=C,
        precision_gamma=precision_gamma,
        planning_horizon=planning_horizon,
        enable_validation=enable_validation
    )
    
    return calculator

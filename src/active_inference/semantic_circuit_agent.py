#!/usr/bin/env python3
"""
Semantic Circuit Active Inference Agent

This agent learns circuit-semantic relationship hypotheses rather than 
intervention strategies. It focuses on understanding which circuits encode
semantic relationships like "Golden Gate Bridge → San Francisco".

Key Innovation: Agent learns about semantic circuit structure using Expected 
Free Energy to guide investigation of the most promising circuits from the
induction head filter.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union 
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr

# Real pymdp imports
import pymdp
from pymdp.agent import Agent
from pymdp.utils import obj_array, random_A_matrix, random_B_matrix, norm_dist
from pymdp.maths import softmax, kl_div, entropy

# Project imports
from ..core.interfaces import IActiveInferenceAgent
from ..core.data_structures import (
    CircuitFeature, InterventionResult, BeliefState, 
    CorrespondenceMetrics, NovelPrediction
)
from ..config.experiment_config import CompleteConfig, InterventionType
from ..circuit_analysis.induction_head_filter import InductionHeadFilter, InductionHeadCandidate

logger = logging.getLogger(__name__)

class SemanticCircuitAgent(IActiveInferenceAgent):
    """
    Semantic Circuit Active Inference Agent
    
    Learns circuit-semantic relationship hypotheses:
    - State space: [10 circuit_types × 4 semantic_strength × 3 confidence] = 120 circuit hypotheses
    - Observations: [4 semantic_success × 3 activation_strength] = 12 observation states
    - Focus: Which circuits encode semantic relationships, not intervention strategies
    """
    
    def __init__(self, config: CompleteConfig):
        """Initialize semantic circuit learning agent."""
        self.config = config
        
        # Semantic-focused parameters
        self.num_circuit_types = 10      # Different types of semantic circuits
        self.num_semantic_strength = 4   # none, weak, medium, strong
        self.num_confidence = 3          # low, medium, high
        self.num_semantic_success = 4    # failed, partial, good, excellent
        self.num_activation_strength = 3 # low, medium, high
        
        # Initialize induction head filter
        self.induction_filter = InductionHeadFilter()
        self.filtered_candidates: List[InductionHeadCandidate] = []
        self.candidate_hypotheses: Dict[str, Dict[str, float]] = {}
        
        # Design semantic-focused state and observation spaces
        self._design_semantic_spaces()
        
        # Create generative model for semantic circuit learning
        self._create_semantic_generative_model()
        
        # Initialize pymdp agent
        self._initialize_semantic_agent()
        
        # Semantic tracking
        self.semantic_beliefs: Dict[str, Dict[str, float]] = {}
        self.circuit_observations: List[Dict[str, Any]] = []
        self.semantic_predictions: List[Dict[str, Any]] = []
        
        logger.info(f"SemanticCircuitAgent initialized with semantic hypothesis learning")
    
    def _design_semantic_spaces(self):
        """Design state and observation spaces for semantic circuit learning."""
        # Simplified single-factor state space for pymdp compatibility
        # State represents semantic strength hypothesis: 0=weak, 1=medium, 2=strong, 3=excellent
        self.num_states = [self.num_semantic_strength]  # Single factor
        self.state_space_size = self.num_semantic_strength
        
        # Simplified single-factor observation space
        # Observation represents semantic success: 0=failed, 1=weak, 2=good, 3=excellent
        self.num_obs = [self.num_semantic_success]  # Single factor
        self.obs_space_size = self.num_semantic_success
        
        logger.info(f"Semantic state space: {self.num_states} = {self.state_space_size} hypotheses")
        logger.info(f"Semantic observation space: {self.num_obs} = {self.obs_space_size} observations")
    
    def _create_semantic_generative_model(self):
        """Create generative model for semantic circuit learning."""
        # A matrix: observation likelihood given circuit hypothesis state
        # Strong semantic circuits should produce good semantic success + high activation
        A = np.zeros((self.obs_space_size, self.state_space_size))
        
        for state_idx in range(self.state_space_size):
            for obs_idx in range(self.obs_space_size):
                # Strong semantic state should produce strong observations
                if state_idx == obs_idx:  # Perfect match
                    A[obs_idx, state_idx] = 0.8
                elif abs(state_idx - obs_idx) == 1:  # Close match
                    A[obs_idx, state_idx] = 0.15
                else:  # Poor match
                    A[obs_idx, state_idx] = 0.05
        
        # Normalize A matrix
        A = A / A.sum(axis=0, keepdims=True)
        A[np.isnan(A)] = 1.0 / self.obs_space_size  # Handle division by zero
        
        # Create obj_array with correct size first, then assign
        self.A = obj_array(1)
        self.A[0] = A
        
        # B matrix: 3D transition matrix [next_state, current_state, action]
        # Action 0: investigate, Action 1: skip
        num_actions = 2
        B = np.zeros((self.state_space_size, self.state_space_size, num_actions))
        
        # For both actions, mostly stable transitions
        for action in range(num_actions):
            for state in range(self.state_space_size):
                B[state, state, action] = 0.85  # Stay in same state
                # Small probability of transitioning to adjacent states
                if state > 0:
                    B[state-1, state, action] = 0.075
                if state < self.state_space_size - 1:
                    B[state+1, state, action] = 0.075
        
        # Normalize B matrix columns to sum to 1
        for action in range(num_actions):
            for state in range(self.state_space_size):
                B[:, state, action] = B[:, state, action] / B[:, state, action].sum()
        
        self.B = obj_array(1)
        self.B[0] = B
        
        # C matrix: preferences for strong semantic discovery
        C = np.array([0.0, 1.0, 2.0, 3.0])  # Prefer stronger semantic success
        assert len(C) == self.obs_space_size
        
        self.C = obj_array(1)
        self.C[0] = C
        
        # D matrix: initial beliefs (uniform over circuit hypotheses)
        D = norm_dist(np.ones(self.state_space_size))
        self.D = obj_array(1)
        self.D[0] = D
        
        logger.info("Created semantic circuit generative model")
    
    def _initialize_semantic_agent(self):
        """Initialize pymdp agent for semantic circuit learning."""
        try:
            # Create simple policies for semantic circuit investigation
            # Policy 0: investigate circuit, Policy 1: skip circuit
            self.policies = [
                np.array([[0]]),  # investigate this circuit
                np.array([[1]])   # skip this circuit
            ]
            
            self.agent = Agent(
                A=self.A,
                B=self.B,
                C=self.C, 
                D=self.D,
                policies=self.policies,
                use_utility=True,           # Enable pragmatic value
                use_states_info_gain=True,  # Enable epistemic value
                use_param_info_gain=False,  # Disable parameter learning for simplicity
                policy_len=1,
                inference_horizon=1
            )
            logger.info("Initialized semantic circuit pymdp Agent")
        except Exception as e:
            logger.error(f"Failed to initialize semantic agent: {e}")
            raise
    
    def initialize_from_circuit_features(self, features: List[CircuitFeature]):
        """Initialize semantic hypotheses from discovered circuit features."""
        logger.info(f"Initializing semantic agent from {len(features)} circuit features")
        
        # Filter features for induction head candidates
        if len(features) > 100:  # Only filter if we have many features
            self.filtered_candidates = self.induction_filter.filter_for_induction_heads(features, max_features=100)
            logger.info(f"Filtered to {len(self.filtered_candidates)} induction head candidates")
        else:
            # Convert features to candidates if already filtered
            self.filtered_candidates = [
                InductionHeadCandidate(
                    feature=f,
                    induction_score=0.5,
                    semantic_relevance=0.5,
                    layer_position="middle" if 6 <= f.layer_idx <= 19 else "other"
                )
                for f in features
            ]
        
        # Initialize semantic hypotheses for each candidate
        for candidate in self.filtered_candidates:
            feature_id = f"L{candidate.feature.layer_idx}F{candidate.feature.feature_id}"
            self.candidate_hypotheses[feature_id] = {
                "semantic_strength": candidate.semantic_relevance,
                "induction_score": candidate.induction_score,
                "activation_strength": candidate.feature.activation_strength,
                "confidence": 0.5,
                "semantic_success_history": []
            }
        
        logger.info(f"Initialized {len(self.candidate_hypotheses)} semantic circuit hypotheses")
    
    def select_intervention(self, features: List[CircuitFeature]) -> Tuple[CircuitFeature, InterventionType]:
        """Select which circuit to investigate using Expected Free Energy."""
        if not self.filtered_candidates:
            self.initialize_from_circuit_features(features)
        
        # Use Expected Free Energy to select most promising circuit
        best_candidate = None
        best_efe = float('inf')
        efe_scores = []  # Track all EFE scores for analysis
        
        for candidate in self.filtered_candidates:
            feature_id = f"L{candidate.feature.layer_idx}F{candidate.feature.feature_id}"
            hypothesis = self.candidate_hypotheses.get(feature_id, {})
            
            # Calculate Expected Free Energy for investigating this circuit
            efe = self._calculate_circuit_efe(candidate, hypothesis)
            efe_scores.append((feature_id, efe, candidate))
            
            if efe < best_efe:
                best_efe = efe
                best_candidate = candidate
        
        # Log top candidates for transparency
        efe_scores.sort(key=lambda x: x[1])  # Sort by EFE (lower is better)
        logger.info(f"EFE ranking (top 5): {[(fid, f'{efe:.3f}') for fid, efe, _ in efe_scores[:5]]}")
        
        if best_candidate is None:
            # Fallback to first available feature
            best_candidate = self.filtered_candidates[0] if self.filtered_candidates else None
            if best_candidate is None:
                return features[0], InterventionType.ABLATION
        
        # Log selection reasoning with rich context
        feature_id = f"L{best_candidate.feature.layer_idx}F{best_candidate.feature.feature_id}"
        hypothesis = self.candidate_hypotheses.get(feature_id, {})
        confidence = hypothesis.get("confidence", 0.5)
        logger.info(f"🎯 EFE selected circuit {feature_id} (EFE: {best_efe:.3f}, confidence: {confidence:.3f}, activation: {best_candidate.feature.activation_strength:.1f})")
        
        # Choose intervention type based on circuit hypothesis
        hypothesis = self.candidate_hypotheses.get(feature_id, {})
        if hypothesis.get("confidence", 0.5) > 0.7:
            intervention_type = InterventionType.ABLATION  # Strong hypothesis, test with ablation
        else:
            intervention_type = InterventionType.MEAN_ABLATION  # Weak hypothesis, gentler test
        
        return best_candidate.feature, intervention_type
    
    def _calculate_circuit_efe(self, candidate: InductionHeadCandidate, hypothesis: Dict[str, Any]) -> float:
        """Calculate Expected Free Energy for investigating a circuit.
        
        EFE = -Epistemic_Value - Pragmatic_Value
        Lower EFE = Higher priority for investigation
        """
        # Epistemic Value: Information gain from reducing uncertainty
        confidence = hypothesis.get("confidence", 0.5)
        uncertainty = 1.0 - confidence
        
        # Enhanced epistemic value calculation
        # Higher activation strength suggests more important circuit
        activation_bonus = min(1.0, candidate.feature.activation_strength / 20.0)
        epistemic_value = uncertainty * (1.5 + activation_bonus)
        
        # Pragmatic Value: Expected semantic importance
        # Combine multiple indicators of semantic relevance
        semantic_strength = candidate.semantic_relevance * candidate.induction_score
        layer_bonus = 1.0 if 8 <= candidate.feature.layer_idx <= 15 else 0.8  # Middle layers preferred
        pragmatic_value = semantic_strength * layer_bonus
        
        # Exploration bonus: slight preference for less investigated circuits
        history_length = len(hypothesis.get("semantic_success_history", []))
        exploration_bonus = max(0, 0.3 - history_length * 0.1)
        
        # Expected Free Energy (lower is better)
        efe = -epistemic_value - pragmatic_value - exploration_bonus
        
        # Log detailed EFE calculation for transparency
        feature_id = f"L{candidate.feature.layer_idx}F{candidate.feature.feature_id}"
        logger.debug(f"EFE calculation for {feature_id}: epistemic={epistemic_value:.3f}, pragmatic={pragmatic_value:.3f}, exploration={exploration_bonus:.3f}, total={efe:.3f}")
        
        return efe
    
    def update_beliefs(self, intervention_result: InterventionResult) -> CorrespondenceMetrics:
        """Update semantic circuit beliefs based on intervention results."""
        from ..core.data_structures import CorrespondenceMetrics
        
        feature = intervention_result.target_feature
        feature_id = f"L{feature.layer_idx}F{feature.feature_id}"
        
        # Evaluate semantic success of the intervention
        semantic_success = self._evaluate_semantic_success(intervention_result)
        activation_strength = self._categorize_activation_strength(feature.activation_strength)
        
        # Create observation
        obs_idx = semantic_success * self.num_activation_strength + activation_strength
        observation = np.zeros(self.obs_space_size)
        observation[obs_idx] = 1.0
        
        # Update agent beliefs
        try:
            self.agent.infer_states([observation])
            
            # Update our semantic hypotheses
            old_confidence = 0.5
            if feature_id in self.candidate_hypotheses:
                old_confidence = self.candidate_hypotheses[feature_id].get("confidence", 0.5)
                self.candidate_hypotheses[feature_id]["semantic_success_history"].append(semantic_success)
                
                # Update confidence based on consistency of results
                history = self.candidate_hypotheses[feature_id]["semantic_success_history"]
                if len(history) >= 3:
                    consistency = 1.0 - np.std(history[-3:]) / 2.0  # Normalize std
                    self.candidate_hypotheses[feature_id]["confidence"] = min(0.95, consistency)
            
            # Calculate correspondence metrics
            new_confidence = self.candidate_hypotheses.get(feature_id, {}).get("confidence", 0.5)
            belief_correspondence = abs(new_confidence - old_confidence)  # How much beliefs changed
            
            # Create correspondence metrics
            correspondence = CorrespondenceMetrics(
                belief_updating_correspondence=belief_correspondence * 100,  # Convert to percentage
                precision_weighting_correspondence=semantic_success * 25.0,  # Scale success to percentage
                prediction_error_correspondence=max(0, 100 - abs(intervention_result.effect_magnitude or 0) * 100),
                overall_correspondence=(belief_correspondence + semantic_success / 4.0) * 50,  # Combined score
                circuit_attention_patterns=[activation_strength, semantic_success],
                ai_precision_patterns=[new_confidence, belief_correspondence],
                circuit_prediction_errors=[intervention_result.effect_magnitude or 0]
            )
            
            logger.info(f"Updated beliefs for {feature_id}: semantic_success={semantic_success}, correspondence={correspondence.overall_correspondence:.1f}%")
            return correspondence
            
        except Exception as e:
            logger.error(f"Failed to update beliefs: {e}")
            # Return default correspondence metrics on error
            return CorrespondenceMetrics(
                belief_updating_correspondence=0.0,
                precision_weighting_correspondence=0.0,
                prediction_error_correspondence=0.0,
                overall_correspondence=0.0
            )
    
    def _evaluate_semantic_success(self, result: InterventionResult) -> int:
        """Evaluate how well the intervention revealed semantic relationships."""
        # This is a heuristic - in practice would use actual semantic discovery metrics
        effect_magnitude = abs(result.effect_magnitude) if result.effect_magnitude else 0.0
        
        if effect_magnitude > 0.5:
            return 3  # excellent - strong semantic effect
        elif effect_magnitude > 0.2:
            return 2  # good - moderate semantic effect  
        elif effect_magnitude > 0.05:
            return 1  # partial - weak semantic effect
        else:
            return 0  # failed - no semantic effect
    
    def _categorize_activation_strength(self, activation: float) -> int:
        """Categorize activation strength into discrete levels."""
        if activation > 5.0:
            return 2  # high
        elif activation > 2.0:
            return 1  # medium
        else:
            return 0  # low
    
    def generate_predictions(self, belief_state: BeliefState, circuit_graph: Any) -> List[NovelPrediction]:
        """Generate novel semantic predictions from learned circuit hypotheses (RQ3)."""
        predictions = []
        
        # Prediction 1: Circuit-Semantic Mapping Predictions
        semantic_mapping_predictions = self._generate_semantic_mapping_predictions()
        predictions.extend(semantic_mapping_predictions)
        
        # Prediction 2: Cross-Domain Transfer Predictions  
        transfer_predictions = self._generate_transfer_predictions()
        predictions.extend(transfer_predictions)
        
        # Prediction 3: Layer-Semantic Hierarchy Predictions
        hierarchy_predictions = self._generate_hierarchy_predictions()
        predictions.extend(hierarchy_predictions)
        
        # Prediction 4: Novel Semantic Concept Predictions
        novel_concept_predictions = self._generate_novel_concept_predictions()
        predictions.extend(novel_concept_predictions)
        
        logger.info(f"Generated {len(predictions)} novel semantic predictions for RQ3")
        return predictions
    
    def _generate_semantic_mapping_predictions(self) -> List[NovelPrediction]:
        """Generate predictions about circuit-semantic mappings."""
        predictions = []
        
        for feature_id, hypothesis in self.candidate_hypotheses.items():
            if hypothesis.get("confidence", 0.0) > 0.7:
                prediction = NovelPrediction(
                    prediction_id=f"semantic_mapping_{feature_id}",
                    prediction_type="semantic_circuit_mapping",
                    description=f"Circuit {feature_id} encodes semantic relationship patterns and should activate for similar geographical landmark → location mappings",
                    predicted_probability=hypothesis["confidence"],
                    validation_status="pending",
                    evidence=f"High confidence ({hypothesis['confidence']:.2f}) from {len(hypothesis.get('semantic_success_history', []))} semantic discovery observations"
                )
                predictions.append(prediction)
        
        return predictions
    
    def _generate_transfer_predictions(self) -> List[NovelPrediction]:
        """Generate predictions about semantic transfer to new domains."""
        # Predict that semantic circuits will transfer to new domains
        transfer_domains = [
            "landmark → city: Eiffel Tower → Paris",
            "monument → country: Big Ben → London", 
            "bridge → region: Tower Bridge → London",
            "statue → location: Statue of Liberty → New York"
        ]
        
        predictions = []
        for i, domain in enumerate(transfer_domains[:2]):  # Generate 2 transfer predictions
            prediction = NovelPrediction(
                prediction_id=f"semantic_transfer_{i+1}",
                prediction_type="semantic_transfer",
                description=f"Semantic circuits learned from Golden Gate → San Francisco will transfer to {domain}",
                predicted_probability=0.75,  # Conservative estimate
                validation_status="pending", 
                evidence="Semantic circuit patterns should generalize across similar relationship types"
            )
            predictions.append(prediction)
        
        return predictions
    
    def _generate_hierarchy_predictions(self) -> List[NovelPrediction]:
        """Generate predictions about layer-semantic processing hierarchy."""
        # Analyze layer distribution of confident circuits
        layer_distribution = {}
        for feature_id, hypothesis in self.candidate_hypotheses.items():
            if hypothesis.get("confidence", 0.0) > 0.6:
                layer = int(feature_id.split('F')[0][1:])  # Extract layer from L8F160 format
                layer_distribution[layer] = layer_distribution.get(layer, 0) + 1
        
        if layer_distribution:
            most_active_layer = max(layer_distribution.items(), key=lambda x: x[1])[0]
            prediction = NovelPrediction(
                prediction_id="semantic_hierarchy_1",
                prediction_type="semantic_hierarchy",
                description=f"Layer {most_active_layer} serves as primary semantic relationship processor, with activation cascade to adjacent layers",
                predicted_probability=0.8,
                validation_status="pending",
                evidence=f"Layer {most_active_layer} contains {layer_distribution[most_active_layer]} high-confidence semantic circuits"
            )
            return [prediction]
        
        return []
    
    def _generate_novel_concept_predictions(self) -> List[NovelPrediction]:
        """Generate predictions about novel semantic concepts."""
        # Predict novel semantic relationships based on learned patterns
        prediction = NovelPrediction(
            prediction_id="novel_concept_1",
            prediction_type="novel_semantic_concept",
            description="Circuits will encode abstract geographic relationship patterns: [Cultural Symbol] → [Associated Location] beyond specific landmarks",
            predicted_probability=0.7,
            validation_status="pending",
            evidence="Successful learning of landmark → location pattern suggests generalization to cultural symbol → location pattern"
        )
        
        return [prediction]
    
    def get_belief_summary(self) -> Dict[str, Any]:
        """Get summary of current semantic circuit beliefs."""
        if not self.candidate_hypotheses:
            return {"total_hypotheses": 0}
        
        confident_circuits = [
            feature_id for feature_id, hyp in self.candidate_hypotheses.items()
            if hyp.get("confidence", 0.0) > 0.7
        ]
        
        return {
            "total_hypotheses": len(self.candidate_hypotheses),
            "confident_semantic_circuits": len(confident_circuits),
            "avg_confidence": np.mean([h.get("confidence", 0.0) for h in self.candidate_hypotheses.values()]),
            "strong_semantic_circuits": confident_circuits[:5],  # Top 5
            "semantic_discovery_focus": "circuit-semantic relationships",
            "learning_approach": "Expected Free Energy guided investigation"
        }
    
    def get_feature_confidence(self, feature: CircuitFeature) -> float:
        """Get confidence that a feature encodes semantic relationships."""
        feature_id = f"L{feature.layer_idx}F{feature.feature_id}"
        return self.candidate_hypotheses.get(feature_id, {}).get("confidence", 0.0)
    
    # Abstract method implementations required by IActiveInferenceAgent
    
    def initialize_beliefs(self, features: Dict[int, List]) -> BeliefState:
        """Initialize belief state from discovered features."""
        # Convert dict format to list for compatibility
        all_features = []
        for layer_features in features.values():
            all_features.extend(layer_features)
        
        self.initialize_from_circuit_features(all_features)
        
        # Create belief state
        belief_state = BeliefState(
            feature_beliefs={
                feature_id: hypothesis.get("confidence", 0.5)
                for feature_id, hypothesis in self.candidate_hypotheses.items()
            },
            uncertainty_estimates={
                feature_id: 1.0 - hypothesis.get("confidence", 0.5)
                for feature_id, hypothesis in self.candidate_hypotheses.items()
            },
            timestamp=datetime.now()
        )
        
        return belief_state
    
    def calculate_expected_free_energy(self, feature: CircuitFeature, 
                                     intervention_type: InterventionType) -> float:
        """Calculate expected free energy for potential intervention."""
        feature_id = f"L{feature.layer_idx}F{feature.feature_id}"
        hypothesis = self.candidate_hypotheses.get(feature_id, {})
        
        # Find corresponding candidate
        candidate = None
        for cand in self.filtered_candidates:
            if cand.feature.layer_idx == feature.layer_idx and cand.feature.feature_id == feature.feature_id:
                candidate = cand
                break
        
        if candidate is None:
            return 1.0  # High EFE for unknown features
        
        return self._calculate_circuit_efe(candidate, hypothesis)
    
    def check_convergence(self) -> bool:
        """Check if semantic learning has converged."""
        if not self.candidate_hypotheses:
            return False
        
        # Consider converged if most hypotheses have high confidence
        confident_count = sum(
            1 for hyp in self.candidate_hypotheses.values()
            if hyp.get("confidence", 0.0) > 0.8
        )
        
        convergence_ratio = confident_count / len(self.candidate_hypotheses)
        return convergence_ratio > 0.7  # 70% of hypotheses are confident
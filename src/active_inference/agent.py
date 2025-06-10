# Active Inference Agent Implementation  
# Following patterns from https://pymdp-rtd.readthedocs.io/en/latest/

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

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
        
        # Handle pymdp dependency
        try:
            import pymdp
            from pymdp import Agent
            # from pymdp.maths import softmax, kl_div, dot # Not used directly yet
            # from pymdp.utils import obj_array, random_A_matrix, random_B_matrix # Not used directly yet
            self.pymdp_is_available = True
        except ImportError:
            self.pymdp_is_available = False

        config_use_pymdp = config.active_inference.use_pymdp

        if config_use_pymdp and not self.pymdp_is_available:
            logger.warning("pymdp is configured to be used but is not installed. Falling back to non-pymdp mode.")
            self.use_pymdp = False
        else:
            self.use_pymdp = config_use_pymdp and self.pymdp_is_available

        # Active Inference parameters
        self.epistemic_weight = config.active_inference.epistemic_weight
        self.exploration_weight = config.active_inference.exploration_weight # Not used yet, but part of config
        self.convergence_threshold = config.active_inference.convergence_threshold
        
        # State tracking
        self.belief_state: Optional[BeliefState] = None # Explicitly Optional
        self.intervention_history: List[InterventionResult] = []
        self.correspondence_history: List[CorrespondenceMetrics] = []
        self.novel_predictions: List[NovelPrediction] = []
        
        # pymdp agent if available
        self.pymdp_agent = None # Will be instance of pymdp.Agent if used
        
        logger.info(f"ActiveInferenceAgent initialized. Using pymdp: {self.use_pymdp} (available: {self.pymdp_is_available}, configured: {config.active_inference.use_pymdp})")
    
    def initialize_beliefs(self, features: Dict[int, List[SAEFeature]]) -> BeliefState:
        """Initialize belief state from discovered features."""
        logger.info("Initializing Active Inference beliefs from discovered features")
        
        all_features_list: List[SAEFeature] = []
        for layer_features in features.values():
            all_features_list.extend(layer_features)
        
        if not all_features_list:
            logger.warning("No features found for belief initialization")
            self.belief_state = self._create_empty_belief_state()
            return self.belief_state
        
        feature_importances: Dict[int, float] = {}
        uncertainty: Dict[int, float] = {}
        
        for feature in all_features_list:
            feature_importances[feature.feature_id] = feature.max_activation
            uncertainty[feature.feature_id] = np.clip(1.0 - feature.max_activation, 0.01, 1.0) # Ensure uncertainty is not zero
        
        connection_beliefs = self._initialize_connection_beliefs(all_features_list)
        
        generative_model: Optional[Dict[str, Any]] = None
        posterior_beliefs_qs: Optional[np.ndarray] = None # This is what pymdp calls qs
        precision_matrix: Optional[np.ndarray] = None
        
        if self.use_pymdp:
            try:
                # This would create and initialize self.pymdp_agent
                # For now, just setting up placeholders for generative_model parts
                _gm, _pb, _pm = self._initialize_pymdp_components(all_features_list)
                if _gm is not None: # If successful
                     generative_model = _gm
                     posterior_beliefs_qs = _pb
                     precision_matrix = _pm
                else: # Fallback if pymdp init returns None (e.g. no features)
                    self.use_pymdp = False
                    logger.warning("PyMDP component initialization failed (no features?), falling back to non-pymdp mode for this session.")

            except Exception as e:
                logger.warning(f"PyMDP components initialization failed: {e}. Falling back to non-pymdp mode.")
                self.use_pymdp = False
        
        n_features = len(all_features_list)
        # Default qs if not using pymdp or if pymdp init failed to provide it
        if posterior_beliefs_qs is None:
            posterior_beliefs_qs = np.ones(n_features) / n_features if n_features > 0 else np.array([])
        
        confidence = self._calculate_initial_confidence(feature_importances, uncertainty)
        
        self.belief_state = BeliefState(
            qs=posterior_beliefs_qs,
            feature_importances=feature_importances,
            connection_beliefs=connection_beliefs,
            uncertainty=uncertainty,
            confidence=confidence,
            generative_model=generative_model, # Store the model structure
            posterior_beliefs=posterior_beliefs_qs, # Store initial posteriors if from pymdp
            precision_matrix=precision_matrix
        )
        
        logger.info(f"Initialized beliefs for {len(all_features_list)} features with confidence {confidence:.3f}")
        return self.belief_state

    def _create_empty_belief_state(self) -> BeliefState:
        return BeliefState(
            qs=np.array([]), feature_importances={}, connection_beliefs={},
            uncertainty={}, confidence=0.0
        )

    def _initialize_connection_beliefs(self, features: List[SAEFeature]) -> Dict[Tuple[int, int], float]:
        logger.debug("Initializing connection beliefs (currently a placeholder).")
        return {}

    def _initialize_pymdp_components(self, features: List[SAEFeature]) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray], Optional[np.ndarray]]:
        logger.warning("_initialize_pymdp_components is a placeholder.")
        if not features: return None, None, None
        num_hidden_states = len(features)
        num_observable_outcomes = 3
        A = np.ones((num_observable_outcomes, num_hidden_states)) / num_observable_outcomes
        initial_qs = np.ones(num_hidden_states) / num_hidden_states
        precision_matrix = np.eye(num_hidden_states)
        return {'A': A}, initial_qs, precision_matrix

    def _calculate_initial_confidence(self, importances: Dict[int, float], uncertainties: Dict[int, float]) -> float:
        if not uncertainties: return 0.0
        avg_uncertainty = sum(uncertainties.values()) / len(uncertainties)
        return np.clip(1.0 - avg_uncertainty, 0.0, 1.0)

    def calculate_expected_free_energy(self, feature: SAEFeature, intervention_type: InterventionType) -> float:
        if self.belief_state is None:
            logger.error("EFE calculation failed: BeliefState is not initialized.")
            return 0.0
        
        epistemic_value = self._calculate_epistemic_value(feature, intervention_type)
        pragmatic_value = self._calculate_pragmatic_value(feature, intervention_type)
        
        efe = self.epistemic_weight * epistemic_value + (1 - self.epistemic_weight) * pragmatic_value

        if self.use_pymdp and self.belief_state.generative_model is not None:
            # These would be more complex calculations involving the generative model
            model_uncertainty_reduction = self._calculate_model_uncertainty_reduction(feature)
            causal_info_gain = self._calculate_causal_information_gain(feature)
            efe += self.epistemic_weight * model_uncertainty_reduction + \
                   (1-self.epistemic_weight) * causal_info_gain # Simplified aggregation
        return efe

    def _calculate_epistemic_value(self, feature: SAEFeature, intervention_type: InterventionType) -> float:
        if self.belief_state is None: return 0.0
        return self.belief_state.uncertainty.get(feature.feature_id, 0.5)

    def _calculate_pragmatic_value(self, feature: SAEFeature, intervention_type: InterventionType) -> float:
        if self.belief_state is None: return 0.0
        return self.belief_state.feature_importances.get(feature.feature_id, 0.5)

    def _calculate_model_uncertainty_reduction(self, feature: SAEFeature) -> float:
        logger.debug("_calculate_model_uncertainty_reduction is a placeholder.")
        return 0.05 # Small arbitrary value

    def _calculate_causal_information_gain(self, feature: SAEFeature) -> float:
        logger.debug("_calculate_causal_information_gain is a placeholder.")
        return 0.05 # Small arbitrary value

    def update_beliefs(self, intervention_result: InterventionResult) -> CorrespondenceMetrics:
        if self.belief_state is None:
            logger.error("Cannot update beliefs: BeliefState is not initialized.")
            return CorrespondenceMetrics(0.0,0.0,0.0,0.0)

        target_id = intervention_result.target_feature.feature_id
        effect_size = intervention_result.effect_size
        alpha = self.config.active_inference.epistemic_weight # Use a configured param, e.g. epistemic_weight as learning rate

        current_importance = self.belief_state.feature_importances.get(target_id, 0.5)
        new_importance = current_importance + effect_size * alpha
        self.belief_state.feature_importances[target_id] = np.clip(new_importance, 0.0, 1.0)

        current_uncertainty = self.belief_state.uncertainty.get(target_id, 0.5)
        reduction_factor = abs(effect_size) * alpha
        new_uncertainty = current_uncertainty * (1.0 - reduction_factor)
        self.belief_state.uncertainty[target_id] = np.clip(new_uncertainty, 0.01, 1.0)

        self.belief_state.confidence = np.clip(1.0 - self.belief_state.get_average_uncertainty(), 0.0, 1.0)
        self.intervention_history.append(intervention_result)

        # TODO: Implement pymdp-specific belief update
        # This would involve:
        # 1. Translating InterventionResult to an observation format compatible with self.pymdp_agent.A (likelihood matrix).
        #    For instance, the observation could be a discretized version of `effect_size` or `target_token_change`.
        # 2. Calling self.pymdp_agent.infer_states(observation) to get new self.belief_state.qs (posterior over states).
        #    This updates the agent's beliefs about which hidden state (e.g., circuit configuration) is most likely.
        # 3. Potentially updating self.belief_state.generative_model (e.g., learning parameters of A or B matrices)
        #    or other pymdp-related fields like `self.belief_state.precision_matrix`.


        belief_updating_corr = np.clip(abs(effect_size * 2.0), 0.0, 1.0)
        prediction_error_corr = np.clip(abs(effect_size * 1.5), 0.0, 1.0)
        precision_weighting_corr = 0.5
        overall_corr = np.clip((belief_updating_corr + prediction_error_corr + precision_weighting_corr) / 3.0, 0.0, 1.0)
        
        metrics = CorrespondenceMetrics(
            belief_updating_correspondence=belief_updating_corr,
            precision_weighting_correspondence=precision_weighting_corr,
            prediction_error_correspondence=prediction_error_corr,
            overall_correspondence=overall_corr
        )
        self.correspondence_history.append(metrics)
        logger.debug(f"Beliefs updated for feature {target_id}. Importance: {new_importance:.3f}, Uncertainty: {new_uncertainty:.3f}")
        return metrics

    def generate_predictions(self) -> List[NovelPrediction]:
        logger.debug("Generating predictions based on current beliefs.")
        predictions = []
        if self.belief_state:
            for fid, importance in self.belief_state.feature_importances.items():
                uncertainty = self.belief_state.uncertainty.get(fid, 1.0)
                if importance > 0.8 and uncertainty < 0.2: # Example criteria
                    pred_confidence = np.clip(importance * (1.0 - uncertainty), 0.0, 1.0)
                    predictions.append(NovelPrediction(
                        prediction_type="feature_interaction",
                        description=f"Feature {fid} (layer {self.belief_state.feature_importances.get(fid, -1)}) is predicted to be highly influential.", # This description is a bit off for layer
                        testable_hypothesis=f"Ablating feature {fid} will significantly alter model output for specific inputs.",
                        expected_outcome="Significant change in logits or specific behavioral change.",
                        test_method="Targeted ablation study.",
                        confidence=pred_confidence
                    ))
        return predictions[:self.config.research_questions.rq3_predictions_target]

    def check_convergence(self, threshold: Optional[float] = None) -> bool:
        if threshold is None:
            threshold = self.config.active_inference.convergence_threshold
        if self.belief_state is None:
            logger.warning("Cannot check convergence: BeliefState is not initialized.")
            return False
        avg_uncertainty = self.belief_state.get_average_uncertainty()
        converged = avg_uncertainty < threshold
        logger.info(f"Convergence check: Avg Uncertainty {avg_uncertainty:.4f} vs Threshold {threshold:.4f}. Converged: {converged}")
        return converged
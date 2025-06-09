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
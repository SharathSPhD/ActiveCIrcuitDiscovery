# Circuit Tracer Implementation - Modular Design
# Following patterns from https://github.com/safety-research/circuit-tracer/tree/main/demos

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import warnings

# Core dependencies
try:
    import transformer_lens
    from transformer_lens import HookedTransformer
    TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    TRANSFORMER_LENS_AVAILABLE = False
    warnings.warn("TransformerLens not available")

try:
    import sae_lens
    from sae_lens import SAE
    SAE_LENS_AVAILABLE = True
except ImportError:
    SAE_LENS_AVAILABLE = False

# Project imports with proper relative imports
try:
    from ..core.interfaces import ICircuitTracer
    from ..core.data_structures import SAEFeature, InterventionResult, AttributionGraph, CircuitNode
    from ..config.experiment_config import CompleteConfig, InterventionType, DeviceType
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from core.interfaces import ICircuitTracer
    from core.data_structures import SAEFeature, InterventionResult, AttributionGraph, CircuitNode
    from config.experiment_config import CompleteConfig, InterventionType, DeviceType

logger = logging.getLogger(__name__)

class CircuitTracer(ICircuitTracer):
    """Circuit discovery implementation following circuit-tracer patterns."""
    
    def __init__(self, config: CompleteConfig):
        """Initialize circuit tracer with configuration."""
        self.config = config
        self.device = self._resolve_device(config.model.device)
        logger.info(f"Initializing CircuitTracer for {config.model.name} on {self.device}")
        
        self.model = self._load_model()
        self.sae_analyzers = {}
        self._load_sae_analyzers()
        
    def _resolve_device(self, device_config: DeviceType) -> str:
        """Resolve device configuration to actual device string."""
        if device_config == DeviceType.AUTO:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_config.value
    
    def _load_model(self) -> HookedTransformer:
        """Load transformer model using TransformerLens."""
        if not TRANSFORMER_LENS_AVAILABLE:
            raise ImportError("TransformerLens required for circuit analysis")
        
        model = HookedTransformer.from_pretrained(
            self.config.model.name,
            device=self.device,
            fold_ln=False,
            center_writing_weights=False,
        )
        model.eval()
        logger.info(f"Loaded {self.config.model.name} successfully")
        return model
    
    def _load_sae_analyzers(self):
        """Load SAE analyzers - auto-discover layers if enabled."""
        if not self.config.sae.enabled:
            logger.info("SAE analysis disabled in configuration")
            return
        
        if self.config.sae.auto_discover_layers:
            logger.info("Auto-discovering active layers...")
            target_layers = self._auto_discover_active_layers()
            # Update config with discovered layers
            self.config.sae.target_layers = target_layers
            logger.info(f"Auto-discovered layers: {target_layers}")
        else:
            target_layers = self.config.sae.target_layers
            logger.info(f"Using configured target layers: {target_layers}")
        
        for layer in target_layers:
            try:
                if SAE_LENS_AVAILABLE:
                    sae_id = f"gpt2-small-res-jb-{layer}"
                    sae = SAE.from_pretrained(sae_id, device=self.device)
                    self.sae_analyzers[layer] = sae
                    logger.info(f"Loaded SAE for layer {layer}")
                else:
                    self._create_fallback_analyzer(layer)
            except Exception as e:
                logger.warning(f"Could not load SAE for layer {layer}: {e}")
                self._create_fallback_analyzer(layer)
    
    def _auto_discover_active_layers(self) -> List[int]:
        """Auto-discover layers with significant activity for sample inputs."""
        # Use sample inputs from config to determine active layers
        sample_inputs = self.config.sae.sample_inputs_for_layer_discovery
        
        start_layer, end_layer = self.config.sae.layer_search_range
        if end_layer == -1:
            end_layer = self.model.cfg.n_layers - 1
        
        layer_activities = {}
        
        for sample_input in sample_inputs:
            tokens = self.model.to_tokens(sample_input)
            with torch.no_grad():
                logits, cache = self.model.run_with_cache(tokens)
                
                for layer in range(start_layer, end_layer + 1):
                    resid_key = f"blocks.{layer}.hook_resid_post"
                    if resid_key in cache:
                        activations = cache[resid_key]
                        # Calculate layer activity (variance in activations)
                        activity = torch.var(activations).item()
                        
                        if layer not in layer_activities:
                            layer_activities[layer] = []
                        layer_activities[layer].append(activity)
        
        # Calculate average activity per layer
        avg_activities = {}
        for layer, activities in layer_activities.items():
            avg_activities[layer] = np.mean(activities)
        
        # Select top layers with highest activity
        sorted_layers = sorted(avg_activities.items(), key=lambda x: x[1], reverse=True)
        
        # Determine number of target layers based on config
        num_potential_layers = len(sorted_layers)
        ratio_based_layers = int(num_potential_layers * self.config.sae.auto_discover_layer_ratio)

        n_target_layers = min(
            self.config.sae.auto_discover_max_layers,
            max(self.config.sae.auto_discover_min_layers, ratio_based_layers)
        )
        # Ensure we don't try to select more layers than available
        n_target_layers = min(n_target_layers, num_potential_layers)

        selected_layers = [layer for layer, _ in sorted_layers[:n_target_layers]]
        selected_layers.sort()  # Keep in order
        
        logger.info(f"Layer activities: {avg_activities}")
        logger.info(f"Selected {len(selected_layers)} most active layers: {selected_layers}")
        
        return selected_layers
    
    def _create_fallback_analyzer(self, layer: int):
        """
        Create a fallback SAE analyzer for a specific layer.

        This method is invoked under two conditions:
        1. `sae_lens` library is not installed or unavailable.
        2. A pre-trained SAE model for the specified layer fails to load (e.g., due to network issues or model unavailability).

        The fallback analyzer consists of:
        - 'encoder': Randomly initialized weights simulating an SAE encoder (shape: d_model, n_features).
                     `n_features` is typically `d_model * config.sae.fallback_sae_feature_multiplier`.
        - 'decoder': Randomly initialized weights simulating an SAE decoder (shape: n_features, d_model).
                     Both encoder and decoder weights are scaled by `config.sae.fallback_sae_weight_scale`.

        Key characteristics and purpose:
        - Synthetic Features: The "features" derived from this fallback are synthetic and NOT learned from data.
                              They are essentially random projections.
        - Execution Flow: Its primary purpose is to enable the basic execution flow of the circuit discovery
                          pipeline and facilitate testing when real SAEs are not available or functional.
                          This ensures that the system can run end-to-end, even if feature analysis is degraded.

        Limitations:
        - Not for Research: This fallback is NOT suitable for genuine mechanistic interpretability research,
                            meaningful circuit discovery, or drawing any valid conclusions about the model's internal
                            mechanisms. The "features" identified will be random and uninformative.
        - Randomness: Discovered "features" will be based on random weight initializations and will not reflect
                      any learned representations of the model.
        """
        d_model = self.model.cfg.d_model
        n_features = d_model * self.config.sae.fallback_sae_feature_multiplier
        
        self.sae_analyzers[layer] = {
            'type': 'fallback', # Clearly mark this as a fallback
            'layer': layer,
            'd_model': d_model,
            'n_features': n_features,
            # Randomly initialized encoder weights (d_model, n_features)
            'encoder': torch.randn(d_model, n_features, device=self.device) * self.config.sae.fallback_sae_weight_scale,
            # Randomly initialized decoder weights (n_features, d_model)
            'decoder': torch.randn(n_features, d_model, device=self.device) * self.config.sae.fallback_sae_weight_scale
        }
        logger.info(f"Created fallback SAE analyzer for layer {layer} (synthetic features only)")
    
    def find_active_features(self, text: str, threshold: float = 0.05) -> Dict[int, List[SAEFeature]]:
        """Find active SAE features for given input text."""
        logger.debug(f"Finding active features for: '{text[:50]}...'")
        
        tokens = self.model.to_tokens(text)
        active_features = {}
        
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(tokens)
            
            for layer in self.config.sae.target_layers:
                if layer not in self.sae_analyzers:
                    continue
                
                resid_key = f"blocks.{layer}.hook_resid_post"
                if resid_key not in cache:
                    continue
                
                activations = cache[resid_key]
                features = self._analyze_activations_with_sae(activations, layer, threshold)
                
                if features:
                    active_features[layer] = features[:self.config.sae.max_features_per_layer]
        
        total_features = sum(len(features) for features in active_features.values())
        logger.info(f"Found {total_features} active features across {len(active_features)} layers")
        return active_features

    def perform_intervention(self, text: str, feature: SAEFeature,
                           intervention_type: InterventionType) -> InterventionResult:
        """Perform intervention on specified feature. (Placeholder)"""
        logger.warning("perform_intervention is a placeholder and not fully implemented.")
        # This is a dummy implementation.
        # In a real scenario, you would apply the intervention and measure its effect.
        original_logits = self.model.run_with_hooks(
            self.model.to_tokens(text),
            return_type="logits"
        )
        # Simulate some effect
        intervened_logits = original_logits.clone()
        intervened_logits[0, -1, np.random.randint(0, intervened_logits.shape[-1])] *= 0.5

        return InterventionResult(
            intervention_type=intervention_type,
            target_feature=feature,
            original_logits=original_logits,
            intervened_logits=intervened_logits,
            effect_size=np.random.rand(),
            target_token_change=np.random.rand() * 2 - 1, # Random change between -1 and 1
            intervention_layer=feature.layer,
            metadata={'notes': 'Placeholder implementation'}
        )

    def build_attribution_graph(self, text: str) -> AttributionGraph:
        """Build complete attribution graph for circuit analysis. (Placeholder)"""
        logger.warning("build_attribution_graph is a placeholder and not fully implemented.")
        # This is a dummy implementation.
        return AttributionGraph(
            input_text=text,
            nodes={}, # No real nodes
            edges={},
            target_output="<dummy_output>",
            confidence=0.1,
            metadata={'notes': 'Placeholder implementation'}
        )

    def get_feature_activations(self, text: str, layer: int) -> torch.Tensor:
        """Get feature activations for specific layer. (Placeholder)"""
        logger.warning("get_feature_activations is a placeholder and not fully implemented.")
        # This is a dummy implementation.
        # It should return actual feature activations if SAEs were properly processed.
        # For now, return random tensor of plausible shape.
        d_model = self.model.cfg.d_model
        # Assuming n_features is related to d_model, like in fallback or real SAEs
        n_features = d_model * self.config.sae.fallback_sae_feature_multiplier

        # Return a tensor of shape (batch_size, num_tokens, n_features) - simplified
        # For now, let's return (1, 1, n_features) as a very rough placeholder
        return torch.rand((1, 1, n_features), device=self.device)

    def _analyze_activations_with_sae(self, activations: torch.Tensor, layer_idx: int, threshold: float) -> List[SAEFeature]:
        """
        Analyzes activations using the SAE for a specific layer to find active features.
        (Placeholder - this method needs full implementation for real feature extraction)

        Args:
            activations: The activations tensor from the model (e.g., hook_resid_post).
            layer_idx: The layer index for which the SAE should be used.
            threshold: The activation threshold to consider a feature active.

        Returns:
            A list of SAEFeature objects that are active.
        """
        logger.debug(f"Analyzing activations for layer {layer_idx} using its SAE analyzer.")
        sae_analyzer = self.sae_analyzers.get(layer_idx)

        if not sae_analyzer:
            logger.warning(f"No SAE analyzer found for layer {layer_idx}. Cannot analyze activations.")
            return []

        # Example of how one might interact with a real SAE (e.g., sae_lens)
        # or the fallback structure. This is highly simplified.
        active_sae_features = []

        if sae_analyzer['type'] == 'fallback':
            # Fallback logic: very basic, not real feature analysis
            # This is just to allow the pipeline to run.
            # For instance, consider a feature active if its random "projection" is high.
            # The fallback 'encoder' is (d_model, n_features)
            # Activations are likely (batch, seq_len, d_model)
            # We'd need to reshape/select activations. Taking mean over batch/seq for simplicity.
            reshaped_activations = activations.reshape(-1, sae_analyzer['d_model']) # (N, d_model)

            # "Project" onto random feature directions
            # (N, d_model) @ (d_model, n_features) -> (N, n_features)
            feature_activations = torch.matmul(reshaped_activations, sae_analyzer['encoder'])

            # Consider a feature active if its max "activation" across N samples exceeds threshold
            # This is a simplification. Real SAEs have specific ways to get feature activations.
            for i in range(sae_analyzer['n_features']):
                max_act_for_feature = torch.max(feature_activations[:, i]).item()
                if max_act_for_feature > threshold:
                    active_sae_features.append(
                        SAEFeature(
                            feature_id=i,
                            layer=layer_idx,
                            activation_threshold=threshold,
                            description=f"Fallback feature {i} for layer {layer_idx}",
                            max_activation=min(1.0, max_act_for_feature), # Cap at 1.0 for consistency
                            examples=[], # No real examples for fallback
                            # feature_vector and decoder_weights could be slices of the fallback tensors
                            feature_vector=sae_analyzer['encoder'][:, i].cpu().numpy(),
                            decoder_weights=sae_analyzer['decoder'][i, :].cpu().numpy()
                        )
                    )
            logger.info(f"Fallback SAE for layer {layer_idx} 'found' {len(active_sae_features)} synthetic features.")

        else: # This would be for a real SAE (e.g., sae_lens.SAE object)
            # This part needs to be implemented to correctly use the sae_lens SAE API
            # For example:
            #   hidden_pre = sae_analyzer.encode(activations) # Get feature activations
            #   # Iterate through features, check against threshold, create SAEFeature objects
            logger.warning(f"Real SAE processing for layer {layer_idx} in _analyze_activations_with_sae is not fully implemented.")
            # Return empty for now to avoid errors with incompatible SAE object interaction
            return []

        return active_sae_features

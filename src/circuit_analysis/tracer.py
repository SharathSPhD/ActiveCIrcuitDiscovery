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
        # Use sample inputs to determine active layers
        sample_inputs = [
            "The Golden Gate Bridge is located in",
            "When I think about cats, I",
            "The capital of France is"
        ]
        
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
        
        # Take top 6-8 layers (or all if fewer layers exist)
        n_target_layers = min(8, max(6, len(sorted_layers) // 2))
        selected_layers = [layer for layer, _ in sorted_layers[:n_target_layers]]
        selected_layers.sort()  # Keep in order
        
        logger.info(f"Layer activities: {avg_activities}")
        logger.info(f"Selected {len(selected_layers)} most active layers: {selected_layers}")
        
        return selected_layers
    
    def _create_fallback_analyzer(self, layer: int):
        """Create fallback analyzer for single layer."""
        d_model = self.model.cfg.d_model
        n_features = d_model * 4
        
        self.sae_analyzers[layer] = {
            'type': 'fallback',
            'layer': layer,
            'd_model': d_model,
            'n_features': n_features,
            'encoder': torch.randn(d_model, n_features, device=self.device) * 0.1,
            'decoder': torch.randn(n_features, d_model, device=self.device) * 0.1
        }
        logger.info(f"Created fallback SAE analyzer for layer {layer}")
    
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

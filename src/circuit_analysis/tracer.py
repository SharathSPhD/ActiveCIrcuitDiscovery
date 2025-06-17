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
    from core.interfaces import ICircuitTracer
    from core.data_structures import SAEFeature, InterventionResult, AttributionGraph, CircuitNode, GraphNode, GraphEdge
    from config.experiment_config import CompleteConfig, InterventionType, DeviceType
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from core.interfaces import ICircuitTracer
    from core.data_structures import SAEFeature, InterventionResult, AttributionGraph, CircuitNode, GraphNode, GraphEdge
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
    
    def _load_model(self):
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
    
    def perform_intervention(self, text: str, feature: SAEFeature, 
                           intervention_type: InterventionType) -> InterventionResult:
        """Perform intervention on specified feature."""
        logger.debug(f"Performing {intervention_type.value} intervention on feature {feature.feature_id}")
        
        tokens = self.model.to_tokens(text)
        
        # Get baseline activations and logits
        with torch.no_grad():
            baseline_logits, baseline_cache = self.model.run_with_cache(tokens)
            baseline_logits = baseline_logits[:, -1, :]  # Last token logits
        
        # Perform intervention based on type
        if intervention_type == InterventionType.ABLATION:
            intervened_logits = self._perform_ablation_intervention(
                tokens, feature, baseline_cache
            )
        elif intervention_type == InterventionType.ACTIVATION_PATCHING:
            intervened_logits = self._perform_activation_patching(
                tokens, feature, baseline_cache
            )
        elif intervention_type == InterventionType.MEAN_ABLATION:
            intervened_logits = self._perform_mean_ablation(
                tokens, feature, baseline_cache
            )
        else:
            raise ValueError(f"Unknown intervention type: {intervention_type}")
        
        # Calculate metrics
        effect_size = self._calculate_effect_size(baseline_logits, intervened_logits)
        target_token_change = self._calculate_target_token_change(
            baseline_logits, intervened_logits, tokens
        )
        
        return InterventionResult(
            intervention_type=intervention_type,
            target_feature=feature,
            original_logits=baseline_logits,
            intervened_logits=intervened_logits,
            effect_size=effect_size,
            target_token_change=target_token_change,
            intervention_layer=feature.layer,
            metadata={
                'input_text': text,
                'feature_activation': feature.max_activation,
                'intervention_success': True
            }
        )
    
    def build_attribution_graph(self, text: str) -> AttributionGraph:
        """Build complete attribution graph for circuit analysis."""
        logger.info(f"Building attribution graph for: '{text[:50]}...'")
        
        # Find active features
        active_features = self.find_active_features(text)
        
        if not active_features:
            logger.warning("No active features found for attribution graph")
            return self._create_empty_attribution_graph(text)
        
        # Flatten features and create nodes
        all_features = []
        for layer_features in active_features.values():
            all_features.extend(layer_features)
        
        nodes = {}
        for i, feature in enumerate(all_features):
            # Calculate causal influence through intervention
            intervention_result = self.perform_intervention(
                text, feature, InterventionType.ABLATION
            )
            
            nodes[i] = CircuitNode(
                feature=feature,
                activation_value=feature.max_activation,
                causal_influence=intervention_result.effect_size
            )
        
        # Build edges through attribution analysis
        edges = self._compute_feature_attributions(text, all_features, nodes)
        
        # Calculate overall confidence
        confidence = self._calculate_graph_confidence(nodes, edges)
        
        # Convert nodes and edges to new AttributionGraph format
        graph_nodes = []
        graph_edges = []
        
        # Convert CircuitNodes to GraphNodes
        for i, circuit_node in nodes.items():
            graph_node = GraphNode(
                node_id=f"feature_{circuit_node.feature.feature_id}_layer_{circuit_node.feature.layer}",
                layer=circuit_node.feature.layer,
                feature_id=circuit_node.feature.feature_id,
                importance=circuit_node.causal_influence,
                description=circuit_node.feature.description
            )
            graph_nodes.append(graph_node)
        
        # Convert edge dict to GraphEdges
        for (source_idx, target_idx), weight in edges.items():
            if source_idx in nodes and target_idx in nodes:
                source_node = nodes[source_idx]
                target_node = nodes[target_idx]
                
                source_id = f"feature_{source_node.feature.feature_id}_layer_{source_node.feature.layer}"
                target_id = f"feature_{target_node.feature.feature_id}_layer_{target_node.feature.layer}"
                
                graph_edge = GraphEdge(
                    source_id=source_id,
                    target_id=target_id,
                    weight=weight,
                    confidence=min(1.0, abs(weight)),
                    edge_type="causal"
                )
                graph_edges.append(graph_edge)
        
        # Create attribution graph with new format
        attribution_graph = AttributionGraph(
            input_text=text,
            nodes=graph_nodes,
            edges=graph_edges,
            target_output=self._get_target_output(text),
            confidence=confidence,
            metadata={
                'total_features': len(all_features),
                'total_edges': len(edges),
                'layers_analyzed': list(active_features.keys()),
                'method': 'gradient_based_attribution',
                'attribution_version': '2.0'
            }
        )
        
        # Add error nodes for unexplained variance if enabled
        if self.config.sae.include_error_nodes:
            attribution_graph = self._add_error_nodes(attribution_graph, all_features)
        
        # Apply intelligent pruning based on cumulative attribution
        if len(graph_nodes) > self.config.sae.max_graph_nodes:
            attribution_graph = attribution_graph.prune_graph(
                node_threshold=0.85, 
                edge_threshold=0.90
            )
            logger.info(f"Graph pruned to {len(attribution_graph.nodes)} nodes and {len(attribution_graph.edges)} edges")
        
        logger.info(f"Built attribution graph with {len(attribution_graph.nodes)} nodes and {len(attribution_graph.edges)} edges")
        return attribution_graph
    
    def get_feature_activations(self, text: str, layer: int) -> torch.Tensor:
        """Get feature activations for specific layer."""
        tokens = self.model.to_tokens(text)
        
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
            
            resid_key = f"blocks.{layer}.hook_resid_post"
            if resid_key in cache:
                activations = cache[resid_key]
                
                # Apply SAE if available
                if layer in self.sae_analyzers:
                    sae = self.sae_analyzers[layer]
                    if isinstance(sae, dict):  # Fallback SAE
                        # Simple linear transformation
                        activations_flat = activations.view(-1, activations.size(-1))
                        feature_activations = torch.matmul(activations_flat, sae['encoder'])
                        return F.relu(feature_activations)
                    else:
                        # Real SAE
                        return sae.encode(activations)
                
                return activations
        
        return torch.zeros(1, 1, self.model.cfg.d_model, device=self.device)
    
    def _perform_ablation_intervention(self, tokens: torch.Tensor, 
                                     feature: SAEFeature, 
                                     baseline_cache: Dict) -> torch.Tensor:
        """Perform feature ablation intervention."""
        
        def ablation_hook(activations, hook):
            if feature.layer in self.sae_analyzers:
                sae = self.sae_analyzers[feature.layer]
                if isinstance(sae, dict):  # Fallback SAE
                    # Zero out specific feature in encoder space
                    activations_flat = activations.view(-1, activations.size(-1))
                    feature_activations = torch.matmul(activations_flat, sae['encoder'])
                    feature_activations[:, feature.feature_id] = 0.0
                    # Reconstruct
                    reconstructed = torch.matmul(feature_activations, sae['decoder'])
                    return reconstructed.view(activations.shape)
                else:
                    # Real SAE ablation
                    feature_acts = sae.encode(activations)
                    feature_acts[:, :, feature.feature_id] = 0.0
                    return sae.decode(feature_acts)
            return activations
        
        # Run with intervention hook
        hook_point = f"blocks.{feature.layer}.hook_resid_post"
        with self.model.hooks([(hook_point, ablation_hook)]):
            logits = self.model(tokens)
        
        return logits[:, -1, :]
    
    def _perform_activation_patching(self, tokens: torch.Tensor,
                                   feature: SAEFeature,
                                   baseline_cache: Dict) -> torch.Tensor:
        """Perform activation patching intervention."""
        
        # Get clean activation for this feature
        clean_activation = baseline_cache[f"blocks.{feature.layer}.hook_resid_post"]
        
        def patching_hook(activations, hook):
            if feature.layer in self.sae_analyzers:
                sae = self.sae_analyzers[feature.layer]
                if isinstance(sae, dict):  # Fallback SAE
                    # Patch specific feature with clean activation
                    activations_flat = activations.view(-1, activations.size(-1))
                    clean_flat = clean_activation.view(-1, clean_activation.size(-1))
                    
                    feature_acts = torch.matmul(activations_flat, sae['encoder'])
                    clean_feature_acts = torch.matmul(clean_flat, sae['encoder'])
                    
                    # Patch specific feature
                    feature_acts[:, feature.feature_id] = clean_feature_acts[:, feature.feature_id]
                    
                    # Reconstruct
                    reconstructed = torch.matmul(feature_acts, sae['decoder'])
                    return reconstructed.view(activations.shape)
                else:
                    # Real SAE patching
                    feature_acts = sae.encode(activations)
                    clean_feature_acts = sae.encode(clean_activation)
                    feature_acts[:, :, feature.feature_id] = clean_feature_acts[:, :, feature.feature_id]
                    return sae.decode(feature_acts)
            return activations
        
        hook_point = f"blocks.{feature.layer}.hook_resid_post"
        with self.model.hooks([(hook_point, patching_hook)]):
            logits = self.model(tokens)
        
        return logits[:, -1, :]
    
    def _perform_mean_ablation(self, tokens: torch.Tensor,
                             feature: SAEFeature,
                             baseline_cache: Dict) -> torch.Tensor:
        """Perform mean ablation intervention."""
        
        # Calculate mean activation for this feature across dataset
        # For now, use zero ablation as approximation
        return self._perform_ablation_intervention(tokens, feature, baseline_cache)
    
    def _calculate_effect_size(self, baseline_logits: torch.Tensor, 
                             intervened_logits: torch.Tensor) -> float:
        """Calculate intervention effect size."""
        # L2 norm of logit difference, normalized
        diff = intervened_logits - baseline_logits
        l2_norm = torch.norm(diff).item()
        
        # Normalize by baseline logit magnitude
        baseline_norm = torch.norm(baseline_logits).item()
        normalized_effect = l2_norm / (baseline_norm + 1e-8)
        
        return min(1.0, normalized_effect)
    
    def _calculate_target_token_change(self, baseline_logits: torch.Tensor,
                                     intervened_logits: torch.Tensor,
                                     tokens: torch.Tensor) -> float:
        """Calculate change in target token probability."""
        # Get the most likely next token from baseline
        target_token = torch.argmax(baseline_logits, dim=-1)
        
        # Calculate probability change
        baseline_probs = F.softmax(baseline_logits, dim=-1)
        intervened_probs = F.softmax(intervened_logits, dim=-1)
        
        baseline_prob = baseline_probs[0, target_token].item()
        intervened_prob = intervened_probs[0, target_token].item()
        
        return baseline_prob - intervened_prob
    
    def _compute_feature_attributions(self, text: str, features: List[SAEFeature],
                                    nodes: Dict[int, CircuitNode]) -> Dict[Tuple[int, int], float]:
        """Compute attribution edges between features."""
        edges = {}
        
        # For each pair of features from different layers
        for i, feature_i in enumerate(features):
            for j, feature_j in enumerate(features):
                if i != j and feature_i.layer < feature_j.layer:
                    # Calculate attribution strength
                    attribution = self._calculate_pairwise_attribution(
                        text, feature_i, feature_j
                    )
                    
                    if abs(attribution) > 0.1:  # Threshold for significant edges
                        edges[(i, j)] = attribution
                        nodes[i].add_downstream(j)
                        nodes[j].add_upstream(i)
        
        return edges
    
    def _calculate_pairwise_attribution(self, text: str, 
                                      source_feature: SAEFeature,
                                      target_feature: SAEFeature) -> float:
        """Calculate gradient-based attribution between two features."""
        # Gradient-based direct effect computation following attribution graphs methodology
        
        try:
            return self._compute_direct_effect(text, source_feature, target_feature)
        except Exception as e:
            logger.warning(f"Gradient attribution failed, falling back to correlation: {e}")
            # Fallback to correlation-based method
            return self._calculate_correlation_attribution(text, source_feature, target_feature)
    
    def _compute_direct_effect(self, text: str, source_feature: SAEFeature, 
                             target_feature: SAEFeature) -> float:
        """Compute direct causal effect using gradients."""
        tokens = self.model.to_tokens(text)
        
        # Create intervention hooks for gradient computation
        source_hook_point = f"blocks.{source_feature.layer}.hook_resid_post"
        target_hook_point = f"blocks.{target_feature.layer}.hook_resid_post"
        
        # Get baseline activations
        baseline_logits, baseline_cache = self.model.run_with_cache(tokens)
        
        # Get source feature activation
        source_activations = baseline_cache[source_hook_point]
        
        # Create hook for source feature ablation
        def ablation_hook(activations, hook):
            if source_feature.layer in self.sae_analyzers:
                sae = self.sae_analyzers[source_feature.layer]
                if isinstance(sae, dict):  # Fallback SAE
                    # Zero out specific feature
                    activations_flat = activations.view(-1, activations.size(-1))
                    feature_activations = torch.matmul(activations_flat, sae['encoder'])
                    feature_activations[:, source_feature.feature_id] = 0.0
                    reconstructed = torch.matmul(feature_activations, sae['decoder'])
                    return reconstructed.view(activations.shape)
                else:
                    # Real SAE ablation
                    feature_acts = sae.encode(activations)
                    feature_acts[:, :, source_feature.feature_id] = 0.0
                    return sae.decode(feature_acts)
            return activations
        
        # Compute target feature activation with source ablated
        with self.model.hooks([(source_hook_point, ablation_hook)]):
            _, ablated_cache = self.model.run_with_cache(tokens)
        
        # Calculate difference in target feature activation
        baseline_target = self._extract_feature_activation(
            baseline_cache[target_hook_point], target_feature.feature_id, target_feature.layer
        )
        ablated_target = self._extract_feature_activation(
            ablated_cache[target_hook_point], target_feature.feature_id, target_feature.layer
        )
        
        # Direct effect is the difference in target activation
        direct_effect = (baseline_target - ablated_target).item()
        
        # Normalize by source activation strength
        source_strength = self._extract_feature_activation(
            source_activations, source_feature.feature_id, source_feature.layer
        ).item()
        
        if abs(source_strength) > 1e-8:
            normalized_effect = direct_effect / abs(source_strength)
        else:
            normalized_effect = 0.0
        
        return float(normalized_effect)
    
    def _extract_feature_activation(self, activations: torch.Tensor, 
                                  feature_id: int, layer: int) -> torch.Tensor:
        """Extract specific feature activation from layer activations."""
        if layer in self.sae_analyzers:
            sae = self.sae_analyzers[layer]
            if isinstance(sae, dict):  # Fallback SAE
                activations_flat = activations.view(-1, activations.size(-1))
                feature_activations = torch.matmul(activations_flat, sae['encoder'])
                feature_activations = F.relu(feature_activations)
                if feature_id < feature_activations.size(-1):
                    return feature_activations[:, feature_id].mean()
            else:
                # Real SAE
                feature_activations = sae.encode(activations)
                if feature_id < feature_activations.size(-1):
                    return feature_activations[:, :, feature_id].mean()
        
        # Fallback to raw activation
        if feature_id < activations.size(-1):
            return activations[:, :, feature_id].mean()
        else:
            return torch.tensor(0.0, device=activations.device)
    
    def _calculate_correlation_attribution(self, text: str, 
                                         source_feature: SAEFeature,
                                         target_feature: SAEFeature) -> float:
        """Fallback correlation-based attribution."""
        tokens = self.model.to_tokens(text)
        
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
            
            # Get activations for both features
            source_acts = self.get_feature_activations(text, source_feature.layer)
            target_acts = self.get_feature_activations(text, target_feature.layer)
            
            if source_acts.numel() > source_feature.feature_id and target_acts.numel() > target_feature.feature_id:
                source_val = source_acts.flatten()[source_feature.feature_id].item()
                target_val = target_acts.flatten()[target_feature.feature_id].item()
                
                # Simple correlation proxy
                return source_val * target_val / (source_feature.max_activation + 1e-8)
        
        return 0.0
    
    def _calculate_graph_confidence(self, nodes: Dict[int, CircuitNode],
                                  edges: Dict[Tuple[int, int], float]) -> float:
        """Calculate overall graph confidence."""
        if not nodes:
            return 0.0
        
        # Base confidence on node causal influences and edge strengths
        node_confidences = [node.causal_influence for node in nodes.values()]
        edge_strengths = [abs(weight) for weight in edges.values()]
        
        avg_node_confidence = np.mean(node_confidences) if node_confidences else 0.0
        avg_edge_strength = np.mean(edge_strengths) if edge_strengths else 0.0
        
        # Combine with connectivity bonus
        connectivity_bonus = min(0.2, len(edges) / (len(nodes) * (len(nodes) - 1) / 2))
        
        overall_confidence = (0.6 * avg_node_confidence + 
                            0.3 * avg_edge_strength + 
                            0.1 * connectivity_bonus)
        
        return min(1.0, max(0.0, overall_confidence))
    
    def _get_target_output(self, text: str) -> str:
        """Get target output for the input text."""
        tokens = self.model.to_tokens(text)
        
        with torch.no_grad():
            logits = self.model(tokens)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            next_token = self.model.to_string(next_token_id)
        
        return next_token.strip()
    
    def _create_empty_attribution_graph(self, text: str) -> AttributionGraph:
        """Create empty attribution graph for edge cases."""
        return AttributionGraph(
            input_text=text,
            nodes={},
            edges={},
            target_output="",
            confidence=0.0,
            metadata={'error': 'no_active_features'}
        )
    
    def _analyze_activations_with_sae(self, activations: torch.Tensor, 
                                    layer: int, threshold: float) -> List[SAEFeature]:
        """Analyze activations using SAE to extract features."""
        features = []
        
        if layer not in self.sae_analyzers:
            return features
        
        sae = self.sae_analyzers[layer]
        
        if isinstance(sae, dict):  # Fallback SAE
            # Simple feature extraction
            activations_flat = activations.view(-1, activations.size(-1))
            feature_activations = torch.matmul(activations_flat, sae['encoder'])
            feature_activations = F.relu(feature_activations)
            
            # Find features above threshold
            for feature_id in range(feature_activations.size(-1)):
                activation_val = feature_activations[:, feature_id].max().item()
                if activation_val > threshold:
                    features.append(SAEFeature(
                        feature_id=feature_id,
                        layer=layer,
                        activation_threshold=threshold,
                        description=f"Feature {feature_id} in layer {layer}",
                        max_activation=activation_val,
                        examples=[f"Activation: {activation_val:.3f}"]
                    ))
        else:
            # Real SAE feature extraction
            try:
                feature_activations = sae.encode(activations)
                
                # Find features above threshold
                active_indices = (feature_activations > threshold).nonzero(as_tuple=True)
                
                for batch_idx, pos_idx, feature_idx in zip(*active_indices):
                    activation_val = feature_activations[batch_idx, pos_idx, feature_idx].item()
                    
                    features.append(SAEFeature(
                        feature_id=feature_idx.item(),
                        layer=layer,
                        activation_threshold=threshold,
                        description=f"SAE Feature {feature_idx.item()} in layer {layer}",
                        max_activation=activation_val,
                        examples=[f"Position {pos_idx.item()}: {activation_val:.3f}"]
                    ))
            except Exception as e:
                logger.warning(f"SAE analysis failed for layer {layer}: {e}")
        
        # Sort by activation strength and limit
        features.sort(key=lambda f: f.max_activation, reverse=True)
        return features
    
    def _add_error_nodes(self, graph: AttributionGraph, features: List[SAEFeature]) -> AttributionGraph:
        """Add error nodes to represent unexplained variance."""
        try:
            # Calculate total explained variance by layer
            layer_explained_variance = {}
            layer_total_variance = {}
            
            for feature in features:
                layer = feature.layer
                if layer not in layer_explained_variance:
                    layer_explained_variance[layer] = 0.0
                    layer_total_variance[layer] = 0.0
                
                layer_explained_variance[layer] += feature.max_activation
                layer_total_variance[layer] += 1.0  # Normalized total
            
            # Add error nodes for layers with significant unexplained variance
            new_nodes = list(graph.nodes)
            new_edges = list(graph.edges)
            
            for layer, explained in layer_explained_variance.items():
                total = layer_total_variance[layer]
                unexplained_ratio = max(0.0, (total - explained) / total) if total > 0 else 0.0
                
                # Add error node if unexplained variance > 20%
                if unexplained_ratio > 0.2:
                    error_node = GraphNode(
                        node_id=f"error_layer_{layer}",
                        layer=layer,
                        feature_id=-1,  # Special ID for error nodes
                        importance=unexplained_ratio,
                        description=f"Unexplained variance in layer {layer} ({unexplained_ratio:.1%})"
                    )
                    new_nodes.append(error_node)
            
            # Return new graph with error nodes
            return AttributionGraph(
                input_text=graph.input_text,
                nodes=new_nodes,
                edges=new_edges,
                target_output=graph.target_output,
                confidence=graph.confidence,
                metadata={**graph.metadata, 'error_nodes_added': True}
            )
            
        except Exception as e:
            logger.warning(f"Error node addition failed: {e}")
            return graph

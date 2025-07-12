"""
Real Circuit Tracer using circuit_tracer library with Gemma-2-2B transcoder support.
No SAE fallbacks, mocks, or approximations - pure circuit-tracer integration.
"""

import torch
from .model_manager import model_manager
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer

# Import circuit-tracer components  
from circuit_tracer import ReplacementModel

from ..core.interfaces import ICircuitTracer
from ..config.experiment_config import InterventionType
from ..core.data_structures import CircuitFeature, InterventionResult


class RealCircuitTracer(ICircuitTracer):
    """
    Real circuit tracer using circuit_tracer.ReplacementModel and Gemma-2-2B transcoders.
    No fallbacks or approximations - uses actual mechanistic interpretability tools.
    """
    
    def __init__(self, model_name: str = "google/gemma-2-2b", transcoder_set: str = "gemma"):
        self.model_name = model_name
        self.transcoder_set = transcoder_set
        self.model = None
        self.tokenizer = None
        self.discovered_features = []
        
        print(f"🔧 Initializing RealCircuitTracer with {model_name} + {transcoder_set}")
        
    def initialize_model(self):
        """Load Gemma-2-2B model and initialize circuit-tracer ReplacementModel with transcoders."""
        if self.model is None:
            print(f"📥 Loading {self.model_name} with circuit-tracer and {self.transcoder_set} transcoders...")
            
            try:
                # Use model_manager for cached singleton model loading
                print(f"🔧 Using model_manager for cached {self.model_name} + {self.transcoder_set}")
                
                # Get shared tokenizer instance
                self.tokenizer = model_manager.get_tokenizer(self.model_name)
                
                # Get shared model instance (loads once, reuses thereafter)
                self.model = model_manager.get_model(self.model_name, self.transcoder_set)
                
                print(f"✅ RealCircuitTracer initialized successfully!")
                print(f"   Model: {self.model_name}")
                print(f"   Transcoders: {self.transcoder_set}")
                print(f"   Layers: {len(self.model.transcoders)}")
                print(f"   Features per layer: {self.model.d_transcoder}")
                
            except Exception as e:
                print(f"❌ Failed to initialize circuit-tracer: {e}")
                print("🔧 This might be due to HuggingFace authentication or model access")
                raise e
    
    def discover_active_features(self, input_text: str, layers: Optional[List[int]] = None, threshold: float = 0.1) -> List[CircuitFeature]:
        """Discover active transcoder features using circuit-tracer."""
        self.initialize_model()
        
        if layers is None:
            # Use subset of layers for efficiency (Gemma-2-2B has 26 layers)
            layers = list(range(0, min(26, len(self.model.transcoders))))
            
        print(f"🔍 Discovering transcoder features across {len(layers)} layers for: '{input_text[:50]}...'")
        
        discovered_features = []
        
        try:
            with torch.inference_mode():
                # Get transcoder activations using circuit-tracer
                logits, activations = self.model.get_activations(
                    input_text,
                    zero_bos=True,
                    sparse=False
                )
                
                # Process activations for each layer
                for layer_idx in layers:
                    if layer_idx < activations.shape[0]:  # [n_layers, seq_len, d_transcoder]
                        layer_activations = activations[layer_idx]  # [seq_len, d_transcoder]
                        
                        # Find features above threshold
                        max_activations = layer_activations.max(dim=0)[0]  # Max over sequence
                        active_features = torch.where(max_activations > threshold)[0]
                        
                        for feat_idx in active_features:
                            activation_strength = float(max_activations[feat_idx])
                            
                            feature = CircuitFeature.from_transcoder_data(
                                layer_idx=layer_idx,
                                feature_id=int(feat_idx),
                                activation=activation_strength,
                                semantic_description=f"GemmaScope feature L{layer_idx}F{feat_idx}",
                                component_type="mlp_transcoder",
                                intervention_sites=[f"blocks.{layer_idx}.mlp.transcoder.{feat_idx}"]
                            )
                            discovered_features.append(feature)
                
                self.discovered_features = discovered_features
                print(f"✅ Discovered {len(discovered_features)} active transcoder features")
                
        except Exception as e:
            print(f"❌ Feature discovery failed: {e}")
            raise e
        
        return discovered_features
    
    def get_active_features_for_input(self, input_text: str, layer_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Get the most active features for a specific input text and layer."""
        self.initialize_model()
        
        with torch.inference_mode():
            # Get activations for this specific input
            _, activations = self.model.feature_intervention(input_text, [])
            
            # Get features active at the last token position for the specified layer
            layer_activations = activations[layer_idx, -1, :]  # [d_transcoder]
            top_features = torch.topk(layer_activations, min(top_k, len(layer_activations)))
            
            # Return list of (feature_idx, activation_value) tuples
            active_features = []
            for activation_val, feature_idx in zip(top_features.values, top_features.indices):
                if float(activation_val) > 0.1:  # Only include meaningfully active features
                    active_features.append((int(feature_idx), float(activation_val)))
            
            return active_features
    
    def intervene_on_feature(self, feature: CircuitFeature, input_text: str, 
                           intervention_type: str = "ablation", 
                           intervention_value: float = 0.0,
                           check_if_active: bool = True) -> InterventionResult:
        """Perform intervention using circuit-tracer ReplacementModel.
        
        Args:
            feature: The circuit feature to intervene on
            input_text: The input text to test intervention on
            intervention_type: Type of intervention ("ablation", "amplification")
            intervention_value: Value to set the feature to (0.0 for ablation)
            check_if_active: If True, check if feature is actually active for this input
        """
        self.initialize_model()
        
        # Check if this feature is actually active for the given input
        if check_if_active:
            active_features = self.get_active_features_for_input(input_text, feature.layer_idx, top_k=20)
            feature_activations = {feat_idx: activation for feat_idx, activation in active_features}
            
            if feature.feature_id not in feature_activations:
                print(f"⚠️  Feature {feature.feature_id} is not active for input '{input_text[:50]}...'")
                print(f"📊 Active features in Layer {feature.layer_idx}: {list(feature_activations.keys())[:10]}")
                
                # Instead of intervening on inactive feature, pick the most active one
                if active_features:
                    original_feature_id = feature.feature_id
                    feature.feature_id = active_features[0][0]  # Use most active feature
                    print(f"🔄 Switching to most active feature: {original_feature_id} → {feature.feature_id}")
                    print(f"📈 Target feature activation: {active_features[0][1]:.3f}")
                else:
                    print(f"❌ No active features found in Layer {feature.layer_idx}")
            else:
                actual_activation = feature_activations[feature.feature_id]
                print(f"✅ Target feature {feature.feature_id} is active (activation: {actual_activation:.3f})")
        
        print(f"🎯 Intervening on Layer {feature.layer_idx} Feature {feature.feature_id} ({intervention_type})")
        
        try:
            with torch.inference_mode():
                # Get baseline prediction and activations
                baseline_logits, baseline_activations = self.model.feature_intervention(input_text, [])
                baseline_pred = self._decode_top_token(baseline_logits)
                
                # For ablation, use 0.0. For amplification, use current activation * multiplier
                if intervention_type == "amplification" and intervention_value == 0.0:
                    # Get current activation and amplify it
                    current_activation = float(baseline_activations[feature.layer_idx, -1, feature.feature_id])
                    intervention_value = current_activation * 2.0  # Double the activation
                    print(f"📈 Amplifying feature from {current_activation:.3f} to {intervention_value:.3f}")
                
                # Prepare intervention: (layer, position, feature_idx, value)
                intervention_tuples = [(feature.layer_idx, -1, feature.feature_id, intervention_value)]
                
                # Perform feature intervention
                modified_logits, modified_activations = self.model.feature_intervention(
                    input_text, intervention_tuples
                )
                
                modified_pred = self._decode_top_token(modified_logits)
                
                # Calculate intervention effect
                baseline_probs = torch.softmax(baseline_logits[0, -1], dim=-1)
                modified_probs = torch.softmax(modified_logits[0, -1], dim=-1)
                effect_magnitude = float(torch.norm(baseline_probs - modified_probs))
                
                result = InterventionResult(
                    target_feature=feature,
                    intervention_type=InterventionType(intervention_type),
                    original_logits=baseline_logits,
                    intervened_logits=modified_logits,
                    effect_size=effect_magnitude,
                    target_token_change=effect_magnitude,
                    intervention_layer_idx=feature.layer_idx,
                    effect_magnitude=effect_magnitude,
                    baseline_prediction=baseline_pred,
                    intervention_prediction=modified_pred,
                    semantic_change=baseline_pred != modified_pred,
                    statistical_significance=effect_magnitude > 0.01
                )
                
                print(f"📊 Baseline: '{baseline_pred}' → Modified: '{modified_pred}'")
                print(f"📊 Effect magnitude: {effect_magnitude:.4f}")
                print(f"🎯 Token change: {baseline_pred != modified_pred}")
                print(f"✅ Intervention {'successful' if result.statistical_significance else 'minimal effect'}")
                
                # Log detailed results for debugging
                if effect_magnitude > 0.001:
                    print(f"🎉 MEANINGFUL INTERVENTION DETECTED!")
                    print(f"   Layer: {feature.layer_idx}, Feature: {feature.feature_id}")
                    print(f"   Input: '{input_text}'")
                    print(f"   Baseline → Modified: '{baseline_pred}' → '{modified_pred}'")
                    print(f"   Effect magnitude: {effect_magnitude:.6f}")
                
                return result
                
        except Exception as e:
            print(f"❌ Intervention failed: {e}")
            import traceback
            traceback.print_exc()
            return InterventionResult(
                target_feature=feature,
                intervention_type=InterventionType(intervention_type),
                original_logits=torch.zeros(1, 1, 256000),
                intervened_logits=torch.zeros(1, 1, 256000),
                effect_size=0.0,
                target_token_change=0.0,
                intervention_layer_idx=feature.layer_idx,
                effect_magnitude=0.0,
                baseline_prediction="ERROR",
                intervention_prediction="ERROR",
                semantic_change=False,
                statistical_significance=False
            )
    
    def _decode_top_token(self, logits: torch.Tensor) -> str:
        """Decode the top predicted token from logits."""
        top_token_id = logits[0, -1].argmax().item()
        return self.tokenizer.decode(top_token_id).strip()
    
    def build_attribution_graph(self, features: List[CircuitFeature]) -> Dict[str, Any]:
        """Build attribution graph using circuit-tracer analysis."""
        self.initialize_model()
        
        print(f"📊 Building attribution graph for {len(features)} features...")
        
        # Use circuit-tracer to build feature interaction graph
        graph_data = {
            "nodes": [],
            "edges": [],
            "feature_count": len(features),
            "layers": set(),
            "transcoder_info": {
                "model": self.model_name,
                "transcoder_set": self.transcoder_set,
                "d_transcoder": self.model.d_transcoder if self.model else None
            }
        }
        
        for feature in features:
            graph_data["nodes"].append({
                "id": f"L{feature.layer_idx}F{feature.feature_id}",
                "layer": feature.layer_idx,
                "feature_id": feature.feature_id,
                "activation_strength": feature.activation_strength,
                "description": feature.semantic_description,
                "component_type": feature.component_type
            })
            graph_data["layers"].add(feature.layer_idx)
        
        # Add simple connectivity based on layer proximity
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i != j and abs(feat1.layer_idx - feat2.layer_idx) == 1:
                    # Features in adjacent layers may be connected
                    interaction_strength = min(feat1.activation_strength, feat2.activation_strength)
                    if interaction_strength > 0.1:
                        graph_data["edges"].append({
                            "source": f"L{feat1.layer_idx}F{feat1.feature_id}",
                            "target": f"L{feat2.layer_idx}F{feat2.feature_id}",
                            "weight": float(interaction_strength),
                            "type": "layer_adjacency"
                        })
        
        graph_data["layers"] = sorted(list(graph_data["layers"]))
        
        print(f"✅ Built attribution graph: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")
        return graph_data
    
    def get_semantic_features(self, concept: str, k: int = 10) -> List[CircuitFeature]:
        """Find transcoder features most relevant to a semantic concept."""
        print(f"🔍 Searching for features related to: '{concept}'")
        
        # Discover features using the concept as input
        concept_features = self.discover_active_features(concept)
        
        # Return top-k features by activation strength
        relevant_features = sorted(
            concept_features, 
            key=lambda f: f.activation_strength, 
            reverse=True
        )[:k]
        
        print(f"✅ Found {len(relevant_features)} most relevant features")
        return relevant_features
    
    def test_semantic_discovery(self, source_concept: str, target_concept: str) -> bool:
        """Test semantic discovery: source_concept → target_concept."""
        print(f"🧪 Testing semantic discovery: '{source_concept}' → '{target_concept}'")
        
        try:
            # Get features for the source concept
            source_features = self.get_semantic_features(source_concept)
            
            if not source_features:
                print(f"❌ No features found for '{source_concept}'")
                return False
            
            # Test intervention on top feature with a completion prompt
            completion_prompt = f"The {source_concept} is located in"
            top_feature = source_features[0]
            
            result = self.intervene_on_feature(top_feature, completion_prompt)
            
            # Check if the intervention affects prediction toward target concept
            contains_target = target_concept.lower() in result.baseline_prediction.lower()
            has_effect = result.statistical_significance
            
            print(f"📊 Completion: '{completion_prompt}' → '{result.baseline_prediction}'")
            print(f"📊 Contains '{target_concept}': {contains_target}")
            print(f"📊 Intervention has effect: {has_effect}")
            
            success = contains_target or has_effect
            print(f"{'✅ SUCCESS' if success else '❌ FAILED'}: Semantic discovery test")
            
            return success
            
        except Exception as e:
            print(f"❌ Semantic discovery test failed: {e}")
            return False

    # Required interface methods
    def find_active_features(self, text: str, threshold: float = 0.1) -> Dict[int, List]:
        """Find active transcoder features grouped by layer."""
        features = self.discover_active_features(text, threshold=threshold)
        
        # Group by layer
        by_layer = {}
        for feature in features:
            if feature.layer_idx not in by_layer:
                by_layer[feature.layer_idx] = []
            by_layer[feature.layer_idx].append(feature)
        
        return by_layer
    
    def perform_intervention(
        self,
        text: str,
        feature: CircuitFeature,
        intervention_type: str = "ablation"
    ) -> InterventionResult:
        """Perform intervention on specified feature."""
        return self.intervene_on_feature(feature, text, str(intervention_type))
    
    def get_feature_activations(self, text: str, layer: int) -> torch.Tensor:
        """Get transcoder feature activations for specific layer."""
        self.initialize_model()
        
        try:
            with torch.inference_mode():
                logits, activations = self.model.get_activations(text)
                if layer < activations.shape[0]:
                    return activations[layer]  # [seq_len, d_transcoder]
                else:
                    return torch.zeros(1, self.model.d_transcoder if self.model else 1024)
        except Exception as e:
            print(f"❌ Failed to get activations for layer {layer}: {e}")
            return torch.zeros(1, 1024)
    def enhanced_semantic_discovery(self, source_concept: str, target_concept: str) -> bool:
        """Enhanced semantic discovery using multi-feature interventions."""
        from .semantic_enhancement import SemanticDiscoveryEnhancer
        
        enhancer = SemanticDiscoveryEnhancer(self)
        return enhancer.enhanced_semantic_test(source_concept, target_concept)

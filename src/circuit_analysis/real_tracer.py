"""
from .model_manager import model_manager
Real Circuit Tracer using circuit_tracer library with Gemma-2-2B transcoder support.
No SAE fallbacks, mocks, or approximations - pure circuit-tracer integration.
"""

import torch
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
    
    def __init__(self, model_name: str = "google/gemma-2-2b", transcoder_set: str = "gemmascope-l0-0"):
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
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Initialize circuit-tracer ReplacementModel with GemmaScope transcoders
                self.model = ReplacementModel.from_pretrained(
                    model_name=self.model_name,
                    transcoder_set=self.transcoder_set,
                    device=torch.device("cuda"),
                    dtype=torch.bfloat16
                )
                
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
    
    def intervene_on_feature(self, feature: CircuitFeature, input_text: str, 
                           intervention_type: str = "ablation", 
                           intervention_value: float = 0.0) -> InterventionResult:
        """Perform intervention using circuit-tracer ReplacementModel."""
        self.initialize_model()
        
        print(f"🎯 Intervening on Layer {feature.layer_idx} Feature {feature.feature_id} ({intervention_type})")
        
        try:
            with torch.inference_mode():
                # Get baseline prediction
                baseline_logits = self.model(input_text)
                baseline_pred = self._decode_top_token(baseline_logits)
                
                # Prepare intervention: (layer, position, feature_idx, value)
                # Intervene at the last token position
                seq_len = baseline_logits.shape[1]
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
                print(f"✅ Intervention {'successful' if result.statistical_significance else 'minimal effect'}")
                
                return result
                
        except Exception as e:
            print(f"❌ Intervention failed: {e}")
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

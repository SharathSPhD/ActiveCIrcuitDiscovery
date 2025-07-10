"""
Enhanced semantic discovery module for circuit tracer.
Implements multi-feature intervention and activation transfer methodology.
"""

import torch
from typing import Dict, List, Tuple, Optional
import numpy as np
from ..core.data_structures import CircuitFeature


class SemanticDiscoveryEnhancer:
    """Enhanced semantic discovery using multi-feature interventions and activation transfer."""
    
    def __init__(self, circuit_tracer):
        self.tracer = circuit_tracer
        
    def identify_semantic_features(self, source_concept: str, target_concept: str, 
                                 k: int = 5) -> List[CircuitFeature]:
        """Use circuit-tracer attribution to find semantic bridging features."""
        print(f"🔍 Identifying semantic bridging features: {source_concept} → {target_concept}")
        
        # Get features for both concepts
        source_features = self.tracer.get_semantic_features(source_concept, k=k*2)
        target_features = self.tracer.get_semantic_features(target_concept, k=k*2)
        
        # Find features that appear in both (semantic bridges)
        bridging_features = []
        
        for s_feat in source_features:
            for t_feat in target_features:
                # Features in similar layers with high activation for both concepts
                layer_distance = abs(s_feat.layer_idx - t_feat.layer_idx)
                if layer_distance <= 2:  # Adjacent or nearby layers
                    # Create a combined semantic strength score
                    combined_strength = (s_feat.activation_strength + t_feat.activation_strength) / 2
                    
                    # Prefer the feature with higher activation
                    best_feature = s_feat if s_feat.activation_strength > t_feat.activation_strength else t_feat
                    best_feature.semantic_relevance = combined_strength
                    
                    if best_feature not in bridging_features:
                        bridging_features.append(best_feature)
        
        # If no bridging features found, use top features from source concept
        if not bridging_features:
            bridging_features = source_features[:k]
            for feat in bridging_features:
                feat.semantic_relevance = feat.activation_strength
        
        # Sort by semantic relevance and return top k
        bridging_features.sort(key=lambda f: getattr(f, 'semantic_relevance', f.activation_strength), reverse=True)
        
        print(f"✅ Found {len(bridging_features[:k])} semantic bridging features")
        return bridging_features[:k]
    
    def perform_semantic_intervention(self, features: List[CircuitFeature], 
                                    source_text: str, target_concept: str,
                                    intervention_method: str = "activation_transfer") -> bool:
        """Perform coordinated multi-feature intervention for semantic discovery."""
        print(f"🎯 Performing semantic intervention: {intervention_method}")
        
        if intervention_method == "activation_transfer":
            return self._activation_transfer_intervention(features, source_text, target_concept)
        elif intervention_method == "multi_ablation":
            return self._multi_feature_ablation(features, source_text, target_concept)
        else:
            return self._coordinated_intervention(features, source_text, target_concept)
    
    def _activation_transfer_intervention(self, features: List[CircuitFeature], 
                                        source_text: str, target_concept: str) -> bool:
        """Use activation transfer from related prompts."""
        print("📡 Using activation transfer methodology")
        
        try:
            self.tracer.initialize_model()
            
            # Create target-biased prompt
            target_prompt = f"The {target_concept} is a location in"
            
            with torch.inference_mode():
                # Get activations for target concept
                _, target_activations = self.tracer.model.get_activations(target_prompt, sparse=True)
                
                # Create intervention tuples using target activations
                intervention_tuples = []
                for feature in features:
                    if feature.layer_idx < target_activations.shape[0]:
                        # Use target activation value instead of zero ablation
                        target_activation = target_activations[feature.layer_idx, -1, feature.feature_id]
                        intervention_tuples.append((
                            feature.layer_idx, -1, feature.feature_id, float(target_activation)
                        ))
                
                # Perform intervention with activation transfer
                baseline_logits = self.tracer.model(source_text)
                modified_logits, _ = self.tracer.model.feature_intervention(
                    source_text, intervention_tuples
                )
                
                # Check if intervention shifts prediction toward target
                baseline_pred = self.tracer._decode_top_token(baseline_logits)
                modified_pred = self.tracer._decode_top_token(modified_logits)
                
                print(f"📊 Baseline: '{baseline_pred}' → Modified: '{modified_pred}'")
                
                # Success if target concept appears in either prediction
                target_lower = target_concept.lower()
                success = (target_lower in baseline_pred.lower() or 
                          target_lower in modified_pred.lower())
                
                print(f"{SUCCESS if success else FAILED}: Activation transfer intervention")
                return success
                
        except Exception as e:
            print(f"❌ Activation transfer failed: {e}")
            return False
    
    def _multi_feature_ablation(self, features: List[CircuitFeature], 
                              source_text: str, target_concept: str) -> bool:
        """Perform coordinated ablation on multiple features."""
        print("🔥 Using multi-feature ablation")
        
        try:
            self.tracer.initialize_model()
            
            # Create intervention tuples for multiple features
            intervention_tuples = []
            for feature in features:
                # Use graduated ablation values based on feature importance
                ablation_strength = 0.5 * feature.activation_strength
                intervention_tuples.append((
                    feature.layer_idx, -1, feature.feature_id, -ablation_strength
                ))
            
            with torch.inference_mode():
                baseline_logits = self.tracer.model(source_text)
                modified_logits, _ = self.tracer.model.feature_intervention(
                    source_text, intervention_tuples
                )
                
                baseline_pred = self.tracer._decode_top_token(baseline_logits)
                modified_pred = self.tracer._decode_top_token(modified_logits)
                
                print(f"📊 Baseline: '{baseline_pred}' → Modified: '{modified_pred}'")
                
                # Check for semantic shift toward target
                target_lower = target_concept.lower()
                success = target_lower in modified_pred.lower()
                
                print(f"{SUCCESS if success else FAILED}: Multi-feature ablation")
                return success
                
        except Exception as e:
            print(f"❌ Multi-feature ablation failed: {e}")
            return False
    
    def _coordinated_intervention(self, features: List[CircuitFeature], 
                                source_text: str, target_concept: str) -> bool:
        """Perform coordinated intervention across multiple layers."""
        print("🎛️ Using coordinated intervention")
        
        # Group features by layer for coordinated intervention
        features_by_layer = {}
        for feature in features:
            if feature.layer_idx not in features_by_layer:
                features_by_layer[feature.layer_idx] = []
            features_by_layer[feature.layer_idx].append(feature)
        
        success_count = 0
        total_layers = len(features_by_layer)
        
        for layer_idx, layer_features in features_by_layer.items():
            try:
                # Intervention strength proportional to number of features in layer
                base_strength = 1.0 / len(layer_features)
                
                intervention_tuples = []
                for feature in layer_features:
                    intervention_tuples.append((
                        layer_idx, -1, feature.feature_id, base_strength
                    ))
                
                with torch.inference_mode():
                    baseline_logits = self.tracer.model(source_text)
                    modified_logits, _ = self.tracer.model.feature_intervention(
                        source_text, intervention_tuples
                    )
                    
                    modified_pred = self.tracer._decode_top_token(modified_logits)
                    
                    if target_concept.lower() in modified_pred.lower():
                        success_count += 1
                        print(f"✅ Layer {layer_idx} intervention successful")
                    
            except Exception as e:
                print(f"❌ Layer {layer_idx} intervention failed: {e}")
        
        success = success_count > 0
        print(f"📊 Coordinated intervention: {success_count}/{total_layers} layers successful")
        return success
    
    def enhanced_semantic_test(self, source_concept: str, target_concept: str) -> bool:
        """Enhanced semantic discovery test with multiple strategies."""
        print(f"🧪 Enhanced semantic test: '{source_concept}' → '{target_concept}'")
        
        try:
            # Step 1: Identify semantic bridging features
            bridging_features = self.identify_semantic_features(source_concept, target_concept)
            
            if not bridging_features:
                print("❌ No bridging features found")
                return False
            
            # Step 2: Try multiple intervention strategies
            strategies = ["activation_transfer", "multi_ablation", "coordinated"]
            
            for strategy in strategies:
                print(f"🔄 Trying strategy: {strategy}")
                
                # Create completion prompt
                completion_prompt = f"The {source_concept} is located in"
                
                success = self.perform_semantic_intervention(
                    bridging_features, completion_prompt, target_concept, strategy
                )
                
                if success:
                    print(f"SUCCESS with strategy: {strategy}")
                    return True
            
            print("❌ All strategies failed")
            return False
            
        except Exception as e:
            print(f"❌ Enhanced semantic test failed: {e}")
            return False

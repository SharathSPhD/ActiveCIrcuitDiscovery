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
                
                # Prepare intervention tuples: (layer, position, feature_idx, value)
                # Use -1 for last token position as per circuit-tracer convention
                intervention_tuples = [(feature.layer_idx, -1, feature.feature_id, intervention_value)]
                
                # Perform feature intervention using correct circuit-tracer API
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
                
                if effect_magnitude > 0.01:
                    print("✅ Intervention meaningful effect")
                else:
                    print("✅ Intervention minimal effect")
                    
                return result
                
        except Exception as e:
            print(f"❌ Intervention failed: {e}")
            return InterventionResult(
                target_feature=feature,
                intervention_type=InterventionType(intervention_type),
                original_logits=torch.zeros((1, 1, 256000)),
                intervened_logits=torch.zeros((1, 1, 256000)),
                effect_size=0.0,
                target_token_change=0.0,
                intervention_layer_idx=feature.layer_idx,
                effect_magnitude=0.0,
                baseline_prediction="ERROR",
                intervention_prediction="ERROR",
                semantic_change=False,
                statistical_significance=False
            )
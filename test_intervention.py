#!/usr/bin/env python3
"""Test circuit interventions to debug why they're not working."""

from circuit_tracer import ReplacementModel
import torch

def test_interventions():
    model = ReplacementModel.from_pretrained(
        model_name="google/gemma-2-2b",
        transcoder_set="gemma",
        device=torch.device("cuda"),
        dtype=torch.bfloat16
    )

    test_text = "The Golden Gate Bridge is located in"

    # Get baseline logits and activations
    print("Getting baseline activations...")
    baseline_logits, activations = model.feature_intervention(test_text, [])

    # Get the most active features in layer 8
    layer_8_activations = activations[8, -1, :]  # Last token position
    top_features = torch.topk(layer_8_activations, 5)

    print("Top 5 most active features in Layer 8:")
    for i, (activation_val, feature_idx) in enumerate(zip(top_features.values, top_features.indices)):
        print(f"  {i+1}. Feature {feature_idx}: {activation_val:.3f}")

    print("\nTesting ablation of actually active features:")
    successful_interventions = []
    
    for i, (activation_val, feature_idx) in enumerate(zip(top_features.values, top_features.indices)):
        feature_idx = int(feature_idx)
        activation_val = float(activation_val)
        
        print(f"\nTest {i+1}: Feature {feature_idx} (activation: {activation_val:.3f})")
        
        # Test ablating this feature
        interventions = [(8, -1, feature_idx, 0.0)]
        modified_logits, _ = model.feature_intervention(test_text, interventions)
        diff = torch.norm(baseline_logits - modified_logits)
        
        # Check token change
        baseline_token = torch.argmax(baseline_logits[0, -1]).item()
        modified_token = torch.argmax(modified_logits[0, -1]).item()
        token_changed = baseline_token != modified_token
        
        print(f"  Logit difference: {diff:.6f}")
        print(f"  Token change: {baseline_token} -> {modified_token} (changed: {token_changed})")
        
        if diff > 0.001:  # If there is a measurable effect
            print(f"  ✅ SUCCESSFUL INTERVENTION!")
            successful_interventions.append((feature_idx, diff, token_changed))
        else:
            print(f"  ❌ Minimal effect")
    
    if not successful_interventions:
        print("\n❌ All ablations had minimal effect. Trying amplification...")
        
        # Try amplifying the most active feature
        most_active_idx = int(top_features.indices[0])
        most_active_val = float(top_features.values[0])
        
        interventions = [(8, -1, most_active_idx, most_active_val * 3)]  # Triple the activation
        modified_logits, _ = model.feature_intervention(test_text, interventions)
        diff = torch.norm(baseline_logits - modified_logits)
        print(f"\nAmplification test - Logit difference: {diff:.6f}")
        
        if diff > 0.001:
            print("✅ Amplification works!")
        else:
            print("❌ Even amplification has no effect")
    else:
        print(f"\n✅ Found {len(successful_interventions)} working interventions!")
        for feature_idx, diff, token_changed in successful_interventions:
            print(f"  - Feature {feature_idx}: effect {diff:.6f}, token change: {token_changed}")

if __name__ == "__main__":
    test_interventions()
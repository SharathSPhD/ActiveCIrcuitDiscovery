#!/usr/bin/env python3
"""
Comprehensive test showing BEFORE vs AFTER intervention effectiveness.
This demonstrates the complete fix for the circuit intervention system.
"""

import sys
sys.path.append(".")

from src.circuit_analysis.real_tracer import RealCircuitTracer
from src.active_inference.semantic_circuit_agent import SemanticCircuitAgent
from src.config.experiment_config import CompleteConfig
from src.core.data_structures import CircuitFeature

def test_before_after_comparison():
    print("🔬 BEFORE vs AFTER: Enhanced Circuit Intervention System")
    print("=" * 80)
    
    # Initialize components
    tracer = RealCircuitTracer()
    config = CompleteConfig()
    agent = SemanticCircuitAgent(config)
    
    test_input = "The Golden Gate Bridge is located in"
    print(f"🔍 Test Input: '{test_input}'")
    print(f"🎯 Expected Output: 'San Francisco' (semantic completion)")
    
    # === BEFORE: Simulate Old System Behavior ===
    print(f"\n" + "="*60)
    print("❌ BEFORE: Old System (Broken)")
    print("="*60)
    
    # Create a fake inactive feature (like the old system selected)
    old_feature = CircuitFeature(
        feature_id=2,
        layer_idx=0,
        activation_strength=0.1,
        description="Early layer feature L0F2 (inactive)",
        max_activation=0.1,
        examples=[]
    )
    
    print(f"🎯 Old Target: Layer {old_feature.layer_idx}, Feature {old_feature.feature_id}")
    print(f"📊 Activation Strength: {old_feature.activation_strength:.3f} (very low)")
    
    # Test if this feature is actually active
    old_actives = tracer.get_active_features_for_input(test_input, old_feature.layer_idx, top_k=10)
    old_active_dict = {idx: act for idx, act in old_actives}
    old_is_active = old_feature.feature_id in old_active_dict
    old_actual_activation = old_active_dict.get(old_feature.feature_id, 0.0)
    
    print(f"❌ Actually Active: {'YES' if old_is_active else 'NO'} (activation: {old_actual_activation:.3f})")
    
    # Perform intervention on inactive feature
    old_result = tracer.intervene_on_feature(old_feature, test_input, check_if_active=False)
    print(f"📊 Old Results:")
    print(f"   Effect Magnitude: {old_result.effect_magnitude:.6f}")
    print(f"   Baseline: '{old_result.baseline_prediction}'")
    print(f"   Modified: '{old_result.intervention_prediction}'")
    print(f"   Token Changed: {old_result.baseline_prediction != old_result.intervention_prediction}")
    print(f"   Significant: {'YES' if old_result.statistical_significance else 'NO'}")
    
    # === AFTER: New Enhanced System ===
    print(f"\n" + "="*60)
    print("✅ AFTER: Enhanced System (Fixed)")
    print("="*60)
    
    # Discover features using real system
    discovered_features = tracer.discover_active_features(test_input, layers=[6, 7, 8, 9, 10], threshold=0.5)
    print(f"🔍 Discovered: {len(discovered_features)} features in middle layers")
    
    # Initialize agent and use Enhanced EFE selection
    agent.initialize_from_circuit_features(discovered_features)
    new_feature, intervention_type = agent.select_intervention_with_active_features(discovered_features, test_input)
    
    print(f"🎯 New Target: Layer {new_feature.layer_idx}, Feature {new_feature.feature_id}")
    print(f"📊 Activation Strength: {new_feature.activation_strength:.3f}")
    
    # Verify this feature is actually active
    new_actives = tracer.get_active_features_for_input(test_input, new_feature.layer_idx, top_k=10)
    new_active_dict = {idx: act for idx, act in new_actives}
    new_is_active = new_feature.feature_id in new_active_dict
    new_actual_activation = new_active_dict.get(new_feature.feature_id, 0.0)
    
    print(f"✅ Actually Active: {'YES' if new_is_active else 'NO'} (activation: {new_actual_activation:.3f})")
    
    # Perform enhanced intervention
    new_result = tracer.intervene_on_feature(new_feature, test_input, check_if_active=True)
    print(f"📊 New Results:")
    print(f"   Effect Magnitude: {new_result.effect_magnitude:.6f}")
    print(f"   Baseline: '{new_result.baseline_prediction}'")
    print(f"   Modified: '{new_result.intervention_prediction}'")
    print(f"   Token Changed: {new_result.baseline_prediction != new_result.intervention_prediction}")
    print(f"   Significant: {'YES' if new_result.statistical_significance else 'NO'}")
    
    # === COMPARISON ===
    print(f"\n" + "="*60)
    print("📊 IMPROVEMENT ANALYSIS")
    print("="*60)
    
    print(f"🎯 Feature Selection:")
    print(f"   Old: L{old_feature.layer_idx}F{old_feature.feature_id} ({'ACTIVE' if old_is_active else 'INACTIVE'})")
    print(f"   New: L{new_feature.layer_idx}F{new_feature.feature_id} ({'ACTIVE' if new_is_active else 'INACTIVE'})")
    
    print(f"\n📈 Activation Strength:")
    print(f"   Old: {old_actual_activation:.3f}")
    print(f"   New: {new_actual_activation:.3f}")
    print(f"   Improvement: {new_actual_activation / max(old_actual_activation, 0.001):.1f}x")
    
    print(f"\n⚡ Effect Magnitude:")
    print(f"   Old: {old_result.effect_magnitude:.6f}")
    print(f"   New: {new_result.effect_magnitude:.6f}")
    if old_result.effect_magnitude > 0:
        effect_improvement = new_result.effect_magnitude / old_result.effect_magnitude
        print(f"   Improvement: {effect_improvement:.1f}x")
    else:
        print(f"   Improvement: INFINITE (0.000000 → {new_result.effect_magnitude:.6f})")
    
    print(f"\n🎯 Layer Targeting:")
    print(f"   Old: Layer {old_feature.layer_idx} (early layer - less semantic)")
    print(f"   New: Layer {new_feature.layer_idx} (middle layer - more semantic)")
    
    print(f"\n✅ Success Rate:")
    old_success = old_result.statistical_significance
    new_success = new_result.statistical_significance
    print(f"   Old: {'SUCCESS' if old_success else 'FAILED'}")
    print(f"   New: {'SUCCESS' if new_success else 'FAILED'}")
    
    # === SUMMARY ===
    print(f"\n" + "="*60)
    print("🏆 FINAL ASSESSMENT")
    print("="*60)
    
    overall_improvement = (
        new_is_active and not old_is_active and
        new_result.effect_magnitude > old_result.effect_magnitude and
        new_success
    )
    
    if overall_improvement:
        print("🎉 COMPLETE SUCCESS: Enhanced system dramatically outperforms old system!")
        print(f"✅ Key Improvements:")
        print(f"   - Target Selection: Inactive → Active features")
        print(f"   - Layer Focus: Early (L{old_feature.layer_idx}) → Middle (L{new_feature.layer_idx})")
        print(f"   - Effect Magnitude: {old_result.effect_magnitude:.6f} → {new_result.effect_magnitude:.6f}")
        print(f"   - Success Rate: {'PASS' if old_success else 'FAIL'} → {'PASS' if new_success else 'FAIL'}")
        
        print(f"\n🔧 Technical Details:")
        print(f"   - Input: '{test_input}'")
        print(f"   - Baseline Prediction: '{new_result.baseline_prediction}'")
        print(f"   - Model: google/gemma-2-2b with GemmaScope transcoders")
        print(f"   - Method: Enhanced EFE + Activity-aware selection")
        
        return True
    else:
        print("⚠️ PARTIAL SUCCESS: Some improvements but needs refinement")
        return False

if __name__ == "__main__":
    try:
        success = test_before_after_comparison()
        print(f"\n{'🎉 INTERVENTION SYSTEM FULLY OPERATIONAL' if success else '⚠️ SYSTEM NEEDS FURTHER TUNING'}")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
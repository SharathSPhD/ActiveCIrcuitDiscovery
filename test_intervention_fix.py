#!/usr/bin/env python3
"""Test that the enhanced intervention system targets active features."""

import sys
sys.path.append(".")

from src.circuit_analysis.real_tracer import RealCircuitTracer
from src.core.data_structures import CircuitFeature

def test_enhanced_intervention():
    print("🧪 Testing Enhanced Intervention (Active Feature Targeting)")
    print("=" * 60)
    
    # Initialize tracer
    tracer = RealCircuitTracer()
    test_input = "The Golden Gate Bridge is located in"
    
    print(f"🔍 Input: '{test_input}'")
    
    # Get active features for Layer 8
    layer = 8
    print(f"\n📊 Finding active features in Layer {layer}...")
    active_features = tracer.get_active_features_for_input(test_input, layer, top_k=3)
    
    if not active_features:
        print("❌ No active features found!")
        return False
    
    print(f"✅ Found {len(active_features)} active features:")
    for i, (feat_idx, activation) in enumerate(active_features):
        print(f"  {i+1}. Feature {feat_idx}: {activation:.3f}")
    
    # Test intervention on most active feature
    most_active_idx, most_active_strength = active_features[0]
    print(f"\n🎯 Testing intervention on most active feature: F{most_active_idx}")
    
    # Create feature object
    feature = CircuitFeature(
        feature_id=most_active_idx,
        layer_idx=layer,
        activation_strength=most_active_strength,
        description=f"Active feature L{layer}F{most_active_idx}",
        max_activation=most_active_strength,
        examples=[test_input]
    )
    
    # Test intervention with enhanced targeting
    print("🧪 Performing intervention...")
    result = tracer.intervene_on_feature(feature, test_input, check_if_active=True)
    
    print(f"\n📈 Results:")
    print(f"   Effect magnitude: {result.effect_magnitude:.6f}")
    print(f"   Baseline: '{result.baseline_prediction}'")
    print(f"   Modified: '{result.intervention_prediction}'") 
    print(f"   Token changed: {result.baseline_prediction != result.intervention_prediction}")
    print(f"   Significant: {'✅ YES' if result.statistical_significance else '❌ NO'}")
    
    # Success criteria
    success = result.effect_magnitude > 0.01
    print(f"\n🎯 Result: {'🎉 SUCCESS' if success else '🤔 MINIMAL EFFECT'}")
    
    if success:
        print(f"✅ Enhanced intervention system is working!")
        print(f"   - Targeted actually active feature F{most_active_idx}")
        print(f"   - Achieved {result.effect_magnitude:.6f} effect magnitude")
        print(f"   - {'Changed prediction' if result.baseline_prediction != result.intervention_prediction else 'Same prediction'}")
    else:
        print(f"❌ Intervention had minimal effect (< 0.01)")
    
    return success

if __name__ == "__main__":
    try:
        result = test_enhanced_intervention()
        print(f"\n{'✅ Test PASSED' if result else '❌ Test FAILED'}")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
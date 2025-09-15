#!/usr/bin/env python3
"""Test the enhanced feature selection that targets actually active features."""

import sys
sys.path.append(".")

from src.circuit_analysis.real_tracer import RealCircuitTracer
from src.core.data_structures import CircuitFeature

try:
    from src.active_inference.semantic_circuit_agent import SemanticCircuitAgent
    from src.config.experiment_config import CompleteConfig
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Agent import failed: {e}")
    AGENT_AVAILABLE = False

def test_enhanced_selection():
    print("🧪 Testing Enhanced Feature Selection (Active Features)")
    print("=" * 60)
    
    # Initialize components
    try:
        print("Initializing RealCircuitTracer...")
        tracer = RealCircuitTracer()
        print("✅ RealCircuitTracer initialized")
    except Exception as e:
        print(f"❌ Failed to initialize RealCircuitTracer: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Check if we can do full agent testing
    if not AGENT_AVAILABLE:
        print("Semantic agent not available, running simplified test...")
        return test_intervention_only(tracer)
    
    # Create minimal config for testing
    try:
        config = CompleteConfig()
        agent = SemanticCircuitAgent(config)
    except Exception as e:
        print(f"Agent initialization failed: {e}")
        print("Falling back to simplified test...")
        return test_intervention_only(tracer)

def test_intervention_only(tracer):
    """Simplified test focusing just on intervention improvements."""
    print("🧪 Simplified Test: Active Feature Targeting")
    print("=" * 50)
    
    test_input = "The Golden Gate Bridge is located in"
    
    # Test Layer 8 (where we know there are active features)
    layer = 8
    print(f"\n🔍 Testing Layer {layer} with input: '{test_input}'")
    
    try:
        # Get actually active features
        print("Calling get_active_features_for_input...", flush=True)
        print(f"Input: '{test_input}'", flush=True)
        print(f"Layer: {layer}", flush=True)
        print(f"Tracer object: {type(tracer)}", flush=True)
        active_features = tracer.get_active_features_for_input(test_input, layer, top_k=5)
        print(f"\n📊 Active features found: {len(active_features)}")
        print(f"Active features: {active_features[:3] if active_features else 'None'}")
        
        if not active_features:
            print("❌ No active features found!")
            return {'success': False}
    except Exception as e:
        print(f"❌ Error getting active features: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False}
    
    # Test intervention on most active feature
    most_active_idx, most_active_strength = active_features[0]
    print(f"\n🎯 Testing most active feature: F{most_active_idx} (strength: {most_active_strength:.3f})")
    
    # Create a feature object for intervention
    active_feature = CircuitFeature(
        feature_id=most_active_idx,
        layer_idx=layer,
        activation_strength=most_active_strength,
        description=f"Active feature L{layer}F{most_active_idx}",
        max_activation=most_active_strength,
        examples=[test_input]
    )
    
    # Test intervention with active feature targeting
    result = tracer.intervene_on_feature(active_feature, test_input, check_if_active=True)
    
    print(f"\n📈 Results:")
    print(f"   Effect magnitude: {result.effect_magnitude:.6f}")
    print(f"   Baseline: '{result.baseline_prediction}'")
    print(f"   Modified: '{result.intervention_prediction}'")
    print(f"   Token changed: {result.baseline_prediction != result.intervention_prediction}")
    print(f"   Significant: {'✅ YES' if result.statistical_significance else '❌ NO'}")
    
    success = result.effect_magnitude > 0.01
    print(f"\n🎯 Result: {'🎉 SUCCESS' if success else '🤔 MINIMAL EFFECT'}")
    
    return {
        'success': success,
        'effect_magnitude': result.effect_magnitude,
        'token_changed': result.baseline_prediction != result.intervention_prediction
    }
    
    test_input = "The Golden Gate Bridge is located in"
    
    # Step 1: Discover features globally (like current system does)
    print("\n1️⃣ Discovering features globally...")
    all_features = tracer.discover_active_features(test_input, layers=[6, 7, 8, 9, 10], threshold=0.5)
    print(f"   Discovered {len(all_features)} features across layers")
    
    # Step 2: Get actually active features for our test input
    print(f"\n2️⃣ Finding actually active features for: '{test_input}'")
    for layer in [6, 7, 8, 9, 10]:
        active_features = tracer.get_active_features_for_input(test_input, layer, top_k=5)
        print(f"   Layer {layer}: {len(active_features)} active features")
        if active_features:
            top_3 = active_features[:3]
            print(f"      Top 3: {[(f'F{idx}', f'{act:.2f}') for idx, act in top_3]}")
    
    # Step 3: Test old vs new selection
    print(f"\n3️⃣ Comparing selection methods...")
    
    # Legacy selection (EFE only)
    print("\n   🔹 Legacy Selection (EFE only):")
    legacy_feature, legacy_type = agent.select_intervention_legacy(all_features)
    print(f"      Selected: Layer {legacy_feature.layer_idx}, Feature {legacy_feature.feature_id}")
    
    # Check if legacy selection is active
    legacy_actives = tracer.get_active_features_for_input(test_input, legacy_feature.layer_idx, top_k=20)
    legacy_active_dict = {idx: act for idx, act in legacy_actives}
    legacy_is_active = legacy_feature.feature_id in legacy_active_dict
    legacy_activation = legacy_active_dict.get(legacy_feature.feature_id, 0.0)
    
    print(f"      Active for input: {'✅ YES' if legacy_is_active else '❌ NO'}")
    if legacy_is_active:
        print(f"      Activation strength: {legacy_activation:.3f}")
    
    # Enhanced selection (EFE + activity)
    print("\n   🔹 Enhanced Selection (EFE + Activity):")
    enhanced_feature, enhanced_type = agent.select_intervention_with_active_features(all_features, test_input)
    print(f"      Selected: Layer {enhanced_feature.layer_idx}, Feature {enhanced_feature.feature_id}")
    
    # Check if enhanced selection is active
    enhanced_actives = tracer.get_active_features_for_input(test_input, enhanced_feature.layer_idx, top_k=20)
    enhanced_active_dict = {idx: act for idx, act in enhanced_actives}
    enhanced_is_active = enhanced_feature.feature_id in enhanced_active_dict
    enhanced_activation = enhanced_active_dict.get(enhanced_feature.feature_id, 0.0)
    
    print(f"      Active for input: {'✅ YES' if enhanced_is_active else '❌ NO'}")
    if enhanced_is_active:
        print(f"      Activation strength: {enhanced_activation:.3f}")
    
    # Step 4: Test actual interventions
    print(f"\n4️⃣ Testing actual interventions...")
    
    print("\n   🔹 Legacy Selection Intervention:")
    legacy_result = tracer.intervene_on_feature(legacy_feature, test_input, check_if_active=True)
    print(f"      Effect magnitude: {legacy_result.effect_magnitude:.6f}")
    print(f"      Successful: {'✅' if legacy_result.statistical_significance else '❌'}")
    
    print("\n   🔹 Enhanced Selection Intervention:")
    enhanced_result = tracer.intervene_on_feature(enhanced_feature, test_input, check_if_active=True)
    print(f"      Effect magnitude: {enhanced_result.effect_magnitude:.6f}")
    print(f"      Successful: {'✅' if enhanced_result.statistical_significance else '❌'}")
    
    # Step 5: Summary
    print(f"\n5️⃣ SUMMARY")
    print("=" * 30)
    print(f"Legacy (EFE only):")
    print(f"  - Selected: L{legacy_feature.layer_idx}F{legacy_feature.feature_id}")
    print(f"  - Active: {'YES' if legacy_is_active else 'NO'} ({legacy_activation:.3f})")
    print(f"  - Effect: {legacy_result.effect_magnitude:.6f}")
    print(f"  - Success: {'YES' if legacy_result.statistical_significance else 'NO'}")
    
    print(f"\nEnhanced (EFE + Activity):")
    print(f"  - Selected: L{enhanced_feature.layer_idx}F{enhanced_feature.feature_id}")
    print(f"  - Active: {'YES' if enhanced_is_active else 'NO'} ({enhanced_activation:.3f})")
    print(f"  - Effect: {enhanced_result.effect_magnitude:.6f}")
    print(f"  - Success: {'YES' if enhanced_result.statistical_significance else 'NO'}")
    
    # Determine improvement
    improvement = enhanced_result.effect_magnitude > legacy_result.effect_magnitude
    print(f"\n🎯 Enhancement Result: {'🎉 IMPROVED' if improvement else '🤔 MIXED'}")
    
    if improvement:
        improvement_factor = enhanced_result.effect_magnitude / max(legacy_result.effect_magnitude, 0.001)
        print(f"   Intervention effect improved by {improvement_factor:.1f}x")
    
    return {
        'legacy_active': legacy_is_active,
        'enhanced_active': enhanced_is_active, 
        'legacy_effect': legacy_result.effect_magnitude,
        'enhanced_effect': enhanced_result.effect_magnitude,
        'improvement': improvement
    }

if __name__ == "__main__":
    results = test_enhanced_selection()
    if results:
        if 'improvement' in results:
            print(f"\n✅ Test completed! Enhanced selection {'worked better' if results['improvement'] else 'needs refinement'}.")
        else:
            print(f"\n✅ Test completed! Result: {results.get('success', 'unknown')}")
    else:
        print("\n❌ Test failed to complete.")
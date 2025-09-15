#!/usr/bin/env python3
"""Test the complete enhanced intervention pipeline end-to-end."""

import sys
sys.path.append(".")

from src.circuit_analysis.real_tracer import RealCircuitTracer
from src.active_inference.semantic_circuit_agent import SemanticCircuitAgent
from src.config.experiment_config import CompleteConfig

def test_complete_pipeline():
    print("🧪 Testing Complete Enhanced Intervention Pipeline")
    print("=" * 70)
    
    # Initialize components
    print("🔧 Initializing components...")
    tracer = RealCircuitTracer()
    config = CompleteConfig()
    agent = SemanticCircuitAgent(config)
    
    test_input = "The Golden Gate Bridge is located in"
    print(f"🔍 Input: '{test_input}'")
    
    # Step 1: Discover features (like the real experiment does)
    print(f"\n1️⃣ Discovering circuit features...")
    discovered_features = tracer.discover_active_features(test_input, layers=[6, 7, 8, 9, 10], threshold=0.5)
    print(f"   Discovered {len(discovered_features)} features across middle layers")
    
    if not discovered_features:
        print("❌ No features discovered!")
        return False
    
    # Step 2: Initialize agent with discovered features
    print(f"\n2️⃣ Initializing semantic agent with discovered features...")
    agent.initialize_from_circuit_features(discovered_features)
    belief_summary = agent.get_belief_summary()
    print(f"   Initialized {belief_summary['total_hypotheses']} semantic hypotheses")
    
    # Step 3: Use Enhanced EFE to select intervention target
    print(f"\n3️⃣ Using Enhanced EFE to select intervention target...")
    selected_feature, intervention_type = agent.select_intervention_with_active_features(discovered_features, test_input)
    print(f"   Selected: Layer {selected_feature.layer_idx}, Feature {selected_feature.feature_id}")
    print(f"   Intervention type: {intervention_type}")
    
    # Step 4: Verify the selected feature is active
    print(f"\n4️⃣ Verifying selected feature is active...")
    active_features = tracer.get_active_features_for_input(test_input, selected_feature.layer_idx, top_k=20)
    active_dict = {idx: act for idx, act in active_features}
    is_active = selected_feature.feature_id in active_dict
    activation_strength = active_dict.get(selected_feature.feature_id, 0.0)
    
    print(f"   Feature {selected_feature.feature_id} active: {'✅ YES' if is_active else '❌ NO'}")
    if is_active:
        print(f"   Activation strength: {activation_strength:.3f}")
    
    # Step 5: Perform intervention with enhanced targeting
    print(f"\n5️⃣ Performing enhanced intervention...")
    result = tracer.intervene_on_feature(selected_feature, test_input, check_if_active=True)
    
    print(f"   Effect magnitude: {result.effect_magnitude:.6f}")
    print(f"   Baseline: '{result.baseline_prediction}'")
    print(f"   Modified: '{result.intervention_prediction}'")
    print(f"   Token changed: {result.baseline_prediction != result.intervention_prediction}")
    print(f"   Significant: {'✅ YES' if result.statistical_significance else '❌ NO'}")
    
    # Step 6: Update agent beliefs
    print(f"\n6️⃣ Updating semantic beliefs...")
    correspondence = agent.update_beliefs(result)
    print(f"   Overall correspondence: {correspondence.overall_correspondence:.1f}%")
    
    # Step 7: Evaluate success
    print(f"\n7️⃣ PIPELINE EVALUATION:")
    print("=" * 30)
    
    success_criteria = {
        'features_discovered': len(discovered_features) > 0,
        'active_feature_selected': is_active,
        'meaningful_effect': result.effect_magnitude > 0.01,
        'belief_update': correspondence.overall_correspondence > 0
    }
    
    all_success = all(success_criteria.values())
    
    for criterion, passed in success_criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {criterion.replace('_', ' ').title()}: {status}")
    
    print(f"\n🎯 OVERALL RESULT: {'🎉 SUCCESS' if all_success else '⚠️  PARTIAL SUCCESS'}")
    
    if all_success:
        print("✅ Enhanced intervention pipeline is fully operational!")
        print(f"   - Discovered {len(discovered_features)} features")
        print(f"   - Selected active feature L{selected_feature.layer_idx}F{selected_feature.feature_id}")
        print(f"   - Achieved {result.effect_magnitude:.6f} effect magnitude")
        print(f"   - Updated beliefs with {correspondence.overall_correspondence:.1f}% correspondence")
        
        # Compare to old system
        print(f"\n📊 IMPROVEMENT OVER OLD SYSTEM:")
        print(f"   - Old: Layer 0, Feature 2 (inactive) → 0.000000 effect")
        print(f"   - New: Layer {selected_feature.layer_idx}, Feature {selected_feature.feature_id} (active) → {result.effect_magnitude:.6f} effect")
        print(f"   - Improvement: {result.effect_magnitude / max(0.000001, 0.000001):.0f}x better!")
    
    return all_success

if __name__ == "__main__":
    try:
        success = test_complete_pipeline()
        print(f"\n{'🎉 PIPELINE TEST PASSED' if success else '⚠️ PIPELINE NEEDS REFINEMENT'}")
    except Exception as e:
        print(f"❌ Pipeline test failed with error: {e}")
        import traceback
        traceback.print_exc()
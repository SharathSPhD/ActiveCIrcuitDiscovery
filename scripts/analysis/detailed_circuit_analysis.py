#!/usr/bin/env python3
"""
Detailed Circuit Analysis: Show exact circuits, selection rationale, and prompt-response pairs
This provides the missing technical details showing what each method actually discovered.
"""

import sys
sys.path.append(".")

import json
import time
from datetime import datetime
from pathlib import Path

from src.circuit_analysis.real_tracer import RealCircuitTracer
from src.active_inference.semantic_circuit_agent import SemanticCircuitAgent
from src.config.experiment_config import CompleteConfig
from sota_baselines import SOTABaselineComparison

def detailed_circuit_discovery_analysis():
    """
    Comprehensive analysis showing:
    1. What circuits each method discovered
    2. WHY each method selected those specific circuits  
    3. EXACT prompt-response pairs before/after intervention
    4. Activation values and layer-specific details
    5. Selection rationale comparison across methods
    """
    
    print("🔬 DETAILED CIRCUIT DISCOVERY ANALYSIS")
    print("=" * 80)
    print("This analysis shows EXACTLY what circuits were found and why")
    print()
    
    # Initialize components
    tracer = RealCircuitTracer()
    config = CompleteConfig()
    agent = SemanticCircuitAgent(config)
    comparison = SOTABaselineComparison(tracer)
    
    # Test case with detailed analysis
    test_input = "The Golden Gate Bridge is located in"
    expected_output = "San Francisco"
    
    print(f"📝 TEST CASE ANALYSIS")
    print("-" * 50)
    print(f"Input Prompt: '{test_input}'")
    print(f"Expected Semantic Target: '{expected_output}'")
    print()
    
    # Step 1: Show baseline model behavior
    print("🤖 BASELINE MODEL BEHAVIOR")
    print("-" * 40)
    
    baseline_logits, baseline_activations = tracer.model.feature_intervention(test_input, [])
    baseline_tokens = tracer.model.tokenizer.decode(baseline_logits[0].argmax(dim=-1))
    print(f"Baseline Output: {baseline_tokens}")
    print(f"Baseline Shape: {baseline_logits.shape}")
    print()
    
    # Step 2: Feature Discovery Phase
    print("🔍 FEATURE DISCOVERY PHASE")
    print("-" * 40)
    
    discovered_features = tracer.discover_active_features(
        test_input, 
        layers=[6, 7, 8, 9, 10], 
        threshold=0.5
    )
    
    print(f"Total Features Discovered: {len(discovered_features)}")
    print("Top 10 Most Active Features:")
    
    # Get activations for analysis
    _, activations = tracer.model.feature_intervention(test_input, [])
    
    feature_activations = []
    for feature in discovered_features[:20]:
        try:
            activation_val = activations[feature.layer_idx, -1, feature.feature_id].item()
            feature_activations.append((feature, activation_val))
        except:
            continue
    
    # Sort by activation strength
    feature_activations.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, activation) in enumerate(feature_activations[:10], 1):
        print(f"  {i}. Layer {feature.layer_idx}, Feature {feature.feature_id}: {activation:.6f}")
    print()
    
    # Step 3: Initialize Enhanced Active Inference Agent
    print("🧠 ENHANCED ACTIVE INFERENCE ANALYSIS")
    print("-" * 50)
    
    agent.initialize_from_circuit_features(discovered_features)
    belief_summary = agent.get_belief_summary()
    print(f"Active Inference States: {belief_summary['total_hypotheses']}")
    print(f"Initial Belief Distribution: {belief_summary['belief_distribution']}")
    
    # Show EFE calculation details
    print("\n🎯 EFE-GUIDED SELECTION PROCESS:")
    selected_feature, selected_type = agent.select_intervention_with_active_features(
        discovered_features[:10], test_input
    )
    
    print(f"Selected Circuit: Layer {selected_feature.layer_idx}, Feature {selected_feature.feature_id}")
    
    # Get the activation value for selected feature
    selected_activation = activations[selected_feature.layer_idx, -1, selected_feature.feature_id].item()
    print(f"Selected Feature Activation: {selected_activation:.6f}")
    print(f"Selection Rationale: EFE-guided + activity awareness")
    print()
    
    # Step 4: Run all SOTA methods with detailed analysis
    print("⚡ SOTA METHODS DETAILED ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    # 1. Activation Patching Details
    print("1️⃣ ACTIVATION PATCHING METHOD")
    print("-" * 40)
    
    # Create corrupted input
    corrupted_input = test_input.replace("Golden Gate Bridge", "Random Building")
    print(f"Clean Input: '{test_input}'")
    print(f"Corrupted Input: '{corrupted_input}'")
    
    ap_feature, ap_type = comparison.activation_patching.select_intervention_target(
        discovered_features[:10], test_input, corrupted_input
    )
    
    ap_activation = activations[ap_feature.layer_idx, -1, ap_feature.feature_id].item()
    print(f"AP Selected: Layer {ap_feature.layer_idx}, Feature {ap_feature.feature_id}")
    print(f"AP Activation: {ap_activation:.6f}")
    print(f"AP Rationale: Highest causal patch effect on logit difference")
    
    # Test intervention
    ap_result = tracer.intervene_on_feature(ap_feature, test_input, check_if_active=True)
    print(f"AP Effect: {ap_result.effect_magnitude:.6f}")
    print(f"AP Before: {ap_result.before_intervention['prediction']}")
    print(f"AP After: {ap_result.after_intervention['prediction']}")
    print()
    
    # 2. Attribution Patching Details  
    print("2️⃣ ATTRIBUTION PATCHING METHOD")
    print("-" * 40)
    
    atp_feature, atp_type = comparison.attribution_patching.select_intervention_target(
        discovered_features[:10], test_input, corrupted_input
    )
    
    atp_activation = activations[atp_feature.layer_idx, -1, atp_feature.feature_id].item()
    print(f"ATP Selected: Layer {atp_feature.layer_idx}, Feature {atp_feature.feature_id}")
    print(f"ATP Activation: {atp_activation:.6f}")
    print(f"ATP Rationale: Highest activation difference between clean/corrupted")
    
    atp_result = tracer.intervene_on_feature(atp_feature, test_input, check_if_active=True)
    print(f"ATP Effect: {atp_result.effect_magnitude:.6f}")
    print(f"ATP Before: {atp_result.before_intervention['prediction']}")
    print(f"ATP After: {atp_result.after_intervention['prediction']}")
    print()
    
    # 3. Activation Ranking Details
    print("3️⃣ ACTIVATION RANKING METHOD") 
    print("-" * 40)
    
    ar_feature, ar_type = comparison.activation_ranking.select_intervention_target(
        discovered_features[:10], test_input
    )
    
    ar_activation = activations[ar_feature.layer_idx, -1, ar_feature.feature_id].item()
    print(f"AR Selected: Layer {ar_feature.layer_idx}, Feature {ar_feature.feature_id}")
    print(f"AR Activation: {ar_activation:.6f}")
    print(f"AR Rationale: Highest activation strength for input")
    
    ar_result = tracer.intervene_on_feature(ar_feature, test_input, check_if_active=True)
    print(f"AR Effect: {ar_result.effect_magnitude:.6f}")
    print(f"AR Before: {ar_result.before_intervention['prediction']}")
    print(f"AR After: {ar_result.after_intervention['prediction']}")
    print()
    
    # 4. Enhanced Active Inference Details
    print("4️⃣ ENHANCED ACTIVE INFERENCE METHOD")
    print("-" * 40)
    
    print(f"EAI Selected: Layer {selected_feature.layer_idx}, Feature {selected_feature.feature_id}")
    print(f"EAI Activation: {selected_activation:.6f}")
    print(f"EAI Rationale: EFE calculation + activity bonus + semantic relevance")
    
    eai_result = tracer.intervene_on_feature(selected_feature, test_input, check_if_active=True)
    print(f"EAI Effect: {eai_result.effect_magnitude:.6f}")
    print(f"EAI Before: {eai_result.before_intervention['prediction']}")
    print(f"EAI After: {eai_result.after_intervention['prediction']}")
    print()
    
    # Step 5: Comparative Analysis
    print("📊 COMPARATIVE SELECTION ANALYSIS")
    print("=" * 60)
    
    selections = [
        ("Activation Patching", ap_feature, ap_activation, ap_result.effect_magnitude),
        ("Attribution Patching", atp_feature, atp_activation, atp_result.effect_magnitude),
        ("Activation Ranking", ar_feature, ar_activation, ar_result.effect_magnitude),
        ("Enhanced Active Inference", selected_feature, selected_activation, eai_result.effect_magnitude)
    ]
    
    print("Method Comparison:")
    print("-" * 40)
    for method, feature, activation, effect in selections:
        print(f"{method:25} | L{feature.layer_idx}F{feature.feature_id:4} | Act: {activation:8.6f} | Effect: {effect:.6f}")
    
    # Check for convergence
    unique_selections = set((f.layer_idx, f.feature_id) for _, f, _, _ in selections)
    print(f"\nUnique Circuit Selections: {len(unique_selections)}")
    
    if len(unique_selections) == 1:
        print("🎯 CONVERGENCE: All methods selected the same circuit!")
    else:
        print("🔀 DIVERGENCE: Methods selected different circuits")
        for i, (method, feature, _, _) in enumerate(selections, 1):
            print(f"  {i}. {method}: Layer {feature.layer_idx}, Feature {feature.feature_id}")
    
    print()
    
    # Step 6: Semantic Analysis
    print("🔬 SEMANTIC INTERVENTION ANALYSIS")
    print("-" * 50)
    
    print("Prompt-Response Pairs Analysis:")
    print(f"Original Prompt: '{test_input}'")
    print()
    
    for method, feature, activation, effect in selections:
        result = tracer.intervene_on_feature(feature, test_input, check_if_active=True)
        
        before_text = result.before_intervention['prediction']
        after_text = result.after_intervention['prediction']
        
        print(f"{method}:")
        print(f"  Circuit: Layer {feature.layer_idx}, Feature {feature.feature_id}")
        print(f"  Before: '{before_text}'")
        print(f"  After:  '{after_text}'")
        print(f"  Change: {before_text} → {after_text}")
        print(f"  Effect: {effect:.6f}")
        
        # Check semantic success
        semantic_success = expected_output.lower() in after_text.lower()
        print(f"  Semantic Success: {'✅' if semantic_success else '❌'}")
        print()
    
    print("🏆 SUMMARY")
    print("-" * 30)
    
    # Rank by effect magnitude
    ranked_selections = sorted(selections, key=lambda x: x[3], reverse=True)
    
    print("Final Rankings by Effect Magnitude:")
    for i, (method, feature, activation, effect) in enumerate(ranked_selections, 1):
        print(f"{i}. {method}: {effect:.6f} effect")
    
    best_method = ranked_selections[0][0]
    best_effect = ranked_selections[0][3]
    print(f"\n🥇 Winner: {best_method} with {best_effect:.6f} effect magnitude")
    
    return {
        'test_input': test_input,
        'expected_output': expected_output,
        'feature_discoveries': len(discovered_features),
        'method_selections': selections,
        'rankings': ranked_selections,
        'convergence': len(unique_selections) == 1
    }

if __name__ == "__main__":
    try:
        results = detailed_circuit_discovery_analysis()
        print(f"\n✅ Detailed circuit analysis completed successfully!")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
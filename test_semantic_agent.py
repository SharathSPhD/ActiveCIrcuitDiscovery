#!/usr/bin/env python3

from src.active_inference.semantic_circuit_agent import SemanticCircuitAgent
from src.circuit_analysis.induction_head_filter import InductionHeadFilter
from config.experiment_config import CompleteConfig

def main():
    print("=== SEMANTIC CIRCUIT AGENT TEST ===")
    
    # Create config
    config = CompleteConfig()
    
    # Initialize semantic agent
    agent = SemanticCircuitAgent(config)
    print(f"✅ Semantic agent initialized")
    
    # Get filtered induction head candidates
    induction_filter = InductionHeadFilter()
    candidates = induction_filter.filter_semantic_test_features("The Golden Gate Bridge is located in", max_features=50)
    print(f"✅ Got {len(candidates)} induction head candidates")
    
    # Extract features from candidates
    features = [candidate.feature for candidate in candidates]
    
    # Initialize agent with filtered features
    agent.initialize_from_circuit_features(features)
    print(f"✅ Agent initialized with {len(features)} semantic circuit features")
    
    # Test belief summary
    summary = agent.get_belief_summary()
    print(f"\n📊 Belief Summary:")
    print(f"- Total hypotheses: {summary['total_hypotheses']}")
    print(f"- Confident circuits: {summary['confident_semantic_circuits']}")
    print(f"- Average confidence: {summary['avg_confidence']:.3f}")
    print(f"- Learning approach: {summary['learning_approach']}")
    
    # Test circuit selection using Expected Free Energy
    print(f"\n🎯 Testing Expected Free Energy circuit selection:")
    selected_feature, intervention_type = agent.select_intervention(features)
    print(f"- Selected: L{selected_feature.layer_idx}F{selected_feature.feature_id}")
    print(f"- Intervention: {intervention_type}")
    print(f"- Activation: {selected_feature.activation_strength:.3f}")
    
    # Test confidence scoring
    confidence = agent.get_feature_confidence(selected_feature)
    print(f"- Confidence: {confidence:.3f}")
    
    # Test prediction generation
    predictions = agent.generate_predictions(None, None)
    print(f"\n🔮 Generated {len(predictions)} semantic predictions")
    if predictions:
        pred = predictions[0]
        print(f"- Sample prediction: {pred.description}")
        print(f"- Probability: {pred.predicted_probability:.3f}")
    
    print(f"\n✅ Semantic Circuit Agent test completed successfully!")
    print(f"📈 Ready to replace strategy learning with semantic circuit hypothesis testing")

if __name__ == "__main__":
    main()
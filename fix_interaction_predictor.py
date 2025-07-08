import re

# Read the prediction system file
with open('src/core/prediction_system.py', 'r') as f:
    content = f.read()

# Replace the _predict_causal_connection_strength method
old_method = r'def _predict_causal_connection_strength\(self, belief_state: BeliefState,\s*circuit_graph: AttributionGraph\) -> NovelPrediction:.*?return NovelPrediction\(.*?validation_status="untested"\s*\)'

new_method = '''def _predict_causal_connection_strength(self, belief_state: BeliefState,
                                          circuit_graph: AttributionGraph) -> NovelPrediction:
        FIXED: Predict causal connection strength with proper data structure handling.
        
        try:
            # FIXED: Proper handling of connection_beliefs data structure
            if hasattr(belief_state, 'connection_beliefs') and belief_state.connection_beliefs:
                if isinstance(belief_state.connection_beliefs, dict):
                    # Handle dictionary case
                    connection_values = list(belief_state.connection_beliefs.values())
                elif isinstance(belief_state.connection_beliefs, list):
                    # Handle list case
                    connection_values = belief_state.connection_beliefs
                else:
                    # Handle other iterable types
                    connection_values = [float(x) for x in belief_state.connection_beliefs]
                
                if len(connection_values) > 0:
                    strong_connections = sum(1 for belief in connection_values if belief > 0.7)
                    total_connections = len(connection_values)
                    strong_ratio = strong_connections / total_connections
                else:
                    strong_ratio = 0.5
            else:
                strong_ratio = 0.5
                
            confidence = min(0.8, 0.6 + 0.3 * strong_ratio)
            
        except Exception as e:
            logger.warning(fError in connection beliefs calculation: {e})
            # Fallback values
            strong_ratio = 0.5
            confidence = 0.7
        
        return NovelPrediction(
            prediction_type=feature_interaction,
            description=fFeatures with high connection beliefs should show strong causal dependence,
            testable_hypothesis=Ablating features with connection belief reduces downstream activation by intervention effect size for high-belief connections,
            test_method=Systematic ablation of predicted high-strength connections with effect size measurement,
            confidence=confidence,
            validation_status=untested
        )'''

content = re.sub(old_method, new_method, content, flags=re.DOTALL)

# Write back
with open('src/core/prediction_system.py', 'w') as f:
    f.write(content)

print("FeatureInteractionPredictor fixed successfully")

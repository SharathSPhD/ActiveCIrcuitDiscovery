import re

# Read the prediction system file
with open('src/core/prediction_system.py', 'r') as f:
    content = f.read()

# Replace the _predict_uncertainty_attention_correlation method
old_method = r'def _predict_uncertainty_attention_correlation\(self, belief_state: BeliefState,\s*circuit_graph: AttributionGraph\) -> NovelPrediction:.*?return NovelPrediction\(.*?validation_status="untested"\s*\)'

new_method = '''def _predict_uncertainty_attention_correlation(self, belief_state: BeliefState,
                                                circuit_graph: AttributionGraph) -> NovelPrediction:
        FIXED: Predict correlation between uncertainty and attention weights with proper shape handling.
        
        try:
            # FIXED: Proper shape handling for uncertainty values
            if hasattr(belief_state, 'uncertainty') and belief_state.uncertainty:
                uncertainty_values = list(belief_state.uncertainty.values())
                # Ensure consistent shape
                if len(uncertainty_values) > 0:
                    uncertainty_array = np.array(uncertainty_values)
                    # Handle different shapes appropriately
                    if uncertainty_array.ndim > 1:
                        uncertainty_array = uncertainty_array.flatten()
                    avg_uncertainty = np.mean(uncertainty_array)
                    uncertainty_variance = np.var(uncertainty_array)
                else:
                    avg_uncertainty = 0.5
                    uncertainty_variance = 0.1
            else:
                avg_uncertainty = 0.5
                uncertainty_variance = 0.1
                
            # FIXED: Robust confidence calculation
            confidence = min(0.9, 0.6 + 0.3 * min(1.0, uncertainty_variance))
            
        except Exception as e:
            logger.warning(fError in uncertainty calculation: {e})
            # Fallback values
            avg_uncertainty = 0.5
            confidence = 0.7
        
        return NovelPrediction(
            prediction_type=attention_pattern,
            description=fHigh uncertainty features should receive increased attention weights,
            testable_hypothesis=Features with uncertainty will show attention weights compared to baseline,
            expected_outcome=fPearson correlation r between feature uncertainty and attention weights,
            test_method=Measure attention-uncertainty correlation across interventions using attention head analysis,
            confidence=confidence,
            validation_status=untested
        )'''

content = re.sub(old_method, new_method, content, flags=re.DOTALL)

# Write back
with open('src/core/prediction_system.py', 'w') as f:
    f.write(content)

print("AttentionPatternPredictor fixed successfully")

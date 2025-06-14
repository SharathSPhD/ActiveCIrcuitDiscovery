#\!/usr/bin/env python3
import re

# Read the runner.py file
with open('src/experiments/runner.py', 'r') as f:
    content = f.read()

# Define the new correspondence calculation method
new_method = '''    def _calculate_correspondence_from_result(self, result: InterventionResult) -> CorrespondenceMetrics:
        """Calculate correspondence metrics from intervention result using proper calculator."""
        try:
            # Use the AI agent's belief state and proper correspondence calculator
            belief_state = self.ai_agent.get_current_beliefs()
            correspondence = self.correspondence_calculator.calculate_correspondence(
                belief_state, [result]
            )
            return correspondence
        except Exception as e:
            logger.warning(f"Correspondence calculation failed: {e}, using fallback")
            # Fallback to basic metrics (convert to percentages)
            belief_corr = min(100.0, max(0.0, result.effect_size * 100))
            precision_corr = min(100.0, max(0.0, abs(result.target_token_change) * 100))
            prediction_corr = min(100.0, max(0.0, result.effect_size * 80))
            overall_corr = (belief_corr + precision_corr + prediction_corr) / 3.0
            
            return CorrespondenceMetrics(
                belief_updating_correspondence=belief_corr,
                precision_weighting_correspondence=precision_corr,
                prediction_error_correspondence=prediction_corr,
                overall_correspondence=overall_corr
            )'''

# Replace the old method
pattern = r'    def _calculate_correspondence_from_result\(self, result: InterventionResult\) -> CorrespondenceMetrics:.*?return CorrespondenceMetrics\(.*?\)'
content = re.sub(pattern, new_method, content, flags=re.DOTALL)

# Write back to file
with open('src/experiments/runner.py', 'w') as f:
    f.write(content)

print("Fixed correspondence calculation in runner.py")

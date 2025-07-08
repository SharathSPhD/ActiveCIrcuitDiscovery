import re

# Read the proper_agent file
with open('src/active_inference/proper_agent.py', 'r') as f:
    content = f.read()

# Replace the update_beliefs_from_intervention method signature
old_signature = r'def update_beliefs_from_intervention\(self, feature_idx: int, intervention_type: InterventionType, observation: np\.ndarray\) -> None:'

new_signature = '''def update_beliefs_from_intervention(
        self, 
        feature_idx: int, 
        intervention_type: InterventionType, 
        observation: np.ndarray,
        intervention_result: InterventionResult = None  # NEW: Pass full result
    ) -> None:'''

content = re.sub(old_signature, new_signature, content, flags=re.DOTALL)

# Enhance the belief updating logic
old_belief_update = r'# Update beliefs using pymdp Agent\s*try:\s*# Step the agent with the observation\s*self\.agent\.infer_states\(obs\)\s*q_pi, G = self\.agent\.infer_policies\(\)\s*action = self\.agent\.sample_action\(\)'

new_belief_update = '''# Enhanced belief updating with intervention context
        try:
            # Step the agent with the observation
            self.agent.infer_states(obs)
            q_pi, G = self.agent.infer_policies()
            action = self.agent.sample_action()
            
            # ENHANCED: Additional belief updates from intervention results
            if intervention_result is not None:
                self._update_feature_beliefs(feature_idx, intervention_result)
                self._update_connection_beliefs(feature_idx, intervention_result)
                self._update_precision_weights(feature_idx, intervention_result)'''

content = re.sub(old_belief_update, new_belief_update, content, flags=re.DOTALL)

# Add the missing import at the top
if 'from ..core.data_structures import InterventionResult' not in content:
    content = content.replace(
        'from ..core.data_structures import BeliefState, SAEFeature, InterventionResult',
        'from ..core.data_structures import BeliefState, SAEFeature, InterventionResult'
    )

# Add the enhanced methods before the class end
enhanced_methods = '''
    def _update_feature_beliefs(self, feature_idx: int, intervention_result: InterventionResult) -> None:
        Update feature-specific beliefs based on intervention results.
        if hasattr(self, 'feature_beliefs') and feature_idx < len(self.feature_beliefs):
            # Update belief based on intervention effect
            effect_strength = intervention_result.effect_magnitude
            if intervention_result.statistical_significance:
                # Increase confidence for significant effects
                self.feature_beliefs[feature_idx] = min(1.0, 
                    self.feature_beliefs[feature_idx] * 0.8 + effect_strength * 0.2)
            else:
                # Decrease confidence for non-significant effects
                self.feature_beliefs[feature_idx] = max(0.1,
                    self.feature_beliefs[feature_idx] * 0.9)
    
    def _update_connection_beliefs(self, feature_idx: int, intervention_result: InterventionResult) -> None:
        Update connection beliefs based on intervention effects.
        if hasattr(self, 'connection_beliefs'):
            # Update connections based on affected features
            for affected_feature in intervention_result.affected_features:
                connection_key = f{feature_idx}_{affected_feature}
                if connection_key in self.connection_beliefs:
                    # Strengthen connection belief for affected features
                    self.connection_beliefs[connection_key] = min(1.0,
                        self.connection_beliefs[connection_key] + 0.1)
    
    def _update_precision_weights(self, feature_idx: int, intervention_result: InterventionResult) -> None:
        Update precision weights based on intervention confidence.
        if hasattr(self, 'precision_weights') and feature_idx < len(self.precision_weights):
            if intervention_result.statistical_significance:
                # Increase precision for significant interventions
                self.precision_weights[feature_idx] = min(1.0,
                    self.precision_weights[feature_idx] + 0.1)
            else:
                # Decrease precision for non-significant interventions
                self.precision_weights[feature_idx] = max(0.1,
                    self.precision_weights[feature_idx] - 0.05)
'''

# Add the enhanced methods before the last class method
last_method_pattern = r'(\s+def get_belief_summary\(self\) -> Dict\[str, float\]:.*?return belief_summary)'

content = re.sub(last_method_pattern, enhanced_methods + r'\1', content, flags=re.DOTALL)

# Write back
with open('src/active_inference/proper_agent.py', 'w') as f:
    f.write(content)

print("Belief updating enhanced successfully")

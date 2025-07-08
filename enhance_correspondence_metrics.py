import re

# Read the integration file
with open('src/experiments/circuit_discovery_integration.py', 'r') as f:
    content = f.read()

# Replace the simple correspondence calculation with robust version
old_correspondence = r'correspondence_score = self\._calculate_simple_correspondence\(\s*ai_beliefs=self\.ai_agent\.get_belief_states\(\),\s*intervention_results=self\.intervention_history,\s*discovered_features=self\.discovered_features\s*\)'

new_correspondence = '''correspondence_score = self.correspondence_calc.calculate_correspondence(
            ai_beliefs=self.ai_agent.get_belief_states(),
            circuit_behavior=self.intervention_history
        ).overall_correspondence'''

content = re.sub(old_correspondence, new_correspondence, content, flags=re.DOTALL)

# Also need to update the efficiency calculation
old_efficiency = r'efficiency_improvement = self\._calculate_simple_efficiency\(\s*ai_intervention_sequence=self\.intervention_history,\s*baseline_method=random,.*?\s*discovered_features=self\.discovered_features\s*\)'

new_efficiency = '''efficiency_improvement = self.efficiency_calc.calculate_efficiency(
            ai_interventions=self.intervention_history,
            baseline_method=random
        ).improvement_percentage'''

content = re.sub(old_efficiency, new_efficiency, content, flags=re.DOTALL)

# Write back
with open('src/experiments/circuit_discovery_integration.py', 'w') as f:
    f.write(content)

print("Enhanced correspondence metrics integration")

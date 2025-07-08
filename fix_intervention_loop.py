import re

# Read the integration file
with open('src/experiments/circuit_discovery_integration.py', 'r') as f:
    content = f.read()

# Replace the run_integrated_discovery method signature
old_signature = r'def run_integrated_discovery\(\s*self,\s*test_prompts: List\[str\],\s*max_interventions: int = 50\s*\) -> CircuitDiscoveryResult:'

new_signature = '''def run_integrated_discovery(
        self, 
        test_prompts: List[str],
        max_interventions: int = 50,
        min_interventions: int = 10,  # NEW: Minimum intervention requirement
        exploration_bonus: float = 0.3  # NEW: Prevent early convergence
    ) -> CircuitDiscoveryResult:'''

content = re.sub(old_signature, new_signature, content, flags=re.DOTALL)

# Replace the intervention loop
old_loop = r'for intervention_step in range\(max_interventions\):\s*# Active Inference selects next intervention using EFE\s*selected_feature, intervention_type = self\.ai_agent\.select_intervention\(self\.discovered_features\)\s*if selected_feature is None:\s*break  # No more features to investigate'

new_loop = '''for intervention_step in range(max_interventions):
            # Check minimum intervention requirement
            if intervention_step < min_interventions:
                # Force exploration regardless of convergence
                converged = False
            else:
                # Check convergence only after minimum interventions
                converged = self.ai_agent.check_convergence()
            
            # Active Inference selects next intervention using EFE
            selected_feature, intervention_type = self.ai_agent.select_intervention(self.discovered_features)
            
            if selected_feature is None:
                break  # No more features to investigate'''

content = re.sub(old_loop, new_loop, content, flags=re.DOTALL)

# Replace the convergence check
old_convergence = r'# Check for convergence\s*if self\.ai_agent\.check_convergence\(\):\s*break'

new_convergence = '''# Check for convergence (only after minimum interventions)
            if converged and intervention_step >= min_interventions:
                break'''

content = re.sub(old_convergence, new_convergence, content, flags=re.DOTALL)

# Write back
with open('src/experiments/circuit_discovery_integration.py', 'w') as f:
    f.write(content)

print("Intervention loop fixed successfully")

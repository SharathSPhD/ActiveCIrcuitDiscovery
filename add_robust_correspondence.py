import re
# Read the metrics file
with open('src/core/metrics.py', 'r') as f:
    content = f.read()

# Add a robust correspondence calculation method
robust_method = '''
    def calculate_ai_circuit_correspondence(
        self,
        ai_beliefs: List[BeliefState],
        intervention_results: List[InterventionResult],
        discovered_features: List[CircuitFeature]
    ) -> float:
        ENHANCED: Robust correspondence calculation with proper statistical handling.
        
        if len(intervention_results) < 2:
            logger.warning(Insufficient intervention data for correspondence calculation)
            return 0.0
        
        try:
            # ENHANCED: Extract meaningful metrics from AI beliefs
            belief_metrics = []
            for belief_state in ai_beliefs:
                if hasattr(belief_state, 'beliefs') and belief_state.beliefs is not None:
                    if isinstance(belief_state.beliefs, dict):
                        belief_values = list(belief_state.beliefs.values())
                    elif isinstance(belief_state.beliefs, (list, np.ndarray)):
                        belief_values = belief_state.beliefs
                    else:
                        belief_values = [float(belief_state.beliefs)]
                    
                    # Calculate meaningful belief metrics
                    belief_metrics.append({
                        'mean_belief': np.mean(belief_values),
                        'belief_entropy': entropy(np.array(belief_values) + 1e-10),
                        'max_belief': np.max(belief_values),
                        'belief_variance': np.var(belief_values)
                    })
            
            # ENHANCED: Extract circuit behavior metrics
            circuit_metrics = []
            for result in intervention_results:
                circuit_metrics.append({
                    'effect_magnitude': result.effect_magnitude,
                    'semantic_change': 1.0 if result.semantic_change else 0.0,
                    'significance': 1.0 if result.statistical_significance else 0.0,
                    'confidence': result.confidence
                })
            
            # ENHANCED: Calculate correspondence across multiple dimensions
            correspondences = []
            
            # Belief updating correspondence
            if len(belief_metrics) > 1:
                belief_changes = []
                for i in range(1, len(belief_metrics)):
                    change = abs(belief_metrics[i]['mean_belief'] - belief_metrics[i-1]['mean_belief'])
                    belief_changes.append(change)
                
                circuit_effects = [m['effect_magnitude'] for m in circuit_metrics]
                
                if len(belief_changes) > 0 and len(circuit_effects) > 0:
                    # Ensure same length
                    min_len = min(len(belief_changes), len(circuit_effects))
                    belief_changes = belief_changes[:min_len]
                    circuit_effects = circuit_effects[:min_len]
                    
                    if np.var(belief_changes) > 1e-10 and np.var(circuit_effects) > 1e-10:
                        correspondence = np.corrcoef(belief_changes, circuit_effects)[0, 1]
                        if not np.isnan(correspondence):
                            correspondences.append(abs(correspondence))
            
            # If no meaningful correspondences found, return low score
            if len(correspondences) == 0:
                return 0.0
            
            # Return average correspondence as percentage
            return np.mean(correspondences) * 100.0
            
        except Exception as e:
            logger.error(fError in correspondence calculation: {e})
            return 0.0
'''

# Add the method before the last method in the class
class_end_pattern = r'(\s+def validate_research_questions\(self, correspondence: CorrespondenceMetrics,.*?return rq_results)'

content = re.sub(class_end_pattern, robust_method + r'\1', content, flags=re.DOTALL)

# Add scipy imports at the top
if 'from scipy.stats import entropy' not in content:
    content = content.replace(
        'import numpy as np',
        'import numpy as np\nfrom scipy.stats import entropy'
    )

# Write back
with open('src/core/metrics.py', 'w') as f:
    f.write(content)

print("Added robust correspondence calculation method")

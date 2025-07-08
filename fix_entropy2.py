import re

# Read the file
with open('src/core/data_structures.py', 'r') as f:
    content = f.read()

# Find and replace the entropy method to handle any qs format
entropy_pattern = r'    def get_entropy\(self\) -> float:.*?return entropy\(self\.qs \+ 1e-10\)'
new_entropy = '''    def get_entropy(self) -> float:
        """Calculate entropy of current belief state."""
        from scipy.stats import entropy
        import numpy as np
        
        # Handle different formats of qs
        if isinstance(self.qs, list):
            # Concatenate all state factors
            all_probs = np.concatenate([np.array(qs_factor).flatten() for qs_factor in self.qs])
        else:
            # Single array
            all_probs = np.array(self.qs).flatten()
        
        # Normalize to ensure it's a proper probability distribution
        all_probs = all_probs / (np.sum(all_probs) + 1e-10)
        return entropy(all_probs + 1e-10)'''

content = re.sub(entropy_pattern, new_entropy, content, flags=re.DOTALL)

# Write back
with open('src/core/data_structures.py', 'w') as f:
    f.write(content)

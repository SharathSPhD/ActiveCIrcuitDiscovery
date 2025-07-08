import re

# Read the file
with open('src/core/data_structures.py', 'r') as f:
    content = f.read()

# Find and replace the entropy method to handle concatenation more carefully
entropy_pattern = r'        # Handle different formats of qs.*?return entropy\(all_probs \+ 1e-10\)'
new_entropy = '''        # Handle different formats of qs
        if isinstance(self.qs, list):
            # Sum entropies of each state factor separately
            total_entropy = 0.0
            for qs_factor in self.qs:
                factor_array = np.array(qs_factor).flatten()
                if len(factor_array) > 0 and np.sum(factor_array) > 0:
                    factor_probs = factor_array / np.sum(factor_array)
                    total_entropy += entropy(factor_probs + 1e-10)
            return total_entropy
        else:
            # Single array
            all_probs = np.array(self.qs).flatten()
            if len(all_probs) > 0 and np.sum(all_probs) > 0:
                all_probs = all_probs / np.sum(all_probs)
                return entropy(all_probs + 1e-10)
            else:
                return 0.0'''

content = re.sub(entropy_pattern, new_entropy, content, flags=re.DOTALL)

# Write back
with open('src/core/data_structures.py', 'w') as f:
    f.write(content)

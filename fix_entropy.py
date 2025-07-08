import re

# Read the file
with open('src/core/data_structures.py', 'r') as f:
    content = f.read()

# Find and replace the corrupted entropy method
entropy_pattern = r'    def get_entropy\(self\) -> float:.*?def get_average_uncertainty'
new_entropy = '''    def get_entropy(self) -> float:
        """Calculate entropy of current belief state."""
        from scipy.stats import entropy
        if isinstance(self.qs, list):
            # Calculate total entropy across all state factors
            total_entropy = 0.0
            for qs_factor in self.qs:
                total_entropy += entropy(qs_factor + 1e-10)
            return total_entropy
        else:
            return entropy(self.qs + 1e-10)
    
    def get_average_uncertainty'''

content = re.sub(entropy_pattern, new_entropy, content, flags=re.DOTALL)

# Write back
with open('src/core/data_structures.py', 'w') as f:
    f.write(content)

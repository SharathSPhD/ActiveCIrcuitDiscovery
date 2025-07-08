import re

# Read the file
with open('src/core/data_structures.py', 'r') as f:
    content = f.read()

# Find and replace the entropy method with pymdp-style version
entropy_pattern = r'    def get_entropy\(self\) -> float:.*?return 0\.0'
new_entropy = '''    def get_entropy(self) -> float:
        """Calculate entropy of current belief state using pymdp approach."""
        import numpy as np
        
        def is_obj_array(arr):
            return hasattr(arr, 'dtype') and arr.dtype == 'object'
        
        def entropy_single(dist):
            """Entropy of a single distribution"""
            from scipy.stats import entropy
            dist_flat = np.asarray(dist).flatten()
            if np.sum(dist_flat) > 0:
                dist_norm = dist_flat / np.sum(dist_flat)
                return entropy(dist_norm + 1e-16)
            return 0.0
        
        try:
            # Use pymdp-style object array detection
            if is_obj_array(self.qs):
                # Multiple factors - sum entropies
                total_entropy = 0.0
                for factor in self.qs:
                    total_entropy += entropy_single(factor)
                return total_entropy
            else:
                # Single factor
                return entropy_single(self.qs)
        except Exception:
            # Fallback
            return 0.0'''

content = re.sub(entropy_pattern, new_entropy, content, flags=re.DOTALL)

# Write back
with open('src/core/data_structures.py', 'w') as f:
    f.write(content)

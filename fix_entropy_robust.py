import re

# Read the file
with open('src/core/data_structures.py', 'r') as f:
    content = f.read()

# Find and replace the entropy method with robust version
entropy_pattern = r'    def get_entropy\(self\) -> float:.*?return 0\.0'
new_entropy = '''    def get_entropy(self) -> float:
        """Calculate entropy of current belief state."""
        from scipy.stats import entropy
        import numpy as np
        
        try:
            # Convert to list to handle any sequence type safely
            if hasattr(self.qs, '__iter__') and not isinstance(self.qs, np.ndarray):
                # It's a sequence (list, tuple, etc.) - iterate through factors
                factors = list(self.qs)
                if len(factors) == 1:
                    # Single factor case
                    factor_array = np.asarray(factors[0]).flatten()
                else:
                    # Multiple factors - calculate entropy for each and sum
                    total_entropy = 0.0
                    for factor in factors:
                        factor_array = np.asarray(factor).flatten()
                        if np.sum(factor_array) > 0:
                            factor_probs = factor_array / np.sum(factor_array)
                            total_entropy += entropy(factor_probs + 1e-10)
                    return total_entropy
            else:
                # Single array case
                factor_array = np.asarray(self.qs).flatten()
            
            # Common final processing for single factor
            if np.sum(factor_array) > 0:
                factor_probs = factor_array / np.sum(factor_array)
                return entropy(factor_probs + 1e-10)
            return 0.0
            
        except Exception:
            # Ultimate fallback
            return 0.0'''

content = re.sub(entropy_pattern, new_entropy, content, flags=re.DOTALL)

# Write back
with open('src/core/data_structures.py', 'w') as f:
    f.write(content)

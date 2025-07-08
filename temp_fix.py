import re

# Read the data structures file
with open('src/core/data_structures.py', 'r') as f:
    content = f.read()

# Add backward compatibility property after the CircuitFeature class
circuit_feature_pattern = r'(class CircuitFeature:.*?def __post_init__\(self\):.*?)'
replacement = r'\1\n    @property\n    def layer(self) -> int:\n        Backward compatibility property for layer access\n        return self.layer_idx\n'

content = re.sub(circuit_feature_pattern, replacement, content, flags=re.DOTALL)

# Write back
with open('src/core/data_structures.py', 'w') as f:
    f.write(content)

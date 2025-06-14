#\!/usr/bin/env python3

# Read the data_structures.py file
with open('src/core/data_structures.py', 'r') as f:
    content = f.read()

# Fix the validation
old_validation = '''    def __post_init__(self):
        if not 0 <= self.max_activation <= 1:
            raise ValueError(fmax_activation must be in [0,1], got {self.max_activation})
        if self.layer < 0:
            raise ValueError(flayer must be non-negative, got {self.layer})'''

new_validation = '''    def __post_init__(self):
        # Allow larger activations for real transformer features
        if self.max_activation < 0:
            raise ValueError(fmax_activation must be non-negative, got {self.max_activation})
        if self.layer < 0:
            raise ValueError(flayer must be non-negative, got {self.layer})'''

content = content.replace(old_validation, new_validation)

# Write back to file
with open('src/core/data_structures.py', 'w') as f:
    f.write(content)

print("Fixed SAE activation threshold validation in data_structures.py")

# Read the data structures file
with open('src/core/data_structures.py', 'r') as f:
    content = f.read()

# Fix the __post_init__ method and property placement
content = content.replace(
    '''    def __post_init__(self):
    @property
    def layer(self) -> int:
        Backward compatibility property for layer access
        return self.layer_idx

        if self.max_activation < 0:''',
    '''    def __post_init__(self):
        if self.max_activation < 0:'''
)

# Add the property after the __post_init__ method
content = content.replace(
    '''        if not self.examples:
            self.examples = [fExample for {self.description}]
        
        # Initialize feature vector if not provided
        if self.feature_vector is None:
            # Create a simple feature vector based on activation strength
            self.feature_vector = np.array([self.activation_strength] * 10)''',
    '''        if not self.examples:
            self.examples = [fExample for {self.description}]
        
        # Initialize feature vector if not provided
        if self.feature_vector is None:
            # Create a simple feature vector based on activation strength
            self.feature_vector = np.array([self.activation_strength] * 10)
    
    @property
    def layer(self) -> int:
        Backward compatibility property for layer access
        return self.layer_idx'''
)

# Write back
with open('src/core/data_structures.py', 'w') as f:
    f.write(content)

print("Fixed data_structures.py indentation")

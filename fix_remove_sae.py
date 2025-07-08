# Read the data structures file
with open('src/core/data_structures.py', 'r') as f:
    content = f.read()

# Remove SAE-specific content and make it purely transcoder-based
content = content.replace(
    'feature_source: str = sae  # \'sae\' or \'transcoder\'',
    'feature_source: str = transcoder  # Always transcoder for circuit-tracer'
)

# Update unique_id to always use T prefix
content = content.replace(
    'source_prefix = T if self.feature_source == transcoder else S',
    'source_prefix = T  # Always transcoder'
)

# Fix the from_transcoder_data method parameter name
content = content.replace('layer_idx=layer,', 'layer_idx=layer_idx,')

# Write back
with open('src/core/data_structures.py', 'w') as f:
    f.write(content)

print("Fixed transcoder data structure issues")

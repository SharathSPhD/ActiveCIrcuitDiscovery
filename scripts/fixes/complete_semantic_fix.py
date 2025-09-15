# Complete fix for multi-token generation + bidirectional semantic validation

import re

with open('run_comprehensive_experiment.py', 'r') as f:
    content = f.read()

# 1. Replace single-token model prediction with multi-token generation
old_prediction_block = re.compile(
    r'                # Get model prediction\n'
    r'                tokens = self\.circuit_tracer\.tokenizer\(prompt, return_tensors="pt"\)\.to\(self\.device\)\n'
    r'                \n'
    r'                with torch\.no_grad\(\):\n'
    r'                    logits = self\.circuit_tracer\.model\(tokens\.input_ids\)\n'
    r'                    probs = torch\.softmax\(logits\[0, -1, :\], dim=-1\)\n'
    r'                    top_tokens = torch\.topk\(probs, 10\)\n'
    r'                \n'
    r'                # Get top predictions\n'
    r'                predictions = \[\]\n'
    r'                for prob, token_id in zip\(top_tokens\.values, top_tokens\.indices\):\n'
    r'                    token = self\.circuit_tracer\.tokenizer\.decode\(\[token_id\]\)\.strip\(\)\n'
    r'                    predictions\.append\(\{\n'
    r'                        "token": token,\n'
    r'                        "probability": float\(prob\)\n'
    r'                    \}\)',
    re.MULTILINE
)

new_prediction_block = '''                # Get model prediction with multi-token generation
                tokens = self.circuit_tracer.tokenizer(prompt, return_tensors=pt).to(self.device)
                
                # Generate multi-token completion for better semantic capture
                with torch.no_grad():
                    # Get both multi-token completion and single-token predictions
                    generated = self.circuit_tracer.model.generate(
                        tokens.input_ids,
                        max_new_tokens=4,
                        do_sample=False
                    )
                    
                    # Extract the completion
                    new_tokens = generated[0][tokens.input_ids.shape[1]:]
                    completion = self.circuit_tracer.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    
                    # Also get single-token predictions for analysis
                    logits = self.circuit_tracer.model(tokens.input_ids)
                    probs = torch.softmax(logits[0, -1, :], dim=-1)
                    top_tokens = torch.topk(probs, 10)
                
                # Get top predictions
                predictions = []
                for prob, token_id in zip(top_tokens.values, top_tokens.indices):
                    token = self.circuit_tracer.tokenizer.decode([token_id]).strip()
                    predictions.append({
                        token: token,
                        probability: float(prob)
                    })'''

content = old_prediction_block.sub(new_prediction_block, content)

# 2. Replace semantic validation with bidirectional logic
old_semantic_block = re.compile(
    r'                # Check for semantic correctness\n'
    r'                san_francisco_terms = \[.*?\]\n'
    r'                semantic_match = any\(term\.lower\(\) in pred\["token"\]\.lower\(\) \n'
    r'                                   for pred in predictions\[:3\] \n'
    r'                                   for term in san_francisco_terms\)',
    re.MULTILINE  < /dev/null |  re.DOTALL
)

new_semantic_block = '''                # Check for semantic correctness - bidirectional Golden Gate <-> San Francisco
                completion_lower = completion.lower()
                prompt_lower = prompt.lower()
                
                # Bidirectional semantic validation
                if golden gate in prompt_lower:
                    # Prompt about Golden Gate -> expect San Francisco/California answers
                    semantic_match = (
                        san francisco in completion_lower or
                        francisco in completion_lower or
                        california in completion_lower or
                        completion_lower.strip() == sf
                    )
                elif san francisco in prompt_lower:
                    # Prompt about San Francisco -> expect Golden Gate Bridge answers
                    semantic_match = (
                        golden gate in completion_lower or
                        bridge in completion_lower
                    )
                else:
                    # General case - accept either direction
                    semantic_match = (
                        san francisco in completion_lower or
                        francisco in completion_lower or
                        california in completion_lower or
                        golden gate in completion_lower or
                        bridge in completion_lower or
                        completion_lower.strip() == sf
                    )'''

content = old_semantic_block.sub(new_semantic_block, content)

# 3. Update logging to show completion instead of predictions[0]
content = re.sub(
    r'logger\.info\(f✅ Semantic discovery successful: {predictions[0]["token"]}\)',
    'logger.info(f✅ Semantic discovery successful: {completion})',
    content
)

content = re.sub(
    r'logger\.info\(f❌ Semantic discovery failed\)',
    'logger.info(f❌ Semantic discovery failed: {completion})',
    content
)

# 4. Add completion to results
content = re.sub(
    r'                    top_prediction: predictions\[0\]\[token\],',
    '''                    top_prediction: predictions[0][token],
                    full_completion: completion,''',
    content
)

# Write the fixed file
with open('run_comprehensive_experiment.py', 'w') as f:
    f.write(content)

print('Applied complete semantic discovery fix: multi-token + bidirectional validation')

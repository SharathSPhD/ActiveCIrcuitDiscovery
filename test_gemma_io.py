#!/usr/bin/env python3
"""
Test script to verify Gemma I/O capture works with 2 test cases.
"""

import sys
sys.path.insert(0, 'src')

from experiments.core.master_workflow import *
import json

def test_gemma_io():
    print('🚀 Testing Enhanced Gemma I/O Capture...')
    
    # Create limited test cases
    test_cases = generate_comprehensive_test_cases()[:2]  # Only first 2 cases
    methods = ['Enhanced Active Inference', 'Activation Patching']  # Only 2 methods
    
    print(f'Testing with {len(test_cases)} test cases and {len(methods)} methods')
    
    # Initialize executor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    executor = AuthenticGemmaModelExecutor(device)
    
    print('Loading Gemma model...')
    executor.load_model_and_transcoders()
    
    print('Executing test cases...')
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f'Processing test case {i+1}: "{test_case.input_text}"')
        
        for method in methods:
            try:
                result = executor.execute_method_on_test_case(method, test_case)
                results.append({
                    'test_case_id': test_case.id,
                    'method': method,
                    'test_prompt': result.test_prompt,
                    'gemma_output': result.gemma_output,
                    'intervention_effect': result.intervention_effect
                })
                print(f'  ✅ {method}: "{result.gemma_output[:50]}..."')
            except Exception as e:
                print(f'  ❌ {method}: Error - {e}')
    
    # Save test results
    with open('test_gemma_io_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\n✅ Test completed! {len(results)} results saved to test_gemma_io_results.json')
    
    # Verify all fields are present
    if results:
        first_result = results[0]
        required_fields = ['test_prompt', 'gemma_output', 'intervention_effect']
        missing = [f for f in required_fields if f not in first_result]
        if missing:
            print(f'❌ Missing fields: {missing}')
        else:
            print('✅ All required fields present')

if __name__ == '__main__':
    test_gemma_io()

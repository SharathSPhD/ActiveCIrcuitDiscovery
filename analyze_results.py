import json

with open('results/archive/workflow_results_20250915_155016/refact4_comprehensive_results.json', 'r') as f:
    data = json.load(f)
    
print('=== DATA STRUCTURE ANALYSIS ===')
test_cases = data.get('test_cases', [])
print('Test cases:', len(test_cases))

for i, test_case in enumerate(test_cases[:2]):
    print(f'\n--- Test Case {i+1} ---')
    print('Input:', test_case.get('input', 'N/A'))
    
    method_selections = test_case.get('method_selections', {})
    print('Methods tested:', list(method_selections.keys()))
    
    for method, result in method_selections.items():
        effect = result.get('effect_magnitude', 'N/A')
        time = result.get('computation_time', 'N/A')
        print(f'  {method}: effect={effect}, time={time}')
    
    intervention_details = test_case.get('intervention_details', [])
    print('Intervention details count:', len(intervention_details))
    for j, detail in enumerate(intervention_details[:3]):
        method = detail.get('method', 'N/A')
        semantic = detail.get('semantic_success', 'N/A')
        print(f'  Detail {j}: Method: {method}, Semantic success: {semantic}')

# Check for identical success rates issue
print('\n=== CHECKING FOR IDENTICAL SUCCESS RATES ===')
method_successes = {}
method_totals = {}

for test_case in test_cases:
    intervention_details = test_case.get('intervention_details', [])
    for detail in intervention_details:
        method = detail.get('method', 'Unknown')
        semantic_success = detail.get('semantic_success', False)
        
        if method not in method_successes:
            method_successes[method] = 0
            method_totals[method] = 0
        
        method_totals[method] += 1
        if semantic_success:
            method_successes[method] += 1

print('\nMethod success rates:')
for method in method_totals:
    rate = (method_successes[method] / method_totals[method]) * 100 if method_totals[method] > 0 else 0
    print(f'{method}: {rate:.1f}% ({method_successes[method]}/{method_totals[method]})')

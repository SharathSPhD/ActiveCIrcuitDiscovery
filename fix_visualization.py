#\!/usr/bin/env python3

# Read the run_complete_experiment.py file
with open('run_complete_experiment.py', 'r') as f:
    content = f.read()

# Fix the visualization call
old_call = '''        visualizer.generate_all_visualizations(
            result=result,
            output_dir=str(vis_dir),
            statistical_validation=statistical_validation
        )'''

new_call = '''        # Generate visualizations with proper parameters
        visualizations = visualizer.generate_all_visualizations(
            result=result,
            statistical_validation=statistical_validation
        )
        
        # Move generated files to output directory if needed
        for viz_type, file_path in visualizations.items():
            print(f"📊 Generated {viz_type}: {file_path}")'''

content = content.replace(old_call, new_call)

# Write back to file
with open('run_complete_experiment.py', 'w') as f:
    f.write(content)

print("Fixed visualization method call in run_complete_experiment.py")

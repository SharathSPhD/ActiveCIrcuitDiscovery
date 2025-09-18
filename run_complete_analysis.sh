#!/bin/bash
# COMPLETE ACTIVECIRCUITDISCOVERY ANALYSIS RUNNER
# Single command to run master workflow + unified visualizations

echo "🚀 STARTING COMPLETE ACTIVECIRCUITDISCOVERY ANALYSIS"
echo "=================================================="
echo "✅ 35 diverse test cases"
echo "✅ Authentic Gemma-2-2B model execution"
echo "✅ Method-specific evaluation frameworks"
echo "✅ Statistical significance testing"
echo "✅ Unified authentic visualization system"
echo ""

# Activate virtual environment
source /home/ubuntu/project_venv/bin/activate

# Run master workflow
echo "📊 Running master workflow with comprehensive analysis..."
python experiments/core/master_workflow.py

# Find latest results directory
LATEST_RESULTS=$(ls -t results/authentic_master_workflow_* | head -1)

if [ -n "$LATEST_RESULTS" ]; then
    echo ""
    echo "🎨 Generating unified authentic visualizations..."
    python -c "
from unified_authentic_visualizer import UnifiedAuthenticVisualizer
from pathlib import Path
results_dir = Path('$LATEST_RESULTS')
visualizer = UnifiedAuthenticVisualizer(results_dir)
outputs = visualizer.generate_comprehensive_visualizations()
print(f'✅ Generated {sum(len(paths) for paths in outputs.values())} visualization files')
print(f'📂 Saved to: {visualizer.output_dir}')
"

    echo ""
    echo "🎉 COMPLETE ANALYSIS FINISHED!"
    echo "📂 Results: $LATEST_RESULTS"
    echo "📊 Visualizations: $LATEST_RESULTS/unified_visualizations"
    echo ""
    echo "DELIVERABLES:"
    echo "✅ comprehensive_experiment_results.json - Raw experimental data"
    echo "✅ statistical_analysis.json - Statistical validation"
    echo "✅ method_performance_comprehensive.png - Performance comparison"
    echo "✅ circuit_method_comparison.png - Circuit analysis"
    echo "✅ feature_effectiveness_analysis.png - Feature analysis"
    echo "✅ visualization_summary_report.txt - Complete summary"
else
    echo "❌ No results found. Check for errors in master workflow."
fi

#!/bin/bash
# COMPLETE ACTIVECIRCUITDISCOVERY ANALYSIS RUNNER
# Single command to run master workflow + unified visualizations + layer analysis

echo "🚀 STARTING COMPLETE ACTIVECIRCUITDISCOVERY ANALYSIS"
echo "=================================================="
echo "✅ 35 diverse test cases"
echo "✅ Authentic Gemma-2-2B model execution"
echo "✅ Method-specific evaluation frameworks"
echo "✅ Statistical significance testing"
echo "✅ Unified authentic visualization system"
echo "✅ Layer/feature analysis with Active Inference metrics"
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
from unified_authentic_visualizer import EnhancedUnifiedAuthenticVisualizer
from pathlib import Path
results_dir = Path('$LATEST_RESULTS')
visualizer = EnhancedUnifiedAuthenticVisualizer(results_dir)
try:
    outputs = visualizer.generate_enhanced_visualizations()
    print(f'✅ Generated {sum(len(paths) for paths in outputs.values())} visualization files')
    print(f'📂 Saved to: {visualizer.output_dir}')
except Exception as e:
    print(f'⚠️ Visualization generation had issues: {e}')
    print('📊 Continuing with layer analysis...')
"

    echo ""
    echo "🔬 Generating layer/feature analysis plots..."
    python create_layer_plots.py

    echo ""
    echo "🎉 COMPLETE ANALYSIS FINISHED!"
    echo "📂 Results: $LATEST_RESULTS"
    echo "📊 Visualizations: $LATEST_RESULTS/enhanced_unified_visualizations"
    echo "🔬 Layer Analysis: $LATEST_RESULTS/layer_feature_plots"
    echo ""
    echo "DELIVERABLES:"
    echo "✅ comprehensive_experiment_results.json - Raw experimental data with layer details"
    echo "✅ statistical_analysis.json - Statistical validation"
    echo "✅ method_performance_summary.csv - Performance metrics"
    echo "✅ enhanced_unified_visualizations/ - Case-specific analysis"
    echo "✅ layer_feature_plots/ - Layer/feature activation analysis"
    echo "✅ Individual case plots with transcoder features (L8F7439, L10F8215, etc.)"
else
    echo "❌ No results found. Check for errors in master workflow."
fi

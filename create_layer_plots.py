#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Load data
with open('results/authentic_master_workflow_20250919_182203/comprehensive_experiment_results.json', 'r') as f:
    data = json.load(f)

print(f"🔍 Analyzing {len(data['test_case_details'])} test cases...")

# Create comprehensive layer analysis plots for all cases
results_dir = Path('results/authentic_master_workflow_20250919_182203/layer_feature_plots')
results_dir.mkdir(exist_ok=True)

# Summary data
all_layer_data = {}
all_features_by_layer = {}

for i, test_case in enumerate(data['test_case_details'][:5]):  # First 5 cases for demonstration
    case_id = test_case['test_case_id']
    test_prompt = test_case['input_text']

    eai_result = test_case['method_results']['Enhanced Active Inference']
    layer_data = eai_result['method_specific_metrics']['layer_activations']
    discovered_features = eai_result['method_specific_metrics']['discovered_features']

    print(f"Case {case_id}: {len(layer_data)} layers, {len(discovered_features)} features")

    # Create individual case plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Feature activation heatmap
    layers = sorted(layer_data.keys())
    activations_matrix = []
    feature_labels = []

    for layer in layers:
        layer_features = layer_data[layer]['features']
        layer_activations = layer_data[layer]['activation_strengths']
        activations_matrix.extend(layer_activations)
        layer_label = layer.replace('layer_', 'L')
        feature_labels.extend([f"{layer_label}:{feat}" for feat in layer_features])

    if activations_matrix:
        n_features = len(activations_matrix)
        heatmap_data = np.array(activations_matrix).reshape(1, -1)

        im = ax1.imshow(heatmap_data, cmap='viridis', aspect='auto')
        ax1.set_xticks(range(n_features))
        ax1.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=6)
        ax1.set_yticks([0])
        ax1.set_yticklabels(['Activation'])
        ax1.set_title(f'Layer-wise Feature Activations\nCase {case_id}: {test_prompt[:50]}...', fontsize=12)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1, shrink=0.6)
        cbar.set_label('Activation Strength', rotation=270, labelpad=15)

    # 2. Layer summary statistics
    layer_means = []
    layer_names = []
    layer_counts = []

    for layer in layers:
        layer_means.append(layer_data[layer]['mean_activation'])
        layer_names.append(layer.replace('layer_', 'L'))
        layer_counts.append(len(layer_data[layer]['features']))

    bars = ax2.bar(layer_names, layer_means, alpha=0.7, color='skyblue')
    ax2.set_ylabel('Mean Activation')
    ax2.set_title('Mean Activation by Layer')
    ax2.tick_params(axis='x', rotation=45)

    # Add count annotations
    for bar, count in zip(bars, layer_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{count}', ha='center', va='bottom', fontsize=8)

    # 3. Top features
    all_activations = []
    all_feat_labels = []
    for layer in layers:
        for feat, act in zip(layer_data[layer]['features'], layer_data[layer]['activation_strengths']):
            all_activations.append(act)
            all_feat_labels.append(f"{layer.replace('layer_', 'L')}:{feat}")

    # Show top 8 features
    if all_activations:
        top_indices = np.argsort(all_activations)[-8:]
        top_activations = [all_activations[i] for i in top_indices]
        top_labels = [all_feat_labels[i] for i in top_indices]

        ax3.barh(range(len(top_activations)), top_activations, alpha=0.7, color='orange')
        ax3.set_yticks(range(len(top_labels)))
        ax3.set_yticklabels(top_labels, fontsize=8)
        ax3.set_title('Top 8 Feature Activations')
        ax3.set_xlabel('Activation Strength')

    # 4. Active Inference metrics
    efe_score = eai_result['method_specific_metrics']['efe_minimization_score']
    belief_corr = eai_result['method_specific_metrics']['belief_correspondence']
    feature_precision = eai_result['method_specific_metrics']['feature_selection_precision']
    kl_div = eai_result['method_specific_metrics']['kl_divergence_mean']

    metrics = ['EFE Score', 'Belief Corr.', 'Feature Prec.', 'KL Div.']
    values = [efe_score, belief_corr, feature_precision, kl_div]

    bars = ax4.bar(metrics, values, alpha=0.7, color=['red', 'green', 'blue', 'orange'])
    ax4.set_ylabel('Metric Value')
    ax4.set_title('Active Inference Metrics')
    ax4.tick_params(axis='x', rotation=45)

    # Add value annotations
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    case_plot_path = results_dir / f'case_{case_id:02d}_layer_feature_analysis.png'
    plt.savefig(case_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Generated plot for case {case_id}: {case_plot_path}")

print(f"\n🎉 Layer/feature analysis plots generated in: {results_dir}")
print(f"📊 Individual case plots showing:")
print("  - Layer-wise feature activation heatmaps")
print("  - Mean activation by layer with feature counts")
print("  - Top feature activations per case")
print("  - Active Inference metrics (EFE, belief correspondence, KL divergence)")
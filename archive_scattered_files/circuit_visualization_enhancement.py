#!/usr/bin/env python3
"""
Circuit Component Visualization Enhancement for Unified Authentic Visualizer
Adds circuit features, layers, and activation visualization
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def visualize_circuit_components_for_case(test_case, case_output_dir: Path) -> List[Path]:
    """Create circuit component visualizations for a specific test case."""
    
    visualization_files = []
    
    # Check if any method has circuit_components data
    has_circuit_data = False
    for method, result in test_case.method_results.items():
        if hasattr(result, 'circuit_components') and result.circuit_components:
            has_circuit_data = True
            break
        elif isinstance(result, dict) and 'circuit_components' in result and result['circuit_components']:
            has_circuit_data = True
            break
    
    if not has_circuit_data:
        return visualization_files
    
    # Create circuit components comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Circuit Components Analysis - Case {test_case.case_id}\n{test_case.input_text}', 
                 fontsize=16, fontweight='bold')
    
    methods = list(test_case.method_results.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(methods)]
    
    # 1. Feature Discovery Summary (Top Left)
    ax1 = axes[0, 0]
    feature_counts = []
    method_names = []
    
    for method in methods:
        result = test_case.method_results[method]
        circuit_data = getattr(result, 'circuit_components', None) or (result.get('circuit_components') if isinstance(result, dict) else None)
        
        if circuit_data and 'discovered_features' in circuit_data:
            feature_counts.append(len(circuit_data['discovered_features']))
            method_names.append(method)
    
    if feature_counts:
        bars = ax1.bar(range(len(method_names)), feature_counts, color=colors[:len(method_names)], alpha=0.8)
        ax1.set_title('Features Discovered per Method', fontweight='bold')
        ax1.set_ylabel('Number of Features')
        ax1.set_xticks(range(len(method_names)))
        ax1.set_xticklabels([m.replace(' ', '\n') for m in method_names], fontsize=9)
        
        # Add value labels on bars
        for bar, count in zip(bars, feature_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontweight='bold')
    
    # 2. Layer Activation Analysis (Top Right)
    ax2 = axes[0, 1]
    layer_data_by_method = {}
    
    for method in methods:
        result = test_case.method_results[method]
        circuit_data = getattr(result, 'circuit_components', None) or (result.get('circuit_components') if isinstance(result, dict) else None)
        
        if circuit_data and 'layer_activations' in circuit_data:
            layer_data_by_method[method] = circuit_data['layer_activations']
    
    if layer_data_by_method:
        # Plot max activations by layer for each method
        for i, (method, layer_data) in enumerate(layer_data_by_method.items()):
            layers = []
            max_activations = []
            
            for layer_name, data in layer_data.items():
                layer_num = int(layer_name.split('_')[1])
                layers.append(layer_num)
                max_activations.append(data['max_activation'])
            
            ax2.plot(layers, max_activations, marker='o', linewidth=2, 
                    label=method, color=colors[i % len(colors)])
        
        ax2.set_title('Max Activation by Layer', fontweight='bold')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Max Activation')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    
    # 3. Circuit Pathway Comparison (Bottom Left)
    ax3 = axes[1, 0]
    pathway_strengths = []
    
    for method in methods:
        result = test_case.method_results[method]
        circuit_data = getattr(result, 'circuit_components', None) or (result.get('circuit_components') if isinstance(result, dict) else None)
        
        if circuit_data and 'circuit_pathway' in circuit_data:
            pathways = circuit_data['circuit_pathway']
            avg_strength = np.mean([p['connection_strength'] for p in pathways]) if pathways else 0
            pathway_strengths.append(avg_strength)
        else:
            pathway_strengths.append(0)
    
    if any(s > 0 for s in pathway_strengths):
        bars = ax3.bar(range(len(methods)), pathway_strengths, color=colors, alpha=0.8)
        ax3.set_title('Average Circuit Connection Strength', fontweight='bold')
        ax3.set_ylabel('Connection Strength')
        ax3.set_xticks(range(len(methods)))
        ax3.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=9)
        
        # Add value labels on bars
        for bar, strength in zip(bars, pathway_strengths):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{strength:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Method-Specific Analysis (Bottom Right)
    ax4 = axes[1, 1]
    method_scores = []
    score_labels = []
    
    for method in methods:
        result = test_case.method_results[method]
        circuit_data = getattr(result, 'circuit_components', None) or (result.get('circuit_components') if isinstance(result, dict) else None)
        
        if circuit_data:
            # Get method-specific analysis score
            if 'efe_guided_selection' in circuit_data:
                score = circuit_data['efe_guided_selection'].get('selected_features', 0) / 100.0
                score_labels.append('EFE Features')
            elif 'patching_analysis' in circuit_data:
                score = circuit_data['patching_analysis'].get('intervention_precision', 0)
                score_labels.append('Patch Precision')
            elif 'attribution_analysis' in circuit_data:
                score = circuit_data['attribution_analysis'].get('feature_attribution_precision', 0)
                score_labels.append('Attribution Prec.')
            elif 'ranking_analysis' in circuit_data:
                score = circuit_data['ranking_analysis'].get('selection_precision', 0)
                score_labels.append('Ranking Prec.')
            else:
                score = 0
                score_labels.append('No Data')
            
            method_scores.append(score)
        else:
            method_scores.append(0)
            score_labels.append('No Data')
    
    if any(s > 0 for s in method_scores):
        bars = ax4.bar(range(len(methods)), method_scores, color=colors, alpha=0.8)
        ax4.set_title('Method-Specific Circuit Analysis', fontweight='bold')
        ax4.set_ylabel('Analysis Score')
        ax4.set_xticks(range(len(methods)))
        ax4.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=9)
        
        # Add value labels on bars
        for bar, score in zip(bars, method_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save circuit components visualization
    circuit_file = case_output_dir / 'circuit_components_analysis.png'
    plt.savefig(circuit_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    visualization_files.append(circuit_file)
    return visualization_files


if __name__ == "__main__":
    print("Circuit visualization enhancement module ready")

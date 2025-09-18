#!/usr/bin/env python3
"""
Generate Circuit-Tracer Style Attribution Graphs
Shows neural network activation pathways for EFE-guided vs SOTA methods
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CircuitTracerVisualizer:
    def __init__(self, results_path):
        """Load REFACT-4 results and prepare for circuit visualization"""
        with open(results_path) as f:
            self.data = json.load(f)

        print(f"📊 Loaded experiment: {self.data['experiment_id']}")
        print(f"🔬 Methods: {list(self.data['method_performance'].keys())}")

    def extract_circuit_pathways(self):
        """Extract transcoder feature pathways for each method"""
        pathways = {}

        for test_case in self.data['test_cases']:
            case_id = test_case['test_case']
            prompt = test_case['input']

            print(f"🧠 Analyzing circuit pathways for: '{prompt[:50]}...'")

            # Extract method-specific circuit selections
            for method, selection in test_case['method_selections'].items():
                if method not in pathways:
                    pathways[method] = {'circuits': [], 'features': [], 'effects': []}

                # Extract transcoder feature info
                circuit_id = selection['circuit']  # e.g., 'L8F3099'
                layer = selection['layer']
                feature_id = selection['feature_id']
                activation = selection['activation']
                effect = selection['effect_magnitude']

                pathways[method]['circuits'].append({
                    'circuit_id': circuit_id,
                    'layer': layer,
                    'feature_id': feature_id,
                    'activation': activation,
                    'effect': effect,
                    'prompt': prompt,
                    'case_id': case_id
                })

        return pathways

    def create_circuit_attribution_graph(self, method_name, pathway_data):
        """Create circuit-tracer style attribution graph showing neural pathways"""
        fig, ax = plt.subplots(figsize=(16, 12))

        # Create network graph representing transcoder feature connections
        G = nx.DiGraph()  # Directed graph to show information flow

        # Add nodes for each transcoder feature
        pos = {}
        node_sizes = []
        node_colors = []
        labels = {}
        edge_weights = []

        # Group circuits by layer for better visualization
        circuits_by_layer = {}
        for circuit in pathway_data['circuits']:
            layer = circuit['layer']
            if layer not in circuits_by_layer:
                circuits_by_layer[layer] = []
            circuits_by_layer[layer].append(circuit)

        # Position nodes by layer (x-axis) and spread vertically
        layer_positions = {}
        for layer, circuits in circuits_by_layer.items():
            layer_positions[layer] = layer * 3  # Spread layers horizontally

            for i, circuit in enumerate(circuits):
                node_id = f"L{circuit['layer']}F{circuit['feature_id']}"
                G.add_node(node_id,
                          layer=circuit['layer'],
                          feature_id=circuit['feature_id'],
                          activation=circuit['activation'],
                          effect=circuit['effect'])

                # Position: x by layer, y by activation strength + spread
                x = layer_positions[layer]
                y = circuit['activation'] * 2 + i * 0.5
                pos[node_id] = (x, y)

                # Node size based on effect magnitude
                node_sizes.append(max(300, abs(circuit['effect']) * 100000))

                # Node color based on layer
                node_colors.append(circuit['layer'])

                # Node label
                labels[node_id] = f"L{circuit['layer']}\nF{circuit['feature_id']}"

        # Add edges to show information flow between layers
        sorted_layers = sorted(circuits_by_layer.keys())
        for i in range(len(sorted_layers) - 1):
            curr_layer = sorted_layers[i]
            next_layer = sorted_layers[i + 1]

            # Connect each feature in current layer to features in next layer
            for curr_circuit in circuits_by_layer[curr_layer]:
                curr_node = f"L{curr_circuit['layer']}F{curr_circuit['feature_id']}"
                for next_circuit in circuits_by_layer[next_layer]:
                    next_node = f"L{next_circuit['layer']}F{next_circuit['feature_id']}"

                    # Edge weight based on combined effect magnitudes
                    weight = (curr_circuit['effect'] + next_circuit['effect']) / 2
                    G.add_edge(curr_node, next_node, weight=weight)
                    edge_weights.append(weight)

        # Draw the circuit attribution graph only if we have nodes
        if len(pos) > 0 and len(node_sizes) == len(pos):
            # Draw nodes
            nx.draw_networkx_nodes(G, pos,
                                 node_size=node_sizes,
                                 node_color=node_colors,
                                 cmap='viridis',
                                 alpha=0.8,
                                 edgecolors='black',
                                 linewidths=1,
                                 ax=ax)

            # Draw edges with varying thickness based on weight
            if edge_weights:
                max_weight = max(edge_weights) if edge_weights else 1
                normalized_weights = [max(0.5, (w / max_weight) * 5) for w in edge_weights]
                nx.draw_networkx_edges(G, pos,
                                     edge_color='gray',
                                     alpha=0.6,
                                     width=normalized_weights,
                                     arrowsize=20,
                                     arrowstyle='->',
                                     ax=ax)

            # Draw labels
            nx.draw_networkx_labels(G, pos, labels,
                                   font_size=9,
                                   font_color='white',
                                   font_weight='bold',
                                   ax=ax)

        # Customize plot
        ax.set_title(f"Circuit-Tracer Attribution Graph: {method_name}\n"
                    f"Neural Pathway Visualization for Transcoder Features",
                    fontsize=16, fontweight='bold', pad=20)

        ax.set_xlabel("Layer Progression →", fontsize=14)
        ax.set_ylabel("Activation Strength", fontsize=14)

        # Add colorbar for layers
        if node_colors:
            sm = plt.cm.ScalarMappable(cmap='viridis',
                                     norm=plt.Normalize(vmin=min(node_colors),
                                                      vmax=max(node_colors)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label('Transformer Layer', fontsize=12)

        # Add text annotation with method details
        total_features = len(pathway_data['circuits'])
        avg_effect = np.mean([c['effect'] for c in pathway_data['circuits']])
        layers_used = sorted(set(c['layer'] for c in pathway_data['circuits']))

        info_text = (f"Method: {method_name}\n"
                    f"Features Discovered: {total_features}\n"
                    f"Layers: {layers_used}\n"
                    f"Avg Effect: {avg_effect:.6f}")

        ax.text(0.02, 0.98, info_text,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
               fontsize=10)

        # Remove axis spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        return fig

    def create_comparative_pathways(self, pathways):
        """Create side-by-side comparison of neural pathways"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()

        methods = list(pathways.keys())
        colors = ['red', 'blue', 'green', 'orange']

        for i, (method, data) in enumerate(pathways.items()):
            ax = axes[i]

            # Extract pathway data
            layers = [c['layer'] for c in data['circuits']]
            activations = [c['activation'] for c in data['circuits']]
            effects = [c['effect'] for c in data['circuits']]

            # Create scatter plot showing pathway distribution
            scatter = ax.scatter(layers, activations,
                               s=[e * 50000 for e in effects],
                               c=colors[i],
                               alpha=0.7,
                               edgecolors='black')

            # Add connection lines between points
            if len(layers) > 1:
                sorted_indices = np.argsort(layers)
                sorted_layers = [layers[j] for j in sorted_indices]
                sorted_activations = [activations[j] for j in sorted_indices]
                ax.plot(sorted_layers, sorted_activations,
                       color=colors[i], alpha=0.5, linewidth=2, linestyle='--')

            ax.set_title(f"{method}\nNeural Pathway Distribution",
                        fontsize=14, fontweight='bold')
            ax.set_xlabel("Transformer Layer")
            ax.set_ylabel("Feature Activation")
            ax.grid(True, alpha=0.3)

            # Add statistics
            avg_layer = np.mean(layers)
            avg_activation = np.mean(activations)
            avg_effect = np.mean(effects)

            stats_text = (f"Avg Layer: {avg_layer:.1f}\n"
                         f"Avg Activation: {avg_activation:.2f}\n"
                         f"Avg Effect: {avg_effect:.6f}")

            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                   fontsize=9)

        plt.suptitle("Circuit-Tracer: Comparative Neural Pathway Analysis\n"
                    "EFE-Guided Active Inference vs SOTA Methods",
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        return fig

    def generate_visualizations(self):
        """Generate all circuit-tracer visualizations"""
        print("\n🎯 GENERATING CIRCUIT-TRACER VISUALIZATIONS")
        print("=" * 60)

        # Extract circuit pathways
        pathways = self.extract_circuit_pathways()

        # Create output directory
        viz_dir = Path("visualizations/circuit_tracer_native")
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Generate individual attribution graphs
        for method, data in pathways.items():
            print(f"\n📊 Creating circuit attribution graph for: {method}")
            fig = self.create_circuit_attribution_graph(method, data)

            # Save as high-res PNG
            safe_name = method.lower().replace(' ', '_').replace('-', '_')
            output_file = viz_dir / f"{safe_name}_circuit_attribution.png"
            fig.savefig(str(output_file), dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"   ✅ Saved: {output_file}")
            plt.close(fig)

        # Generate comparative visualization
        print(f"\n🔄 Creating comparative pathway analysis")
        comp_fig = self.create_comparative_pathways(pathways)

        comp_file = viz_dir / "comparative_neural_pathways.png"
        comp_fig.savefig(str(comp_file), dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
        print(f"   ✅ Saved: {comp_file}")
        plt.close(comp_fig)

        # Generate summary report
        print(f"\n📝 Generating circuit analysis report")
        report_file = viz_dir / "circuit_analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write("CIRCUIT-TRACER ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Experiment: {self.data['experiment_id']}\n")
            f.write(f"Generated: {len(pathways)} method visualizations\n\n")

            for method, data in pathways.items():
                circuits = data['circuits']
                layers = [c['layer'] for c in circuits]
                effects = [c['effect'] for c in circuits]
                activations = [c['activation'] for c in circuits]

                f.write(f"{method}:\n")
                f.write(f"  - Transcoder Features: {len(circuits)}\n")
                f.write(f"  - Layer Range: {min(layers)}-{max(layers)}\n")
                f.write(f"  - Avg Effect Magnitude: {np.mean(effects):.6f}\n")
                f.write(f"  - Avg Activation: {np.mean(activations):.3f}\n")
                f.write(f"  - Circuit IDs: {[c['circuit_id'] for c in circuits]}\n\n")

        print(f"   ✅ Saved: {report_file}")

        # Final summary
        print(f"\n🎉 CIRCUIT-TRACER VISUALIZATIONS COMPLETE!")
        print(f"📁 Output directory: {viz_dir}")
        print(f"📊 Generated {len(pathways)} attribution graphs + 1 comparative view")

        # Key insights
        print(f"\n🧠 KEY CIRCUIT-TRACER INSIGHTS:")
        for method, data in pathways.items():
            circuits = data['circuits']
            layers = [c['layer'] for c in circuits]
            effects = [c['effect'] for c in circuits]

            print(f"   {method}:")
            print(f"     - Neural pathway spans layers: {min(layers)}-{max(layers)}")
            print(f"     - Average effect magnitude: {np.mean(effects):.6f}")
            print(f"     - Transcoder features activated: {len(circuits)}")
            print(f"     - Primary circuit: {circuits[0]['circuit_id'] if circuits else 'None'}")


if __name__ == "__main__":
    # Initialize visualizer with REFACT-4 results
    results_path = "results/archive/workflow_results_20250915_155016/refact4_comprehensive_results.json"

    print("🔬 CIRCUIT-TRACER NEURAL PATHWAY VISUALIZER")
    print("=" * 60)
    print("Generating authentic circuit-tracer style visualizations...")
    print("Showing transcoder feature activations and neural pathways\n")

    visualizer = CircuitTracerVisualizer(results_path)
    visualizer.generate_visualizations()
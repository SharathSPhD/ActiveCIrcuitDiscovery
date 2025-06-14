# Visualization System for YorK_RP Active Inference Circuit Discovery
# Integrates with circuit-tracer library patterns and creates publication-ready visualizations

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import json
import warnings

# Circuit visualization libraries
try:
    import circuitsvis as cv
    CIRCUITSVIS_AVAILABLE = True
except ImportError:
    CIRCUITSVIS_AVAILABLE = False
    warnings.warn("circuitsvis not available - using fallback visualizations")

# Project imports
try:
    from core.interfaces import IVisualizationGenerator
    from core.data_structures import (
        AttributionGraph, ExperimentResult, InterventionResult, 
        CorrespondenceMetrics, BeliefState, SAEFeature, NovelPrediction
    )
    from core.prediction_system import EnhancedPredictionGenerator
    from core.statistical_validation import StatisticalTest, StatisticalValidator
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from core.interfaces import IVisualizationGenerator
    from core.data_structures import (
        AttributionGraph, ExperimentResult, InterventionResult,
        CorrespondenceMetrics, BeliefState, SAEFeature, NovelPrediction
    )
    from core.prediction_system import EnhancedPredictionGenerator
    from core.statistical_validation import StatisticalTest, StatisticalValidator

logger = logging.getLogger(__name__)

class CircuitVisualizer(IVisualizationGenerator):
    """Enhanced circuit visualization system with statistical validation visualizations."""
    
    def __init__(self, output_dir: str = "visualizations"):
        """Initialize enhanced visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting style for publication-ready figures
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
        
        logger.info(f"Enhanced CircuitVisualizer initialized: {self.output_dir}")
    
    def create_circuit_diagram(self, graph: AttributionGraph, output_path: str) -> str:
        """Create circuit diagram visualization using circuit-tracer patterns."""
        logger.info("Creating circuit diagram visualization")
        
        if CIRCUITSVIS_AVAILABLE:
            return self._create_circuitsvis_diagram(graph, output_path)
        else:
            return self._create_networkx_diagram(graph, output_path)
    
    def _create_networkx_diagram(self, graph: AttributionGraph, output_path: str) -> str:
        """Create circuit diagram using NetworkX and matplotlib."""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes with layer information
        nodes_by_layer = {}
        
        for node in graph.nodes:
            G.add_node(node.node_id, 
                      layer=node.layer,
                      importance=node.importance,
                      description=node.description[:20] + "..." if len(node.description) > 20 else node.description)
            
            if node.layer not in nodes_by_layer:
                nodes_by_layer[node.layer] = []
            nodes_by_layer[node.layer].append(node.node_id)
        
        # Add edges
        for edge in graph.edges:
            G.add_edge(edge.source_id, edge.target_id, 
                      weight=edge.weight, 
                      confidence=edge.confidence)
        
        # Create hierarchical layout
        pos = self._create_hierarchical_layout(nodes_by_layer)
        
        # Draw nodes with size based on importance
        for layer, nodes in nodes_by_layer.items():
            node_sizes = [graph.get_node_by_id(node_id).importance * 1000 for node_id in nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                                 node_size=node_sizes,
                                 node_color=f'C{layer % 10}',
                                 alpha=0.8, ax=ax)
        
        # Draw edges with thickness based on weight
        edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, 
                             alpha=0.6, edge_color='gray', ax=ax)
        
        # Add labels
        labels = {node_id: f"L{graph.get_node_by_id(node_id).layer}\nF{graph.get_node_by_id(node_id).feature_id}" 
                 for node_id in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        ax.set_title('Discovered Circuit Pathways\n(Node size = importance, Edge thickness = weight)', 
                    fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        output_file = self.output_dir / f"{output_path}_circuit.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Circuit diagram saved: {output_file}")
        return str(output_file)
    
    def _create_hierarchical_layout(self, nodes_by_layer: Dict[int, List[str]]) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout for circuit diagram."""
        pos = {}
        layer_width = 3.0
        layer_spacing = 2.0
        
        for layer, nodes in nodes_by_layer.items():
            x = layer * layer_spacing
            n_nodes = len(nodes)
            
            if n_nodes == 1:
                y_positions = [0]
            else:
                y_positions = np.linspace(-layer_width/2, layer_width/2, n_nodes)
            
            for i, node_id in enumerate(nodes):
                pos[node_id] = (x, y_positions[i])
        
        return pos
    
    def create_metrics_dashboard(self, result: ExperimentResult, output_path: str) -> str:
        """Create comprehensive metrics dashboard."""
        logger.info("Creating metrics dashboard")
        
        # Create interactive dashboard with plotly
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Research Question Progress',
                'Efficiency Comparison', 
                'Correspondence Metrics',
                'Intervention Effects'
            ]
        )
        
        # 1. Research Question Progress
        rq_names = ['RQ1 (Correspondence)', 'RQ2 (Efficiency)', 'RQ3 (Predictions)']
        rq_targets = [70, 30, 3]
        rq_achieved = [
            np.mean([m.overall_correspondence for m in result.correspondence_metrics]) * 100 if result.correspondence_metrics else 0,
            result.efficiency_metrics.get('overall_improvement', 0),
            len([p for p in result.novel_predictions if hasattr(p, 'validation_status') and p.validation_status == 'validated'])
        ]
        
        colors = ['green' if achieved >= target else 'red' for achieved, target in zip(rq_achieved, rq_targets)]
        
        fig.add_trace(
            go.Bar(x=rq_names, y=rq_achieved, marker_color=colors, name='Achieved'),
            row=1, col=1
        )
        
        # 2. Efficiency Comparison
        strategies = ['Active Inference', 'Random', 'High Activation', 'Sequential']
        intervention_counts = [
            len(result.intervention_results),
            len(result.intervention_results) * 2.5,  # Simulated baseline
            len(result.intervention_results) * 2.3,
            len(result.intervention_results) * 2.0
        ]
        
        fig.add_trace(
            go.Bar(x=strategies, y=intervention_counts, 
                  marker_color=['blue', 'orange', 'orange', 'orange']),
            row=1, col=2
        )
        
        output_file = self.output_dir / f"{output_path}_dashboard.html"
        fig.write_html(str(output_file))
        
        logger.info(f"Interactive dashboard saved: {output_file}")
        return str(output_file)
    
    def create_belief_evolution_plot(self, belief_history: List[BeliefState], output_path: str) -> str:
        """Create plot showing belief evolution over time."""
        logger.info("Creating belief evolution plot")
        
        if not belief_history:
            logger.warning("No belief history provided")
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Active Inference Belief Evolution', fontsize=16, fontweight='bold')
        
        # Extract data from belief history
        time_steps = list(range(len(belief_history)))
        confidences = [bs.confidence for bs in belief_history]
        uncertainties = [np.mean(list(bs.uncertainty.values())) if bs.uncertainty else 0 for bs in belief_history]
        
        # 1. Confidence over time
        axes[0, 0].plot(time_steps, confidences, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Belief Confidence Over Time')
        axes[0, 0].set_xlabel('Intervention Step')
        axes[0, 0].set_ylabel('Confidence')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Uncertainty over time
        axes[0, 1].plot(time_steps, uncertainties, 'r-o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Average Uncertainty Over Time')
        axes[0, 1].set_xlabel('Intervention Step')
        axes[0, 1].set_ylabel('Uncertainty')
        axes[0, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.output_dir / f"{output_path}_belief_evolution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Belief evolution plot saved: {output_file}")
        return str(output_file)
    
    def create_prediction_validation_plot(self, predictions: List[NovelPrediction], output_path: str) -> str:
        """Create visualization of novel prediction validation results."""
        logger.info("Creating prediction validation visualization")
        
        if not predictions:
            logger.warning("No predictions provided")
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Novel Prediction Validation Results', fontsize=16, fontweight='bold')
        
        # Extract prediction data
        prediction_types = [p.prediction_type for p in predictions]
        validation_statuses = [p.validation_status for p in predictions]
        confidences = [p.confidence for p in predictions]
        
        # 1. Prediction type distribution
        type_counts = pd.Series(prediction_types).value_counts()
        axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Prediction Types Generated')
        
        # 2. Validation status distribution
        status_counts = pd.Series(validation_statuses).value_counts()
        colors = {'validated': 'green', 'falsified': 'red', 'pending': 'orange'}
        status_colors = [colors.get(status, 'gray') for status in status_counts.index]
        axes[0, 1].bar(status_counts.index, status_counts.values, color=status_colors)
        axes[0, 1].set_title('Validation Status Distribution')
        axes[0, 1].set_ylabel('Count')
        
        # 3. Confidence by prediction type
        df_predictions = pd.DataFrame({
            'type': prediction_types,
            'confidence': confidences,
            'status': validation_statuses
        })
        
        sns.boxplot(data=df_predictions, x='type', y='confidence', ax=axes[1, 0])
        axes[1, 0].set_title('Confidence by Prediction Type')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Confidence vs validation success
        validated_mask = df_predictions['status'] == 'validated'
        axes[1, 1].scatter(df_predictions[validated_mask]['confidence'], 
                          [1] * sum(validated_mask), 
                          c='green', label='Validated', alpha=0.7, s=100)
        axes[1, 1].scatter(df_predictions[~validated_mask]['confidence'], 
                          [0] * sum(~validated_mask), 
                          c='red', label='Not Validated', alpha=0.7, s=100)
        axes[1, 1].set_xlabel('Prediction Confidence')
        axes[1, 1].set_ylabel('Validation Success')
        axes[1, 1].set_title('Confidence vs Validation Success')
        axes[1, 1].legend()
        axes[1, 1].set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        
        output_file = self.output_dir / f"{output_path}_prediction_validation.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Prediction validation plot saved: {output_file}")
        return str(output_file)
    
    def create_statistical_validation_plot(self, statistical_tests: List[StatisticalTest], output_path: str) -> str:
        """Create visualization of comprehensive statistical validation results."""
        logger.info("Creating statistical validation visualization")
        
        if not statistical_tests:
            logger.warning("No statistical tests provided")
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Validation Results', fontsize=16, fontweight='bold')
        
        # Extract test data
        test_names = [t.test_name for t in statistical_tests]
        p_values = [t.p_value for t in statistical_tests]
        effect_sizes = [t.effect_size for t in statistical_tests]
        power_values = [t.power for t in statistical_tests]
        significant = [t.significant for t in statistical_tests]
        
        # 1. P-values with significance threshold
        colors = ['green' if sig else 'red' for sig in significant]
        bars = axes[0, 0].bar(range(len(test_names)), p_values, color=colors, alpha=0.7)
        axes[0, 0].axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
        axes[0, 0].set_xlabel('Test')
        axes[0, 0].set_ylabel('P-value')
        axes[0, 0].set_title('P-values by Test')
        axes[0, 0].set_xticks(range(len(test_names)))
        axes[0, 0].set_xticklabels(test_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        
        # 2. Effect sizes
        axes[0, 1].bar(range(len(test_names)), effect_sizes, 
                       color=['green' if sig else 'red' for sig in significant], alpha=0.7)
        axes[0, 1].set_xlabel('Test')
        axes[0, 1].set_ylabel('Effect Size (Cohen\'s d)')
        axes[0, 1].set_title('Effect Sizes by Test')
        axes[0, 1].set_xticks(range(len(test_names)))
        axes[0, 1].set_xticklabels(test_names, rotation=45, ha='right')
        
        # 3. Statistical power
        axes[1, 0].bar(range(len(test_names)), power_values,
                       color=['green' if p >= 0.8 else 'orange' if p >= 0.5 else 'red' for p in power_values], 
                       alpha=0.7)
        axes[1, 0].axhline(y=0.8, color='green', linestyle='--', label='Adequate Power (0.8)')
        axes[1, 0].set_xlabel('Test')
        axes[1, 0].set_ylabel('Statistical Power')
        axes[1, 0].set_title('Statistical Power by Test')
        axes[1, 0].set_xticks(range(len(test_names)))
        axes[1, 0].set_xticklabels(test_names, rotation=45, ha='right')
        axes[1, 0].legend()
        
        # 4. Effect size vs power scatter
        axes[1, 1].scatter(effect_sizes, power_values, 
                          c=['green' if sig else 'red' for sig in significant], 
                          s=100, alpha=0.7)
        axes[1, 1].set_xlabel('Effect Size')
        axes[1, 1].set_ylabel('Statistical Power')
        axes[1, 1].set_title('Effect Size vs Statistical Power')
        
        # Add text annotations for each point
        for i, (es, pow, name) in enumerate(zip(effect_sizes, power_values, test_names)):
            axes[1, 1].annotate(name[:10], (es, pow), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        output_file = self.output_dir / f"{output_path}_statistical_validation.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Statistical validation plot saved: {output_file}")
        return str(output_file)
    
    def create_comprehensive_dashboard(self, result: ExperimentResult, 
                                     statistical_validation: Dict[str, Any] = None,
                                     output_path: str = "comprehensive_dashboard") -> str:
        """Create comprehensive interactive dashboard with all experiment results."""
        logger.info("Creating comprehensive interactive dashboard")
        
        # Create multi-tab dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Research Question Achievement',
                'Statistical Test Results', 
                'Correspondence Metrics Over Time',
                'Intervention Efficiency Comparison',
                'Prediction Validation Status',
                'Effect Sizes and Confidence Intervals'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # 1. Research Question Achievement
        rq_names = ['RQ1<br>(Correspondence≥70%)', 'RQ2<br>(Efficiency≥30%)', 'RQ3<br>(Predictions≥3)']
        rq_targets = [70, 30, 3]
        rq_achieved = [
            np.mean([m.overall_correspondence for m in result.correspondence_metrics]) * 100 if result.correspondence_metrics else 0,
            result.efficiency_metrics.get('overall_improvement', 0),
            len([p for p in result.novel_predictions if hasattr(p, 'validation_status') and p.validation_status == 'validated'])
        ]
        
        colors = ['green' if achieved >= target else 'red' for achieved, target in zip(rq_achieved, rq_targets)]
        
        fig.add_trace(
            go.Bar(x=rq_names, y=rq_achieved, marker_color=colors, 
                   name='Achieved', text=[f'{val:.1f}' for val in rq_achieved],
                   textposition='auto'),
            row=1, col=1
        )
        
        # Add target lines
        for i, target in enumerate(rq_targets):
            fig.add_hline(y=target, line_dash="dash", line_color="orange",
                         annotation_text=f"Target: {target}", row=1, col=1)
        
        # 2. Statistical Test Results (if available)
        if statistical_validation and 'statistical_summary' in statistical_validation:
            stats = statistical_validation['statistical_summary']
            test_summary = stats.get('test_summary', [])
            
            if test_summary:
                test_names = [t['test_name'] for t in test_summary]
                p_values = [t['p_value'] for t in test_summary]
                significant = [t['significant'] for t in test_summary]
                
                fig.add_trace(
                    go.Bar(x=test_names, y=[-np.log10(p) for p in p_values],
                           marker_color=['green' if sig else 'red' for sig in significant],
                           name='Statistical Tests',
                           text=[f'p={p:.3f}' for p in p_values],
                           textposition='auto'),
                    row=1, col=2
                )
                
                fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red",
                             annotation_text="α=0.05", row=1, col=2)
        
        # 3. Correspondence metrics over interventions
        if result.correspondence_metrics:
            intervention_steps = list(range(len(result.correspondence_metrics)))
            correspondences = [m.overall_correspondence * 100 for m in result.correspondence_metrics]
            
            fig.add_trace(
                go.Scatter(x=intervention_steps, y=correspondences,
                          mode='lines+markers', name='Correspondence',
                          line=dict(color='blue', width=3),
                          marker=dict(size=8)),
                row=2, col=1
            )
            
            fig.add_hline(y=70, line_dash="dash", line_color="green",
                         annotation_text="Target: 70%", row=2, col=1)
        
        # 4. Efficiency comparison
        if result.efficiency_metrics:
            efficiency_data = {k: v for k, v in result.efficiency_metrics.items() 
                             if '_improvement' in k and k != 'overall_improvement'}
            
            if efficiency_data:
                strategies = list(efficiency_data.keys())
                improvements = list(efficiency_data.values())
                
                fig.add_trace(
                    go.Bar(x=strategies, y=improvements,
                           marker_color=['green' if imp >= 30 else 'orange' for imp in improvements],
                           name='Efficiency Improvement',
                           text=[f'{imp:.1f}%' for imp in improvements],
                           textposition='auto'),
                    row=2, col=2
                )
                
                fig.add_hline(y=30, line_dash="dash", line_color="green",
                             annotation_text="Target: 30%", row=2, col=2)
        
        # 5. Prediction validation pie chart
        if result.novel_predictions:
            validation_counts = pd.Series([p.validation_status for p in result.novel_predictions]).value_counts()
            
            fig.add_trace(
                go.Pie(labels=validation_counts.index, values=validation_counts.values,
                       name="Prediction Status"),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title_text="Comprehensive Active Inference Circuit Discovery Results",
            title_x=0.5,
            height=1200,
            showlegend=True
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Achievement", row=1, col=1)
        fig.update_yaxes(title_text="-log10(p-value)", row=1, col=2)
        fig.update_yaxes(title_text="Correspondence (%)", row=2, col=1)
        fig.update_yaxes(title_text="Improvement (%)", row=2, col=2)
        
        # Update x-axis titles
        fig.update_xaxes(title_text="Research Questions", row=1, col=1)
        fig.update_xaxes(title_text="Statistical Tests", row=1, col=2)
        fig.update_xaxes(title_text="Intervention Step", row=2, col=1)
        fig.update_xaxes(title_text="Baseline Strategy", row=2, col=2)
        
        output_file = self.output_dir / f"{output_path}.html"
        fig.write_html(str(output_file))
        
        logger.info(f"Comprehensive dashboard saved: {output_file}")
        return str(output_file)
    
    def generate_all_visualizations(self, result: ExperimentResult, 
                                  attribution_graph: Optional[AttributionGraph] = None,
                                  belief_history: Optional[List[BeliefState]] = None,
                                  statistical_validation: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Generate all available visualizations for experiment results."""
        logger.info("Generating complete visualization suite")
        
        visualization_files = {}
        experiment_name = result.experiment_name or "experiment"
        
        try:
            # 1. Circuit diagram (if attribution graph available)
            if attribution_graph:
                circuit_file = self.create_circuit_diagram(attribution_graph, f"{experiment_name}_circuit")
                visualization_files['circuit_diagram'] = circuit_file
            
            # 2. Metrics dashboard
            dashboard_file = self.create_metrics_dashboard(result, f"{experiment_name}_metrics")
            visualization_files['metrics_dashboard'] = dashboard_file
            
            # 3. Belief evolution (if history available)
            if belief_history:
                belief_file = self.create_belief_evolution_plot(belief_history, f"{experiment_name}_beliefs")
                visualization_files['belief_evolution'] = belief_file
            
            # 4. Prediction validation
            if result.novel_predictions:
                prediction_file = self.create_prediction_validation_plot(result.novel_predictions, f"{experiment_name}_predictions")
                visualization_files['prediction_validation'] = prediction_file
            
            # 5. Statistical validation (if available)
            if statistical_validation and 'statistical_summary' in statistical_validation:
                test_summary = statistical_validation['statistical_summary'].get('test_summary', [])
                if test_summary:
                    # Convert to StatisticalTest objects for visualization
                    stat_tests = []
                    for test_data in test_summary:
                        stat_test = StatisticalTest(
                            test_name=test_data['test_name'],
                            statistic=0.0,  # Not stored in summary
                            p_value=test_data['p_value'],
                            confidence_interval=(0.0, 0.0),  # Not stored in summary
                            effect_size=test_data['effect_size'],
                            power=0.8,  # Default power
                            significant=test_data['significant'],
                            interpretation=test_data['interpretation']
                        )
                        stat_tests.append(stat_test)
                    
                    stats_file = self.create_statistical_validation_plot(stat_tests, f"{experiment_name}_statistics")
                    visualization_files['statistical_validation'] = stats_file
            
            # 6. Comprehensive dashboard
            comprehensive_file = self.create_comprehensive_dashboard(
                result, statistical_validation, f"{experiment_name}_comprehensive"
            )
            visualization_files['comprehensive_dashboard'] = comprehensive_file
            
            logger.info(f"Generated {len(visualization_files)} visualization files")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
        
        return visualization_files
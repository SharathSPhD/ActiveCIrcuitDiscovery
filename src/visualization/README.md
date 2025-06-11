# Visualization System

This directory contains the comprehensive visualization system for the ActiveCircuitDiscovery project, creating publication-ready visualizations of circuit discovery results, Active Inference behavior, and statistical validation outcomes.

## Overview

The visualization system provides:
- **Circuit diagram visualization** with interactive and static formats
- **Statistical validation plots** with comprehensive test results
- **Active Inference behavior visualization** showing belief evolution
- **Research question dashboards** with progress tracking
- **Novel prediction validation plots** with detailed analysis
- **Publication-ready outputs** in multiple formats (PNG, PDF, HTML)

## Main Files

### `visualizer.py`
The core `CircuitVisualizer` class implementing the `IVisualizationGenerator` interface:

**Key Features:**
- **Multi-format output** supporting static and interactive visualizations
- **Circuit-tracer integration** for proper circuit diagram rendering
- **Statistical visualization** with comprehensive test result displays
- **Interactive dashboards** using Plotly for exploration
- **Publication-ready styling** with professional formatting
- **Comprehensive visualization suite** generating all experiment visualizations

**Core Methods:**
- `create_circuit_diagram()` - Generate circuit pathway visualizations
- `create_metrics_dashboard()` - Create research question progress dashboards
- `create_belief_evolution_plot()` - Visualize Active Inference belief changes
- `create_prediction_validation_plot()` - Show novel prediction validation results
- `create_statistical_validation_plot()` - Display statistical test outcomes
- `create_comprehensive_dashboard()` - Generate complete interactive dashboard

## Visualization Types

### 1. Circuit Diagrams

Creates visual representations of discovered neural circuits:

```python
def create_circuit_diagram(self, graph: AttributionGraph, output_path: str):
    """Create circuit diagram using NetworkX or CircuitsVis."""
    
    # Create hierarchical layout based on layers
    pos = self._create_hierarchical_layout(nodes_by_layer)
    
    # Draw nodes with size based on importance
    for layer, nodes in nodes_by_layer.items():
        node_sizes = [graph.get_node_by_id(node_id).importance * 1000 
                     for node_id in nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                             node_size=node_sizes,
                             node_color=f'C{layer % 10}',
                             alpha=0.8)
    
    # Draw edges with thickness based on weight
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, 
                         alpha=0.6, edge_color='gray')
```

**Features:**
- **Hierarchical layout** organizing nodes by transformer layer
- **Size-coded nodes** representing feature importance
- **Weight-coded edges** showing connection strength
- **Color-coded layers** for easy layer identification
- **Interactive elements** when using CircuitsVis

### 2. Statistical Validation Plots

Comprehensive visualization of statistical test results:

```python
def create_statistical_validation_plot(self, statistical_tests: List[StatisticalTest]):
    """Create comprehensive statistical validation visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. P-values with significance threshold
    colors = ['green' if sig else 'red' for sig in significant]
    axes[0, 0].bar(range(len(test_names)), p_values, color=colors)
    axes[0, 0].axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
    axes[0, 0].set_yscale('log')
    
    # 2. Effect sizes with Cohen's d interpretation
    axes[0, 1].bar(range(len(test_names)), effect_sizes, color=colors)
    
    # 3. Statistical power analysis
    power_colors = ['green' if p >= 0.8 else 'orange' if p >= 0.5 else 'red' 
                   for p in power_values]
    axes[1, 0].bar(range(len(test_names)), power_values, color=power_colors)
    axes[1, 0].axhline(y=0.8, color='green', linestyle='--', label='Adequate Power')
    
    # 4. Effect size vs power relationship
    axes[1, 1].scatter(effect_sizes, power_values, c=colors, s=100)
```

**Components:**
- **P-value significance testing** with alpha threshold lines
- **Effect size analysis** with Cohen's d interpretation
- **Statistical power assessment** with adequacy indicators  
- **Relationship analysis** showing effect size vs power correlations

### 3. Active Inference Behavior Visualization

Shows how Active Inference beliefs evolve during circuit discovery:

```python
def create_belief_evolution_plot(self, belief_history: List[BeliefState]):
    """Visualize Active Inference belief evolution over time."""
    
    # Extract evolution data
    time_steps = list(range(len(belief_history)))
    confidences = [bs.confidence for bs in belief_history]
    uncertainties = [np.mean(list(bs.uncertainty.values())) for bs in belief_history]
    
    # Plot confidence evolution
    axes[0, 0].plot(time_steps, confidences, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_title('Belief Confidence Over Time')
    
    # Plot uncertainty reduction
    axes[0, 1].plot(time_steps, uncertainties, 'r-o', linewidth=2, markersize=6)
    axes[0, 1].set_title('Average Uncertainty Over Time')
```

**Metrics Tracked:**
- **Belief confidence** evolution over interventions
- **Uncertainty reduction** as evidence accumulates
- **Feature importance** changes with learning
- **Connection strength** development over time

### 4. Research Question Dashboards

Interactive dashboards showing progress toward research question targets:

```python
def create_comprehensive_dashboard(self, result: ExperimentResult):
    """Create comprehensive interactive dashboard with all results."""
    
    # Research question achievement
    rq_names = ['RQ1<br>(Correspondence≥70%)', 'RQ2<br>(Efficiency≥30%)', 'RQ3<br>(Predictions≥3)']
    rq_achieved = [correspondence_pct, efficiency_pct, validated_predictions]
    
    colors = ['green' if achieved >= target else 'red' 
             for achieved, target in zip(rq_achieved, rq_targets)]
    
    fig.add_trace(go.Bar(x=rq_names, y=rq_achieved, marker_color=colors,
                        text=[f'{val:.1f}' for val in rq_achieved]))
    
    # Add target threshold lines
    for i, target in enumerate(rq_targets):
        fig.add_hline(y=target, line_dash="dash", line_color="orange")
```

**Dashboard Components:**
- **Research question progress** with target thresholds
- **Statistical test summaries** with significance indicators
- **Correspondence metrics** over intervention timeline
- **Efficiency comparisons** across baseline methods
- **Prediction validation status** with detailed breakdowns

### 5. Novel Prediction Validation

Detailed analysis of novel prediction generation and validation:

```python
def create_prediction_validation_plot(self, predictions: List[NovelPrediction]):
    """Visualize novel prediction validation results."""
    
    # Prediction type distribution
    type_counts = pd.Series(prediction_types).value_counts()
    axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
    
    # Validation status distribution
    status_counts = pd.Series(validation_statuses).value_counts()
    colors = {'validated': 'green', 'falsified': 'red', 'pending': 'orange'}
    status_colors = [colors.get(status, 'gray') for status in status_counts.index]
    axes[0, 1].bar(status_counts.index, status_counts.values, color=status_colors)
    
    # Confidence vs validation success
    validated_mask = df_predictions['status'] == 'validated'
    axes[1, 1].scatter(df_predictions[validated_mask]['confidence'], 
                      [1] * sum(validated_mask), c='green', label='Validated')
    axes[1, 1].scatter(df_predictions[~validated_mask]['confidence'], 
                      [0] * sum(~validated_mask), c='red', label='Not Validated')
```

**Analysis Components:**
- **Prediction type distribution** showing generation balance
- **Validation status breakdown** with success rates
- **Confidence analysis** by prediction type
- **Success correlation** between confidence and validation outcome

## Usage Examples

### Basic Visualization Creation

```python
from visualization.visualizer import CircuitVisualizer
from core.data_structures import AttributionGraph, ExperimentResult

# Initialize visualizer
visualizer = CircuitVisualizer("output/visualizations")

# Create circuit diagram
attribution_graph = tracer.build_attribution_graph(text)
circuit_file = visualizer.create_circuit_diagram(
    attribution_graph, "golden_gate_circuit"
)

print(f"Circuit diagram saved: {circuit_file}")
```

### Comprehensive Visualization Suite

```python
# Generate all available visualizations
visualization_files = visualizer.generate_all_visualizations(
    result=experiment_result,
    attribution_graph=attribution_graph,
    belief_history=agent.belief_history,
    statistical_validation=statistical_validation_results
)

print("Generated visualizations:")
for viz_type, file_path in visualization_files.items():
    print(f"  {viz_type}: {file_path}")

# Output:
# circuit_diagram: /output/visualizations/experiment_circuit.png
# metrics_dashboard: /output/visualizations/experiment_metrics.html  
# belief_evolution: /output/visualizations/experiment_beliefs.png
# prediction_validation: /output/visualizations/experiment_predictions.png
# statistical_validation: /output/visualizations/experiment_statistics.png
# comprehensive_dashboard: /output/visualizations/experiment_comprehensive.html
```

### Interactive Dashboard Creation

```python
# Create comprehensive interactive dashboard
dashboard_file = visualizer.create_comprehensive_dashboard(
    experiment_result, 
    statistical_validation=statistical_results,
    output_path="comprehensive_analysis"
)

print(f"Interactive dashboard: {dashboard_file}")
# Opens in browser for exploration
```

### Custom Styling and Formats

```python
# Configure visualization settings
visualizer_config = VisualizationConfig(
    enhanced_plots=True,
    interactive_dashboards=True,
    publication_ready=True,
    save_formats=["png", "pdf", "html"]
)

# Create publication-ready circuit diagram
circuit_diagram = visualizer.create_circuit_diagram(
    attribution_graph, 
    "publication_circuit"
)

# Automatically saves in multiple formats:
# - publication_circuit.png (high-resolution)
# - publication_circuit.pdf (vector format)
# - publication_circuit.html (interactive)
```

### Statistical Validation Visualization

```python
from core.statistical_validation import perform_comprehensive_validation

# Perform statistical validation
statistical_validation = perform_comprehensive_validation(experiment_result)

# Extract statistical tests for visualization
if 'statistical_summary' in statistical_validation:
    test_summary = statistical_validation['statistical_summary'].get('test_summary', [])
    
    # Convert to StatisticalTest objects
    stat_tests = []
    for test_data in test_summary:
        stat_test = StatisticalTest(
            test_name=test_data['test_name'],
            p_value=test_data['p_value'],
            effect_size=test_data['effect_size'],
            significant=test_data['significant'],
            # ... other fields
        )
        stat_tests.append(stat_test)
    
    # Create statistical visualization
    stats_file = visualizer.create_statistical_validation_plot(
        stat_tests, "experiment_statistics"
    )
    
    print(f"Statistical validation plot: {stats_file}")
```

## Advanced Features

### Circuit-tracer Integration

When CircuitsVis is available, the visualizer uses advanced circuit visualization:

```python
def _create_circuitsvis_diagram(self, graph: AttributionGraph, output_path: str):
    """Create advanced circuit diagram using CircuitsVis."""
    
    # Convert to CircuitsVis format
    nodes = []
    for node in graph.nodes:
        nodes.append({
            'id': node.node_id,
            'layer': node.layer,
            'importance': node.importance,
            'feature_id': node.feature_id
        })
    
    edges = []
    for edge in graph.edges:
        edges.append({
            'source': edge.source_id,
            'target': edge.target_id,
            'weight': edge.weight,
            'confidence': edge.confidence
        })
    
    # Create interactive circuit visualization
    circuit_vis = cv.attention_patterns(
        attention=attention_data,
        tokens=tokens,
        head_names=head_names
    )
    
    return circuit_vis
```

### Publication-Ready Styling

Automatic styling for publication-ready figures:

```python
# Publication style configuration
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,  # High resolution
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Professional color schemes
sns.set_palette("husl")  # Visually distinct colors
```

### Multi-Format Output

Automatic generation in multiple formats for different use cases:

```python
def _save_multi_format(self, fig, output_path: str, formats: List[str]):
    """Save figure in multiple formats."""
    for fmt in formats:
        if fmt == 'png':
            fig.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
        elif fmt == 'pdf':
            fig.savefig(f"{output_path}.pdf", bbox_inches='tight')
        elif fmt == 'html' and hasattr(fig, 'write_html'):
            fig.write_html(f"{output_path}.html")
        elif fmt == 'svg':
            fig.savefig(f"{output_path}.svg", bbox_inches='tight')
```

## Configuration Options

### Visualization Configuration
```yaml
visualization:
  enhanced_plots: true                    # Use advanced plotting features
  interactive_dashboards: true            # Generate interactive dashboards
  publication_ready: true                 # Apply publication styling
  save_formats: ["png", "pdf", "html"]    # Output formats
```

### Styling Options
- **Color schemes**: Professional color palettes for different visualization types
- **Font sizes**: Scalable typography for different output sizes
- **Resolution**: High DPI output for publication quality
- **Layout**: Automatic layout optimization for different aspect ratios

## Integration with Other Components

### Experiment Runner Integration
```python
# Automatic visualization generation
if self.config.experiment.generate_visualizations:
    visualizer = CircuitVisualizer(self.output_dir / "visualizations")
    
    # Generate comprehensive visualization suite
    visualization_files = visualizer.generate_all_visualizations(
        result, attribution_graph, belief_history, statistical_validation
    )
    
    # Add to experiment metadata
    result.metadata['visualizations'] = visualization_files
```

### Statistical Validation Integration
```python
# Visualize statistical validation results
from core.statistical_validation import StatisticalValidator

validator = StatisticalValidator()
validation_results = validator.validate_correspondence_significance(
    correspondence_metrics, target_threshold=70.0
)

# Convert to visualization format
statistical_tests = [
    StatisticalTest(
        test_name="Correspondence Validation",
        p_value=validation_results['p_value'],
        effect_size=validation_results['effect_size'],
        significant=validation_results['significant']
    )
]

stats_plot = visualizer.create_statistical_validation_plot(
    statistical_tests, "correspondence_validation"
)
```

### Active Inference Integration
```python
# Visualize belief evolution from AI agent
belief_history = agent.belief_history
correspondence_history = agent.correspondence_history

belief_plot = visualizer.create_belief_evolution_plot(
    belief_history, "ai_belief_evolution"
)
```

## Performance Optimization

### Lazy Loading
- Visualizations generated only when requested
- Large datasets processed in chunks
- Memory-efficient plot generation

### Caching
- Plot style caching for consistent appearance
- Data preprocessing caching for repeated visualizations
- Template caching for dashboard generation

### Parallel Generation
- Multiple visualizations generated concurrently
- Asynchronous plot saving
- Background processing for large datasets

## Dependencies

### Required
- `matplotlib` - Static plot generation and styling
- `seaborn` - Enhanced statistical plotting
- `plotly` - Interactive visualization and dashboards
- `networkx` - Graph layout and network visualization
- `pandas` - Data manipulation for plotting
- `numpy` - Numerical operations

### Optional
- `circuitsvis` - Advanced circuit visualization (when available)
- `kaleido` - Static image export for Plotly (when needed)

### Project Dependencies
- `core.data_structures` - All visualization data structures
- `core.statistical_validation` - Statistical test result structures

## Error Handling

### Graceful Degradation
- Falls back to basic plots when advanced libraries unavailable
- Continues with available formats when some export formats fail
- Provides meaningful error messages for missing dependencies

### Data Validation
- Validates input data structures before visualization
- Handles empty or malformed data gracefully
- Provides informative warnings for data quality issues

## Future Enhancements

### Planned Features
- **3D circuit visualization** for complex multi-layer circuits
- **Animated visualizations** showing temporal evolution
- **Interactive exploration tools** for circuit analysis
- **Custom styling themes** for different publication venues

### Research Extensions
- **Comparative visualizations** across different models
- **Real-time monitoring dashboards** for live experiments
- **Collaborative visualization** with shared annotations
- **AR/VR circuit exploration** for immersive analysis

The visualization system provides comprehensive, publication-ready visualizations that make the complex results of Active Inference-guided circuit discovery accessible to both researchers and broader audiences, supporting scientific communication and insight discovery.
# Interactive Web-based Visualization for Circuit Discovery
# Following circuit-tracer patterns for interactive graph exploration

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np

# Web framework imports
try:
    import dash
    from dash import dcc, html, Input, Output, State, callback_context
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import networkx as nx
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

# Project imports
try:
    from core.data_structures import AttributionGraph, GraphNode, GraphEdge
    from core.interfaces import IVisualizationBackend
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from core.data_structures import AttributionGraph, GraphNode, GraphEdge
    from core.interfaces import IVisualizationBackend

logger = logging.getLogger(__name__)

class InteractiveWebVisualizer(IVisualizationBackend):
    """Interactive web-based visualization using Plotly Dash."""
    
    def __init__(self, config=None):
        """Initialize web visualizer."""
        if not DASH_AVAILABLE:
            raise ImportError("Dash and Plotly required for web visualization")
        
        self.config = config
        self.app = None
        self.current_graph = None
        self.annotations = {}
        self.selected_nodes = set()
        self.layout_cache = {}
        
        logger.info("InteractiveWebVisualizer initialized")
    
    def create_interactive_graph(self, graph: AttributionGraph, 
                               output_path: Optional[str] = None) -> str:
        """Create interactive web interface for graph exploration."""
        self.current_graph = graph
        
        # Create Dash app with custom styling
        from visualization.web_assets import get_external_stylesheets, inject_custom_css
        
        self.app = dash.Dash(__name__, external_stylesheets=get_external_stylesheets())
        inject_custom_css(self.app)
        
        # Setup layout
        self.app.layout = self._create_layout()
        
        # Setup callbacks
        self._setup_callbacks()
        
        # Save standalone HTML if requested
        if output_path:
            self._save_standalone_html(output_path)
            return output_path
        
        return "Interactive graph created. Use run_server() to start."
    
    def run_server(self, host='127.0.0.1', port=8050, debug=False):
        """Run the interactive visualization server."""
        if self.app is None:
            raise ValueError("No graph loaded. Call create_interactive_graph first.")
        
        logger.info(f"Starting web visualizer at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)
    
    def _create_layout(self) -> html.Div:
        """Create the main dashboard layout."""
        return html.Div([
            # Header
            html.Div([
                html.H1("ActiveCircuitDiscovery - Interactive Graph Explorer", 
                       className="header-title"),
                html.P(f"Input: {self.current_graph.input_text}", 
                      className="header-subtitle"),
                html.P(f"Target: {self.current_graph.target_output} | "
                      f"Confidence: {self.current_graph.confidence:.2f}",
                      className="header-info"),
            ], className="header"),
            
            # Controls panel
            html.Div([
                html.Div([
                    html.Label("Node Size:"),
                    dcc.Slider(
                        id='node-size-slider',
                        min=5, max=50, value=20, step=5,
                        marks={i: str(i) for i in range(5, 51, 10)}
                    ),
                ], className="control-group"),
                
                html.Div([
                    html.Label("Edge Threshold:"),
                    dcc.Slider(
                        id='edge-threshold-slider',
                        min=0, max=1, value=0.1, step=0.1,
                        marks={i/10: f"{i/10:.1f}" for i in range(0, 11, 2)}
                    ),
                ], className="control-group"),
                
                html.Div([
                    html.Label("Layout:"),
                    dcc.Dropdown(
                        id='layout-dropdown',
                        options=[
                            {'label': 'Force-directed', 'value': 'spring'},
                            {'label': 'Hierarchical', 'value': 'hierarchical'},
                            {'label': 'Circular', 'value': 'circular'},
                            {'label': 'Layer-based', 'value': 'shell'}
                        ],
                        value='spring'
                    ),
                ], className="control-group"),
                
                html.Div([
                    html.Button("Reset View", id="reset-button", className="control-button"),
                    html.Button("Export Graph", id="export-button", className="control-button"),
                    html.Button("Save Annotations", id="save-annotations-button", className="control-button"),
                ], className="button-group"),
                
            ], className="controls-panel"),
            
            # Main visualization area
            html.Div([
                dcc.Graph(
                    id='main-graph',
                    figure=self._create_graph_figure(),
                    style={'height': '70vh'}
                ),
            ], className="graph-container"),
            
            # Info panels
            html.Div([
                # Selected node info
                html.Div([
                    html.H3("Selected Node"),
                    html.Div(id="node-info", children="Click a node to see details"),
                ], className="info-panel"),
                
                # Graph statistics
                html.Div([
                    html.H3("Graph Statistics"),
                    html.Div(id="graph-stats", children=self._get_graph_stats()),
                ], className="info-panel"),
                
                # Annotation panel
                html.Div([
                    html.H3("Annotations"),
                    dcc.Textarea(
                        id='annotation-input',
                        placeholder='Add annotation for selected node...',
                        style={'width': '100%', 'height': 100}
                    ),
                    html.Button("Add Annotation", id="add-annotation-button"),
                    html.Div(id="annotation-list"),
                ], className="info-panel"),
                
            ], className="info-container"),
            
            # Hidden components for data storage
            dcc.Store(id='graph-data'),
            dcc.Store(id='selected-node-data'),
            html.Div(id='export-download'),
            
        ], className="main-container")
    
    def _create_graph_figure(self, node_size=20, edge_threshold=0.1, 
                           layout_type='spring') -> go.Figure:
        """Create the main graph visualization."""
        # Create NetworkX graph for layout computation
        G = nx.DiGraph()
        
        # Add nodes
        node_positions = {}
        node_texts = []
        node_colors = []
        node_sizes = []
        
        for node in self.current_graph.nodes:
            G.add_node(node.node_id, **{
                'layer': node.layer,
                'importance': node.importance,
                'description': node.description
            })
            
            # Color by layer
            node_colors.append(node.layer)
            # Size by importance
            node_sizes.append(max(5, node.importance * node_size))
            # Text labels
            node_texts.append(f"L{node.layer}<br>F{node.feature_id}<br>{node.importance:.2f}")
        
        # Add edges with threshold filtering
        edge_x = []
        edge_y = []
        edge_texts = []
        
        for edge in self.current_graph.edges:
            if abs(edge.weight) >= edge_threshold:
                G.add_edge(edge.source_id, edge.target_id, weight=edge.weight)
        
        # Compute layout
        pos = self._compute_layout(G, layout_type)
        
        # Extract node positions
        node_x = []
        node_y = []
        for node in self.current_graph.nodes:
            if node.node_id in pos:
                x, y = pos[node.node_id]
                node_x.append(x)
                node_y.append(y)
            else:
                node_x.append(0)
                node_y.append(0)
        
        # Extract edge positions
        for edge in self.current_graph.edges:
            if (abs(edge.weight) >= edge_threshold and 
                edge.source_id in pos and edge.target_id in pos):
                
                x0, y0 = pos[edge.source_id]
                x1, y1 = pos[edge.target_id]
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='rgba(50,50,50,0.5)'),
            hoverinfo='none',
            showlegend=False,
            name='Edges'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Layer"),
                line=dict(width=2, color='white')
            ),
            text=node_texts,
            textposition="middle center",
            textfont=dict(size=8, color='white'),
            hovertemplate='<b>%{text}</b><br>Layer: %{marker.color}<extra></extra>',
            showlegend=False,
            name='Nodes'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Circuit Attribution Graph - {self.current_graph.input_text}",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text=f"Nodes: {len(self.current_graph.nodes)} | "
                         f"Edges: {len([e for e in self.current_graph.edges if abs(e.weight) >= edge_threshold])} | "
                         f"Confidence: {self.current_graph.confidence:.2f}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
    
    def _compute_layout(self, G: nx.DiGraph, layout_type: str) -> Dict[str, Tuple[float, float]]:
        """Compute node positions using NetworkX layouts."""
        cache_key = f"{layout_type}_{len(G.nodes)}_{len(G.edges)}"
        
        if cache_key in self.layout_cache:
            return self.layout_cache[cache_key]
        
        try:
            if layout_type == 'spring':
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif layout_type == 'hierarchical':
                # Layer-based hierarchical layout
                layers = {}
                for node, data in G.nodes(data=True):
                    layer = data.get('layer', 0)
                    if layer not in layers:
                        layers[layer] = []
                    layers[layer].append(node)
                
                pos = {}
                for layer_idx, layer in enumerate(sorted(layers.keys())):
                    layer_nodes = layers[layer]
                    for node_idx, node in enumerate(layer_nodes):
                        x = layer_idx
                        y = node_idx - len(layer_nodes) / 2
                        pos[node] = (x, y)
            elif layout_type == 'circular':
                pos = nx.circular_layout(G)
            elif layout_type == 'shell':
                # Shell layout by layer
                layers = {}
                for node, data in G.nodes(data=True):
                    layer = data.get('layer', 0)
                    if layer not in layers:
                        layers[layer] = []
                    layers[layer].append(node)
                
                shells = [layers[layer] for layer in sorted(layers.keys())]
                pos = nx.shell_layout(G, nlist=shells)
            else:
                pos = nx.spring_layout(G)
            
            self.layout_cache[cache_key] = pos
            return pos
            
        except Exception as e:
            logger.warning(f"Layout computation failed: {e}")
            # Fallback to random layout
            pos = {node: (np.random.random(), np.random.random()) for node in G.nodes()}
            return pos
    
    def _setup_callbacks(self):
        """Setup interactive callbacks for the dashboard."""
        
        @self.app.callback(
            Output('main-graph', 'figure'),
            [Input('node-size-slider', 'value'),
             Input('edge-threshold-slider', 'value'),
             Input('layout-dropdown', 'value'),
             Input('reset-button', 'n_clicks')]
        )
        def update_graph(node_size, edge_threshold, layout_type, reset_clicks):
            return self._create_graph_figure(node_size, edge_threshold, layout_type)
        
        @self.app.callback(
            Output('node-info', 'children'),
            [Input('main-graph', 'clickData')]
        )
        def update_node_info(click_data):
            if click_data is None:
                return "Click a node to see details"
            
            try:
                point_index = click_data['points'][0]['pointIndex']
                if point_index < len(self.current_graph.nodes):
                    node = self.current_graph.nodes[point_index]
                    return html.Div([
                        html.P(f"Node ID: {node.node_id}"),
                        html.P(f"Layer: {node.layer}"),
                        html.P(f"Feature ID: {node.feature_id}"),
                        html.P(f"Importance: {node.importance:.3f}"),
                        html.P(f"Description: {node.description}"),
                    ])
            except Exception as e:
                logger.warning(f"Node info update failed: {e}")
            
            return "Error loading node information"
        
        @self.app.callback(
            Output('export-download', 'children'),
            [Input('export-button', 'n_clicks')]
        )
        def export_graph(n_clicks):
            if n_clicks and n_clicks > 0:
                try:
                    export_data = self.current_graph.to_json()
                    filename = f"circuit_graph_{self.current_graph.input_text[:20]}.json"
                    
                    return html.A(
                        'Download Graph JSON',
                        href=f"data:application/json;charset=utf-8,{export_data}",
                        download=filename,
                        target="_blank"
                    )
                except Exception as e:
                    logger.warning(f"Graph export failed: {e}")
                    return html.P("Export failed")
            
            return ""
    
    def _get_graph_stats(self) -> html.Div:
        """Get graph statistics for display."""
        if not self.current_graph:
            return html.P("No graph loaded")
        
        total_nodes = len(self.current_graph.nodes)
        total_edges = len(self.current_graph.edges)
        
        # Layer distribution
        layer_counts = {}
        for node in self.current_graph.nodes:
            layer = node.layer
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        
        # Edge weight statistics
        edge_weights = [abs(edge.weight) for edge in self.current_graph.edges]
        avg_weight = np.mean(edge_weights) if edge_weights else 0
        max_weight = np.max(edge_weights) if edge_weights else 0
        
        return html.Div([
            html.P(f"Total Nodes: {total_nodes}"),
            html.P(f"Total Edges: {total_edges}"),
            html.P(f"Layers: {len(layer_counts)}"),
            html.P(f"Avg Edge Weight: {avg_weight:.3f}"),
            html.P(f"Max Edge Weight: {max_weight:.3f}"),
            html.P(f"Graph Confidence: {self.current_graph.confidence:.3f}"),
        ])
    
    def _save_standalone_html(self, output_path: str):
        """Save interactive visualization as standalone HTML."""
        try:
            # Create a simplified standalone version
            fig = self._create_graph_figure()
            fig.write_html(output_path, include_plotlyjs=True)
            logger.info(f"Standalone HTML saved to {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save standalone HTML: {e}")

class VisualizationFactory:
    """Factory for creating visualization backends."""
    
    @staticmethod
    def create_visualizer(backend_type: str, config=None) -> IVisualizationBackend:
        """Create visualization backend by type."""
        if backend_type == 'interactive' or backend_type == 'web':
            return InteractiveWebVisualizer(config)
        elif backend_type == 'static':
            # Import static visualizer
            from visualization.visualizer import CircuitVisualizer
            return CircuitVisualizer(config)
        else:
            raise ValueError(f"Unknown visualization backend: {backend_type}")
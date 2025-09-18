#!/usr/bin/env python3
"""
Circuit-Tracer JSON Server
Serves the circuit visualizations from JSON files
"""

import http.server
import socketserver
import json
from pathlib import Path
import threading
import time

class CircuitTracerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_html_page()
        elif self.path.startswith('/api/graph/'):
            case_id = self.path.split('/')[-1]
            self.send_graph_data(case_id)
        else:
            super().do_GET()

    def send_html_page(self):
        """Send the main circuit-tracer visualization page"""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Circuit-Tracer REFACT-4 Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; padding: 20px; background: white; border-radius: 8px; }
        .case-selector { margin: 20px 0; text-align: center; }
        .case-button { margin: 0 10px; padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .case-button:hover { background: #45a049; }
        .case-button.active { background: #2196F3; }
        .visualization { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; }
        .node { fill: #69b3a2; stroke: #333; stroke-width: 2px; }
        .node.selected { fill: #ff6b6b; }
        .link { stroke: #999; stroke-opacity: 0.6; stroke-width: 2px; }
        .label { font-size: 12px; fill: #333; text-anchor: middle; }
        .metadata { background: #e8f4f8; padding: 15px; border-radius: 4px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Circuit-Tracer REFACT-4 Results</h1>
            <p>Enhanced Active Inference vs Baseline Methods - Neural Circuit Discovery</p>
        </div>

        <div class="case-selector">
            <button class="case-button active" onclick="loadCase('1')">Case 1: Golden Gate Bridge</button>
            <button class="case-button" onclick="loadCase('2')">Case 2: Eiffel Tower</button>
            <button class="case-button" onclick="loadCase('3')">Case 3: Big Ben</button>
        </div>

        <div id="metadata" class="metadata"></div>
        <div id="visualization" class="visualization"></div>
    </div>

    <script>
        let currentCase = '1';

        function loadCase(caseId) {
            currentCase = caseId;

            // Update button states
            document.querySelectorAll('.case-button').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            // Load and display the case data
            fetch(`/api/graph/${caseId}`)
                .then(response => response.json())
                .then(data => {
                    displayMetadata(data.metadata);
                    visualizeCircuit(data.graph);
                })
                .catch(error => {
                    console.error('Error loading case:', error);
                    document.getElementById('metadata').innerHTML = '<p>Error loading case data</p>';
                });
        }

        function displayMetadata(metadata) {
            const metadataDiv = document.getElementById('metadata');
            metadataDiv.innerHTML = `
                <h3>📊 Test Case ${metadata.case_id}</h3>
                <p><strong>Input:</strong> "${metadata.input_text}"</p>
                <p><strong>Features Discovered:</strong> ${metadata.features_discovered}</p>
                <p><strong>Active Features:</strong> ${metadata.active_features_count}</p>
                <p><strong>Selected Features:</strong> ${metadata.selected_features_count}</p>
            `;
        }

        function visualizeCircuit(graph) {
            // Clear previous visualization
            d3.select('#visualization').selectAll('*').remove();

            if (!graph.nodes || graph.nodes.length === 0) {
                d3.select('#visualization').append('p').text('No circuit data available for this case.');
                return;
            }

            const width = 800;
            const height = 600;

            const svg = d3.select('#visualization')
                .append('svg')
                .attr('width', width)
                .attr('height', height);

            // Create force simulation
            const simulation = d3.forceSimulation(graph.nodes)
                .force('link', d3.forceLink(graph.edges).id(d => d.id).distance(100))
                .force('charge', d3.forceManyBody().strength(-300))
                .force('center', d3.forceCenter(width / 2, height / 2));

            // Add edges
            const link = svg.append('g')
                .selectAll('line')
                .data(graph.edges)
                .join('line')
                .attr('class', 'link')
                .attr('stroke-width', d => Math.sqrt(d.weight || 1) * 2);

            // Add nodes
            const node = svg.append('g')
                .selectAll('circle')
                .data(graph.nodes)
                .join('circle')
                .attr('class', d => d.selected ? 'node selected' : 'node')
                .attr('r', d => Math.sqrt(d.activation) * 10 + 5)
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));

            // Add labels
            const label = svg.append('g')
                .selectAll('text')
                .data(graph.nodes)
                .join('text')
                .attr('class', 'label')
                .text(d => d.id)
                .attr('dy', -15);

            // Add tooltips
            node.append('title')
                .text(d => `${d.id}\\nActivation: ${d.activation.toFixed(3)}\\nSelected: ${d.selected}`);

            simulation.on('tick', () => {
                link.attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);

                node.attr('cx', d => d.x)
                    .attr('cy', d => d.y);

                label.attr('x', d => d.x)
                     .attr('y', d => d.y);
            });

            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }

            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }

            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
        }

        // Load initial case
        loadCase('1');
    </script>
</body>
</html>
        """

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())

    def send_graph_data(self, case_id):
        """Send graph data for a specific case"""
        try:
            json_file = Path(f"visualizations/circuit_tracer_native/json_graphs/case_{case_id}.json")
            if json_file.exists():
                with open(json_file) as f:
                    data = json.load(f)

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            else:
                self.send_error(404, f"Case {case_id} not found")
        except Exception as e:
            self.send_error(500, f"Error loading case {case_id}: {str(e)}")

def start_server(port=8080):
    """Start the circuit-tracer server"""
    print(f"🚀 Starting Circuit-Tracer Server on port {port}...")

    with socketserver.TCPServer(("", port), CircuitTracerHandler) as httpd:
        print(f"🌐 Server running at http://127.0.0.1:{port}")
        print("📊 Serving REFACT-4 circuit discovery visualizations")
        print("🔗 Available endpoints:")
        print(f"   - http://127.0.0.1:{port}/ (main visualization)")
        print(f"   - http://127.0.0.1:{port}/api/graph/1 (Case 1 data)")
        print(f"   - http://127.0.0.1:{port}/api/graph/2 (Case 2 data)")
        print(f"   - http://127.0.0.1:{port}/api/graph/3 (Case 3 data)")
        print("\n✨ Ready to visualize Enhanced Active Inference results!")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")

if __name__ == "__main__":
    start_server()
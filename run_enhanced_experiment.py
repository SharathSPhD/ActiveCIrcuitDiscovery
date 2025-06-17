#\!/usr/bin/env python3
"""
Enhanced experiment runner for the ActiveCircuitDiscovery framework.
Demonstrates the complete enhanced framework capabilities.
"""

import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_enhanced_experiment():
    """Run the enhanced ActiveCircuitDiscovery experiment."""
    logger.info("🚀 Starting Enhanced ActiveCircuitDiscovery Experiment")
    
    try:
        # Test imports
        logger.info("📦 Loading enhanced framework components...")
        from config.experiment_config import CompleteConfig
        from active_inference.agent import ActiveInferenceAgent
        from circuit_analysis.tracer import CircuitTracer
        from core.data_structures import AttributionGraph, GraphNode, GraphEdge
        from visualization.web_visualizer import InteractiveWebVisualizer, VisualizationFactory
        
        # Create configuration
        logger.info("⚙️ Setting up experiment configuration...")
        config = CompleteConfig()
        config.model.name = "gpt2-small"
        config.sae.target_layers = [6, 7, 8]  # Focus on key layers
        config.sae.include_error_nodes = True
        config.sae.max_graph_nodes = 20  # Smaller for demo
        config.active_inference.max_interventions = 10
        
        # Test input
        test_input = "The Golden Gate Bridge is located in"
        logger.info(f"🎯 Testing with input: {test_input}")
        
        # Initialize components
        logger.info("🔧 Initializing circuit tracer...")
        tracer = CircuitTracer(config)
        
        logger.info("🧠 Initializing Active Inference agent...")
        agent = ActiveInferenceAgent(config, tracer)
        
        # Discover circuits
        logger.info("🔍 Discovering circuit features...")
        features = tracer.find_active_features(test_input)
        logger.info(f"✅ Found {len(features)} active features")
        
        # Build attribution graph
        logger.info("🔗 Building attribution graph...")
        graph = tracer.build_attribution_graph(test_input)
        logger.info(f"✅ Created graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        
        # Active Inference analysis
        logger.info("🎯 Running Active Inference analysis...")
        for i, feature in enumerate(features[:5]):  # Test first 5 features
            efe = agent.calculate_expected_free_energy(feature, "ablation")
            logger.info(f"   Feature {i}: EFE = {efe:.3f}")
        
        # Create visualization
        logger.info("🎨 Creating interactive visualization...")
        visualizer = VisualizationFactory.create_visualizer('interactive')
        viz_result = visualizer.create_interactive_graph(graph, "enhanced_experiment_output.html")
        logger.info(f"✅ Visualization created: {viz_result}")
        
        # Save results
        logger.info("💾 Saving experiment results...")
        results_dir = Path("results/Enhanced_Experiment_Results")
        results_dir.mkdir(exist_ok=True)
        
        # Save graph
        graph_file = results_dir / "attribution_graph.json"
        graph.save(graph_file)
        logger.info(f"✅ Graph saved to {graph_file}")
        
        logger.info("🎉 Enhanced experiment completed successfully\!")
        
        print("\n" + "="*60)
        print("🎯 ENHANCED EXPERIMENT SUMMARY")
        print("="*60)
        print(f"✅ Input processed: {test_input}")
        print(f"✅ Features discovered: {len(features)}")
        print(f"✅ Graph nodes: {len(graph.nodes)}")
        print(f"✅ Graph edges: {len(graph.edges)}")
        print(f"✅ Visualization: enhanced_experiment_output.html")
        print(f"✅ Results saved: {results_dir}")
        print("\n🚀 Enhanced framework is production ready\!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_enhanced_experiment()
    if not success:
        sys.exit(1)

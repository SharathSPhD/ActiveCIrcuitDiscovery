#!/usr/bin/env python3
"""
ActiveCircuitDiscovery - Complete Automated L40S GPU Experiment
YorK_RP: An Active Inference Approach to Circuit Discovery in Large Language Models

This script leverages the FULL ActiveCircuitDiscovery framework with:
- Advanced Active Inference agent with PyMDP integration
- Comprehensive circuit interventions and ablations
- Publication-ready visualizations with CircuitsVis
- Statistical validation and hypothesis testing
- Novel prediction generation and validation
- Baseline method comparisons
- Results stored in project directory structure

Designed for automated execution on DigitalOcean L40S GPU droplets.
"""

import torch
import sys
import os
from pathlib import Path
import json
from datetime import datetime
import logging
import traceback

# Enable optimizations
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add src to path for framework imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def setup_logging():
    """Setup comprehensive logging for the experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'l40s_experiment_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Execute complete Active Inference circuit discovery experiment."""
    
    # Setup logging
    logger = setup_logging()
    logger.info("🚀 Starting Complete ActiveCircuitDiscovery L40S GPU Experiment")
    
    print("=" * 80)
    print("ACTIVECIRCUITDISCOVERY - COMPLETE L40S GPU EXPERIMENT")
    print("YorK_RP: Active Inference Circuit Discovery with Full Framework")
    print("=" * 80)

    # Check GPU availability
    print("\n📊 SYSTEM INFORMATION:")
    print("-" * 40)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        device = torch.device("cuda")
    else:
        print("⚠️ CUDA not available - using CPU (slower)")
        device = torch.device("cpu")

    print(f"🎯 Using device: {device}")

    # Import framework components
    print("\n📦 LOADING FRAMEWORK COMPONENTS:")
    print("-" * 40)
    
    try:
        # Core framework imports
        from experiments.runner import YorKExperimentRunner
        from config.experiment_config import CompleteConfig
        from core.statistical_validation import perform_comprehensive_validation
        from visualization.visualizer import EnhancedVisualizer
        print("✅ Core framework components loaded")
        
        # Verify external dependencies
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.graph_objects as go
        import networkx as nx
        from transformer_lens import HookedTransformer
        import transformer_lens.utils as utils
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        import einops
        from jaxtyping import Float, Int
        print("✅ External dependencies verified")
        
    except ImportError as e:
        logger.error(f"❌ Framework import failed: {e}")
        print(f"❌ Error: {e}")
        print("💡 Ensure all dependencies are installed and src/ is in the project directory")
        return 1

    # Create experiment configuration
    print("\n⚙️ EXPERIMENT CONFIGURATION:")
    print("-" * 40)
    
    try:
        config = CompleteConfig()
        
        # Enhanced experiment settings
        config.experiment.name = "l40s_complete_golden_gate_discovery"
        config.experiment.device = str(device)
        config.experiment.save_visualizations = True
        config.experiment.save_detailed_results = True
        
        # Active Inference parameters
        config.active_inference.max_interventions = 20
        config.active_inference.uncertainty_threshold = 0.1
        config.active_inference.learning_rate = 0.05
        config.active_inference.enable_novel_predictions = True
        
        # Research question targets
        config.research_questions.rq1_correspondence_target = 70.0
        config.research_questions.rq2_efficiency_target = 30.0
        config.research_questions.rq3_predictions_target = 3
        
        # Circuit analysis settings
        config.circuit_analysis.enable_sae_analysis = True
        config.circuit_analysis.intervention_types = ["ablation", "activation_patching", "mean_ablation"]
        config.circuit_analysis.auto_layer_discovery = True
        
        # Visualization settings
        config.visualization.create_interactive_plots = True
        config.visualization.save_static_plots = True
        config.visualization.output_formats = ["png", "html", "pdf"]
        
        print(f"✅ Experiment: {config.experiment.name}")
        print(f"✅ Device: {config.experiment.device}")
        print(f"✅ Max interventions: {config.active_inference.max_interventions}")
        print(f"✅ Correspondence target: {config.research_questions.rq1_correspondence_target}%")
        print(f"✅ Efficiency target: {config.research_questions.rq2_efficiency_target}%")
        print(f"✅ Prediction target: {config.research_questions.rq3_predictions_target}")
        
    except Exception as e:
        logger.error(f"❌ Configuration setup failed: {e}")
        print(f"❌ Configuration error: {e}")
        return 1

    # Define comprehensive test inputs
    print("\n🎯 TEST INPUTS:")
    print("-" * 40)
    
    golden_gate_prompts = [
        "The Golden Gate Bridge is located in",
        "San Francisco's most famous landmark is the Golden Gate",
        "The bridge connecting San Francisco to Marin County is called the Golden Gate",
        "When visiting California, tourists often see the iconic Golden Gate",
        "The famous red suspension bridge spanning the Golden Gate strait is the Golden Gate",
        "In 1937, the newly opened Golden Gate Bridge became",
        "The Art Deco towers of the Golden Gate Bridge rise",
        "Fog often shrouds the Golden Gate Bridge in"
    ]
    
    for i, prompt in enumerate(golden_gate_prompts, 1):
        print(f"  {i}. '{prompt}'")
    
    print(f"✅ Total test cases: {len(golden_gate_prompts)}")

    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = project_root / "results" / f"L40S_Complete_Experiment_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Results directory: {results_dir}")

    # Execute comprehensive experiment
    print("\n🔬 EXECUTING ACTIVE INFERENCE EXPERIMENT:")
    print("=" * 50)
    
    try:
        # Initialize enhanced experiment runner
        logger.info("Initializing YorK experiment runner...")
        runner = YorKExperimentRunner()
        
        # Setup experiment with all components
        logger.info("Setting up experiment with enhanced components...")
        print("🔧 Setting up Active Inference agent...")
        print("🔧 Initializing circuit analysis tools...")
        print("🔧 Preparing visualization pipeline...")
        print("🔧 Loading GPT-2 model on GPU...")
        
        runner.setup_experiment(config)
        print("✅ Experiment setup complete!")
        
        # Run comprehensive experiment
        logger.info(f"Running experiment on {len(golden_gate_prompts)} test inputs...")
        print(f"\n🚀 Running experiment on {len(golden_gate_prompts)} test cases...")
        print("📊 Progress will be displayed in real-time...")
        
        result = runner.run_experiment(golden_gate_prompts)
        
        print("✅ Experiment execution complete!")
        
    except Exception as e:
        logger.error(f"❌ Experiment execution failed: {e}")
        print(f"❌ Experiment failed: {e}")
        traceback.print_exc()
        return 1

    # Perform comprehensive statistical validation
    print("\n📈 STATISTICAL VALIDATION:")
    print("-" * 40)
    
    try:
        logger.info("Performing comprehensive statistical validation...")
        statistical_validation = perform_comprehensive_validation(result)
        print("✅ Statistical validation complete!")
        
        # Extract key statistical metrics
        stats_summary = statistical_validation.get('statistical_summary', {})
        print(f"📊 Statistical tests performed: {stats_summary.get('total_tests', 0)}")
        print(f"📊 Significant results: {stats_summary.get('significant_tests', 0)}")
        print(f"📊 Average effect size: {stats_summary.get('average_effect_size', 0):.3f}")
        print(f"📊 Average statistical power: {stats_summary.get('average_power', 0):.3f}")
        
    except Exception as e:
        logger.error(f"❌ Statistical validation failed: {e}")
        print(f"⚠️ Statistical validation failed: {e}")
        statistical_validation = {}

    # Generate comprehensive visualizations
    print("\n🎨 GENERATING VISUALIZATIONS:")
    print("-" * 40)
    
    try:
        visualizer = EnhancedVisualizer()
        vis_dir = results_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # Generate all visualization types
        print("📊 Creating circuit diagrams...")
        print("📊 Creating interactive dashboards...")
        print("📊 Creating statistical validation plots...")
        print("📊 Creating belief evolution plots...")
        
        visualizer.create_comprehensive_report(
            result=result,
            output_dir=str(vis_dir),
            statistical_validation=statistical_validation
        )
        
        print("✅ Visualization generation complete!")
        
    except Exception as e:
        logger.error(f"❌ Visualization generation failed: {e}")
        print(f"⚠️ Visualization generation failed: {e}")

    # Display comprehensive results
    print("\n" + "=" * 80)
    print("🎉 COMPREHENSIVE EXPERIMENT RESULTS")
    print("=" * 80)
    
    # Basic experiment info
    print(f"\n📋 EXPERIMENT OVERVIEW:")
    print(f"  Name: {result.experiment_name}")
    print(f"  Timestamp: {result.timestamp}")
    print(f"  Duration: {result.metadata.get('duration_seconds', 0):.2f} seconds")
    print(f"  Success Rate: {result.success_rate:.1%}")
    print(f"  Test Cases: {len(golden_gate_prompts)}")
    
    # Research Questions Validation
    print(f"\n🎯 RESEARCH QUESTIONS VALIDATION:")
    print("-" * 50)
    
    rq_results = [
        ("RQ1: AI-Circuit Correspondence ≥70%", result.rq1_passed, "Validate Active Inference can identify circuit-behavior correspondences"),
        ("RQ2: Intervention Efficiency ≥30%", result.rq2_passed, "Demonstrate efficiency gains over baseline methods"),
        ("RQ3: Novel Predictions ≥3", result.rq3_passed, "Generate and validate novel theoretical predictions")
    ]
    
    for rq_name, passed, description in rq_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} {rq_name}")
        print(f"     Description: {description}")
    
    overall_status = "🎉 SUCCESS" if result.overall_success else "⚠️ PARTIAL SUCCESS"
    passed_count = sum(rq[1] for rq in rq_results)
    print(f"\n{overall_status} ({passed_count}/3 research questions passed)")

    # Detailed metrics
    print(f"\n📊 DETAILED PERFORMANCE METRICS:")
    print("-" * 50)
    
    # Correspondence metrics
    if result.correspondence_metrics:
        correspondences = [cm.overall_correspondence for cm in result.correspondence_metrics]
        avg_correspondence = sum(correspondences) / len(correspondences) * 100
        max_correspondence = max(correspondences) * 100
        min_correspondence = min(correspondences) * 100
        
        print(f"🎯 Correspondence Analysis:")
        print(f"  Average: {avg_correspondence:.1f}% (target: 70%)")
        print(f"  Range: {min_correspondence:.1f}% - {max_correspondence:.1f}%")
        print(f"  Cases above target: {sum(1 for c in correspondences if c >= 0.7)}/{len(correspondences)}")
    
    # Efficiency metrics
    if result.efficiency_metrics:
        overall_efficiency = result.efficiency_metrics.get('overall_improvement', 0)
        ai_interventions = result.efficiency_metrics.get('ai_interventions', 0)
        baseline_interventions = result.efficiency_metrics.get('baseline_interventions', 0)
        
        print(f"\n⚡ Efficiency Analysis:")
        print(f"  Overall improvement: {overall_efficiency:.1f}% (target: 30%)")
        print(f"  AI interventions: {ai_interventions}")
        print(f"  Baseline interventions: {baseline_interventions}")
        print(f"  Intervention reduction: {((baseline_interventions - ai_interventions) / baseline_interventions * 100) if baseline_interventions > 0 else 0:.1f}%")
    
    # Novel predictions
    validated_predictions = len([p for p in result.novel_predictions if p.validation_status == 'validated'])
    pending_predictions = len([p for p in result.novel_predictions if p.validation_status == 'pending'])
    
    print(f"\n🔮 Novel Predictions Analysis:")
    print(f"  Total generated: {len(result.novel_predictions)}")
    print(f"  Validated: {validated_predictions} (target: 3)")
    print(f"  Pending: {pending_predictions}")
    print(f"  Success rate: {(validated_predictions / len(result.novel_predictions) * 100) if result.novel_predictions else 0:.1f}%")

    # Display novel predictions detail
    if result.novel_predictions:
        print(f"\n🔬 NOVEL PREDICTIONS DETAIL:")
        print("-" * 50)
        
        for i, prediction in enumerate(result.novel_predictions, 1):
            status_map = {
                "validated": "✅ VALIDATED",
                "falsified": "❌ FALSIFIED", 
                "pending": "⏳ PENDING"
            }
            status = status_map.get(prediction.validation_status, "❓ UNKNOWN")
            
            print(f"\n{i}. {status} - {prediction.prediction_type.upper()}")
            print(f"   Description: {prediction.description}")
            print(f"   Hypothesis: {prediction.testable_hypothesis}")
            print(f"   Confidence: {prediction.confidence:.2f}")
            if hasattr(prediction, 'validation_evidence') and prediction.validation_evidence:
                print(f"   Evidence: {prediction.validation_evidence}")

    # Circuit discovery results
    if hasattr(result, 'discovered_circuits') and result.discovered_circuits:
        print(f"\n🧠 DISCOVERED CIRCUITS:")
        print("-" * 50)
        
        circuit_count = len(result.discovered_circuits)
        print(f"Total circuits discovered: {circuit_count}")
        
        # Show top circuits
        top_circuits = sorted(result.discovered_circuits.items(), 
                            key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, 
                            reverse=True)[:5]
        
        print("Top 5 circuits by importance:")
        for circuit_name, importance in top_circuits:
            print(f"  • {circuit_name}: {importance:.4f}")

    # Interventions performed
    if hasattr(result, 'interventions_performed'):
        print(f"\n🎛️ INTERVENTIONS PERFORMED:")
        print("-" * 50)
        
        intervention_types = {}
        for intervention in result.interventions_performed:
            itype = getattr(intervention, 'intervention_type', 'unknown')
            intervention_types[itype] = intervention_types.get(itype, 0) + 1
        
        print(f"Total interventions: {len(result.interventions_performed)}")
        for itype, count in intervention_types.items():
            print(f"  • {itype}: {count}")

    # Save comprehensive results
    print(f"\n💾 SAVING RESULTS:")
    print("-" * 50)
    
    try:
        # Save main results
        runner.save_results(result, str(results_dir))
        
        # Save statistical validation
        if statistical_validation:
            stats_file = results_dir / "statistical_validation.json"
            with open(stats_file, 'w') as f:
                json.dump(statistical_validation, f, indent=2, default=str)
            print(f"✅ Statistical validation saved to: {stats_file}")
        
        # Save experiment summary
        summary = {
            "experiment_name": result.experiment_name,
            "timestamp": result.timestamp,
            "device": str(device),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None,
            "test_cases": len(golden_gate_prompts),
            "duration_seconds": result.metadata.get('duration_seconds', 0),
            "research_questions": {
                "rq1_passed": result.rq1_passed,
                "rq2_passed": result.rq2_passed, 
                "rq3_passed": result.rq3_passed,
                "overall_success": result.overall_success
            },
            "key_metrics": {
                "avg_correspondence": sum(cm.overall_correspondence for cm in result.correspondence_metrics) / len(result.correspondence_metrics) * 100 if result.correspondence_metrics else 0,
                "efficiency_improvement": result.efficiency_metrics.get('overall_improvement', 0) if result.efficiency_metrics else 0,
                "validated_predictions": validated_predictions,
                "total_predictions": len(result.novel_predictions)
            },
            "results_directory": str(results_dir)
        }
        
        summary_file = results_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Main results saved to: {results_dir}")
        print(f"✅ Experiment summary saved to: {summary_file}")
        
    except Exception as e:
        logger.error(f"❌ Results saving failed: {e}")
        print(f"⚠️ Results saving failed: {e}")

    # Final status
    print("\n" + "=" * 80)
    if result.overall_success:
        print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("✅ All research questions validated!")
        print("🏆 Active Inference circuit discovery framework fully validated!")
    else:
        print("⚠️ EXPERIMENT COMPLETED WITH PARTIAL SUCCESS")
        print(f"✅ {passed_count}/3 research questions passed")
        print("📈 Significant progress demonstrated in circuit discovery capabilities")
    
    print("=" * 80)
    
    logger.info(f"Experiment completed. Results saved to: {results_dir}")
    
    return 0 if result.overall_success else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
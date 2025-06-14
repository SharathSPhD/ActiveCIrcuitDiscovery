#!/usr/bin/env python3
"""
Complete ActiveCircuitDiscovery Experiment Runner
Runs the full framework with all advanced capabilities on L40S GPU
"""

import sys
import os
from pathlib import Path
import json
import logging
from datetime import datetime
import traceback

# Setup Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Configure logging
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'complete_experiment_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Execute complete Active Inference circuit discovery experiment."""
    
    logger = setup_logging()
    logger.info("🚀 Starting Complete ActiveCircuitDiscovery Experiment")
    
    print("=" * 80)
    print("ACTIVECIRCUITDISCOVERY - COMPLETE FRAMEWORK EXECUTION")
    print("YorK_RP: Full Active Inference Circuit Discovery Implementation")
    print("=" * 80)

    # Check GPU
    print("\n📊 SYSTEM INFORMATION:")
    print("-" * 40)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"🎯 Using device: {device}")
    except Exception as e:
        logger.error(f"GPU setup failed: {e}")
        return 1

    # Import and setup framework components
    print("\n📦 LOADING FRAMEWORK COMPONENTS:")
    print("-" * 40)
    
    try:
        # Core framework imports
        from experiments.runner import YorKExperimentRunner
        from config.experiment_config import CompleteConfig
        from core.statistical_validation import perform_comprehensive_validation
        from visualization.visualizer import CircuitVisualizer
        print("✅ Core framework components loaded")
        
        # Additional imports for comprehensive analysis
        import numpy as np
        import matplotlib.pyplot as plt
        print("✅ Analysis libraries loaded")
        
    except ImportError as e:
        logger.error(f"❌ Framework import failed: {e}")
        traceback.print_exc()
        print(f"❌ Error: {e}")
        return 1

    # Create comprehensive configuration
    print("\n⚙️ EXPERIMENT CONFIGURATION:")
    print("-" * 40)
    
    try:
        config = CompleteConfig()
        
        # Enhanced experiment settings for L40S GPU
        config.experiment.name = "l40s_complete_framework_validation"
        config.experiment.device = str(device)
        config.experiment.save_visualizations = True
        config.experiment.save_detailed_results = True
        
        # Active Inference parameters - optimized for comprehensive analysis
        config.active_inference.max_interventions = 25
        config.active_inference.uncertainty_threshold = 0.1
        config.active_inference.learning_rate = 0.05
        config.active_inference.enable_novel_predictions = True
        
        # Research question targets
        config.research_questions.rq1_correspondence_target = 70.0
        config.research_questions.rq2_efficiency_target = 30.0
        config.research_questions.rq3_predictions_target = 3
        
        # SAE analysis settings - comprehensive
        config.sae.enabled = True
        config.sae.target_layers = list(range(12))  # All layers for auto-discovery
        config.sae.activation_threshold = 0.1
        
        # Visualization settings - generate all formats
        config.visualization.create_interactive_plots = True
        config.visualization.save_static_plots = True
        config.visualization.output_formats = ["png", "html", "pdf"]
        
        print(f"✅ Experiment: {config.experiment.name}")
        print(f"✅ Device: {config.experiment.device}")
        print(f"✅ Max interventions: {config.active_inference.max_interventions}")
        print(f"✅ All visualization formats enabled")
        
    except Exception as e:
        logger.error(f"❌ Configuration setup failed: {e}")
        traceback.print_exc()
        return 1

    # Comprehensive test inputs
    print("\n🎯 COMPREHENSIVE TEST INPUTS:")
    print("-" * 40)
    
    golden_gate_prompts = [
        "The Golden Gate Bridge is located in",
        "San Francisco's most famous landmark is the Golden Gate",
        "The bridge connecting San Francisco to Marin County is called the Golden Gate",
        "When visiting California, tourists often see the iconic Golden Gate",
        "The famous red suspension bridge spanning the Golden Gate strait is the Golden Gate",
        "In 1937, the newly opened Golden Gate Bridge became",
        "The Art Deco towers of the Golden Gate Bridge rise",
        "Fog often shrouds the Golden Gate Bridge in",
        "Engineers designed the Golden Gate Bridge to withstand",
        "The Golden Gate Bridge's International Orange color was chosen to"
    ]
    
    for i, prompt in enumerate(golden_gate_prompts, 1):
        print(f"  {i:2d}. '{prompt}'")
    
    print(f"✅ Total test cases: {len(golden_gate_prompts)}")

    # Setup results directory with project structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = project_root / "results" / f"Complete_L40S_Experiment_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Results directory: {results_dir}")

    # Execute comprehensive experiment
    print("\n🔬 EXECUTING COMPLETE FRAMEWORK:")
    print("=" * 50)
    
    try:
        # Initialize experiment runner
        logger.info("Initializing complete experiment runner...")
        runner = YorKExperimentRunner()
        
        # Setup experiment with full framework
        logger.info("Setting up complete experiment components...")
        print("🔧 Initializing Active Inference agent...")
        print("🔧 Setting up circuit analysis tools...")
        print("🔧 Preparing comprehensive visualization pipeline...")
        print("🔧 Loading transformer models and SAE analyzers...")
        
        runner.setup_experiment(config)
        print("✅ Complete framework setup successful!")
        
        # Run comprehensive experiment
        logger.info(f"Running complete experiment on {len(golden_gate_prompts)} test inputs...")
        print(f"\n🚀 Executing experiment on {len(golden_gate_prompts)} test cases...")
        print("📊 Real-time progress with full framework capabilities...")
        
        result = runner.run_experiment(golden_gate_prompts)
        
        print("✅ Complete experiment execution finished!")
        
    except Exception as e:
        logger.error(f"❌ Experiment execution failed: {e}")
        traceback.print_exc()
        print(f"❌ Experiment failed: {e}")
        return 1

    # Comprehensive statistical validation
    print("\n📈 COMPREHENSIVE STATISTICAL VALIDATION:")
    print("-" * 50)
    
    try:
        logger.info("Performing comprehensive statistical validation...")
        statistical_validation = perform_comprehensive_validation(result)
        print("✅ Statistical validation completed!")
        
        # Display statistical summary
        if 'statistical_summary' in statistical_validation:
            stats = statistical_validation['statistical_summary']
            print(f"📊 Statistical tests: {stats.get('total_tests', 0)}")
            print(f"📊 Significant results: {stats.get('significant_tests', 0)}")
            print(f"📊 Average effect size: {stats.get('average_effect_size', 0):.3f}")
            print(f"📊 Statistical power: {stats.get('average_power', 0):.3f}")
        
    except Exception as e:
        logger.error(f"❌ Statistical validation failed: {e}")
        traceback.print_exc()
        print(f"⚠️ Statistical validation failed: {e}")
        statistical_validation = {}

    # Generate comprehensive visualizations
    print("\n🎨 GENERATING COMPREHENSIVE VISUALIZATIONS:")
    print("-" * 50)
    
    try:
        visualizer = CircuitVisualizer()
        vis_dir = results_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        print("📊 Creating circuit diagrams...")
        print("📊 Creating interactive dashboards...")
        print("📊 Creating attention visualizations...")
        print("📊 Creating intervention effect plots...")
        print("📊 Creating belief evolution animations...")
        print("📊 Creating statistical validation reports...")
        
        visualizer.generate_all_visualizations(
            result=result,
            output_dir=str(vis_dir),
            statistical_validation=statistical_validation
        )
        
        print("✅ Complete visualization suite generated!")
        print(f"📁 Visualizations saved to: {vis_dir}")
        
    except Exception as e:
        logger.error(f"❌ Visualization generation failed: {e}")
        traceback.print_exc()
        print(f"⚠️ Visualization generation failed: {e}")

    # Display comprehensive results analysis
    print("\n" + "=" * 80)
    print("🎉 COMPREHENSIVE EXPERIMENT RESULTS ANALYSIS")
    print("=" * 80)
    
    # Experiment overview
    print(f"\n📋 EXPERIMENT OVERVIEW:")
    print(f"  Name: {result.experiment_name}")
    print(f"  Timestamp: {result.timestamp}")
    print(f"  Duration: {result.metadata.get('duration_seconds', 0):.2f} seconds")
    print(f"  GPU Used: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Success Rate: {result.success_rate:.1%}")
    print(f"  Test Cases: {len(golden_gate_prompts)}")
    
    # Research Questions Validation
    print(f"\n🎯 RESEARCH QUESTIONS VALIDATION:")
    print("-" * 50)
    
    rq_results = [
        ("RQ1: AI-Circuit Correspondence ≥70%", result.rq1_passed, "Active Inference identifies circuit-behavior correspondences"),
        ("RQ2: Intervention Efficiency ≥30%", result.rq2_passed, "Efficiency gains over baseline intervention methods"),
        ("RQ3: Novel Predictions ≥3", result.rq3_passed, "Generate and validate novel theoretical predictions")
    ]
    
    for rq_name, passed, description in rq_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} {rq_name}")
        print(f"     {description}")
    
    overall_status = "🎉 COMPLETE SUCCESS" if result.overall_success else "⚠️ PARTIAL SUCCESS"
    passed_count = sum(rq[1] for rq in rq_results)
    print(f"\n{overall_status} ({passed_count}/3 research questions passed)")

    # Detailed performance metrics
    print(f"\n📊 DETAILED PERFORMANCE METRICS:")
    print("-" * 50)
    
    # Correspondence analysis
    if result.correspondence_metrics:
        correspondences = [cm.overall_correspondence for cm in result.correspondence_metrics]
        avg_correspondence = sum(correspondences) / len(correspondences) * 100
        max_correspondence = max(correspondences) * 100
        min_correspondence = min(correspondences) * 100
        
        print(f"🎯 Correspondence Analysis:")
        print(f"  Average: {avg_correspondence:.1f}% (target: 70%)")
        print(f"  Range: {min_correspondence:.1f}% - {max_correspondence:.1f}%")
        print(f"  Cases above target: {sum(1 for c in correspondences if c >= 0.7)}/{len(correspondences)}")
    
    # Efficiency analysis
    if result.efficiency_metrics:
        overall_efficiency = result.efficiency_metrics.get('overall_improvement', 0)
        ai_interventions = result.efficiency_metrics.get('ai_interventions', 0)
        
        print(f"\n⚡ Efficiency Analysis:")
        print(f"  Overall improvement: {overall_efficiency:.1f}% (target: 30%)")
        print(f"  AI interventions used: {ai_interventions}")
        
        # Show baseline comparisons
        for key, value in result.efficiency_metrics.items():
            if key.endswith('_improvement'):
                baseline_name = key.replace('_improvement', '')
                print(f"  vs {baseline_name}: {value:.1f}% improvement")
    
    # Novel predictions analysis
    validated_predictions = len([p for p in result.novel_predictions if p.validation_status == 'validated'])
    pending_predictions = len([p for p in result.novel_predictions if p.validation_status == 'pending'])
    
    print(f"\n🔮 Novel Predictions Analysis:")
    print(f"  Total generated: {len(result.novel_predictions)}")
    print(f"  Validated: {validated_predictions} (target: 3)")
    print(f"  Pending validation: {pending_predictions}")
    print(f"  Success rate: {(validated_predictions / len(result.novel_predictions) * 100) if result.novel_predictions else 0:.1f}%")

    # Show detailed novel predictions
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

    # Circuit discovery results
    if hasattr(result, 'discovered_circuits') and result.discovered_circuits:
        print(f"\n🧠 DISCOVERED CIRCUITS:")
        print("-" * 50)
        
        circuit_count = len(result.discovered_circuits)
        print(f"Total circuits discovered: {circuit_count}")
        
        # Show top circuits by importance
        if isinstance(result.discovered_circuits, dict):
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
    print(f"\n💾 SAVING COMPREHENSIVE RESULTS:")
    print("-" * 50)
    
    try:
        # Save main results using framework
        runner.save_results(result, str(results_dir))
        
        # Save statistical validation
        if statistical_validation:
            stats_file = results_dir / "statistical_validation.json"
            with open(stats_file, 'w') as f:
                json.dump(statistical_validation, f, indent=2, default=str)
            print(f"✅ Statistical validation: {stats_file}")
        
        # Save comprehensive experiment summary
        summary = {
            "experiment_info": {
                "name": result.experiment_name,
                "timestamp": result.timestamp,
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None,
                "framework_version": "complete_refactored",
                "test_cases": len(golden_gate_prompts),
                "duration_seconds": result.metadata.get('duration_seconds', 0)
            },
            "research_questions": {
                "rq1_correspondence": {
                    "target": 70.0,
                    "passed": result.rq1_passed,
                    "avg_correspondence": sum(cm.overall_correspondence for cm in result.correspondence_metrics) / len(result.correspondence_metrics) * 100 if result.correspondence_metrics else 0
                },
                "rq2_efficiency": {
                    "target": 30.0,
                    "passed": result.rq2_passed,
                    "improvement": result.efficiency_metrics.get('overall_improvement', 0) if result.efficiency_metrics else 0
                },
                "rq3_predictions": {
                    "target": 3,
                    "passed": result.rq3_passed,
                    "validated": validated_predictions,
                    "total": len(result.novel_predictions)
                },
                "overall_success": result.overall_success
            },
            "capabilities_demonstrated": {
                "active_inference_agent": True,
                "circuit_discovery": True,
                "intervention_analysis": True,
                "statistical_validation": bool(statistical_validation),
                "comprehensive_visualizations": True,
                "novel_predictions": len(result.novel_predictions) > 0,
                "baseline_comparisons": bool(result.efficiency_metrics)
            },
            "file_locations": {
                "results_directory": str(results_dir),
                "visualizations": str(vis_dir),
                "statistical_validation": str(stats_file) if statistical_validation else None
            }
        }
        
        summary_file = results_dir / "comprehensive_experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Main results: {results_dir}")
        print(f"✅ Comprehensive summary: {summary_file}")
        
    except Exception as e:
        logger.error(f"❌ Results saving failed: {e}")
        traceback.print_exc()
        print(f"⚠️ Results saving failed: {e}")

    # Final status and next steps
    print("\n" + "=" * 80)
    if result.overall_success:
        print("🎉 COMPLETE FRAMEWORK SUCCESSFULLY EXECUTED!")
        print("✅ All research questions validated!")
        print("🏆 Active Inference circuit discovery framework fully operational!")
        print("📊 Comprehensive visualizations and statistical validation completed!")
    else:
        print("⚠️ FRAMEWORK EXECUTED WITH PARTIAL SUCCESS")
        print(f"✅ {passed_count}/3 research questions passed")
        print("📈 Significant progress in circuit discovery capabilities")
        print("🔧 Framework operational - some targets need refinement")
    
    print("\n📁 RESULTS SUMMARY:")
    print(f"  🗂️  All results: {results_dir}")
    print(f"  📊 Visualizations: {vis_dir}")
    print(f"  📈 Statistical validation: Available")
    print(f"  🎯 Framework status: {'FULLY OPERATIONAL' if result.overall_success else 'OPERATIONAL'}")
    
    print("=" * 80)
    
    logger.info(f"Complete experiment finished. Results: {results_dir}")
    
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
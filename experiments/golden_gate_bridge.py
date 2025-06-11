#!/usr/bin/env python3
"""
Enhanced Golden Gate Bridge Circuit Discovery Experiment
=======================================================

Complete Active Inference approach to circuit discovery using the Golden Gate Bridge
as a canonical example in mechanistic interpretability research.

This enhanced experiment validates all three research questions with:
- Proper correspondence metrics calibration [0,100%]
- Enhanced novel prediction generation and validation
- Comprehensive statistical testing and validation
- Complete circuit tracer and Active Inference agent

Research Questions Validated:
- RQ1: Active Inference correspondence with circuit behavior (≥70% target)
- RQ2: Efficiency improvement over baseline methods (≥30% target) 
- RQ3: Novel predictions from Active Inference analysis (≥3 target)

Usage:
    python experiments/golden_gate_bridge.py
    
    # With statistical validation:
    python experiments/golden_gate_bridge.py --with-stats
    
    # Or from project root:
    python -m experiments.golden_gate_bridge
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import logging
from datetime import datetime

# Import enhanced ActiveCircuitDiscovery components
try:
    from experiments.runner import YorKExperimentRunner
    from core.data_structures import ExperimentResult
    from config.experiment_config import CompleteConfig
    from circuit_analysis.tracer import CircuitTracer
    from active_inference.agent import ActiveInferenceAgent
    from core.statistical_validation import perform_comprehensive_validation
    from core.prediction_system import EnhancedPredictionGenerator
    
    print("[SUCCESS] All enhanced ActiveCircuitDiscovery components imported successfully")
    ENHANCED_MODE = True
    
except ImportError as e:
    print(f"[WARNING] Enhanced components import error: {e}")
    print("Trying basic imports...")
    
    try:
        # Fallback to basic imports
        import torch
        import transformer_lens
        import numpy as np
        
        print("[FALLBACK] Using basic transformer analysis")
        ENHANCED_MODE = False
        
    except ImportError as e2:
        print(f"[CRITICAL] Cannot import basic dependencies: {e2}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('golden_gate_bridge_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_golden_gate_bridge_experiment():
    """Run the complete Golden Gate Bridge circuit discovery experiment."""
    print("ActiveCircuitDiscovery: Golden Gate Bridge Circuit Experiment")
    print("=" * 70)
    print()
    
    logger.info("Starting Golden Gate Bridge circuit discovery experiment")
    
    try:
        # Initialize experiment runner
        print("Initializing experiment runner...")
        runner = YorKExperimentRunner()
        
        # Setup experiment components
        print("Setting up experiment components...")
        runner.setup_experiment()
        
        # Define Golden Gate Bridge test inputs
        test_inputs = [
            "The Golden Gate Bridge is located in",
            "San Francisco's most famous landmark is the",
            "The bridge connecting San Francisco to Marin County is called the",
            "When visiting California, tourists often see the iconic",
            "The famous red suspension bridge in San Francisco is known as the"
        ]
        
        # Run complete experiment
        print("Running complete Golden Gate Bridge experiment...")
        print("   This includes:")
        print("   - Auto-discovery of active features across ALL layers")
        print("   - Active Inference guided circuit discovery (should need fewer interventions)")
        print("   - Baseline method comparisons (should need more interventions)")
        print("   - Research question validation")
        print("   - Circuit visualizations")
        print()
        
        results = runner.run_experiment(test_inputs)
        
        # Print results summary
        print_experiment_summary(results)
        
        # Save results
        output_dir = runner.config.experiment.output_dir
        runner.save_results(results, output_dir)
        
        # Enhanced statistical validation
        if ENHANCED_MODE:
            from core.statistical_validation import perform_comprehensive_validation
            statistical_validation = perform_comprehensive_validation(results)
            print("\n📊 Statistical Validation:")
            print("-" * 40)
            if 'statistical_summary' in statistical_validation:
                stats = statistical_validation['statistical_summary']
                print(f"   Tests performed: {stats.get('total_tests', 0)}")
                print(f"   Significant results: {stats.get('significant_tests', 0)}")
                print(f"   Average effect size: {stats.get('average_effect_size', 0):.3f}")
        
        # Research question validation
        rq_validation = runner.validate_research_questions(
            results.correspondence_metrics,
            results.efficiency_metrics,
            len([p for p in results.novel_predictions if hasattr(p, 'validation_status') and p.validation_status == 'validated'])
        )
        
        print("\nResearch Question Validation:")
        print("-" * 40)
        
        rq_details = {
            'rq1': {'description': 'Active Inference correspondence with circuit behavior', 'target': '70%'},
            'rq2': {'description': 'Efficiency improvement over baseline methods', 'target': '30%'},
            'rq3': {'description': 'Novel predictions from Active Inference analysis', 'target': '3+'}
        }
        
        for rq_name in ['rq1', 'rq2', 'rq3']:
            passed_key = f"{rq_name}_passed"
            achieved_key = f"{rq_name}_achieved"
            
            status = "PASSED" if rq_validation.get(passed_key, False) else "FAILED"
            achieved = rq_validation.get(achieved_key, 0)
            target = rq_details[rq_name]['target']
            description = rq_details[rq_name]['description']
            
            print(f"{rq_name.upper()}: {status}")
            print(f"   Target: {target}")
            print(f"   Achieved: {achieved}")
            print(f"   Description: {description}")
            print()
        
        # Overall success
        overall_success = rq_validation.get('overall_success', False)
        success_rate = rq_validation.get('success_rate', 0.0)
        
        if overall_success:
            print("🎉 EXPERIMENT SUCCESSFUL!")
            print(f"   Success rate: {success_rate:.1%}")
            print("   ✅ Active Inference approach validated")
            print("   ✅ Efficiency improvements demonstrated")
            print("   ✅ Novel insights discovered")
        else:
            print("⚠️  Partial Success")
            print(f"   Success rate: {success_rate:.1%}")
            print("   Some research questions need further investigation")
        
        # Output information
        print(f"\n📁 Results saved to: {output_dir}")
        print(f"📊 Experiment log: golden_gate_bridge_experiment.log")
        
        print("\n✅ Experiment completed successfully!")
        
        return {
            'success': overall_success,
            'results': results,
            'output_dir': str(output_dir),
            'rq_validation': rq_validation
        }
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        print(f"\n❌ Experiment failed: {e}")
        print("Check the log file for detailed error information")
        
        # Try fallback mode
        if 'FALLBACK_MODE' in globals() and FALLBACK_MODE:
            print("\n🔄 Attempting fallback demonstration...")
            return run_fallback_demo()
        
        return {
            'success': False,
            'error': str(e)
        }

def run_fallback_demo():
    """Run a basic demonstration when full system is not available."""
    print("Fallback Demo: Basic Circuit Analysis")
    print("-" * 40)
    
    try:
        import torch
        import transformer_lens
        
        # Initialize basic components
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model = transformer_lens.HookedTransformer.from_pretrained("gpt2")
        model.to(device)
        print(f"Model loaded: GPT-2 ({model.cfg.n_layers} layers)")
        
        # Test inputs
        test_inputs = [
            "The Golden Gate Bridge is located in",
            "San Francisco's most famous landmark is the"
        ]
        
        print("\nAnalyzing Golden Gate Bridge circuits...")
        
        for i, text in enumerate(test_inputs):
            print(f"\n🔍 Input {i+1}: '{text}'")
            
            # Get model prediction
            tokens = model.to_tokens(text)
            with torch.no_grad():
                logits, cache = model.run_with_cache(tokens)
                
            # Get top predictions
            probs = torch.softmax(logits[0, -1], dim=-1)
            top_tokens = torch.topk(probs, 5)
            
            print("Top predictions:")
            for j, (prob, token_id) in enumerate(zip(top_tokens.values, top_tokens.indices)):
                token_str = model.to_string(token_id)
                print(f"  {j+1}. '{token_str}' ({prob:.3f})")
            
            # Analyze layer activations
            print("\nLayer activation analysis:")
            for layer in range(min(3, model.cfg.n_layers)):
                activations = cache[f'blocks.{layer}.hook_resid_post']
                max_activation = torch.max(torch.abs(activations)).item()
                print(f"  Layer {layer}: max activation = {max_activation:.3f}")
        
        print("\n✅ Fallback demo completed successfully!")
        print("Install full dependencies for complete Active Inference analysis.")
        
        return {
            'success': True,
            'mode': 'fallback',
            'message': 'Basic circuit analysis completed'
        }
        
    except Exception as e:
        print(f"❌ Fallback demo failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def print_experiment_summary(results: ExperimentResult):
    """Print a comprehensive summary of experiment results."""
    print("\n" + "=" * 60)
    print("📊 EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"Experiment: {results.experiment_name}")
    print(f"Timestamp: {results.timestamp}")
    print(f"Duration: {results.metadata.get('duration_seconds', 0):.1f} seconds")
    
    print(f"\n🎯 Research Question Results:")
    print(f"   RQ1 (Correspondence): {'✅ PASSED' if results.rq1_passed else '❌ FAILED'}")
    print(f"   RQ2 (Efficiency): {'✅ PASSED' if results.rq2_passed else '❌ FAILED'}")
    print(f"   RQ3 (Predictions): {'✅ PASSED' if results.rq3_passed else '❌ FAILED'}")
    print(f"   Overall Success: {'✅ YES' if results.overall_success else '❌ NO'}")
    
    print(f"\n📈 Key Metrics:")
    print(f"   Correspondence metrics: {len(results.correspondence_metrics)}")
    print(f"   Intervention results: {len(results.intervention_results)}")
    print(f"   Novel predictions: {len(results.novel_predictions)}")
    
    if results.efficiency_metrics:
        print(f"\n⚡ Efficiency Analysis:")
        for metric, value in results.efficiency_metrics.items():
            if 'improvement' in metric:
                print(f"   {metric}: {value:.1f}%")
    
    print("=" * 60)

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Golden Gate Bridge Circuit Discovery")
    parser.add_argument("--demo", action="store_true", 
                       help="Run fallback demo instead of full experiment")
    parser.add_argument("--full", action="store_true", 
                       help="Run full experiment (default)")
    parser.add_argument("--with-stats", action="store_true",
                       help="Include comprehensive statistical validation")
    
    args = parser.parse_args()
    
    if args.demo:
        result = run_fallback_demo()
    else:
        # Run full experiment by default
        result = run_golden_gate_bridge_experiment()
        
        # Enhanced statistical validation if requested
        if args.with_stats and ENHANCED_MODE and result['success']:
            print("\n" + "="*70)
            print("🔬 COMPREHENSIVE STATISTICAL ANALYSIS")
            print("="*70)
            try:
                from core.statistical_validation import perform_comprehensive_validation
                validation = perform_comprehensive_validation(result['results'])
                
                if 'statistical_summary' in validation:
                    stats = validation['statistical_summary']
                    print(f"\n📈 Statistical Summary:")
                    print(f"   Total tests: {stats.get('total_tests', 0)}")
                    print(f"   Significant: {stats.get('significant_tests', 0)}")
                    print(f"   Significance rate: {stats.get('significance_rate', 0):.1%}")
                    print(f"   Avg effect size: {stats.get('average_effect_size', 0):.3f}")
                    print(f"   Avg power: {stats.get('average_power', 0):.3f}")
                
                print("\n✅ Enhanced statistical validation completed")
            except Exception as e:
                print(f"⚠️  Statistical validation failed: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Enhanced Active Inference Circuit Discovery Experiment
Demonstrates the complete enhanced system with all improvements.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.runner import YorKExperimentRunner
from config.experiment_config import CompleteConfig
from core.statistical_validation import perform_comprehensive_validation

def main():
    """Run enhanced Active Inference circuit discovery experiment."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Enhanced Active Inference Circuit Discovery Experiment")
    
    # Create enhanced configuration
    config = CompleteConfig()
    config.experiment.name = "enhanced_golden_gate_discovery"
    config.active_inference.max_interventions = 15
    config.research_questions.rq1_correspondence_target = 70.0
    config.research_questions.rq2_efficiency_target = 30.0
    config.research_questions.rq3_predictions_target = 3
    
    # Test inputs for Golden Gate Bridge circuit discovery
    test_inputs = [
        "The Golden Gate Bridge is located in",
        "San Francisco's most famous landmark is the",
        "The bridge connecting San Francisco to Marin County is called the",
        "When visiting California, tourists often see the iconic",
        "The famous red suspension bridge in San Francisco is known as the"
    ]
    
    try:
        # Initialize enhanced experiment runner
        logger.info("Initializing enhanced experiment runner...")
        runner = YorKExperimentRunner()
        
        # Setup experiment with enhanced components
        logger.info("Setting up enhanced experiment components...")
        runner.setup_experiment(config)
        
        # Run experiment
        logger.info(f"Running experiment on {len(test_inputs)} test inputs...")
        result = runner.run_experiment(test_inputs)
        
        # Perform comprehensive statistical validation
        logger.info("Performing comprehensive statistical validation...")
        statistical_validation = perform_comprehensive_validation(result)
        
        # Display results
        print("\n" + "="*80)
        print("ENHANCED ACTIVE INFERENCE CIRCUIT DISCOVERY RESULTS")
        print("="*80)
        
        print(f"\nExperiment: {result.experiment_name}")
        print(f"Duration: {result.metadata.get('duration_seconds', 0):.2f} seconds")
        print(f"Success Rate: {result.success_rate:.1%}")
        
        print("\n" + "-"*50)
        print("RESEARCH QUESTIONS VALIDATION")
        print("-"*50)
        
        rq_results = [
            ("RQ1 (Correspondence ≥70%)", result.rq1_passed),
            ("RQ2 (Efficiency ≥30%)", result.rq2_passed), 
            ("RQ3 (Predictions ≥3)", result.rq3_passed)
        ]
        
        for rq_name, passed in rq_results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{rq_name}: {status}")
        
        overall_status = "🎉 SUCCESS" if result.overall_success else "⚠️ PARTIAL"
        print(f"\nOverall Result: {overall_status}")
        
        print("\n" + "-"*50)
        print("DETAILED METRICS")
        print("-"*50)
        
        # Correspondence metrics
        if result.correspondence_metrics:
            avg_correspondence = sum(cm.overall_correspondence for cm in result.correspondence_metrics) / len(result.correspondence_metrics)
            print(f"Average Correspondence: {avg_correspondence*100:.1f}%")
        
        # Efficiency metrics
        if result.efficiency_metrics:
            overall_efficiency = result.efficiency_metrics.get('overall_improvement', 0)
            print(f"Overall Efficiency Improvement: {overall_efficiency:.1f}%")
            ai_interventions = result.efficiency_metrics.get('ai_interventions', 0)
            print(f"AI Interventions Used: {ai_interventions}")
        
        # Novel predictions
        validated_predictions = len([p for p in result.novel_predictions if p.validation_status == 'validated'])
        print(f"Novel Predictions: {len(result.novel_predictions)} generated, {validated_predictions} validated")
        
        print("\n" + "-"*50)
        print("STATISTICAL VALIDATION")
        print("-"*50)
        
        # Display statistical test results
        if 'statistical_summary' in statistical_validation:
            stats_summary = statistical_validation['statistical_summary']
            print(f"Statistical Tests Performed: {stats_summary.get('total_tests', 0)}")
            print(f"Significant Results: {stats_summary.get('significant_tests', 0)}")
            print(f"Average Effect Size: {stats_summary.get('average_effect_size', 0):.3f}")
            print(f"Average Statistical Power: {stats_summary.get('average_power', 0):.3f}")
        
        print("\n" + "-"*50)
        print("NOVEL PREDICTIONS DETAIL")
        print("-"*50)
        
        for i, prediction in enumerate(result.novel_predictions, 1):
            status_emoji = "✅" if prediction.validation_status == "validated" else "❌" if prediction.validation_status == "falsified" else "⏳"
            print(f"\n{i}. {status_emoji} {prediction.prediction_type.upper()}")
            print(f"   Description: {prediction.description}")
            print(f"   Hypothesis: {prediction.testable_hypothesis}")
            print(f"   Confidence: {prediction.confidence:.2f}")
            print(f"   Status: {prediction.validation_status}")
        
        # Save results
        output_dir = Path("experiment_results") / result.experiment_name
        runner.save_results(result, str(output_dir))
        
        print(f"\n📁 Results saved to: {output_dir}")
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return result
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
    if result and result.overall_success:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure
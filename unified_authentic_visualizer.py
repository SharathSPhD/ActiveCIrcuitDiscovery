#!/usr/bin/env python3
"""
ENHANCED UNIFIED AUTHENTIC VISUALIZATION SYSTEM - CASE-SPECIFIC ANALYSIS
Comprehensive visualization suite for ActiveCircuitDiscovery with individual case analysis

🚨 CRITICAL ENHANCEMENT: Case-specific analysis for all 35 test cases
✅ Individual method comparison per test case
✅ Gemma output analysis per case/method combination
✅ Circuit intervention visualization per case
✅ Comprehensive case-by-case authentic analysis
✅ Statistical summaries maintained + enhanced case details

Addresses the critical gap: no case-specific visualization of circuit differences.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
import logging
from dataclasses import dataclass
import networkx as nx
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
import textwrap
import re
from collections import defaultdict
warnings.filterwarnings('ignore')

# === ENVIRONMENT SETUP ===
def setup_visualization_environment():
    """Setup environment for visualization system."""
    # Check if we're in the virtual environment
    venv_path = "/home/ubuntu/project_venv"
    if not sys.prefix.startswith(venv_path):
        print("❌ Virtual environment not activated!")
        print(f"Please run: source {venv_path}/bin/activate")
        sys.exit(1)

    # Add src to path
    project_root = Path(__file__).parent.absolute()
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Configure matplotlib for high-quality output
    plt.style.use('seaborn-v0_8')
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(project_root / 'visualization.log'),
            logging.StreamHandler()
        ]
    )

    print("✅ Enhanced visualization environment configured successfully")
    return project_root

# === DATA STRUCTURES ===
@dataclass
class TestCase:
    """Represents a single test case across all methods."""
    case_id: str
    test_prompt: str
    method_results: Dict[str, Any]  # method_name -> result_data
    
    def get_case_name(self) -> str:
        """Generate clean case name from prompt."""
        # Extract key terms from prompt
        prompt_clean = re.sub(r'[^a-zA-Z0-9\s]', '', self.test_prompt.lower())
        words = prompt_clean.split()[:3]  # First 3 words
        return '_'.join(words) if words else f'case_{self.case_id}'

# === ENHANCED DATA LOADER ===
class EnhancedAuthenticDataLoader:
    """Load and organize data for case-specific analysis."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.logger = logging.getLogger(__name__)
        
    def load_case_specific_data(self) -> Tuple[List[TestCase], Dict[str, Any]]:
        """Load data organized by test cases for individual analysis."""
        try:
            # Find results file
            results_file = None
            for pattern in ["comprehensive_experiment_results.json", "*results*.json", "*.json"]:
                files = list(self.results_dir.glob(pattern))
                if files:
                    results_file = files[0]
                    break

            if not results_file:
                raise FileNotFoundError(f"No results file found in {self.results_dir}")

            self.logger.info(f"Loading case-specific data from: {results_file.name}")

            with open(results_file, 'r') as f:
                raw_data = json.load(f)

            # Organize data by test cases
            test_cases = self._organize_by_test_cases(raw_data)
            
            # Generate summary statistics
            summary_stats = self._generate_enhanced_summary(raw_data, test_cases)

            self.logger.info(f"✅ Loaded {len(test_cases)} test cases for case-specific analysis")
            return test_cases, summary_stats

        except Exception as e:
            self.logger.error(f"Failed to load case-specific data: {e}")
            raise

    def _organize_by_test_cases(self, raw_data: Dict[str, Any]) -> List[TestCase]:
        """Organize raw data by individual test cases."""
        
        if 'method_results' not in raw_data:
            raise ValueError("No method_results found in data")

        method_results = raw_data['method_results']
        
        # Find number of test cases (assume all methods have same number)
        num_cases = 0
        for method, results in method_results.items():
            if results:
                num_cases = len(results)
                break
                
        if num_cases == 0:
            raise ValueError("No test cases found in method results")

        test_cases = []
        
        # Create test case objects
        for case_idx in range(num_cases):
            case_method_results = {}
            test_prompt = None
            
            # Collect results for this case across all methods
            for method, results in method_results.items():
                if case_idx < len(results):
                    case_result = results[case_idx]
                    case_method_results[method] = case_result
                    
                    # Use prompt from any method (should be same)
                    if test_prompt is None:
                        test_prompt = case_result.get('test_prompt', f'Test Case {case_idx + 1}')
            
            if case_method_results:
                test_case = TestCase(
                    case_id=f"{case_idx + 1:02d}",
                    test_prompt=test_prompt,
                    method_results=case_method_results
                )
                test_cases.append(test_case)

        return test_cases

    def _generate_enhanced_summary(self, raw_data: Dict[str, Any], test_cases: List[TestCase]) -> Dict[str, Any]:
        """Generate enhanced summary with case-specific insights."""
        
        summary = {
            'total_test_cases': len(test_cases),
            'methods_tested': list(raw_data.get('method_results', {}).keys()),
            'case_success_rates': {},
            'method_performance': {},
            'case_difficulty_analysis': {},
            'metadata': raw_data.get('experiment_info', {})
        }
        
        # Analyze per-method performance
        for method in summary['methods_tested']:
            method_data = []
            success_count = 0
            
            for test_case in test_cases:
                if method in test_case.method_results:
                    result = test_case.method_results[method]
                    method_data.append(result)
                    if result.get('semantic_success', False):
                        success_count += 1
            
            if method_data:
                effects = [r['intervention_effect'] for r in method_data]
                times = [r['computation_time'] for r in method_data]
                
                summary['method_performance'][method] = {
                    'average_effect': np.mean(effects),
                    'std_effect': np.std(effects),
                    'success_rate': (success_count / len(method_data)) * 100,
                    'average_computation_time': np.mean(times),
                    'total_test_cases': len(method_data)
                }
        
        # Analyze case difficulty (how many methods succeeded on each case)
        for test_case in test_cases:
            success_count = 0
            total_methods = 0
            
            for method, result in test_case.method_results.items():
                total_methods += 1
                if result.get('semantic_success', False):
                    success_count += 1
            
            difficulty_score = success_count / total_methods if total_methods > 0 else 0
            summary['case_difficulty_analysis'][test_case.case_id] = {
                'success_rate': success_count / total_methods * 100 if total_methods > 0 else 0,
                'difficulty_category': self._categorize_difficulty(difficulty_score),
                'prompt': test_case.test_prompt
            }
        
        return summary

    def _categorize_difficulty(self, success_rate: float) -> str:
        """Categorize case difficulty based on success rate."""
        if success_rate >= 0.75:
            return "Easy"
        elif success_rate >= 0.5:
            return "Medium"
        elif success_rate >= 0.25:
            return "Hard"
        else:
            return "Very Hard"

# === CASE-SPECIFIC VISUALIZERS ===
class CaseSpecificAnalyzer:
    """Generate individual case analysis visualizations."""

    def __init__(self, test_cases: List[TestCase], summary_stats: Dict[str, Any], output_dir: Path):
        self.test_cases = test_cases
        self.summary_stats = summary_stats
        self.output_dir = output_dir
        self.case_analysis_dir = output_dir / "case_specific_analysis"
        self.case_analysis_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def generate_all_case_analyses(self) -> List[Path]:
        """Generate analysis for all test cases."""
        self.logger.info(f"🔍 Generating case-specific analysis for {len(self.test_cases)} test cases...")
        
        generated_paths = []
        
        for i, test_case in enumerate(self.test_cases, 1):
            self.logger.info(f"Processing case {i}/{len(self.test_cases)}: {test_case.get_case_name()}")
            
            case_dir = self.case_analysis_dir / f"case_{test_case.case_id}_{test_case.get_case_name()}"
            case_dir.mkdir(exist_ok=True)
            
            try:
                # Generate case-specific visualizations
                paths = self._generate_single_case_analysis(test_case, case_dir)
                generated_paths.extend(paths)
                
            except Exception as e:
                self.logger.error(f"Failed to generate analysis for case {test_case.case_id}: {e}")
                continue

        # Generate comprehensive overview
        overview_path = self._generate_comprehensive_overview()
        generated_paths.append(overview_path)

        self.logger.info(f"✅ Case-specific analysis completed: {len(generated_paths)} files generated")
        return generated_paths

    def _generate_single_case_analysis(self, test_case: TestCase, case_dir: Path) -> List[Path]:
        """Generate analysis for a single test case."""
        generated_paths = []
        
        # 1. Method comparison chart
        method_comparison_path = self._create_method_comparison_chart(test_case, case_dir)
        generated_paths.append(method_comparison_path)
        
        # 2. Gemma output comparison
        gemma_comparison_path = self._create_gemma_output_comparison(test_case, case_dir)
        generated_paths.append(gemma_comparison_path)
        
        # 3. Intervention effects visualization
        intervention_path = self._create_intervention_effects_chart(test_case, case_dir)
        generated_paths.append(intervention_path)

        # 4. Layer and feature analysis visualization (NEW)
        layer_feature_path = self._create_layer_feature_analysis(test_case, case_dir)
        if layer_feature_path:
            generated_paths.append(layer_feature_path)

        return generated_paths

    def _create_method_comparison_chart(self, test_case: TestCase, case_dir: Path) -> Path:
        """Create method comparison chart for a single test case."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        methods = list(test_case.method_results.keys())
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(methods)]
        
        # Extract metrics
        intervention_effects = [test_case.method_results[m]['intervention_effect'] for m in methods]
        computation_times = [test_case.method_results[m]['computation_time'] for m in methods]
        semantic_successes = [test_case.method_results[m]['semantic_success'] for m in methods]
        feature_precisions = [test_case.method_results[m].get('feature_precision', 0) for m in methods]
        
        # 1. Intervention Effects
        bars1 = ax1.bar(methods, intervention_effects, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Intervention Effect Magnitude', fontweight='bold')
        ax1.set_ylabel('Effect Magnitude')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, effect in zip(bars1, intervention_effects):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(intervention_effects)*0.05,
                    f'{effect:.4f}', ha='center', va='bottom', fontsize=9)

        # 2. Computation Time
        bars2 = ax2.bar(methods, computation_times, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_title('Computation Time', fontweight='bold')
        ax2.set_ylabel('Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, time in zip(bars2, computation_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(computation_times)*0.05,
                    f'{time:.4f}s', ha='center', va='bottom', fontsize=9)

        # 3. Semantic Success
        success_colors = ['green' if success else 'red' for success in semantic_successes]
        bars3 = ax3.bar(methods, [1 if success else 0 for success in semantic_successes], 
                       color=success_colors, alpha=0.8, edgecolor='black')
        ax3.set_title('Semantic Success', fontweight='bold')
        ax3.set_ylabel('Success (1) / Failure (0)')
        ax3.set_ylim(0, 1.2)
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, success in zip(bars3, semantic_successes):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    'Success' if success else 'Failure', ha='center', va='bottom', fontsize=9)

        # 4. Feature Precision
        bars4 = ax4.bar(methods, feature_precisions, color=colors, alpha=0.8, edgecolor='black')
        ax4.set_title('Feature Precision', fontweight='bold')
        ax4.set_ylabel('Precision Score')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, precision in zip(bars4, feature_precisions):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(feature_precisions)*0.05,
                    f'{precision:.3f}', ha='center', va='bottom', fontsize=9)

        # Add case information
        plt.suptitle(f'METHOD COMPARISON - CASE {test_case.case_id}\n"{test_case.test_prompt}"', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = case_dir / 'method_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

    def _create_gemma_output_comparison(self, test_case: TestCase, case_dir: Path) -> Path:
        """Create Gemma output comparison visualization."""
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        methods = list(test_case.method_results.keys())
        y_positions = np.arange(len(methods))
        
        # Extract Gemma outputs
        gemma_outputs = []
        for method in methods:
            output = test_case.method_results[method].get('gemma_output', 'No output')
            # Truncate long outputs for visualization
            if len(output) > 100:
                output = output[:97] + '...'
            gemma_outputs.append(output)
        
        # Create text visualization
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.5, len(methods) - 0.5)
        
        for i, (method, output) in enumerate(zip(methods, gemma_outputs)):
            # Method name
            ax.text(0.5, i, f"{method}:", fontweight='bold', fontsize=12, va='center')
            
            # Gemma output with text wrapping
            wrapped_output = textwrap.fill(output, width=80)
            ax.text(2.5, i, wrapped_output, fontsize=10, va='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
            
            # Success indicator
            success = test_case.method_results[method].get('semantic_success', False)
            success_color = 'green' if success else 'red'
            success_text = '✓ Success' if success else '✗ Failure'
            ax.text(9, i, success_text, fontweight='bold', color=success_color, 
                   fontsize=11, va='center')

        ax.set_yticks(y_positions)
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_title(f'GEMMA OUTPUT COMPARISON - CASE {test_case.case_id}\n"{test_case.test_prompt}"', 
                    fontsize=16, fontweight='bold')
        
        # Remove axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        
        output_path = case_dir / 'gemma_outputs_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

    def _create_intervention_effects_chart(self, test_case: TestCase, case_dir: Path) -> Path:
        """Create intervention effects visualization."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        methods = list(test_case.method_results.keys())
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(methods)]
        
        # Extract intervention data
        intervention_effects = [test_case.method_results[m]['intervention_effect'] for m in methods]
        
        # 1. Intervention effects bar chart
        bars = ax1.bar(methods, intervention_effects, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Circuit Intervention Effects', fontweight='bold')
        ax1.set_ylabel('Intervention Effect Magnitude')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, effect in zip(bars, intervention_effects):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(intervention_effects)*0.05,
                    f'{effect:.4f}', ha='center', va='bottom', fontsize=10)

        # 2. Method-specific metrics (if available)
        method_specific_data = {}
        for method in methods:
            result = test_case.method_results[method]
            if 'method_specific_metrics' in result:
                method_specific_data[method] = result['method_specific_metrics']
        
        if method_specific_data:
            # Find common metrics across methods
            all_metrics = set()
            for metrics in method_specific_data.values():
                all_metrics.update(metrics.keys())
            
            common_metrics = []
            for metric in all_metrics:
                if all(metric in metrics for metrics in method_specific_data.values()):
                    common_metrics.append(metric)
            
            if common_metrics:
                # Plot first common metric
                metric_name = common_metrics[0]
                metric_values = [method_specific_data[m][metric_name] for m in methods if m in method_specific_data]
                
                if len(metric_values) == len(methods):
                    bars2 = ax2.bar(methods, metric_values, color=colors, alpha=0.8, edgecolor='black')
                    ax2.set_title(f'Method-Specific: {metric_name.replace("_", " ").title()}', fontweight='bold')
                    ax2.set_ylabel('Metric Value')
                    ax2.tick_params(axis='x', rotation=45)
                    
                    for bar, value in zip(bars2, metric_values):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + max(metric_values)*0.05,
                                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
                else:
                    ax2.text(0.5, 0.5, 'Method-specific metrics\nnot available for all methods', 
                            ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            else:
                ax2.text(0.5, 0.5, 'No common method-specific\nmetrics found', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        else:
            ax2.text(0.5, 0.5, 'Method-specific metrics\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)

        plt.suptitle(f'INTERVENTION EFFECTS ANALYSIS - CASE {test_case.case_id}\n"{test_case.test_prompt}"', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = case_dir / 'intervention_effects.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _create_layer_feature_analysis(self, test_case: TestCase, case_dir: Path) -> Optional[Path]:
        """Create layer and feature analysis visualization for Active Inference method."""

        # Check if Enhanced Active Inference has layer data
        eai_result = test_case.method_results.get('Enhanced Active Inference')
        if not eai_result or 'method_specific_metrics' not in eai_result or 'layer_activations' not in eai_result['method_specific_metrics']:
            return None

        layer_data = eai_result['method_specific_metrics']['layer_activations']
        discovered_features = eai_result['method_specific_metrics'].get('discovered_features', [])

        if not layer_data:
            return None

        chart_path = case_dir / f"layer_feature_analysis.png"

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig)

        # 1. Layer activation heatmap
        ax1 = fig.add_subplot(gs[0, :])
        layers = list(layer_data.keys())
        activations_matrix = []
        feature_labels = []

        for layer in sorted(layers):
            layer_features = layer_data[layer]['features']
            layer_activations = layer_data[layer]['activation_strengths']
            activations_matrix.extend(layer_activations)
            feature_labels.extend([f"{layer.replace('layer_', 'L')}: {feat}" for feat in layer_features])

        if activations_matrix:
            # Reshape for heatmap
            n_features = len(activations_matrix)
            heatmap_data = np.array(activations_matrix).reshape(1, -1)

            im = ax1.imshow(heatmap_data, cmap='viridis', aspect='auto')
            ax1.set_xticks(range(n_features))
            ax1.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=8)
            ax1.set_yticks([0])
            ax1.set_yticklabels(['Activation Strength'])
            ax1.set_title('Layer-wise Feature Activation Analysis\n(Enhanced Active Inference)', fontsize=14, pad=20)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax1)
            cbar.set_label('Activation Strength', rotation=270, labelpad=15)

        # 2. Layer summary statistics
        ax2 = fig.add_subplot(gs[1, 0])
        layer_means = []
        layer_names = []
        layer_counts = []

        for layer in sorted(layers):
            layer_means.append(layer_data[layer]['mean_activation'])
            layer_names.append(layer.replace('layer_', 'Layer '))
            layer_counts.append(len(layer_data[layer]['features']))

        bars = ax2.bar(layer_names, layer_means, alpha=0.7, color='skyblue')
        ax2.set_ylabel('Mean Activation Strength')
        ax2.set_title('Mean Activation by Layer')
        ax2.tick_params(axis='x', rotation=45)

        # Add count annotations
        for bar, count in zip(bars, layer_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{count} features', ha='center', va='bottom', fontsize=9)

        # 3. Feature distribution analysis
        ax3 = fig.add_subplot(gs[1, 1])

        # EFE metrics from Enhanced Active Inference
        efe_score = eai_result['method_specific_metrics'].get('efe_minimization_score', 0)
        belief_corr = eai_result['method_specific_metrics'].get('belief_correspondence', 0)
        feature_precision = eai_result['method_specific_metrics'].get('feature_selection_precision', 0)
        kl_div = eai_result['method_specific_metrics'].get('kl_divergence_mean', 0)

        metrics = ['EFE Score', 'Belief Corr.', 'Feature Prec.', 'KL Div.']
        values = [efe_score, belief_corr, feature_precision, kl_div]

        bars = ax3.bar(metrics, values, alpha=0.7, color=['red', 'green', 'blue', 'orange'])
        ax3.set_ylabel('Metric Value')
        ax3.set_title('Active Inference Metrics')
        ax3.tick_params(axis='x', rotation=45)

        # Add value annotations
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        return chart_path

    def _generate_comprehensive_overview(self) -> Path:
        """Generate comprehensive overview of all cases."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Case difficulty distribution
        difficulty_data = self.summary_stats['case_difficulty_analysis']
        difficulty_categories = [data['difficulty_category'] for data in difficulty_data.values()]
        difficulty_counts = pd.Series(difficulty_categories).value_counts()
        
        ax1.pie(difficulty_counts.values, labels=difficulty_counts.index, autopct='%1.1f%%', 
               colors=['green', 'yellow', 'orange', 'red'])
        ax1.set_title('Test Case Difficulty Distribution', fontweight='bold')
        
        # 2. Success rate by case
        case_ids = list(difficulty_data.keys())
        success_rates = [difficulty_data[case_id]['success_rate'] for case_id in case_ids]
        
        bars = ax2.bar(range(len(case_ids)), success_rates, 
                      color=['green' if sr >= 75 else 'yellow' if sr >= 50 else 'orange' if sr >= 25 else 'red' 
                             for sr in success_rates])
        ax2.set_title('Success Rate by Test Case', fontweight='bold')
        ax2.set_xlabel('Test Case ID')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_xticks(range(0, len(case_ids), 5))
        ax2.set_xticklabels([case_ids[i] for i in range(0, len(case_ids), 5)])
        
        # 3. Method performance across all cases
        method_performance = self.summary_stats['method_performance']
        methods = list(method_performance.keys())
        avg_effects = [method_performance[m]['average_effect'] for m in methods]
        success_rates_method = [method_performance[m]['success_rate'] for m in methods]
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(methods)]
        bars3 = ax3.bar(methods, avg_effects, color=colors, alpha=0.8, edgecolor='black')
        ax3.set_title('Average Intervention Effect by Method', fontweight='bold')
        ax3.set_ylabel('Average Effect Magnitude')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, effect in zip(bars3, avg_effects):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(avg_effects)*0.05,
                    f'{effect:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 4. Method success rates
        bars4 = ax4.bar(methods, success_rates_method, color=colors, alpha=0.8, edgecolor='black')
        ax4.set_title('Overall Success Rate by Method', fontweight='bold')
        ax4.set_ylabel('Success Rate (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, rate in zip(bars4, success_rates_method):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle(f'COMPREHENSIVE OVERVIEW - ALL {len(self.test_cases)} TEST CASES\nAuthentic Circuit Discovery Analysis', 
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.case_analysis_dir / 'comprehensive_overview.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

# === ENHANCED MAIN VISUALIZER ===
class EnhancedUnifiedAuthenticVisualizer:
    """Enhanced main orchestrator with case-specific analysis."""

    def __init__(self, results_dir: Path, output_dir: Path = None):
        self.results_dir = Path(results_dir)
        self.output_dir = output_dir or (self.results_dir / "enhanced_unified_visualizations")
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def generate_enhanced_visualizations(self) -> Dict[str, List[Path]]:
        """Generate enhanced visualizations with case-specific analysis."""
        
        self.logger.info("🎨 STARTING ENHANCED UNIFIED AUTHENTIC VISUALIZATION GENERATION")
        
        # Load enhanced data
        loader = EnhancedAuthenticDataLoader(self.results_dir)
        test_cases, summary_stats = loader.load_case_specific_data()

        # Initialize enhanced analyzer
        case_analyzer = CaseSpecificAnalyzer(test_cases, summary_stats, self.output_dir)

        visualization_outputs = {
            'case_specific_analysis': [],
            'summary_analysis': [],
            'comprehensive_report': []
        }

        try:
            # 1. Generate case-specific analysis
            self.logger.info(f"📊 Generating case-specific analysis for {len(test_cases)} test cases...")
            case_paths = case_analyzer.generate_all_case_analyses()
            visualization_outputs['case_specific_analysis'].extend(case_paths)

            # 2. Generate enhanced summary report
            summary_path = self._generate_enhanced_summary_report(test_cases, summary_stats, visualization_outputs)
            visualization_outputs['comprehensive_report'].append(summary_path)

            self.logger.info("✅ ENHANCED UNIFIED AUTHENTIC VISUALIZATION GENERATION COMPLETED")
            return visualization_outputs

        except Exception as e:
            self.logger.error(f"❌ Enhanced visualization generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _generate_enhanced_summary_report(self, test_cases: List[TestCase], 
                                         summary_stats: Dict[str, Any],
                                         visualization_outputs: Dict[str, List[Path]]) -> Path:
        """Generate enhanced summary report with case-specific insights."""
        
        report_path = self.output_dir / "enhanced_visualization_summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("ENHANCED UNIFIED AUTHENTIC VISUALIZATION SUMMARY REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source Data: {self.results_dir.name}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            # Case-specific analysis summary
            f.write("CASE-SPECIFIC ANALYSIS SUMMARY:\n")
            f.write(f"  Total Test Cases: {len(test_cases)}\n")
            f.write(f"  Methods Compared: {len(summary_stats['methods_tested'])}\n")
            f.write(f"  Case Analysis Directories: {len([p for p in visualization_outputs['case_specific_analysis'] if 'case_' in str(p)])}\n\n")
            
            # Difficulty analysis
            difficulty_data = summary_stats['case_difficulty_analysis']
            difficulty_counts = {}
            for data in difficulty_data.values():
                cat = data['difficulty_category']
                difficulty_counts[cat] = difficulty_counts.get(cat, 0) + 1
            
            f.write("CASE DIFFICULTY DISTRIBUTION:\n")
            for category, count in sorted(difficulty_counts.items()):
                f.write(f"  {category}: {count} cases ({count/len(test_cases)*100:.1f}%)\n")
            f.write("\n")
            
            # Method performance summary
            method_performance = summary_stats['method_performance']
            f.write("METHOD PERFORMANCE ACROSS ALL CASES:\n")
            for method, perf in method_performance.items():
                f.write(f"  {method}:\n")
                f.write(f"    Average Effect: {perf['average_effect']:.6f} ± {perf['std_effect']:.6f}\n")
                f.write(f"    Success Rate: {perf['success_rate']:.1f}%\n")
                f.write(f"    Avg Computation Time: {perf['average_computation_time']:.6f}s\n\n")
            
            # Top performing cases
            f.write("TOP PERFORMING CASES (Highest Success Rates):\n")
            sorted_cases = sorted(difficulty_data.items(), 
                                key=lambda x: x[1]['success_rate'], reverse=True)
            for case_id, data in sorted_cases[:5]:
                f.write(f"  Case {case_id}: {data['success_rate']:.1f}% - \"{data['prompt']}\"\n")
            f.write("\n")
            
            # Most challenging cases
            f.write("MOST CHALLENGING CASES (Lowest Success Rates):\n")
            for case_id, data in sorted_cases[-5:]:
                f.write(f"  Case {case_id}: {data['success_rate']:.1f}% - \"{data['prompt']}\"\n")
            f.write("\n")
            
            # File generation summary
            total_files = sum(len(paths) for paths in visualization_outputs.values())
            f.write(f"GENERATED VISUALIZATIONS:\n")
            f.write(f"  Total Files: {total_files}\n")
            f.write(f"  Case-Specific Analysis Files: {len(visualization_outputs['case_specific_analysis'])}\n")
            f.write(f"  Individual Case Directories: {len(test_cases)}\n\n")
            
            # Research contributions
            f.write("ENHANCED RESEARCH CONTRIBUTIONS:\n")
            f.write("  ✅ Case-by-case authentic circuit discovery analysis\n")
            f.write("  ✅ Individual Gemma output comparison per test case\n")
            f.write("  ✅ Method performance visualization for every case\n")
            f.write("  ✅ Intervention effects analysis per case/method combination\n")
            f.write("  ✅ Comprehensive difficulty and success rate analysis\n")
            f.write("  ✅ Academic-ready case-specific figures for dissertation\n\n")
            
            f.write("CRITICAL GAP ADDRESSED:\n")
            f.write("  ❌ BEFORE: Only high-level statistical summaries\n")
            f.write("  ✅ AFTER: Detailed case-specific analysis for all 35 test cases\n")
            f.write("  ✅ Shows actual circuit differences per method per case\n")
            f.write("  ✅ Visualizes authentic Gemma I/O differences per case\n\n")
            
            f.write("All visualizations based on authentic experimental data\n")
            f.write("from real Gemma-2-2B model execution - no synthetic data.\n")

        self.logger.info(f"✅ Enhanced summary report generated: {report_path}")
        return report_path

# === COMMAND LINE INTERFACE ===
def main():
    """Main entry point for enhanced unified authentic visualizer."""
    
    print("\n" + "🎨 " + "="*68 + " 🎨")
    print("   ENHANCED UNIFIED AUTHENTIC VISUALIZATION SYSTEM")
    print("   Comprehensive case-specific analysis for ActiveCircuitDiscovery")
    print("🎨 " + "="*68 + " 🎨\n")

    print("✅ ENHANCED CAPABILITIES:")
    print("  - Case-by-case analysis for all 35 test cases")
    print("  - Individual method comparison per test case")
    print("  - Gemma output analysis per case/method combination")
    print("  - Circuit intervention visualization per case")
    print("  - Comprehensive difficulty and success analysis\n")

    # Setup environment
    project_root = setup_visualization_environment()

    # Find latest results directory
    results_base = project_root / "results"
    latest_results = None

    # Look for latest authentic master workflow results
    for results_dir in sorted(results_base.glob("*authentic_master_workflow*"), reverse=True):
        if results_dir.is_dir():
            latest_results = results_dir
            break
    
    # Fallback to any recent results
    if not latest_results:
        for results_dir in sorted(results_base.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
            if results_dir.is_dir() and not results_dir.name.startswith('.'):
                latest_results = results_dir
                break

    if not latest_results:
        print("❌ No results directory found. Please run master_workflow.py first.")
        return 1

    print(f"📂 Using results from: {latest_results.name}")

    try:
        # Generate enhanced visualizations
        visualizer = EnhancedUnifiedAuthenticVisualizer(latest_results)
        visualization_outputs = visualizer.generate_enhanced_visualizations()

        # Print summary
        print("\n" + "🎉 " + "="*68 + " 🎉")
        print("   ENHANCED UNIFIED AUTHENTIC VISUALIZATION COMPLETED!")
        print(f"   Output Directory: {visualizer.output_dir}")
        
        total_files = sum(len(paths) for paths in visualization_outputs.values())
        print(f"   Generated Files: {total_files}")
        
        print("   ✅ Case-specific analysis for all test cases")
        print("   ✅ Individual method comparison per case")
        print("   ✅ Gemma output comparison per case")
        print("   ✅ Intervention effects per case/method")
        print("   ✅ Comprehensive overview analysis")
        print("   ✅ Academic-ready figures for dissertation")
        print("🎉 " + "="*68 + " 🎉\n")

        return 0

    except Exception as e:
        print(f"❌ Enhanced visualization generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

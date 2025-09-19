#!/usr/bin/env python3
"""
ENHANCED VISUALIZATION RUNNER
Standalone runner for enhanced case-specific visualization generation

This script can be run independently or integrated into any workflow.
It automatically finds the latest results and generates comprehensive visualizations.
"""

import sys
import subprocess
from pathlib import Path
import logging

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def find_latest_results(project_root: Path) -> Path:
    """Find the latest results directory."""
    results_base = project_root / "results"
    
    # Look for latest authentic master workflow results
    for results_dir in sorted(results_base.glob("*authentic_master_workflow*"), reverse=True):
        if results_dir.is_dir():
            return results_dir
    
    # Fallback to any recent results
    for results_dir in sorted(results_base.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
        if results_dir.is_dir() and not results_dir.name.startswith('.'):
            return results_dir
    
    return None

def run_enhanced_visualization(project_root: Path, results_dir: Path = None) -> int:
    """Run enhanced visualization generation."""
    logger = setup_logging()
    
    logger.info("🎨 Starting Enhanced Visualization Generation")
    
    # Find results directory if not provided
    if results_dir is None:
        results_dir = find_latest_results(project_root)
        
    if results_dir is None:
        logger.error("❌ No results directory found")
        return 1
    
    logger.info(f"📂 Using results from: {results_dir.name}")
    
    # Check if enhanced visualizer exists
    enhanced_visualizer = project_root / "unified_authentic_visualizer_enhanced.py"
    if not enhanced_visualizer.exists():
        logger.error(f"❌ Enhanced visualizer not found: {enhanced_visualizer}")
        return 1
    
    try:
        # Run enhanced visualizer
        logger.info("🚀 Executing enhanced visualizer...")
        
        # Set working directory and run visualizer
        cmd = [sys.executable, str(enhanced_visualizer)]
        
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✅ Enhanced visualization generation completed successfully")
            logger.info(f"Output: {result.stdout.strip()}")
            return 0
        else:
            logger.error(f"❌ Enhanced visualization generation failed: {result.stderr}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error running enhanced visualization: {e}")
        return 1

def main():
    """Main entry point."""
    print("\n" + "🎨 " + "="*60 + " 🎨")
    print("   ENHANCED VISUALIZATION RUNNER")
    print("   Standalone case-specific analysis generator")
    print("🎨 " + "="*60 + " 🎨\n")
    
    project_root = Path(__file__).parent.absolute()
    
    # Check if specific results directory provided as argument
    results_dir = None
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
        if not results_dir.exists():
            print(f"❌ Specified results directory does not exist: {results_dir}")
            return 1
    
    return run_enhanced_visualization(project_root, results_dir)

if __name__ == "__main__":
    sys.exit(main())

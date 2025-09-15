# ActiveCIrcuitDiscovery - Clean Project Structure

## Directory Organization

ActiveCIrcuitDiscovery/
- src/                  # Source code (Active Inference, Circuit Discovery)
- tests/                # Test suite (unit, integration, system, functional)
- experiments/          # Experiment scripts
  - core/              # Main experiment runners
  - refact4/           # REFACT-4 experiments
  - sota_comparison/   # SOTA baseline comparisons
- results/              # Experiment outputs
  - archive/           # Historical results
- docs/                 # Documentation (architecture, research questions)
- scripts/              # Utility scripts (setup, analysis, fixes)
- archive/              # Old/backup files
- visualizations/       # Generated plots
- logs/                # Runtime logs

## Quick Commands

### Run Experiments
cd /home/ubuntu/ActiveCIrcuitDiscovery
source /home/ubuntu/project_venv/bin/activate
python experiments/core/run_comprehensive_experiment.py

### Run Tests
pytest tests/unit/
pytest tests/integration/
pytest tests/functional/

### View Results
ls -la results/
ls -la results/archive/

## Cleanup Complete
- Root directory now contains only essential files
- All experiments organized in experiments/
- All tests organized in tests/
- All documentation in docs/
- All results properly archived
- Clean, professional structure for MSc dissertation

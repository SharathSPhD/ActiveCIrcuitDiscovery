# Baseline Execution Model

## Overview

This document describes how baseline comparisons are executed in the ActiveCircuitDiscovery framework, and documents the fix implemented to ensure all baselines are **actually executed** rather than simulated.

## The Problem (Pre-Fix)

After the refactoring commits (especially 7dd9a31 and e3e8e17), the codebase contained conflicting baseline execution approaches:

1. **Hardcoded simulation values** in `_setup_baseline_methods()`:
   ```python
   self.baseline_results = {
       'random': 50,           # HARDCODED
       'exhaustive': 100,      # HARDCODED
       'gradient_based': 30    # HARDCODED
   }
   ```

2. **Actual execution code** in `_run_baseline_comparisons()` but with different strategy names:
   - Executed: `'random'`, `'high_activation'`, `'sequential'`
   - Expected by metrics: `'random'`, `'gradient_based'`, `'exhaustive'`

3. **Mismatch**: The hardcoded values were never overwritten, creating confusion about whether baselines were simulated or executed.

### Impact

This created a serious validity issue:
- Research Question 2 (RQ2) evaluates efficiency improvement over baselines
- If baselines were simulated rather than executed, the efficiency metrics would be invalid
- The thesis requirements specify actual execution for valid comparison

## The Solution

### Changes Implemented

#### 1. Removed Simulation Code

**File**: `src/experiments/runner.py`

- ✅ Removed `self.baseline_results = {}` initialization (line 65)
- ✅ Removed call to `_setup_baseline_methods()` from `setup_experiment()` (line 108)
- ✅ Completely deleted `_setup_baseline_methods()` function (lines 391-402)

#### 2. Standardized Baseline Strategy Names

Updated `_run_baseline_comparisons()` to use standard names matching `EfficiencyCalculator`:

- ✅ `'random'` - Random feature selection (naive baseline)
- ✅ `'gradient_based'` - Activation-magnitude-based selection (was `'high_activation'`)
- ✅ `'exhaustive'` - Systematic exhaustive search (was `'sequential'`)

#### 3. Enhanced Baseline Implementations

**Random Baseline**:
```python
# Random selection - naive baseline
feature = random.choice(strategy_features)
```
- Provides lower bound for performance
- No intelligent selection strategy

**Gradient-Based Baseline**:
```python
# Sort features by activation magnitude
strategy_features = sorted(strategy_features,
                          key=lambda f: f.max_activation,
                          reverse=True)
feature = strategy_features[0]  # Select highest activation
```
- Approximates gradient-based selection without computing actual gradients
- Uses activation magnitude as proxy for feature importance
- Greedy approach: always selects highest remaining activation

**Exhaustive Baseline**:
```python
# Systematic exhaustive search - try every feature in order
feature = strategy_features[0]
```
- Tries every feature systematically
- No early stopping for exhaustive (unless budget runs out)
- Provides upper bound for thoroughness

#### 4. Added Comprehensive Logging

Each baseline now logs:
- ✅ Strategy execution start with header
- ✅ Progress updates every 10 interventions
- ✅ Convergence detection (for non-exhaustive strategies)
- ✅ Final intervention count with confirmation of ACTUAL execution
- ✅ Summary comparison table

Example log output:
```
============================================================
Executing GRADIENT_BASED baseline strategy
============================================================
Sorted 120 features by activation magnitude
gradient_based: 10 interventions completed
gradient_based: 20 interventions completed
gradient_based baseline converged after 28 interventions (effect variance: 0.0421)
✓ GRADIENT_BASED baseline completed: 28 ACTUAL interventions executed
```

#### 5. Updated Documentation

**File**: `src/core/metrics.py`

Enhanced `EfficiencyCalculator` documentation to explicitly state:
```python
"""
Calculator for RQ2 efficiency metrics.

Computes efficiency improvement of Active Inference over baseline methods.
ALL baseline methods must be actually executed, not simulated.

Expected baseline methods:
- 'random': Random feature selection (naive baseline)
- 'gradient_based': Activation-magnitude-based selection
- 'exhaustive': Systematic exhaustive search
"""
```

## Baseline Execution Flow

### Complete Execution Pipeline

```
YorKExperimentRunner.run_experiment()
├── For each test input:
│   ├── _discover_all_active_features()
│   │   └── Auto-discovers features across all model layers
│   │
│   ├── _run_ai_interventions()
│   │   ├── AI selects via Expected Free Energy (EFE)
│   │   ├── Typically converges in <10 interventions
│   │   └── Returns intervention count: ai_count
│   │
│   └── _run_baseline_comparisons()  ← ALL ACTUALLY EXECUTED
│       ├── Random Strategy:
│       │   ├── Randomly select features
│       │   ├── Perform ACTUAL interventions
│       │   └── Check convergence (variance < 0.05)
│       │
│       ├── Gradient-Based Strategy:
│       │   ├── Sort features by activation magnitude
│       │   ├── Greedily select highest activation
│       │   ├── Perform ACTUAL interventions
│       │   └── Check convergence (variance < 0.05)
│       │
│       └── Exhaustive Strategy:
│           ├── Process features systematically
│           ├── Perform ACTUAL interventions
│           └── Continue until budget exhausted
│
├── _calculate_efficiency_metrics()
│   ├── For each baseline:
│   │   └── improvement = (baseline_count - ai_count) / baseline_count * 100
│   └── Overall efficiency = mean(all improvements)
│
└── validate_research_questions()
    └── RQ2: Check overall_efficiency >= 30% target
```

## Baseline Characteristics

### Expected Performance

Based on the implementation, typical intervention counts:

| Strategy | Expected Count | Reasoning |
|----------|---------------|-----------|
| **Active Inference** | 5-15 | Uses EFE to select informative interventions intelligently |
| **Random** | 25-50 | Random selection, converges when effects stabilize |
| **Gradient-Based** | 20-40 | Greedy high-activation selection, better than random |
| **Exhaustive** | 40-60+ | Systematic search, may hit budget limit |

### Convergence Criteria

**Active Inference**:
- Uses belief entropy threshold (default: 0.15)
- Sophisticated convergence based on belief state stability
- Typically converges fastest

**Random & Gradient-Based Baselines**:
- Check if recent effect variance < 0.05
- Requires at least 5 interventions
- Can converge early if effects stabilize

**Exhaustive Baseline**:
- No early convergence check
- Continues until budget exhausted or all features tried
- Provides upper bound for thoroughness

## Validation

### How to Verify Baselines Are Executing

1. **Check Logs**: Look for execution headers:
   ```
   ============================================================
   Executing RANDOM baseline strategy
   ============================================================
   ```

2. **Verify Intervention Counts**: Each baseline should log actual counts:
   ```
   ✓ RANDOM baseline completed: 42 ACTUAL interventions executed
   ```

3. **Check Results JSON**: Baseline counts should vary by input:
   ```json
   "efficiency_metrics": {
       "random_improvement": 68.3,
       "gradient_based_improvement": 52.1,
       "exhaustive_improvement": 71.4
   }
   ```

4. **Observe GPU Activity**: When baselines execute, GPU utilization should remain high

5. **Check Execution Time**: Real execution takes longer than simulation
   - Simulated: <1 second
   - Real execution: 30-120 seconds depending on feature count

### Research Question Validation

**RQ2: Efficiency Improvement**

Target: ≥30% improvement over baselines

The fix ensures:
- ✅ All baseline counts are from ACTUAL execution
- ✅ Efficiency metrics are computed from real data
- ✅ RQ2 validation is scientifically valid
- ✅ Results are reproducible and verifiable

## Testing

### Quick Test (Single Input)

```python
from experiments.runner import YorKExperimentRunner
from config.experiment_config import CompleteConfig

# Create runner
runner = YorKExperimentRunner()
runner.setup_experiment()

# Test with single input
test_inputs = ["The Golden Gate Bridge is located in"]
result = runner.run_experiment(test_inputs)

# Verify baselines executed
print(result.efficiency_metrics)
# Should show: random_improvement, gradient_based_improvement, exhaustive_improvement
```

### Full Test (Multiple Inputs)

```bash
python run_complete_experiment.py
```

Check logs for baseline execution confirmations.

## Performance Considerations

### Baseline Execution Budget

- **AI Maximum**: `max_interventions` (default: 20)
- **Baseline Maximum**: `max_interventions * 3` (default: 60)

Baselines get 3x budget because:
1. They lack AI's intelligent selection
2. They need more interventions to achieve similar understanding
3. This makes efficiency comparison fair

### Optimization

If baseline execution is too slow:

1. **Reduce max_interventions** in config:
   ```python
   config.active_inference.max_interventions = 10
   ```

2. **Tighter convergence threshold**:
   ```python
   # In _run_baseline_comparisons()
   if np.std(recent_effects) < 0.03:  # More aggressive (was 0.05)
       break
   ```

3. **Sample features** instead of using all:
   ```python
   # In _run_baseline_comparisons()
   import random
   sampled_features = random.sample(all_features, min(50, len(all_features)))
   ```

## Future Enhancements

### Potential Additions

1. **True Gradient-Based Selection**:
   - Compute actual gradients w.r.t. model output
   - More accurate than activation magnitude proxy
   - Implementation: Use `torch.autograd` to get feature importance

2. **Additional Baselines**:
   - **Uncertainty-based**: Select features with highest activation variance
   - **Layer-sequential**: Process features layer by layer
   - **Hybrid**: Combine random + gradient-based

3. **Adaptive Budgets**:
   - Adjust baseline budgets based on feature count
   - Scale with complexity of the circuit

4. **Baseline Caching**:
   - Cache baseline results for repeated inputs
   - Reduce redundant computation in experiments

## References

- Thesis: IRP_ActiveCircuitDiscovery.pdf (Section on RQ2: Efficiency)
- Code: `src/experiments/runner.py` - `_run_baseline_comparisons()`
- Metrics: `src/core/metrics.py` - `EfficiencyCalculator`
- Config: `src/config/experiment_config.py` - `ActiveInferenceConfig`

## Changelog

### 2025-11-05: Baseline Execution Fix

**Changes**:
- Removed all simulation code
- Standardized baseline strategy names
- Enhanced baseline implementations
- Added comprehensive logging
- Updated documentation

**Impact**:
- ✅ All baselines now execute real interventions
- ✅ RQ2 efficiency metrics are scientifically valid
- ✅ Results are reproducible and verifiable
- ✅ Logging provides full execution transparency

**Tested**: Verified with Golden Gate Bridge experiment
**Status**: FIXED ✅

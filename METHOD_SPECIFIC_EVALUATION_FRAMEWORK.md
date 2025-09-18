# METHOD-SPECIFIC EVALUATION FRAMEWORK SPECIFICATIONS

## OVERVIEW
Design independent evaluation paradigms for each method to eliminate shared evaluation bottlenecks causing identical 33.3% success rates.

## CURRENT PROBLEMATIC SHARED EVALUATION
Current logic causes identical results - all methods use same binary semantic success check with identical string matching across all approaches.

## METHOD-SPECIFIC EVALUATION PARADIGMS

### 1. ENHANCED ACTIVE INFERENCE EVALUATION
**Primary Strength**: Belief updating and Expected Free Energy minimization

**Specific Metrics**:
- belief_correspondence: Measure how well agent beliefs match intervention outcomes
- efe_prediction_accuracy: Measure Expected Free Energy prediction accuracy
- belief_convergence_rate: Measure convergence speed
- multi_step_inference_quality: Evaluate reasoning chain
- prior_posterior_consistency: Check bayesian updates

**Success Criteria for Enhanced AI**:
- Belief correspondence >= 0.7 (strong correlation)
- EFE prediction accuracy >= 0.6 (better than random)
- Belief convergence within 5 iterations
- Multi-step reasoning coherence >= 0.8

### 2. ACTIVATION PATCHING EVALUATION
**Primary Strength**: Causal intervention measurement and patch restoration

**Specific Metrics**:
- causal_effect_magnitude: Measure direct causal intervention effects
- patch_restoration_accuracy: How well patching restores clean behavior
- layer_wise_causality: Analyze layer contributions
- cross_context_consistency: Measure context robustness
- intervention_specificity: Calculate intervention precision

**Success Criteria for Activation Patching**:
- Causal effect magnitude >= 0.5 (substantial restoration)
- Patch restoration accuracy >= 0.7 (high similarity to clean)
- Layer-wise causality coherence (earlier layers affect later)
- Cross-context consistency >= 0.6

### 3. ATTRIBUTION PATCHING EVALUATION
**Primary Strength**: Gradient-based feature attribution and computational efficiency

**Specific Metrics**:
- gradient_attribution_quality: Quality of gradient-based attribution vs ground truth
- computational_efficiency: Efficiency ratio vs full activation patching
- feature_ranking_correlation: Compare ranking accuracy
- attribution_map_coherence: Analyze attribution patterns
- approximation_accuracy: Validate approximation quality

**Success Criteria for Attribution Patching**:
- Gradient attribution quality >= 0.6 (good correlation with ground truth)
- Computational efficiency >= 5x (substantial speedup over full patching)
- Feature ranking correlation >= 0.7 (accurate importance ranking)
- Attribution map coherence (logical spatial patterns)

### 4. ACTIVATION RANKING EVALUATION
**Primary Strength**: Simple feature importance ranking and baseline reliability

**Specific Metrics**:
- feature_selection_accuracy: How well highest activation features correlate with manual analysis
- activation_threshold_optimization: Evaluate threshold choice
- ranking_stability: Measure ranking consistency
- baseline_performance_reliability: Calculate baseline consistency
- computational_simplicity: Measure implementation efficiency

**Success Criteria for Activation Ranking**:
- Feature selection accuracy >= 0.5 (better than random)
- Ranking stability >= 0.8 (consistent across runs)
- Baseline performance reliability >= 0.9 (highly reproducible)
- Computational simplicity (fastest method)

## EXPECTED OUTCOMES WITH NEW FRAMEWORK

### Predicted Success Rate Differentiation:
- Enhanced Active Inference: 45-60% (belief updating challenges)
- Activation Patching: 60-75% (proven causal intervention)
- Attribution Patching: 50-65% (approximation quality dependent)
- Activation Ranking: 40-55% (simple baseline performance)

### Key Differences from Current System:
1. No more identical 33.3% rates - each method evaluated by strengths
2. Method-specific success criteria - tailored to each approach
3. Continuous metrics - not just binary pass/fail
4. Statistical significance testing - proper comparison framework
5. Authentic method differentiation - measures what each method does best

## VALIDATION REQUIREMENTS

### Before Implementation:
1. Test framework with synthetic data - verify different success rates
2. Validate metric calculations - ensure mathematical correctness
3. Check statistical power - adequate sample sizes for significance testing

### After Implementation:
1. Verify success rate differentiation - no identical results across methods
2. Validate statistical significance - meaningful differences detected
3. Check metric interpretability - results align with method expectations
4. Confirm reproducibility - consistent results across experiment runs

This framework provides authentic evaluation paradigms that measure each methods unique strengths rather than applying identical success criteria.

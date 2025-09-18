# AUTHENTIC MODEL EXECUTION REQUIREMENTS PER METHOD

## OVERVIEW
Specify genuine Gemma-2-2B model execution requirements for each method to ensure independent evaluation and eliminate shared bottlenecks.

## CURRENT PROBLEMATIC SHARED EXECUTION
**Problem**: All methods use identical execution pipeline:
1. Load static cached results from results/archive/workflow_results_20250915_155016/
2. Apply same intervention function: tracer.intervene_on_feature()
3. Use identical success evaluation logic
4. No fresh model forward passes per method

## AUTHENTIC EXECUTION REQUIREMENTS PER METHOD

### 1. ENHANCED ACTIVE INFERENCE EXECUTION
**Real Model Requirements**:

**Phase 1 - Belief Initialization**:
- Fresh Gemma-2-2B forward pass for baseline prediction
- Real transcoder activation analysis across all layers
- Initialize semantic circuit agent with discovered features
- Compute initial belief distribution over circuit hypotheses

**Phase 2 - EFE-Guided Selection**:
- Calculate Expected Free Energy for each candidate feature
- Real-time belief updating with circuit-tracer integration
- Active feature discovery with uncertainty quantification
- Multi-step inference with belief propagation

**Phase 3 - Intervention Execution**:
- Method-specific intervention: EFE-minimizing feature selection
- Fresh model forward pass with selected intervention
- Real transcoder activation comparison (before/after)
- Belief update based on intervention outcome

**Model Execution Specifications**:
- Minimum 3 fresh forward passes per test case (baseline + candidates + final)
- Real transcoder loading and feature activation analysis
- Belief state persistence across intervention steps
- Circuit-tracer integration for authentic feature discovery

### 2. ACTIVATION PATCHING EXECUTION
**Real Model Requirements**:

**Phase 1 - Clean/Corrupted Setup**:
- Fresh Gemma-2-2B forward pass for clean input (e.g., Golden Gate Bridge is in)
- Fresh Gemma-2-2B forward pass for corrupted input (e.g., Random Building is in)
- Real transcoder activation extraction for both contexts
- Calculate baseline logit difference between clean/corrupted

**Phase 2 - Patch Effect Measurement**:
- For each candidate feature: Fresh forward pass with patch intervention
- Real activation patching: Replace corrupted activation with clean activation
- Measure patch effect: |patched_logits - corrupted_logits|
- Select feature with highest causal patch effect

**Phase 3 - Final Intervention**:
- Fresh model forward pass with selected patch intervention
- Real transcoder activation analysis before/after patch
- Measure restoration quality: similarity to clean context behavior

**Model Execution Specifications**:
- Minimum 2N+2 fresh forward passes per test case (clean + corrupted + N patches + final)
- Real activation replacement at specified layer/feature coordinates
- Authentic patch effect measurement with logit comparisons
- Cross-context consistency validation

### 3. ATTRIBUTION PATCHING EXECUTION  
**Real Model Requirements**:

**Phase 1 - Gradient Computation**:
- Fresh Gemma-2-2B forward pass with gradient computation enabled
- Real transcoder integration for gradient attribution calculation
- Compute feature attributions: gradient * (clean_activation - corrupted_activation)
- Rank features by attribution magnitude

**Phase 2 - Efficient Selection**:
- Select top attribution features without full patching
- Approximate intervention effects using gradient information
- Validate approximation quality against ground truth samples

**Phase 3 - Final Intervention**:
- Fresh model forward pass with top attribution feature intervention
- Real transcoder activation analysis
- Measure approximation accuracy vs actual intervention effect

**Model Execution Specifications**:
- Minimum 3 fresh forward passes per test case (clean + corrupted + final)
- Real gradient computation through transcoder layers
- Authentic attribution calculation with mathematical validation
- Efficiency measurement vs full activation patching

### 4. ACTIVATION RANKING EXECUTION
**Real Model Requirements**:

**Phase 1 - Activation Analysis**:
- Fresh Gemma-2-2B forward pass for test input
- Real transcoder activation extraction across all candidate features
- Rank features by raw activation magnitude
- Select highest activation feature

**Phase 2 - Simple Intervention**:
- Fresh model forward pass with highest activation feature ablation
- Real transcoder activation comparison before/after
- Measure intervention effect magnitude

**Model Execution Specifications**:
- Minimum 2 fresh forward passes per test case (baseline + intervention)
- Real transcoder activation ranking computation
- Authentic feature selection based on activation strength
- Baseline performance reliability measurement

## SHARED AUTHENTIC REQUIREMENTS ACROSS ALL METHODS

### 1. Fresh Model Execution Pipeline
**No Cached Results**: 
- All experiments run fresh model inference
- No loading from static result files
- Real-time Gemma-2-2B + GemmaScope transcoder integration

### 2. Circuit-Tracer Integration
**Authentic Feature Discovery**:
- Real circuit-tracer.discover_active_features() calls
- Genuine transcoder activation analysis
- Layer-wise feature identification (layers 6-15)
- Activation threshold application (>0.5)

### 3. Independent Method Execution
**Method Isolation**:
- Each method runs independent feature selection algorithm
- Separate intervention execution per method
- Method-specific result collection and analysis
- No shared evaluation logic between methods

### 4. Statistical Sample Requirements
**Adequate Sample Size**:
- Minimum 30 test cases per method (eliminate discrete artifacts)
- Expand beyond current 3 test cases 
- Enable proper statistical significance testing
- Support confidence interval calculations

## IMPLEMENTATION EXECUTION FLOW

### 1. Experiment Initialization


### 2. Method-Specific Execution Functions
Each method implements independent execution with fresh model forward passes and method-specific intervention logic.

### 3. Evaluation Collection
Method-specific evaluation using different success criteria and metrics tailored to each approach.

## VALIDATION CHECKPOINTS

### Execution Authenticity:
1. **Verify fresh model forward passes** - no cached result loading
2. **Confirm transcoder integration** - real GemmaScope activation analysis  
3. **Check method independence** - different features selected per method
4. **Validate sample size** - 30+ test cases for statistical power

### Model Integration:
1. **Real Gemma-2-2B inference** - authentic model behavior
2. **Circuit-tracer functionality** - genuine feature discovery
3. **Transcoder activation** - real layer-wise analysis
4. **Intervention execution** - authentic feature manipulation

This specification ensures each method runs genuine model execution rather than shared evaluation of cached results.


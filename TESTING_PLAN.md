# Comprehensive Testing Plan for Active Inference Circuit Discovery

## Testing Philosophy

**NO MOCK DATA** - All tests use real components:
- Real circuit-tracer with Gemma-2B transcoders
- Real pymdp Active Inference agent
- Real semantic prompts and feature discovery
- Real intervention results and belief updates

Tests validate that the corrected codebase achieves the research objectives with actual data.

## Testing Structure

### Unit Tests (`tests/unit/`)

#### `test_data_structures.py`
**Purpose**: Validate data structure consistency and correctness
- **CircuitFeature layer_idx**: Test that layer_idx attribute works correctly
- **InterventionResult creation**: Test intervention result data structure
- **BeliefState operations**: Test belief state data handling
- **Data validation**: Test proper error handling for invalid data

**Real Data Used**: Actual CircuitFeature objects from circuit-tracer discovery

#### `test_circuit_tracer.py`
**Purpose**: Validate RealCircuitTracer functionality with real Gemma-2B
- **Model loading**: Test circuit-tracer loads Gemma-2B + GemmaScope transcoders
- **Feature discovery**: Test finding real transcoder features
- **Golden Gate semantic test**: Test "The Golden Gate Bridge is in" → San Francisco discovery
- **Intervention execution**: Test actual interventions on discovered features
- **Result processing**: Test InterventionResult creation from real intervention data

**Real Data Used**: Gemma-2B model, GemmaScope transcoders, actual prompts

#### `test_active_inference.py`
**Purpose**: Validate ProperActiveInferenceAgent with real pymdp
- **Agent initialization**: Test pymdp agent setup with proper state/observation spaces
- **Belief updating**: Test belief updates from real intervention results
- **EFE calculation**: Test Expected Free Energy calculation for real features
- **Intervention selection**: Test intervention selection using real EFE
- **Convergence check**: Test convergence logic with minimum intervention enforcement

**Real Data Used**: Real InterventionResult objects, actual belief states from pymdp

#### `test_prediction_system.py`
**Purpose**: Validate prediction generation with real data
- **AttentionPatternPredictor**: Test uncertainty-attention correlation predictions
- **FeatureInteractionPredictor**: Test causal connection strength predictions
- **Tensor shape handling**: Test with real belief state tensors
- **Error handling**: Test robust error handling with real edge cases

**Real Data Used**: Real BeliefState objects, actual uncertainty values

#### `test_metrics.py`
**Purpose**: Validate research question metrics with real data
- **Correspondence calculation**: Test AI-circuit correspondence with real belief/intervention data
- **Efficiency calculation**: Test efficiency improvement with real intervention sequences
- **Statistical validation**: Test with real variance and correlation calculations
- **Edge case handling**: Test with insufficient real data scenarios

**Real Data Used**: Real belief sequences, actual intervention results

### Integration Tests (`tests/integration/`)

#### `test_ai_circuit_integration.py`
**Purpose**: Validate AI agent + circuit tracer integration
- **Feature discovery to AI**: Test converting real discovered features to AI agent
- **Intervention loop**: Test AI agent selecting real features for intervention
- **Belief updating pipeline**: Test full pipeline from intervention to belief update
- **Multi-intervention sequence**: Test minimum 10 interventions with real data

**Real Data Used**: Full pipeline with real Gemma-2B features and pymdp agent

#### `test_intervention_pipeline.py`
**Purpose**: Validate complete intervention pipeline
- **Minimum intervention enforcement**: Test ≥10 interventions performed
- **Exploration bonus**: Test prevention of early convergence
- **Belief variance**: Test meaningful belief changes across interventions
- **Statistical significance**: Test intervention results have proper variance

**Real Data Used**: Real intervention sequences, actual belief evolution

#### `test_research_questions.py`
**Purpose**: Validate research question calculations end-to-end
- **RQ1 correspondence**: Test ≥70% AI-circuit correspondence calculation
- **RQ2 efficiency**: Test ≥30% efficiency improvement calculation  
- **RQ3 predictions**: Test ≥3 novel predictions generation
- **Integrated validation**: Test all RQs with real multi-intervention data

**Real Data Used**: Complete experiment data from real runs

### System Tests (`tests/system/`)

#### `test_full_experiment.py`
**Purpose**: Validate complete experiment pipeline
- **End-to-end execution**: Test full experiment runs without crashes
- **Research objective achievement**: Test all RQs can pass with corrected code
- **Data consistency**: Test no attribute errors or type mismatches throughout
- **Completion verification**: Test experiment completes with valid results

**Real Data Used**: Full experiment with real Gemma-2B, real prompts, real results

#### `test_semantic_discovery.py`
**Purpose**: Validate semantic feature discovery
- **Golden Gate discovery**: Test "The Golden Gate Bridge is in" → San Francisco
- **Feature activation**: Test semantic features activate for relevant prompts
- **Intervention effects**: Test semantic interventions change model behavior
- **Discovery success rate**: Test reasonable semantic discovery success

**Real Data Used**: Real semantic prompts, actual GemmaScope transcoder features

#### `test_objectives_validation.py`
**Purpose**: Validate research objectives are achievable
- **70% correspondence**: Test corrected code can achieve ≥70% correspondence
- **30% efficiency**: Test corrected code can achieve ≥30% efficiency improvement
- **3+ predictions**: Test corrected code generates ≥3 novel predictions
- **Statistical validity**: Test all metrics calculated with proper significance

**Real Data Used**: Real experiment runs demonstrating objective achievement

## Test Implementation Strategy

### Phase 1: Unit Tests
1. Create test directory structure
2. Implement data structure tests with real CircuitFeature objects
3. Implement circuit tracer tests with real Gemma-2B loading
4. Implement Active Inference tests with real pymdp agent
5. Implement prediction system tests with real belief states
6. Implement metrics tests with real intervention data

### Phase 2: Integration Tests  
1. Test AI + circuit integration with real feature discovery
2. Test intervention pipeline with real multi-intervention sequences
3. Test research question calculations with real experimental data

### Phase 3: System Tests
1. Test full experiment pipeline end-to-end
2. Test semantic discovery with real prompts
3. Test research objectives validation with real achievement

## Test Execution

Each test will:
1. **Use real components** - circuit-tracer, pymdp, Gemma-2B transcoders
2. **Test actual functionality** - no mocks, no fake data
3. **Validate corrections** - ensure all fixes work with real data
4. **Verify objectives** - confirm research goals are achievable

## Success Criteria

**Unit Tests**: All components work correctly with real data
**Integration Tests**: All component interactions work properly  
**System Tests**: Complete experiment achieves all research objectives

- ✅ No syntax errors or runtime crashes
- ✅ Minimum 10 interventions performed
- ✅ Meaningful belief updates from real interventions
- ✅ Proper tensor shape handling
- ✅ RQ1: ≥70% AI-circuit correspondence achievable
- ✅ RQ2: ≥30% efficiency improvement achievable  
- ✅ RQ3: ≥3 novel predictions generated
- ✅ Statistical validation with proper variance and significance

## Test Files Structure

```
tests/
├── unit/
│   ├── test_data_structures.py
│   ├── test_circuit_tracer.py
│   ├── test_active_inference.py
│   ├── test_prediction_system.py
│   └── test_metrics.py
├── integration/
│   ├── test_ai_circuit_integration.py
│   ├── test_intervention_pipeline.py
│   └── test_research_questions.py
└── system/
    ├── test_full_experiment.py
    ├── test_semantic_discovery.py
    └── test_objectives_validation.py
```

This testing approach ensures the corrected codebase works correctly with real data and achieves the stated research objectives.
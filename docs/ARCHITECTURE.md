# ActiveCircuitDiscovery - System Architecture

## Overview

ActiveCircuitDiscovery implements Enhanced Active Inference for circuit discovery in Large Language Models. The architecture integrates real mechanistic interpretability tools (`circuit-tracer`) with principled Active Inference (`pymdp`) for efficient neural circuit discovery.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ActiveCircuitDiscovery System                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐    ┌──────────────────────┐               │
│  │   Active Inference  │    │   Circuit Discovery  │               │
│  │                     │    │                      │               │
│  │ ProperActiveInference│◄──►│ RealCircuitTracer    │               │
│  │ Agent (pymdp)       │    │ (circuit-tracer)     │               │
│  │                     │    │                      │               │
│  │ - EFE calculation   │    │ - Gemma-2-2B model   │               │
│  │ - Belief updating   │    │ - Transcoder features │               │
│  │ - Policy inference  │    │ - Real interventions  │               │
│  └─────────────────────┘    └──────────────────────┘               │
│             │                           │                          │
│             └───────────┬───────────────┘                          │
│                         ▼                                          │
│             ┌─────────────────────┐                                │
│             │ Integration Layer   │                                │
│             │                     │                                │
│             │ CircuitDiscovery    │                                │
│             │ Integration         │                                │
│             │                     │                                │
│             │ - EFE-guided        │                                │
│             │   intervention      │                                │
│             │ - Belief-circuit    │                                │
│             │   correspondence    │                                │
│             │ - RQ validation     │                                │
│             └─────────────────────┘                                │
│                         │                                          │
│                         ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  Experiment Results                         │   │
│  │                                                             │   │
│  │ • Circuit Graphs      • Novel Predictions                  │   │
│  │ • RQ Validation       • Statistical Validation             │   │
│  │ • Intervention Effects• Visualization Dashboards           │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Active Inference Framework (`src/active_inference/`)

#### ProperActiveInferenceAgent (`proper_agent.py`)
```python
class ProperActiveInferenceAgent:
    """Real pymdp.Agent integration for circuit discovery."""
    
    # State Space Design
    state_space = [
        component_id,      # Circuit component identifier (0-N)
        importance_level,  # Learned importance (0-4: low to critical)
        intervention_type  # Last intervention type (0-2: ablation/patch/scale)
    ]
    
    # Observation Space Design  
    observation_space = [
        effect_magnitude,     # Intervention effect (0-4: none to severe)
        semantic_change,      # Semantic behavior change (0-1: no/yes)
        statistical_significance  # Significance test result (0-1: no/yes)
    ]
```

**Key Methods**:
- `infer_states()`: Variational message passing for state inference
- `infer_policies()`: Policy inference using Expected Free Energy
- `sample_action()`: Sample intervention based on policy distribution
- `update_A()`: Update observation model from intervention results
- `calculate_expected_free_energy()`: EFE calculation for intervention selection

#### CircuitGenerativeModelBuilder (`generative_model.py`)
```python
# Generative Model Components
A_matrices = {
    # P(observation | state)
    'effect_magnitude': P(intervention_effect | component_importance),
    'semantic_change': P(semantic_change | component_type),
    'significance': P(statistical_sig | effect_magnitude)
}

B_matrices = {
    # P(next_state | current_state, action)
    'component_dynamics': P(next_importance | current_importance, intervention),
    'intervention_effects': P(next_type | current_type, selected_action)
}

C_matrices = {
    # Preferences (goal-directed behavior)
    'discovery_goals': [high_effect_preference, semantic_preservation, efficiency]
}

D_matrices = {
    # Prior beliefs
    'component_priors': uniform_importance_prior,
    'intervention_priors': exploration_bias_prior
}
```

#### ActiveInferenceCycle (`inference_cycle.py`)
```python
class ActiveInferenceCycle:
    """Complete perception-action cycle for circuit discovery."""
    
    def perceive(self, intervention_result):
        """Convert intervention results to observations."""
        
    def plan(self):
        """EFE-based planning for next intervention."""
        
    def act(self):
        """Execute selected intervention."""
        
    def learn(self, observation):
        """Update generative model from intervention evidence."""
```

### 2. Circuit Discovery Framework (`src/circuit_analysis/`)

#### RealCircuitTracer (`real_tracer.py`)
```python
class RealCircuitTracer:
    """Real circuit discovery using circuit-tracer + Gemma-2-2B."""
    
    def __init__(self):
        self.model = "google/gemma-2-2b"  # Transcoder support
        self.replacement_model = ReplacementModel()  # Real circuit-tracer
        self.transcoders = {}  # Layer-wise transcoders
        
    def find_active_features(self, prompt, threshold=0.1):
        """Discover active transcoder features."""
        # Golden Gate Bridge → San Francisco emerges naturally
        
    def perform_intervention(self, feature, intervention_type):
        """Real circuit intervention (no approximations)."""
        # Uses ReplacementModel for actual intervention
```

**Transcoder Feature Discovery**:
- **Natural Emergence**: Golden Gate Bridge → San Francisco features discovered automatically
- **Semantic Richness**: Location, landmark, geographical features
- **Layer Analysis**: Feature distribution across transformer layers
- **Activation Patterns**: Real activation analysis via transcoders

### 3. Integration Framework (`src/experiments/`)

#### CircuitDiscoveryIntegration (`circuit_discovery_integration.py`)
```python
class CircuitDiscoveryIntegration:
    """Integrates Active Inference with circuit discovery."""
    
    def run_integrated_discovery(self, test_prompts, max_interventions=50):
        """Main integration loop."""
        
        for step in range(max_interventions):
            # Active Inference selects intervention using EFE
            feature_idx, intervention_type = self.ai_agent.select_intervention()
            
            # Circuit tracer performs real intervention
            result = self.circuit_tracer.perform_intervention(feature, intervention_type)
            
            # AI agent updates beliefs from intervention evidence
            observation = self._convert_intervention_to_observation(result)
            self.ai_agent.update_beliefs_from_intervention(observation)
            
            # Check for convergence
            if self.ai_agent.check_convergence():
                break
                
        return self._generate_results()
```

### 4. Data Structures & Metrics (`src/core/`)

#### Core Data Structures (`data_structures.py`)
```python
@dataclass
class TranscoderFeature:
    """Transcoder feature replacing SAE features."""
    layer_idx: int
    feature_idx: int
    component_type: str  # 'attention', 'mlp', 'residual'
    activation_strength: float
    semantic_description: str
    intervention_sites: List[str]

@dataclass  
class InterventionResult:
    """Results from circuit intervention."""
    intervention_type: str
    target_feature: TranscoderFeature
    effect_magnitude: float
    baseline_prediction: str
    intervention_prediction: str
    semantic_change: bool
    statistical_significance: bool

@dataclass
class CircuitDiscoveryResult:
    """Complete experiment results."""
    discovered_features: List[TranscoderFeature]
    ai_intervention_sequence: List[Dict[str, Any]]
    final_beliefs: Dict[str, float]
    correspondence_score: float
    efficiency_improvement: float
    novel_predictions: List[Dict[str, Any]]
    total_interventions: int
    convergence_achieved: bool
```

#### Research Question Metrics (`metrics.py`)
```python
class CorrespondenceCalculator:
    """Calculate AI-circuit correspondence for RQ1."""
    
    def calculate_ai_circuit_correspondence(self, ai_beliefs, intervention_results, features):
        """Measure belief-behavior correspondence (target: ≥70%)."""

class EfficiencyCalculator:
    """Calculate intervention efficiency for RQ2."""
    
    def calculate_intervention_efficiency(self, ai_sequence, baseline_method, features):
        """Measure efficiency improvement (target: ≥30%)."""

class StatisticalValidator:
    """Statistical validation for all metrics."""
    
    def validate_research_questions(self, results):
        """Bootstrap confidence intervals and hypothesis testing."""
```

## Data Flow Architecture

### 1. Initialization Phase
```
Gemma-2-2B Model Loading
         ↓
Circuit-Tracer Integration
         ↓
Transcoder Loading (All Layers)
         ↓
Initial Feature Discovery
         ↓
Active Inference Agent Initialization
         ↓
Generative Model Construction
```

### 2. Discovery Loop
```
Current Beliefs State
         ↓
EFE Calculation → Policy Inference → Action Selection
         ↓
Intervention Execution (Real Circuit Manipulation)
         ↓
Effect Measurement → Observation Formation
         ↓
Belief Updating (Variational Message Passing)
         ↓
Convergence Check → Continue/Terminate
```

### 3. Validation Phase
```
Final Results Collection
         ↓
RQ1: Correspondence Calculation
         ↓
RQ2: Efficiency Comparison
         ↓
RQ3: Prediction Generation & Validation
         ↓
Statistical Significance Testing
         ↓
Result Visualization & Export
```

## Deployment Architecture

### Local Development Environment
```
Local Machine (CPU/GPU)
├── Python 3.8+
├── PyTorch 2.7.1
├── circuit-tracer>=0.1.0
├── pymdp==0.0.1
└── Development Testing
```

### Production Environment (L40S GPU)
```
DigitalOcean L40S Droplet
├── Ubuntu 22.04 LTS
├── NVIDIA L40S (46GB memory)
├── CUDA 12.1
├── PyTorch 2.7.1+cu121
├── Gemma-2-2B Model (~13GB)
├── Circuit-Tracer Transcoders (~15GB)
└── Production Experiments
```

### Memory Architecture (L40S GPU)
```
Total GPU Memory: 46GB
├── Gemma-2-2B Model: ~13GB (fp16)
├── Transcoders (All Layers): ~15GB
├── Activation Cache: ~8GB
├── Gradient Computation: ~5GB
├── Working Memory: ~3GB
└── System Overhead: ~2GB
```

## Interface Architecture

### External Interfaces
```python
# Main Experiment Interface
def run_integrated_experiment() -> Dict[str, Any]:
    """Main entry point for complete experiment."""

# Component Interfaces  
class IActiveInferenceAgent(ABC):
    """Abstract interface for Active Inference agents."""
    
class ICircuitTracer(ABC):
    """Abstract interface for circuit discovery tools."""
```

### Internal Communication
```python
# Agent-Tracer Communication
observation = tracer.perform_intervention(feature, intervention_type)
agent.update_beliefs_from_intervention(observation)

# Belief-Circuit Correspondence
correspondence = calculate_correspondence(agent.beliefs, circuit_behavior)

# EFE-Guided Selection
feature_idx, intervention = agent.select_intervention_by_efe()
```

## Security & Error Handling

### Error Handling Strategy
```python
# Graceful degradation for missing components
try:
    from circuit_tracer import ReplacementModel
except ImportError:
    raise ImportError("circuit_tracer required for production use")

# GPU memory management
torch.cuda.empty_cache()
if torch.cuda.memory_allocated() > 0.9 * torch.cuda.max_memory_allocated():
    trigger_memory_cleanup()

# Intervention safety checks
def validate_intervention_safety(feature, intervention_type):
    """Ensure intervention won't damage model state."""
```

### Data Validation
```python
# Input validation
def validate_transcoder_feature(feature: TranscoderFeature):
    """Validate feature data structure."""
    
# Result validation  
def validate_experiment_results(results: CircuitDiscoveryResult):
    """Ensure results meet quality standards."""
```

## Performance Characteristics

### Computational Complexity
- **Feature Discovery**: O(L × F) where L=layers, F=features per layer
- **EFE Calculation**: O(S × A) where S=states, A=actions  
- **Belief Updating**: O(S²) for variational message passing
- **Overall**: O(I × (L × F + S²)) where I=interventions

### Memory Usage Patterns
- **Peak Usage**: During intervention execution (~40GB)
- **Steady State**: Model + transcoders loaded (~28GB)
- **Minimum**: Base model only (~13GB)

### Expected Performance
- **Initialization Time**: 3-5 minutes (model + transcoder loading)
- **Per Intervention**: 30-60 seconds
- **Total Experiment**: 30-60 minutes (50 interventions)
- **Convergence**: Typically 25-40 interventions

---

**Architecture Version**: v2.0
**Last Updated**: 2025-01-07
**Status**: Production Ready
**Deployment Target**: DigitalOcean L40S GPU
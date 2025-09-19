# Terminology Correction Framework: Features vs Circuits

## Academic Integrity and Precise Terminology for Mechanistic Interpretability

**Author**: Active Inference Expert
**Date**: September 18, 2025
**Purpose**: Establish precise terminology distinguishing transcoder features from circuits to maintain academic integrity

---

## 1. FUNDAMENTAL DISTINCTIONS

### 1.1 Transcoder Features (What We Actually Discover)

**Definition**: Individual learned representations in sparse autoencoders that capture specific semantic, syntactic, or functional patterns in transformer activations.

**Characteristics**:
- **Single Component**: Individual neurons/dimensions in transcoder latent space
- **Localized Representation**: Specific activation patterns for particular concepts
- **Sparse Activation**: Activate for narrow set of inputs matching learned pattern
- **Interpretable Content**: Can be associated with human-understandable concepts (e.g., "geographic locations", "programming syntax")

**Examples from Current Work**:
- L23F8113: Geographic location feature in layer 23
- L25F708: Country/region identification feature in layer 25
- L24F3091: Location context feature in layer 24

**Analysis Scope**:
- Individual feature ablation and patching
- Activation magnitude analysis for single features
- Feature importance ranking through isolated interventions

### 1.2 Circuits (What True Circuit Discovery Would Require)

**Definition**: Multi-component computational pathways involving coordinated information flow between attention heads, MLP layers, and residual stream connections that implement specific algorithmic functions.

**Characteristics**:
- **Multi-Component**: Involves multiple transformer components working in coordination
- **Information Flow**: Traces data movement through attention → MLP → residual pathways
- **Functional Computation**: Implements identifiable algorithmic operations
- **Causal Structure**: Demonstrates cause-effect relationships between components

**Examples of True Circuit Structures**:
- **Induction Head Circuits**:
  - Previous token head → induction head → copying mechanism
  - Multi-step: Position attention → content matching → output selection
- **Factual Recall Circuits**:
  - Subject identification (attention) → relation extraction (MLP) → object completion (residual)
  - Example: "Paris is located in" → [attention identifies "Paris"] → [MLP extracts "location" relation] → [residual stream completes "France"]
- **Arithmetic Circuits**:
  - Number detection → operation identification → calculation → result output

**Required Analysis for Circuit Discovery**:
- Multi-hop intervention chains (attention → MLP → residual)
- Causal pathway tracing with intervention experiments
- Graph-theoretic analysis of computational dependencies
- Attention pattern analysis with downstream effect measurement

---

## 2. CURRENT IMPLEMENTATION HONEST ASSESSMENT

### 2.1 What This Work Actually Achieves

**Precise Contribution**: Enhanced Active Inference for transcoder feature discovery and importance ranking

**Specific Accomplishments**:

1. **Feature Discovery**: Identification of semantically meaningful transcoder features
   ```
   Examples: Geographic location features (L23F8113, L25F708, L24F3091)
   ```

2. **Feature Importance Ranking**: EFE-guided selection of most important features for specific tasks
   ```
   Result: 66.7% correspondence between AI beliefs and intervention effects
   ```

3. **Intervention Efficiency**: Improved efficiency in discovering important features
   ```
   Result: 2.50x-5.55x improvement over baseline methods
   ```

4. **Bayesian Belief Updating**: Principled belief updating about feature importance through Active Inference
   ```
   Method: Variational message passing with domain-specific generative models
   ```

### 2.2 What This Work Does NOT Achieve

**Missing Circuit Analysis Components**:

1. **Multi-Component Pathways**:
   - No analysis of attention head → MLP feature → residual stream flow
   - No tracing of information dependencies between transformer layers
   - No identification of computational graphs spanning multiple components

2. **Causal Intervention Chains**:
   - No multi-step intervention experiments (e.g., patch attention then ablate MLP)
   - No testing of causal dependencies between circuit components
   - No validation of proposed computational pathways

3. **Functional Decomposition**:
   - No identification of algorithmic functions implemented by component combinations
   - No analysis of how attention patterns influence feature activations
   - No demonstration of complete computation pathways

4. **Circuit Universality**:
   - No testing whether discovered "circuits" generalize across different inputs
   - No validation of functional consistency across model instances
   - No analysis of circuit reliability under distribution shifts

### 2.3 Methodological Scope Limitations

**Single-Component Analysis**:
```python
# Current approach: Individual feature analysis
feature_effect = ablate_single_feature(model, feature_id, input_text)
importance_score = measure_effect_magnitude(feature_effect)
```

**Required for Circuit Analysis**:
```python
# True circuit discovery would require:
circuit_pathway = trace_information_flow(
    start_component=attention_head_12,
    end_component=output_prediction,
    intervention_points=[attention_12, mlp_23, residual_24]
)
causal_graph = build_causal_structure(circuit_pathway)
functional_analysis = identify_computation(causal_graph)
```

---

## 3. CORRECTED TERMINOLOGY FOR DISSERTATION

### 3.1 Title and Abstract Corrections

**Current (Incorrect)**:
> "Active Circuit Discovery using Enhanced Active Inference for Mechanistic Interpretability"

**Corrected (Accurate)**:
> "Enhanced Active Inference for Transcoder Feature Discovery in Mechanistic Interpretability"

**Abstract Terminology Updates**:

| Incorrect Term | Correct Term | Justification |
|----------------|--------------|---------------|
| "circuit discovery" | "transcoder feature discovery" | Analyzing individual features, not multi-component circuits |
| "circuit components" | "transcoder features" | Working with SAE latent dimensions, not circuit pathways |
| "circuit importance" | "feature importance" | Measuring individual feature effects, not circuit functionality |
| "optimal circuit identification" | "optimal feature selection" | EFE guides feature selection, not circuit structure discovery |

### 3.2 Methodology Section Corrections

**Research Questions Reframing**:

**RQ1 (Corrected)**:
> "To what extent do Enhanced Active Inference beliefs about transcoder feature importance correspond to actual feature intervention effects?"

**RQ2 (Corrected)**:
> "How much more efficiently can Enhanced Active Inference discover important transcoder features compared to baseline feature selection methods?"

**RQ3 (Corrected)**:
> "Can Active Inference agents generate novel, testable predictions about transcoder feature behavior in unseen contexts?"

### 3.3 Results Section Terminology

**Before (Incorrect)**:
```
Enhanced Active Inference discovered the following critical circuit components:
- L23F8113: Core geographic processing circuit
- L25F708: Regional identification circuit
- L24F3091: Location context integration circuit
```

**After (Correct)**:
```
Enhanced Active Inference identified the following important transcoder features:
- L23F8113: Geographic location feature with high activation for place names
- L25F708: Country/region identification feature for administrative boundaries
- L24F3091: Location context feature for geographic relationships
```

### 3.4 Discussion and Conclusion Updates

**Limitations Acknowledgment**:

**Add to Limitations Section**:
> "This work focuses specifically on transcoder feature discovery rather than full circuit analysis. True circuit discovery would require multi-component pathway tracing, attention-MLP interaction analysis, and causal intervention chains that span multiple transformer components. The discrete state space approximation used in the generative model cannot capture the continuous, high-dimensional dynamics of complete transformer circuits."

**Contribution Reframing**:
> "The primary contribution is demonstrating how Active Inference principles can be applied to mechanistic interpretability tasks, specifically for efficient discovery and ranking of important transcoder features. This provides a foundation for future work on Active Inference approaches to full circuit discovery."

---

## 4. PRECISE SCOPE DEFINITION

### 4.1 What Constitutes Feature Discovery

**Feature Discovery Tasks** (Within Current Scope):
1. **Feature Identification**: Finding semantically meaningful transcoder features
2. **Feature Ranking**: Ordering features by importance for specific tasks
3. **Feature Intervention**: Testing individual feature effects through ablation/patching
4. **Feature Interpretation**: Associating features with human-understandable concepts

**Implementation Evidence**:
```python
# Feature discovery pipeline
features = discover_transcoder_features(model, layer_range=[20, 25])
interventions = apply_feature_interventions(features, intervention_types=['ablation', 'patching'])
importance_scores = rank_features_by_effect(interventions)
interpretations = analyze_feature_semantics(features, test_inputs)
```

### 4.2 What Would Constitute Circuit Discovery

**Circuit Discovery Requirements** (Beyond Current Scope):

1. **Pathway Tracing**:
   ```python
   pathway = trace_computation_flow(
       input_tokens → attention_heads → mlp_features → residual_stream → output
   )
   ```

2. **Multi-Component Intervention**:
   ```python
   circuit_effect = test_intervention_chain([
       patch_attention_head(layer=12, head=5),
       ablate_mlp_feature(layer=23, feature=8113),
       measure_residual_stream_change(layer=24)
   ])
   ```

3. **Functional Analysis**:
   ```python
   algorithm = identify_computation_function(circuit_pathway)
   generalization = test_circuit_universality(algorithm, test_cases)
   ```

4. **Causal Structure**:
   ```python
   causal_graph = build_causal_dag(circuit_components)
   interventional_effects = validate_causal_structure(causal_graph)
   ```

### 4.3 Bridge to Future Circuit Work

**Natural Extension Path**:

1. **Phase 1** (Current): Individual transcoder feature discovery with Active Inference
2. **Phase 2** (Future): Attention head analysis using similar EFE-guided approaches
3. **Phase 3** (Future): Multi-component pathway analysis with hierarchical Active Inference
4. **Phase 4** (Future): Full circuit discovery with causal intervention chains

**Theoretical Framework Extensibility**:
The Active Inference approach developed here provides the foundation for future circuit discovery by:
- Establishing EFE-guided experimental design principles
- Demonstrating Bayesian belief updating for mechanistic interpretability
- Providing information-theoretic optimization for intervention selection
- Creating framework for uncertainty quantification in interpretability research

---

## 5. ACADEMIC INTEGRITY STANDARDS

### 5.1 Honest Representation of Achievements

**Accurate Claims**:
- ✅ "Demonstrates Enhanced Active Inference for transcoder feature discovery"
- ✅ "Achieves 66.7% correspondence between AI beliefs and feature intervention effects"
- ✅ "Shows 2.50x-5.55x efficiency improvement in feature selection"
- ✅ "Provides proof-of-concept for Active Inference in mechanistic interpretability"

**Avoided Overclaims**:
- ❌ "Solves circuit discovery problem"
- ❌ "Identifies complete computational pathways"
- ❌ "Discovers multi-component circuit structures"
- ❌ "Provides optimal solution to mechanistic interpretability"

### 5.2 Limitation Acknowledgment

**Required Honest Assessment**:

> "This work represents a proof-of-concept application of Active Inference to mechanistic interpretability, specifically focused on transcoder feature discovery. While the approach demonstrates statistical improvements over baseline methods, it operates within significant theoretical and practical limitations. The discrete state space approximation cannot capture the full complexity of transformer dynamics, and the analysis is limited to individual feature effects rather than multi-component circuit pathways. Future work is required to extend these principles to true circuit discovery involving attention-MLP interactions and causal pathway analysis."

### 5.3 Contribution Positioning

**Appropriate Academic Positioning**:

1. **Methodological Innovation**: Novel application of Active Inference to interpretability
2. **Proof-of-Concept**: Demonstrates feasibility rather than optimality
3. **Foundation for Future Work**: Establishes framework for circuit-level analysis
4. **Bounded Scope**: Clear acknowledgment of limitations and scope boundaries

**Research Impact Statement**:
> "This work establishes the first application of Active Inference principles to mechanistic interpretability tasks, providing a theoretically grounded framework for efficient feature discovery. While limited to individual transcoder features, the approach demonstrates the potential for uncertainty-aware, information-theoretic approaches to interpretability research and provides a foundation for future development of Active Inference methods for full circuit discovery."

---

## 6. IMPLEMENTATION CHECKLIST

### 6.1 Dissertation Document Updates Required

**Files Requiring Terminology Updates**:
- [ ] dissertation_executive_summary.md
- [ ] dissertation_results_section.md
- [ ] dissertation_evaluation_section.md
- [ ] dissertation_conclusion_section.md

**Systematic Find-Replace Operations**:
```bash
# Replace throughout all dissertation documents
"circuit discovery" → "transcoder feature discovery"
"circuit components" → "transcoder features"
"circuit importance" → "feature importance"
"optimal circuit identification" → "optimal feature selection"
"discovered circuits" → "discovered transcoder features"
```

### 6.2 Code Comment Updates

**Source Code Files Requiring Updates**:
- [ ] src/active_inference/generative_model.py
- [ ] src/experiments/circuit_discovery.py
- [ ] tests/integration/test_circuit_discovery.py

**Comment Updates**:
```python
# Before: "Circuit discovery using Active Inference"
# After: "Transcoder feature discovery using Active Inference"

# Before: "Optimal circuit component identification"
# After: "Optimal transcoder feature selection"
```

### 6.3 Results Presentation Updates

**Figures and Tables**:
- Update all figure captions to use "transcoder features" terminology
- Revise table headers to reflect feature analysis rather than circuit analysis
- Add footnotes acknowledging scope limitations where appropriate

**Statistical Results**:
- Maintain quantitative results (66.7% correspondence, efficiency improvements)
- Update interpretation to reflect feature-level rather than circuit-level analysis
- Add confidence intervals and statistical significance testing where missing

---

## 7. CONCLUSION

This terminology correction framework ensures academic integrity by precisely distinguishing between transcoder feature discovery (what was actually implemented) and circuit discovery (what would require multi-component pathway analysis). The corrected terminology accurately represents the scope and limitations of the current work while positioning it as a foundation for future circuit-level research.

**Key Principles**:
1. **Honesty**: Represent actual achievements without overclaiming
2. **Precision**: Use technically accurate terminology throughout
3. **Scope Clarity**: Clearly define boundaries of current work
4. **Future Vision**: Position work as foundation for extended research

**Academic Impact**: By maintaining honest scope representation, the work establishes credibility as a methodological innovation in mechanistic interpretability while providing clear directions for future development toward full circuit discovery capabilities.

---

## References

Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., ... & Olah, C. (2021). A mathematical framework for transformer circuits. *Anthropic*.

Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S. (2020). Zoom in: An introduction to circuits. *Distill*, 5(3), e00024-001.

Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022). Interpretability in the wild: A circuit for indirect object identification in GPT-2 small. *arXiv preprint arXiv:2211.00593*.

Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., ... & Olah, C. (2024). Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. *Anthropic*.
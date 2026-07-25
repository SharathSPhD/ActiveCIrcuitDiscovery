# Brief 1: Anthropic mech-interp lineage (verified 2026-07-25)

## Zoom In (Distill, Mar 10 2020) — Olah, Cammarata, Schubert, Goh, Petrov, Carter
- Three claims: (1) Features are the fundamental unit, correspond to directions; (2) features connect via weights forming circuits (computational subgraphs); (3) universality — analogous features/circuits recur across models (curve detectors in AlexNet, InceptionV1, VGG19, ResNet).
- Curve detectors: InceptionV1 mixed3b, ~60px radius, orientation families tiling 360°; seven lines of evidence incl. hand-reimplementing weights. Companion "Curve Detectors" catalogs units e.g. 3b:379.
- Polysemanticity: e.g. InceptionV1 4e:55 fires for cat faces + car fronts + cat legs; hypothesized cause = superposition.

## Toy Models of Superposition (Sept 14 2022) + Towards Monosemanticity (Oct 4 2023)
- Superposition: networks represent more features than neurons as near-orthogonal directions; sparsity controls regime; phase changes; geometric polytopes (antipodal pairs, pentagons, tetrahedra). Polysemanticity is superposition seen from the neuron basis.
- Towards Monosemanticity: SAE (MSE + L1) on 8B MLP-activation samples from a 1-layer transformer (512-neuron MLP); flagship run A/1 = 4,096 features; features for Arabic script, DNA, base64, Hebrew; single neuron mixes citations/dialogue/HTTP/Korean.

## Scaling Monosemanticity (May 21 2024) — Templeton et al.
- SAEs of 1M/4M/34M features on residual stream of a middle layer of Claude 3 Sonnet. Notable features: Golden Gate Bridge 34M/31164353 (multilingual + images), code-error 1M/1013764, addition 1M/697189, internal-conflict 1M/284095; unsafe code, bioweapons, scam emails, deception, sycophancy features.
- Steering: clamping to large multiples of max activation changes behavior; scam-email feature overrides harmlessness training. Caveat: "mere existence of these features should not update our views on how dangerous models are."

## Mapping the Mind (May 21 2024) + Golden Gate Claude (May 23 2024)
- Mapping the Mind: popular writeup; feature geometry mirrors semantics (Alcatraz, Golden State Warriors near GGB).
- Golden Gate Claude: live on claude.ai for exactly 24h; created by clamping GGB feature to 10× its max activation ("precise, surgical change", not fine-tuning/prompting). $10 → pay bridge toll; love story → car crossing bridge; self-description → the bridge.

## Circuit Tracing: Methods (Mar 27 2025) — Ameisen, Lindsey, et al.
- Cross-layer transcoder (CLT): feature reads residual stream at one layer, writes to MLP outputs of that + all subsequent layers (per-layer decoders). vs residual SAEs: SAE describes what's represented; transcoder REPLACES the MLP computation (input→output), features become causal computational units. vs PLT: PLT writes only own layer.
- Replacement model = MLPs swapped for CLT (attention untouched). Local replacement model (per-prompt): freezes attention patterns + layernorm denominators from original forward pass, adds ERROR NODES (true MLP output − CLT reconstruction) → exactly reproduces the model on that prompt; linear in feature activations → every edge an exact linear direct effect.
- Pruning: ~10× node reduction retaining ~80% of explained behavior. Scale: CLTs 300K–10M feats (18-layer model), up to 30M on Claude 3.5 Haiku; largest 18L CLT matches next-token on ~50% of pretraining prompts.
- Limitations: QK attention unexplained, error nodes hide computation, inhibitory circuits hard to read, per-prompt local only, mechanistic-faithfulness residual uncertainty. "Causally faithful" = exact within frozen local replacement model; still must validate by perturbation.

## Biology of an LLM (Mar 27 2025) — Lindsey et al., Claude 3.5 Haiku
- Dallas→Texas→Austin multi-step: internal Texas features; swap with California → "Sacramento". Poetry planning: rhyme candidates planned at newline before line ("rabbit"/"habit"), injection redirects ~70% of cases. Multilingual: shared semantic features, English default. Addition 36+59=95: lookup-table heuristics ("_6+_9→ends in 5"); model self-reports the school algorithm — capability without introspective access. Medical dx: preeclampsia features w/o the word. Hallucination: "can't answer" on by default, suppressed by known-entity features; forcing "known answer" → confident fabricated citations. Refusals: specific harm features → general "harmful request" abstraction from finetuning. BOMB jailbreak: assembles word letter-by-letter w/o representing it; refusal available only at sentence boundary. Unfaithful CoT: signatures for faithful vs bullshitting vs motivated reasoning. Hidden goals readable in graphs, wired into Assistant persona.
- Honesty: satisfying insight on ~25% of prompts attempted; graphs are hypotheses validated by interventions; diagrams are "highly distilled, subjectively determined simplifications".

## circuit-tracer (open source, May 29 2025)
- github.com/safety-research/circuit-tracer, MIT; by Anthropic Fellows Michael Hanna & Mateusz Piotrowski, mentors Ameisen & Lindsey; Neuronpedia integration by Decode Research (Johnny Lin; Curt Tigges).
- Provides ReplacementModel.from_pretrained, attribute, pruning, feature interventions, viz, TransformerLens/nnsight backends. Launch models: Gemma-2-2B, Llama-3.2-1B (PLTs); by mid-2026 also Qwen-3, Gemma-3 270M–27B, Llama-3.1-8B, CLTs for Gemma-2-2B (426K/2.5M), Llama-3.2-1B (524K), GPT-OSS-20B. Neuronpedia frontend: neuronpedia.org/gemma-2-2b/graph.

## Neuronpedia
- Run by Johnny Lin / Decode Research; GemmaScope demo at neuronpedia.org/gemma-scope; graph landscape review Aug 2025 at neuronpedia.org/graph/info.
- Embed format (docs.neuronpedia.org/embed-iframe): https://neuronpedia.org/{model}/{layer}-{sae}/{index}?embed=true with embedplots, embedexplanation, embedtest, defaulttesttext. Width ≥640px else stacks. Example live: https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-transcoder-16k/16369?embed=true&embedexplanation=true&embedplots=true&embedtest=true (source set gemmascope-transcoder-16k). NOTE: verify specific feature meaning in browser before the talk.

## GemmaScope (DeepMind, Jul 31 2024; arXiv:2408.05147)
- 400+ JumpReLU SAEs on every layer/sublayer of Gemma 2 2B & 9B, >30M features; ~15% of Gemma-2-9B training compute; ~20 PiB activations. GemmaScope transcoders (16k/layer) power circuit-tracer Gemma graphs.

## 2025–26 updates
- Sharkey et al. "Open Problems in Mechanistic Interpretability" arXiv:2501.16496.
- Lindsey "Emergent Introspective Awareness" (Oct 2025, transformer-circuits.pub/2025/introspection): concept injection; ~20% detection in best models.
- Circuits Updates through June 2026; ecosystem: Gemma-3 27B, GPT-OSS-20B etc.

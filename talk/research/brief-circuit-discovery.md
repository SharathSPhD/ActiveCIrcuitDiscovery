# Brief 2: Circuit discovery literature (verified 2026-07-25)

## IOI — Wang et al. 2022 (arXiv:2211.00593, ICLR 2023)
- 26 attention heads, 7 classes, GPT-2 small: duplicate token (0.1, 3.0), previous token (2.2, 4.11), induction (5.5, 6.9), S-inhibition (7.3, 7.9, 8.6, 8.10), name movers (9.6, 9.9, 10.0), negative name movers (10.7, 11.10), backup name movers (~8 heads, L9–11; self-repair/Hydra).
- Path patching; ABC corruption dataset; mean ablation. Criteria: faithfulness (~87% of logit diff recovered, model mean ≈3.55 — verify before quoting), completeness (adversarial subsets; negative/backup heads expose gaps), minimality.

## ACDC — Conmy et al. 2023 (arXiv:2304.14997, NeurIPS spotlight)
- Reverse topological greedy edge pruning; remove edge if KL increase < τ. One forward pass per candidate edge; ~32k edges for GPT-2-small edge graphs → tens of thousands of passes, hours of GPU. Recovers canonical circuits (ROC vs ground truth). Limits: greedy, no interactions, τ/metric sensitivity (KL sign-blind → misses negative name movers), not adaptive — uniform budget spend.

## EAP — Syed, Rager, Conmy 2023 (arXiv:2310.10348)
- ΔL ≈ (e_corrupt − e_clean)ᵀ ∇ₑ L(clean); ALL edge scores from 2 forward + 1 backward pass (10³–10⁴× cheaper than ACDC); better mean AUC than ACDC + subnetwork probing. Follow-ups: EAP-IG (Hanna et al., arXiv:2403.17806, COLM 2024, integrated gradients, faithfulness curves not overlap), EAP-GP (arXiv:2502.06852).

## Transcoders
- Dunefsky, Chlenski, Nanda 2024 (arXiv:2406.11944, NeurIPS): transcoder approximates MLP input→output; attribution factorizes into input-invariant (virtual weights: enc·dec) × input-dependent (scalar activation); circuit analysis THROUGH MLPs tractable; Pareto matches/beats SAEs; GPT-2 Greater-Than case study.
- Paulo, Shabalin, Belrose 2025 (arXiv:2501.18823): skip transcoders Pareto-dominate; transcoder latents more interpretable than SAE latents (Llama-3.2, Pythia); field should switch for circuit analysis.

## Sparse feature circuits — Marks et al. 2024 (arXiv:2403.19647, ICLR 2025 spotlight)
- SAE-feature-level causal graphs w/ attribution patching incl. error nodes; Pythia-70M, Gemma-2-2B. SHIFT: human trims spurious features (Bias-in-Bios gender) → better worst-group accuracy.

## Steering
- ActAdd (Turner et al. arXiv:2308.10248); RepE (Zou et al. 2310.01405); CAA (Rimsky 2312.06681).
- AxBench (Wu et al. 2501.17148, ICML 2025): prompting > all representation steering; SAE steering near bottom; ReFT-r1 competitive. Canonical "steering underperforms prompting".
- Tan et al. 2407.12404: steering vectors high variance, brittle OOD. Large multipliers push activations off-manifold — key critique for m=10 steering.
- Chalnev, Siu, Conmy 2411.02193 (verify): SAE-targeted steering to limit side effects.

## Faithfulness debates
- Zhang & Nanda 2309.16042: metric (logit diff vs prob vs KL) + corruption choice change localization conclusions; recommend contrastive logit-diff + token replacement.
- Miller, Chughtai, Saunders 2407.08734: faithfulness not robust — zero vs mean vs resample ablation, positions, node vs edge flip optimal circuits; AutoCircuit lib.
- Causal scrubbing (Chan et al. 2022, AF post): resample ablation gold standard.
- Zero ablation maximally OOD; mean keeps 1st moments; resample most on-distribution.
- ACD positioning: motivates explicit observation models over trusting one scalar.

## Adaptive/budgeted intervention selection (ACD's slot)
- Interpretability agents: FIND benchmark (2309.03886), MAIA (2404.14394, ICML 2024) — adaptive but heuristic, no belief state, no formal budget/EIG objective.
- Mask-optimization circuit finding: subnetwork probing (2104.03514), Edge Pruning (2406.16778, scales to CodeLlama-13B).
- Bayesian active causal discovery: Tong&Koller 2001, Murphy 2001, causal bandits (Lattimore 2016), ABCD-Strategy (Agrawal 2019 AISTATS), Scherrer 2109.02429, Tigas 2022 (2203.02016, NeurIPS: BOED over targets+values, DiBS posterior), Tigas 2023 (2302.10607, ICML), Toth 2206.02063 Active Bayesian Causal Inference.
- One-liner: prior work is exhaustive (ACDC), amortized-linear (EAP), or agentic-informal (MAIA); ACD = sequential Bayesian experimental design over interventions under explicit budget (verify "first" phrasing).

## Layer division of labor
- Tigges et al. 2407.10827 (NeurIPS 2024): IOI circuits consistent across Pythia 70M–2.8B and training; supports GPT-2→Gemma transfer of motifs.
- No canonical published Gemma-2-2B IOI circuit (verified absence by search).
- ROME (2202.05262): early–middle MLPs at subject last token store facts; late attention retrieves. Geva 2012.14913 (MLP key-value memories), Geva 2304.14767 (subject enrichment early MLPs → attribute extraction by late attention).

## 2025–26
- Attribution graphs (Mar 27 2025) + circuit-tracer open source (May 29 2025) + Neuronpedia graph frontend; "Circuits Research Landscape" Aug 2025 (neuronpedia.org/graph/info).
- MIB benchmark (2504.13151, verify). Surveys: Bereska & Gavves 2404.14082, Sharkey 2501.16496, Rai 2407.02646 (verify IDs).
- Goodfire Ember (Llama-3.3-70B SAEs, Dec 2024; DeepSeek-R1 SAEs); EleutherAI sparsify/skip-transcoders; auto-interp (2410.13928).

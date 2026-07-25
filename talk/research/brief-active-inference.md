# Brief 3: Active inference canon, critiques, audience (verified 2026-07-25)

## Canonical EFE
- Da Costa 2020 (2001.07203, JMP): G(π) = Risk (KL[Q(s_τ,A|π)‖P(s_τ,A)]) + Ambiguity (E_Q H[P(o|s,A)]) — EXACT. Epistemic+pragmatic decomposition exact only up to vanishing "expected evidence bound"; with parameter learning: pragmatic + evidence-bound(≈0) + salience (state info gain) + NOVELTY (param info gain over A).
- Friston 2017 process theory: epistemic value = expected information gain = mutual information = expected Bayesian surprise. Vectorized: G = o·(ln o − U) + s·H, H = −diag(AᵀlnA). Risk-2017 in outcome space; Risk-2020 in state space; differ by evidence bound.
- Slide identity: EFE = risk+ambiguity = −(epistemic+pragmatic).

## Critiques the audience wields
- Millidge/Tschantz/Buckley "Whence the EFE?" (2004.08128): EFE ≠ VFE of future. FEF (honest extension) DISCOURAGES exploration; EFE = FEF − expected info gain; exploratory drive comes from prior-for-posterior swap, a modeling choice not theorem. FEEF alternative.
- Sajid 2021 "demystified" (1909.10863): AIF vs RL matched POMDPs; belief-based, native epistemic drive, reward = observation w/ C prior; wins reward-free; pressure: small tasks, correct model handed.
- Dark room (Friston/Thornton/Clark 2012): resolved via phenotype-relative priors → feeds tautology worry.
- EFE = BOED + expected utility (Sajid/Da Costa/Parr/Friston 2110.04074): flat preferences → Lindley EIG; no ambiguity → expected utility. Deflationary: sum of two known objectives.
- Gershman 2019 (1901.07945): FEP unfalsifiable as principle; bite in generative model choices. Predictive coding not entailed.
- Bruineberg "Emperor's New Markov Blankets" (BBS 2022): Pearl vs Friston blankets. Biehl 2001.06408 technical critique; Aguilera 2105.11203 FEP physics particularity.

## Single-step vs deep; BOED lineage
- Sophisticated inference (2006.04120): recursive Bellman-like EFE, beliefs-about-beliefs, tree search w/ pruning (~1/16 threshold). Vanilla = O(|U|^T) policy enumeration.
- Single-step EFE ≈ salience: myopic → epistemic term dominates = expected info gain about states (Parr & Friston 2017 J R Soc Interface; Itti & Baldi Bayesian surprise).
- Lindley 1956 (EIG); MacKay 1992 (info-based objective functions, active learning). EFE-epistemic = EIG = I(s;o|π) proven Friston 2015 "Active inference and epistemic value" Eq. 4. Schwartenbeck 2019 eLife e41703: curiosity/novelty unified (goal-directed + info-seeking).

## pymdp facts (repo verified, v1.0.3; JOSS 7(73):4098, arXiv:2201.03904)
- main = JAX rewrite; NumPy JOSS implementation under pymdp/legacy/. Legacy Agent defaults: policy_len=1, inference_horizon=1, gamma=16.0 (policy precision), alpha=16.0 (action precision, stochastic selection), use_utility=True, use_states_info_gain=True, use_param_info_gain=FALSE by default, inference_algo="VANILLA" (fixed-point iteration, mean-field), lr_pA=1.0. Policies = cartesian product over policy_len.
- A-update verified: qA[m] = pA[m] + lr·spm_cross(obs,qs)·(A>0).
- EFE assembly (control.py update_posterior_policies): G = utility + states_info_gain (+ pA/pB info gain if enabled); q_pi = softmax(gamma·G + lnE).
- Salience via spm_MDP_G = H[Q(o|π)] − E_Q H[P(o|s)].
- NOVELTY: calc_pA_info_gain uses spm_wnorm(pA) = 1/Σa − 1/a; gated by use_param_info_gain.
- **ACD's agent sets use_param_info_gain=True, pA = A·10 concentration, lr_pA=η — novelty term IS active.** (verified in src/active_inference/pomdp_agent.py lines 307–326)
- Utility = E_Q(o)[ln softmax(C)]; utility + state info gain = −(risk-over-observations + ambiguity) EXACTLY (entropy cancellation). Default carve-up is utility+epistemic; risk is over OBSERVATIONS not states.
- Limitations experts cite: discrete only, exhaustive policy evaluation, mean-field across factors, no C-learning in legacy.

## AI/ML active inference + interpretability intersection
- Deep AIF: Ueltzhöffer 1709.02341, Fountas 2006.04176, Millidge 1907.03876, Mazzaglia contrastive 2021 + review Entropy 24:301.
- VERSES: Ecosystems of Intelligence 2212.01354; renormalizing generative models 2407.20292; AXIOM 2505.24784.
- HEADLINE: as of Jul 2026 the AIF×mech-interp intersection is essentially EMPTY. Nearest: MAIA (no formalism), ACDC (not decision-theoretic), FEPS 2411.14991 (interpretability OF AIF agents — reverse), Pezzulo TiCS 2024 "Generating meaning" (passive AI critique), multi-LLM AIF 2412.10425. Recommend scanning IWAI TOCs before claiming "first-ever"; phrase as "to our knowledge".

## BOED / active causal discovery
- Tigas 2203.02016 (targets+values, NeurIPS 2022); Rainforth 2302.14545 "Modern BED" review (EIG, nested MC, DAD amortized); Sussex 2105.14024 near-optimal multi-perturbation; EPIG 2304.08151.
- Bridge punchline: BOED's EIG IS the EFE epistemic term; EFE = EIG + built-in preference term, applied to "which ablation next".

## Audience
- Active Inference Institute (Daniel Friedman, ~2000 members); ActInf GuestStreams (>#100); Textbook Group reads Parr/Pezzulo/Friston 2022 (align A/B/C/D notation to it).
- IWAI: Oxford 2024, Montreal 2025 (CCIS 2857), Madrid 2026 (7th, CSIC, Oct 14–16, theme "Foundations", invited Friston/Rao/Pezzulo).
- Likely hard questions by person: Friston (is your generative model a model of the network or relabeling?), Parr (what are A/B/C/D concretely; does message passing correspond?), Pezzulo (does agent framing buy predictions?), Da Costa (show objective + conditions), Heins (EFE actually used? show epistemic vs pragmatic), Tschantz (why discrete pymdp vs learned continuous?), Buckley (what does AIF predict that probing doesn't?), Millidge (reduce to known estimator or show novelty), Sajid (uncertainty about the network or just fit?), Smith (validation: parameter recovery/model comparison?), Ramstead (where's the blanket/boundary?), Friedman (open + reproducible?).

export type QA = {
  id: string;
  tag: 'EFE theory' | 'Mech interp' | 'Statistics' | 'Scaling & safety' | 'Rapid fire';
  q: string;
  concede: string;
  reply: string;
};

export const QAS: QA[] = [
  // ---------------- EFE THEORY ----------------
  {
    id: 'q1', tag: 'EFE theory',
    q: 'With policy length 1, your EFE collapses to salience plus preference-seeking. Why call this active inference rather than greedy information-gain acquisition?',
    concede: 'Correct on the mathematics: with a one-step horizon there is no path integral, no recursion, and the epistemic term is exactly the salience of epistemic-foraging models. Nothing here needs the deep-policy machinery.',
    reply: 'Three things distinguish it from greedy info-gain even at depth one. First, the objective carries three terms, not one: state information gain, parameter information gain (novelty — the likelihood is being learned), and preference satisfaction, in a single currency with no tuned trade-off weight. Second, action semantics enter through the transition model rather than through the acquisition function — ablation vs steering differ in EFE because their B-matrices differ in entropy, which is how a process theory should encode "what an experiment does". Third, the same architecture extends to depth k by swapping the policy space, with sophisticated-inference tree search as the known upgrade path; a greedy acquisition function has no such continuation. We claim the shallowest member of the family, and say so.',
  },
  {
    id: 'q2', tag: 'EFE theory',
    q: 'Your C vector "prefers high KL". Isn\'t that just a reward function with new labels — and then the whole thing is a bandit in Bayesian clothing?',
    concede: 'C is a goal specification, chosen by us, and preferring large causal effects is a design decision — the discovery objective enters by fiat, exactly as reward would.',
    reply: 'The difference is architectural, not rhetorical. A bandit maximises that reward; this agent trades it against two information terms with no exploration hyperparameter, learns its likelihood online, and — the key empirical exhibit — modulates its actions by belief state: on Gemma it steers 94% of steps after an exploratory opening; on Llama, same C, same priors, it declines to steer 70% of the time because its posterior never licenses it. The paper\'s own bandit baseline needed hand-built uncertainty bonuses, decay schedules and layer priors to be competitive, and its plain-UCB ablation shows those priors were doing the work on Llama. The claim is not "preferences aren\'t rewards"; it is that risk + double epistemic value in one objective buys adaptive behaviour that reward-plus-bonus heuristics had to be tuned into.',
  },
  {
    id: 'q3', tag: 'EFE theory',
    q: 'Millidge, Tschantz and Buckley showed EFE is not the variational free energy of the future — the exploratory term is inserted by a prior-for-posterior swap, not derived. Why build on an objective with contested foundations?',
    concede: 'We take "Whence the EFE?" seriously and make no claim that EFE is uniquely mandated by the FEP. If FEF or FEEF were used instead, the exploratory behaviour would differ, and that is a choice we inherited rather than resolved.',
    reply: 'For this application the contested step is actually the selling point. Circuit discovery needs exploration — an objective that discourages it (FEF) is simply the wrong tool for experiment selection, whatever its derivational purity. We use EFE as what Sajid et al. proved it to be: Bayesian optimal experimental design plus expected utility, a seventy-year-old, well-understood objective wearing process-theory notation. The paper\'s grounding is Lindley and MacKay as much as Friston; if the audience prefers to read the whole system as adaptive BOED with a learned likelihood and action-dependent transitions, nothing in the results changes. The foundations debate is about which objective follows from the FEP — not about whether this objective selects informative experiments, which is the only property we use.',
  },
  {
    id: 'q4', tag: 'EFE theory',
    q: 'So it IS just Bayesian experimental design with extra steps. What does the active-inference framing add beyond branding?',
    concede: 'With flat preferences, discrete EFE reduces exactly to Lindley design — we cite that reduction rather than hide it, and a pure-BOED implementation of the same loop would be a fair baseline we did not run.',
    reply: 'Three concrete additions over vanilla BOED as usually practised. (1) Preferences and information gain live in one objective, so the agent can trade "learn about the circuit" against "confirm the high-impact features" without a scalarisation hyperparameter. (2) The generative model is richer than a BOED acquisition loop: action-conditioned transitions encode intervention semantics, and Dirichlet learning gives the novelty drive — BOED classically optimises over a fixed likelihood. (3) The framing imports a mature process vocabulary — precision, convergence as belief stability, epistemic-to-pragmatic handoffs — that made the emergent behaviours (Gemma\'s explore-then-steer, Llama\'s caution) predictable and diagnosable. Branding would be claiming novelty for the estimator; the paper claims novelty for the application and is explicit about the lineage.',
  },
  {
    id: 'q5', tag: 'EFE theory',
    q: 'Why not sophisticated inference? Sequential experiments have value-of-information interactions a myopic agent cannot see.',
    concede: 'True, and the paper lists single-step planning as a limitation. Complementary intervention sequences — ablate two features that back each other up, or patch then steer the same feature — are invisible to a depth-one agent.',
    reply: 'It was a deliberate first-rung choice: vanilla policy enumeration is exponential in horizon over a ~150-pair action space, and the interpretability payload was already novel. The right upgrade is exactly the one this community built — sophisticated inference\'s recursive tree search with pruning, which handles the branching by thresholding policy posteriors. The infrastructure is ready for it: beliefs, likelihood learning and the intervention engine are horizon-agnostic; only infer_policies changes. Note also that some sequential structure sneaks in even at depth one, because the Dirichlet updates make later EFE evaluations depend on earlier observations — the agent at step 15 is not the agent at step 1.',
  },
  {
    id: 'q6', tag: 'EFE theory',
    q: 'The Llama multi-step failure was fixed by refining the state space, not the objective. Doesn\'t that show the generative model does all the work and EFE is along for the ride?',
    concede: 'The diagnosis is exactly that: three layer-role bins could not represent where Llama\'s causal mass sat, and no objective on a state space that cannot express the answer will find it. The 6-bin fix was chosen by a human, post hoc.',
    reply: 'That is the correct active-inference reading, not an objection to it — model evidence lives in the model, and this community has said for a decade that behaviour is only as good as the generative model licensing it. What the episode actually demonstrates is the diagnostic value of the framing: the failure localised to a nameable model component (the s₁ discretisation), predicted a specific fix, and the fix quadrupled efficiency without touching EFE, γ, η or C — while the hyperparameter sweeps show those knobs do nothing. A heuristic that failed the same way would have offered no such handle. The honest next step is structure learning — letting the agent expand its own state space under model-evidence pressure — which is a known active-inference research line, not an exotic add-on.',
  },
  {
    id: 'q7', tag: 'EFE theory',
    q: 'Did you actually enable the parameter-information-gain (novelty) term in pymdp, or are you learning A without being curious about it?',
    concede: 'A fair implementation audit — pymdp defaults use_param_info_gain to False, and many published "learning" agents forget it.',
    reply: 'Enabled. The agent constructs pymdp\'s Agent with use_utility, use_states_info_gain and use_param_info_gain all True, passes pA initialised at 10×A (mild confidence in the calibrated prior), and learns with η = lr_pA = 1.0 — it is in the paper\'s appendix hyperparameter table ("param info gain: enabled") and checkable in pomdp_agent.py. So the EFE carries the full Da Costa four-term decomposition minus the vanishing evidence bound: pragmatic value, state info gain, and Dirichlet novelty. The novelty term matters here: it draws probes toward candidates whose observation mapping is under-evidenced, which is visibly what drives coverage across layers early in episodes.',
  },
  {
    id: 'q8', tag: 'EFE theory',
    q: 'The B-matrix "intervention semantics" are hand-designed priors. The agent\'s celebrated action policy just reads your assumptions back to you.',
    concede: 'The entropy ordering H(B_abl) > H(B_patch) > H(B_steer) is a hand-specified prior, and the 0.50/0.70/0.90 diagonals are round numbers, stated as such in the appendix.',
    reply: 'Two defences, one empirical, one methodological. Empirically: the same priors produce opposite policies on the two models — explore-then-steer on Gemma, ablation-dominant on Llama — so the behaviour cannot be a readout of B alone; it is B interacting with observed likelihood statistics through inference. A schedule baked into priors could not reverse itself. Methodologically: encoding "what an ablation does to your beliefs" as a transition prior is precisely how a process model should represent action semantics — the alternative is an acquisition function with the same assumptions hidden in code. The rows are symmetric specifically so utility stays action-neutral and the epistemic terms differentiate actions; that is a falsifiable design whose predicted signature (ablate first, steer when confident) can fail, and on Llama partially does.',
  },
  {
    id: 'q9', tag: 'EFE theory',
    q: 'The environment is a frozen, deterministic network. What are your hidden states states OF, and what is q(s) tracking — is this perception or curve-fitting?',
    concede: 'The transformer is deterministic given the prompt; "importance" and "causal influence" are not stochastic latent causes in the world but graded properties we defined, made noisy only by discretisation and measurement granularity.',
    reply: 'The hidden states are epistemic states of the analyst — exactly as in Bayesian experimental design, where θ is a fixed unknown and the posterior tracks knowledge, not physics. Uncertainty here is real but epistemic: before probing, the KL of an intervention is unknown; observations resolve it partially (bins), and beliefs compress evidence into a reusable summary. If one insists active inference requires stochastic environments, then all of BOED falls too — Lindley\'s experiments were deterministic chemistry often enough. What we would concede as substantive: the state space is ours, not learned, and richer ontologies (position-resolved roles, feature-interaction states) are the growth direction. The blanket question has a clean answer too: the boundary is the intervention API — actions cross it one way, KL observations the other.',
  },
  {
    id: 'q10', tag: 'EFE theory',
    q: 'Preferring high KL is preferring disruption. Won\'t the agent chase features that merely break the model — off-manifold vandalism — rather than task-relevant circuitry?',
    concede: 'KL at the final token is a blunt preference: it cannot distinguish "this feature computes the answer" from "zeroing this feature deranges the output distribution", and at steering multiplier 10 some probes are certainly off-distribution.',
    reply: 'Within the ablation-only protocol the criticism has a bound: effects are measured against the oracle on the same KL currency, so "vandalism" is available to every method equally and the comparison stays fair. The steering analysis then confronts the worry directly with the random-feature control — and partially vindicates it, which we report as the H2b null: at high multipliers, amplification is substantially generic. That is the paper policing its own preference function. The forward fix is preference engineering, which this framework makes explicit rather than implicit: C over task-aligned observables (logit-diff toward the correct token, faithfulness-curve area) drops in without touching the agent. The deeper point for this audience: the C vector is where interpretability\'s "what counts as explanation" debate becomes a single, inspectable object.',
  },
  {
    id: 'q11', tag: 'EFE theory',
    q: 'Why a discrete pymdp agent at all? A continuous or amortised generative model would scale and avoid the discretisation that caused your worst failure.',
    concede: 'The KL-threshold sensitivity (82% → 60% under −20% bins) and the Llama multi-step collapse are both discretisation injuries; a continuous observation model would not have them.',
    reply: 'Discrete-state active inference was chosen for interpretability of the interpreter: every belief, likelihood row and preference is a readable number — which is how we could publish the full generative model and let you audit it, and how the failure was diagnosable at all. It is also the formulation with mature, exact EFE machinery (pymdp), whereas continuous EFE still involves sampling approximations whose own biases would confound a first study. The measured costs are known and bounded: thresholds are the sensitive knob, agent hyperparameters are flat. Hybrid schemes — learned quantile bins, or a continuous likelihood with discrete states — are the obvious second iteration, and nothing in the architecture resists them.',
  },
  {
    id: 'q12', tag: 'EFE theory',
    q: 'γ = 16, fixed. No precision dynamics, no affective modulation — isn\'t fixed precision un-active-inference?',
    concede: 'Yes, precision is static; there is no γ update from expected model evidence, no confidence-modulated policy temperature.',
    reply: 'We swept γ across a sixteen-fold range (4 to 64) and bounded efficiency did not move on either model — in this regime EFE differences between candidates are large relative to 1/γ, so policy selection is effectively argmax and precision dynamics would have nothing to grip. That is worth knowing: the interesting precision phenomena this community studies live where EFE landscapes are flat, and per-step candidate landscapes here are not. Where precision dynamics would matter is across episodes — modulating trust in the learned A as evidence accumulates — which is a genuinely open extension.',
  },
  {
    id: 'q13', tag: 'EFE theory',
    q: 'Your agent holds beliefs per candidate feature with a shared 36-state core — a mean-field assumption across features. Circuits are interactions; where is the joint hypothesis space?',
    concede: 'There is no posterior over circuits — no representation of "feature A and B are redundant" or "A gates B". Beliefs factorise across candidates, so backup-head phenomena and OR-structure are invisible by construction.',
    reply: 'That factorisation is what makes the problem tractable at 2,200 candidates, and it mirrors the field\'s own practice — attribution scores, ACDC edges and EAP rankings are all per-component marginals too; nobody currently maintains joint circuit posteriors at this scale. The honest framing: ACD upgrades the per-component marginal from a static score to a learned belief with calibrated uncertainty. Joint structure is exactly what multi-step policies would begin to expose (ablate A, then test B conditional on the outcome), which ties this limitation to the planning-depth one: both point at the same next paper. If someone in the room wants to build hierarchical state spaces over feature ensembles — that is a collaboration, not a rebuttal.',
  },
  {
    id: 'q14', tag: 'EFE theory',
    q: 'Ryan Smith would ask: where is the model validation? Parameter recovery, model comparison against a simpler ideal observer — or is the agent purely illustrative?',
    concede: 'No parameter-recovery study and no formal model comparison against, say, a Kalman-style ideal observer with the same observations. The validation offered is behavioural and comparative (baselines, sensitivity, action-matched controls) rather than the computational-psychiatry standard.',
    reply: 'Some of the standard\'s substance is present under other names: the hyperparameter sweeps are a flatness analysis showing conclusions don\'t depend on fitted constants; the A-convergence curve shows the learned likelihood stabilises rather than oscillating (a parameter-recovery surrogate given ground truth is unknown); and the plain-UCB baseline is an ablated model comparison — same observations, no generative model — quantifying what the model buys. What is genuinely missing and worth doing: simulate a synthetic "circuit" with known ground-truth importance, verify the agent recovers it, and compare marginal likelihoods across generative-model variants. That is a clean IWAI-style follow-up.',
  },
  {
    id: 'q15', tag: 'EFE theory',
    q: 'The agent\'s prior observation comes from graph importance — the same graph EAP ranks. You\'ve given the agent a compressed copy of the baseline and are surprised it doesn\'t beat the original?',
    concede: 'Sharp, and essentially correct as a description: the κ-scaled prior discretises the graph signal into four bins, so the agent starts with a lossy version of what EAP uses losslessly.',
    reply: 'That asymmetry is the experiment, though — the question was never "can 4-bin beliefs out-rank a continuous score at its own game", it was "what does adaptivity add on top of a shared hypothesis source". Given identical graph information, the agent converts it into calibrated uncertainty plus online corrections from real interventions; EAP spends it in fixed order and cannot notice when reality disagrees (its Llama multi-step CI dips to 62.8% on one prompt — no mechanism to recover). The right reading of "EAP wins in-distribution" is that the pruned candidate set is EAP\'s home turf, as the paper says. The interesting deployment regime — where attribution is stale, prompts shift, or intervention types have different costs — is where a belief-updating policy can diverge from its prior, and that is the follow-up experiment worth designing together.',
  },
  {
    id: 'q16', tag: 'EFE theory',
    q: 'Convergence is declared at rolling belief-KL < 0.01. Arbitrary threshold, arbitrary window?',
    concede: 'Both are conventional choices, not derived quantities; a different window or threshold would declare convergence earlier or later.',
    reply: 'Convergence detection is diagnostic, not load-bearing: episodes run their full budget regardless, so no result depends on the flag. Its role is the auditing use-case — a signal that further budget is unlikely to revise beliefs, letting an operator stop early. A principled replacement is easy to state in this framework: stop when expected information gain of the best remaining probe falls below its cost — a proper Bayesian stopping rule that EFE hands you for free. Worth implementing; the current flag is the placeholder version.',
  },

  // ---------------- MECH INTERP ----------------
  {
    id: 'q17', tag: 'Mech interp',
    q: 'EAP beats the agent everywhere. Why should the interpretability community care about a selector that loses to a one-pass gradient ranking?',
    concede: 'On these benchmarks, in-distribution, with candidates drawn from the attribution graph itself, EAP is the better ablation-KL selector, full stop — 91.9% and 95.2% on IOI, unbeaten anywhere.',
    reply: 'Because the comparison is EAP on its own home turf, and the agent still gets within 10 points on Gemma while carrying capabilities EAP lacks by construction: calibrated uncertainty per feature, a multi-action repertoire, online correction when observations contradict attribution, and an audit trail of beliefs. Static rankings have no error bars and no way to notice they are wrong; an auditing workflow needs both. The paper\'s contribution claim was always the policy architecture, stated in the abstract — no efficiency SOTA is claimed. And note the one regime already visible where static ranking wobbles: Llama multi-step EAP\'s CI stretches down to 62.8% because one prompt\'s attribution was misleading; a belief-updating selector is the only class of method that can, in principle, recover mid-episode.',
  },
  {
    id: 'q18', tag: 'Mech interp',
    q: 'KL at the final token is a contested effect measure — the faithfulness-metrics literature shows conclusions flip between KL, logit-diff, and ablation schemes. Why KL?',
    concede: 'Fully aware: Miller et al. show published circuit faithfulness swings wildly across ablation and metric choices, and KL is sign-blind — it cannot see suppressive components like IOI\'s negative name movers as different from promoting ones.',
    reply: 'KL was chosen because it is task-agnostic (works identically across IOI, multi-step, five domains without designing contrastive pairs), strictly non-negative (bounded oracle efficiency needs a non-negative currency), and the native scale of the observation model. The design protects against metric fragility in two ways: everything is comparative — every method, including the oracle, is scored on the same KL, so metric biases cancel in the ratio; and the discretisation sensitivity sweep directly tests how conclusions respond to redefining "how much KL matters", finding the ordering stable. Extending the observation model with a signed logit-diff modality is a two-line change to A and C, and would let the agent hunt suppressive circuitry specifically — a genuinely nice extension this framing makes modular.',
  },
  {
    id: 'q19', tag: 'Mech interp',
    q: 'Your own H2b is null — circuit-selected features are not significantly more steerable than random ones. Why keep steering in the action space at all?',
    concede: 'At the prediction-change level, selectivity did not separate from the control (11 vs 8, 9 vs 6, Fisher n.s.), and multiplier 10 is off-distribution — consistent with the broader steering-reliability literature (AxBench, Tan et al.).',
    reply: 'Steering stays because its role in the agent is epistemic economics, not concept demonstration: it is the low-transition-entropy, high-pragmatic-value action — the "confirm" probe — and removing it collapses the action space the multi-action analysis exists to study. The KL selectivity signal is actually present (0.367 vs 0.086 mean KL at m=10, 4×) even though top-1 flips don\'t reach significance at n=50; and the concept-amplification analysis shows steering redirects mass onto coherent concept tokens rather than noise. What the null kills is a specific strong claim — "graph importance predicts steerability" — and the paper lets it die publicly. That is the corrective the steering literature has been asking for: dose-response curves with matched controls instead of demos.',
  },
  {
    id: 'q20', tag: 'Mech interp',
    q: 'Transcoder replacement models have error nodes and frozen attention — your candidate set structurally misses attention-pattern computation and everything in the reconstruction error. Discovery over what, exactly?',
    concede: 'Correct: candidates are transcoder features only. QK-circuit formation, and any computation living in the CLT/PLT reconstruction error, are outside the agent\'s hypothesis space — the same blind spots Anthropic documents for attribution graphs themselves.',
    reply: 'ACD inherits its substrate\'s ontology deliberately — it is an intervention-selection layer over circuit-tracer, and fixing the substrate\'s coverage is upstream research (better transcoders, attention attribution) that the selection layer benefits from automatically. Note the framework itself is substrate-agnostic: the hidden factors are (importance, locality, causal strength) over any intervenable unit — attention heads, SAE latents, or edges would slot in with a different intervention API. The budget-allocation problem this paper formalises exists on every substrate; we instantiated it on the one with a public causal API and pre-trained dictionaries for two open models.',
  },
  {
    id: 'q21', tag: 'Mech interp',
    q: 'Interventions happen at a single token position and effects are read at the final position only. Positional circuit structure — which IOI is famous for — is invisible.',
    concede: 'True: features whose causal role is specific to earlier positions are only probed through their final-position footprint, and the observation model has no positional resolution.',
    reply: 'The candidate features themselves are position-indexed (layer, position, feature), so positional structure exists in the hypothesis space — what is missing is position-resolved observation. The paper lists a position-resolved observation model as the natural extension; in this architecture it is literally one more modality (KL measured at intermediate positions or per-position logit shifts) plus corresponding A rows. Worth noting the recovered structure is still position-sensible: top IOI features sit at the indirect-object and end positions in late layers, which is where path-patching analyses put the name-mover action.',
  },
  {
    id: 'q22', tag: 'Mech interp',
    q: 'You measure selection efficiency within a pre-pruned candidate set. The real discovery problem — what survives pruning — was solved by circuit-tracer before your agent woke up.',
    concede: 'Yes: the evaluation measures budget allocation within an already-pruned shortlist (~top features by influence, capped per layer), and a different pruning threshold would shift the absolute numbers.',
    reply: 'Deliberate division of labour, stated in the abstract: attribution proposes, interventions dispose, and ACD owns only the disposal budget. Whole-circuit discovery from scratch under a 20-probe budget is information-theoretically hopeless (2,200 candidates, 3 actions); every practical pipeline — Anthropic\'s included — prunes first and verifies second. The agent\'s job begins where cheap evidence ends. The pruning threshold is also shared by every baseline including the oracle, so within-set comparisons are unaffected; only generalisation of the absolute efficiencies is at stake, which the multi-model, multi-task replication addresses.',
  },
  {
    id: 'q23', tag: 'Mech interp',
    q: 'Why is there no ACDC comparison? It is the canonical automated circuit-discovery baseline.',
    concede: 'There is none, and the paper flags it as a real limitation rather than an oversight.',
    reply: 'Substrate mismatch: ACDC operates on the attention-head/MLP edge graph with per-edge corruption patching; ACD operates on transcoder-feature nodes with a fixed per-feature intervention budget. A like-for-like run requires porting the POMDP to edge-level units — a different observation model and intervention engine — which is future work, not a table row. The closest fair established-method comparison available on this substrate is the direct EAP ranking, which we implement carefully (direct-to-logit attribution, distinct from greedy connectivity) and lose to honestly. Comparing incommensurable metrics — ACDC\'s ROC against ground-truth edges vs our oracle-KL efficiency — would have manufactured a win or a loss at will; declining to do so is the defensible choice.',
  },
  {
    id: 'q24', tag: 'Mech interp',
    q: 'Do you know what any of your discovered features actually mean? There are no auto-interp labels, no feature dashboards in the paper.',
    concede: 'The paper does not label features semantically — no auto-interp pass, no Neuronpedia cross-referencing of the specific discovered indices. Discovery here is causal-topological (which features matter, where), not semantic (what they encode).',
    reply: 'That separation was scoped on purpose: semantic labelling has its own literature and failure modes, and grafting it on would have diluted the selection-policy claim. But the pieces connect trivially: every discovered feature is a GemmaScope transcoder latent addressable on Neuronpedia (the Part I embed shows exactly that dashboard type), so a semantic layer is one API call per feature. The layer-level findings already carry meaning without labels: name-mover-like concentration at layers 24–25 for IOI, early-layer recruitment for multi-hop reasoning, replicated across architectures — structure consistent with the established division of labour from the factual-recall literature.',
  },
  {
    id: 'q25', tag: 'Mech interp',
    q: 'Gemma-2-2B late-layer "name movers" — is that actually the same circuit Wang et al. found in GPT-2, or a coincidence of depth fractions?',
    concede: 'Not established at head level: our units are MLP-transcoder features, not attention heads, so identity with the 26-head GPT-2 circuit is not something this method can assert.',
    reply: 'What is established: the causal-mass profile matches — top single features at layers 24–25 of 26 (~92–96% depth), where GPT-2\'s name movers sit at layers 9–10 of 12 (~75–83%), and the cross-scale circuit-consistency literature (Tigges et al. 2024) finds IOI component roles stable across model families and scale. The transcoder features are best read as the MLP-side shadow of the name-mover pathway — the features feeding and reading the move — which is precisely the resolution at which the substrate operates. A head-level confirmation on Gemma would need attention attribution, which the current circuit-tracer stack does not expose.',
  },

  // ---------------- STATISTICS ----------------
  {
    id: 'q26', tag: 'Statistics',
    q: 'Five IOI prompts. Three multi-step prompts. You cannot be serious about p-values at n = 3–5.',
    concede: 'The n is small and the paper says so in its limitations — n ≥ 30 is the standard aspiration, and several CIs are embarrassingly wide (Llama IOI POMDP: [17.4, 90.2]).',
    reply: 'The inferential machinery is matched to the n rather than pretending otherwise: exact paired permutation tests (no distributional assumptions; at n=5 the best attainable one-sided p is 1/32 = 0.031, which the Gemma result achieves — meaning the agent beat random on every prompt), percentile bootstrap CIs on everything, and non-significant results reported as non-acceptances (Llama H1, H2b). Note also each "prompt" aggregates 20 interventions × up to 7 methods × 10 random trials — hundreds of GPU-measured KLs per cell; prompt-level n understates the measurement mass. But yes: scaling to 30+ prompts is straightforward compute (~40–60 s per prompt per method) and is the single highest-value robustness upgrade. The permutation logic was chosen exactly so that larger n strengthens rather than reinterprets the analysis.',
  },
  {
    id: 'q27', tag: 'Statistics',
    q: 'Bootstrap CIs from three prompts are close to meaningless — resampling 3 items gives 10 distinct multisets.',
    concede: 'Statistically fair: at n=3 the percentile bootstrap is coarse and its coverage is nominal at best. Those intervals are honest about order-of-magnitude uncertainty, not precise inference.',
    reply: 'Which is why no claim in the paper rests on a multi-step CI: the multi-step section\'s conclusions are the per-prompt diagnostics themselves (4.4%, 15.2%, 55.9% — reported individually, no aggregation hiding) and a mechanism finding (early-layer mass × coarse bins) that was then causally tested by changing the bins and observing the predicted recovery. That is a stronger epistemic move than a tighter CI: intervene on the hypothesised cause of your own failure. The CIs are printed because hiding dispersion at small n is worse than printing coarse dispersion.',
  },
  {
    id: 'q28', tag: 'Statistics',
    q: 'Dozens of comparisons across tables and no multiple-comparison correction anywhere. How many of these results survive Bonferroni?',
    concede: 'No formal correction is applied, and with this many cells some nominal orderings are certainly noise.',
    reply: 'The confirmatory claims are few and pre-registered as hypotheses — H1 through H4, one test each — and the two that pass (H1-Gemma at p=0.031, H2a at p<10⁻⁸) survive any correction you like over that four-test family (H2a by seven orders of magnitude; H1-Gemma is the family\'s only marginal member and is supported by the per-prompt sweep). Everything else — domain tables, sensitivity grids, action distributions — is explicitly descriptive, presented with CIs rather than stars precisely so it will not be read as a forest of significant findings. The paper\'s discipline is in what it declines to accept: H1-Llama, H2b, and H3-multi-step are all reported as failures despite nominally favourable-looking numbers elsewhere.',
  },
  {
    id: 'q29', tag: 'Statistics',
    q: 'The agent is "deterministic given candidates" but action selection is a softmax sample. Where is the seed variance analysis for the agent itself?',
    concede: 'Correct that pymdp\'s stochastic action selection introduces run-to-run variance which is not swept in the paper; random baselines are averaged over 10 trials but the agent is run once per configuration.',
    reply: 'At γ=16 with per-step EFE gaps the size observed here, the softmax is near-degenerate — the γ sweep (4 to 64, no change to one decimal place in bounded efficiency) is operating evidence that policy sampling noise is not driving results: multiplying the temperature by 16 is a far larger perturbation than resampling at fixed temperature. The hyperparameter grid also functions as a multi-run stability check (nine agent configurations on Gemma IOI within 0.1 of each other). A formal seed sweep is cheap and worth adding; the prediction from the γ analysis is variance well under a percentage point.',
  },

  // ---------------- SCALING & SAFETY ----------------
  {
    id: 'q30', tag: 'Scaling & safety',
    q: 'What actually changes at 70B? Attribution graphs get worse, candidate sets get huge, and your 36-state belief looks quaint.',
    concede: 'Untested beyond 2.6B, and two costs clearly grow: graph construction (already the 18 s bottleneck) and candidate volume; error-node mass also grows, shrinking the visible hypothesis space.',
    reply: 'The scaling logic actually favours adaptive selection as models grow: candidate sets expand faster than verification budgets, so the gap between "what attribution proposes" and "what you can afford to test" — the exact gap ACD manages — widens. The agent\'s own compute is microscopic (36-state operations; milliseconds against 30 ms GPU probes) and constant in model size; only the substrate scales. The genuine open problems at scale are substrate problems — CLT quality, attention attribution — plus one agent problem: the state space should get richer (more role bins, interaction factors) as networks deepen, which is the structure-learning agenda again. Anthropic\'s stack already runs attribution graphs on frontier models; the selection layer rides along.',
  },
  {
    id: 'q31', tag: 'Scaling & safety',
    q: 'The auditing pitch assumes budget-constrained verification. Labs have compute. Why not just run the oracle — ablate everything?',
    concede: 'For a 2B model and one prompt, exhaustive ablation of a pruned set is genuinely affordable — our own oracle did it (2,200 ablations ≈ a minute). At that scale the budget constraint is soft.',
    reply: 'The constraint hardens along three axes at once: model scale (per-probe cost grows), prompt coverage (auditing is per-behaviour, and behaviours × prompts multiply), and action space (steering sweeps and patches multiply the per-feature cost — our multi-action space is already 3×, a multiplier sweep 18×). Exhaustive verification across a deployment-scale behaviour suite is not a compute line item anyone pays today — which is why current practice verifies a hand-chosen handful of features, i.e. an implicit, unprincipled selection policy. The proposal is to make that policy explicit, uncertainty-calibrated and audit-loggable. When compute is truly unconstrained, the agent gracefully degrades into the oracle; the interesting regimes are everywhere else.',
  },
  {
    id: 'q32', tag: 'Scaling & safety',
    q: 'The discussion gestures at deception detection. That is a big cheque for a method that lost to a gradient ranking on IOI.',
    concede: 'It is future-work language, and the paper claims no deception result. Nothing here demonstrates detection of anything adversarial.',
    reply: 'The specific relevance is narrower and more defensible: deception-style audits are the worst case for static attribution — the behaviours are rare, prompt-sensitive, and the auditor cannot trust that a single gradient pass on a benign prompt ranks the right features. Sequential probing with belief updating is the natural shape for "find the features whose causal role changes across contexts", and the multi-action repertoire (does this feature respond to steering the way an honest-circuit feature should?) is a probe vocabulary static rankings do not have. The honest status: architecture-appropriate, evidence-pending — and this audience\'s own work on epistemically-driven agents is exactly what would move it.',
  },
  {
    id: 'q33', tag: 'Scaling & safety',
    q: 'Your agent\'s priors (D biased to "unimportant", C toward high KL) could systematically under-probe exactly the subtle features a safety audit exists to find.',
    concede: 'Real risk: a sparsity prior plus disruption-seeking preferences steers budget away from low-KL, high-subtlety circuitry — precision-critical but quiet features would be deprioritised.',
    reply: 'Three mitigations, two already in the machinery. The novelty term actively fights the prior: features with under-evidenced observation mappings attract probes regardless of expected KL — curiosity is pointed at the unexplained, not the loud. Dirichlet learning means a quiet feature that produces a surprising bin reshapes the likelihood and earns follow-up. And because priors are explicit objects, an auditor can run the complementary agent — C inverted to prefer confirming unimportance, or D flattened — and diff the discovered sets; try that with an implicit heuristic. The general principle stands though: any budgeted auditor has a prior, and the only safe version is one whose prior is inspectable. That is an argument for this architecture, not against it.',
  },

  // ---------------- RAPID FIRE ----------------
  {
    id: 'q34', tag: 'Rapid fire',
    q: 'One sentence: transcoder vs SAE?',
    concede: '',
    reply: 'An SAE learns a sparse dictionary that describes what a representation contains; a transcoder learns a sparse replacement for what an MLP does — input to output — so causal paths through the MLP become explicit weights instead of gradients through a black box.',
  },
  {
    id: 'q35', tag: 'Rapid fire',
    q: 'Why Gemma-2-2B and Llama-3.2-1B specifically?',
    concede: '',
    reply: 'They were the two models circuit-tracer shipped with public per-layer transcoders at release — GemmaScope\'s 16k-per-layer set for Gemma, the community mntss set for Llama — i.e. the only open models where causally-correct feature interventions were available off the shelf, and conveniently a deep-vs-shallow architectural contrast (26 vs 16 layers) that ended up mattering.',
  },
  {
    id: 'q36', tag: 'Rapid fire',
    q: 'What exactly is an attribution-graph edge?',
    concede: '',
    reply: 'Within the per-prompt local replacement model — transcoders in place of MLPs, attention patterns and layernorm denominators frozen, error nodes added — the network is linear in feature activations, so an edge is the exact linear direct effect of one feature\'s activation on another\'s pre-activation (or on a logit). Exact in that surrogate; hypothesis about the real model, which is why interventions remain the ground truth.',
  },
  {
    id: 'q37', tag: 'Rapid fire',
    q: 'Total compute for the paper?',
    concede: '',
    reply: 'Modest by design: ~40–60 s per prompt per method on a single GPU (attribution ~18 s dominates; interventions ~30 ms each), full multi-model suite in hours on a DGX Spark GB10 — and the whole pipeline fits in 5 GB VRAM in float32, which is why every experiment reproduces on a free Colab T4 via the repo\'s notebooks.',
  },
  {
    id: 'q38', tag: 'Rapid fire',
    q: 'Is the code actually runnable by us, and where do we poke the generative model?',
    concede: '',
    reply: 'github.com/SharathSPhD/ActiveCIrcuitDiscovery — MIT; the agent is one file (src/active_inference/pomdp_agent.py) with A/B/C/D construction in ~150 readable lines, dumped verbatim to results/generative_model.json (the matrices rendered in Part II). Flags expose γ, η, pragmatic weight, layer-role bins and discretisation scales; pymdp pinned; three Colab notebooks reproduce IOI, steering and the agent comparison on a free T4. Fork the B-matrix and disagree with us empirically — that is the point of shipping it.',
  },
  {
    id: 'q39', tag: 'Rapid fire',
    q: 'If EFE selection only clearly wins on Gemma, what is the one-line takeaway you want us to leave with?',
    concede: '',
    reply: 'That circuit discovery is finally posed as what it always was — sequential experimental design under uncertainty — with a full generative-model treatment on real frontier-adjacent substrates, honest controls, and every failure diagnosable in model terms; the numbers say the first, shallowest agent is already competitive, and this room knows better than anyone what the deeper members of that family can do.',
  },
];

import type { Metadata } from 'next';
import Link from 'next/link';
import { Kicker, Callout, Deep, LinkCards, Fig, Eq, M } from '../../components/Prose';
import { LoopSVG, DictionarySVG } from '../../components/DiagramsPart2';
import GenModelInspector from '../../components/GenModelInspector';

export const metadata: Metadata = {
  title: 'Part II — The Bridge: Circuit Discovery as Active Inference',
  description:
    'The conceptual mapping and the full POMDP machinery: generative model, per-step EFE, Dirichlet learning, and an honest account of design choices.',
};

const CANON_CARDS = [
  {
    src: 'JMP · 2020',
    title: 'Active inference on discrete state-spaces',
    desc: 'Da Costa et al. — the exact discrete formulation ACD instantiates: risk+ambiguity, epistemic+pragmatic, and the novelty term for Dirichlet-learned likelihoods.',
    href: 'https://arxiv.org/abs/2001.07203',
  },
  {
    src: 'Neural Comput · 2017',
    title: 'Active inference: a process theory',
    desc: 'Friston et al. — epistemic value = expected information gain = mutual information; the softmax(−γG) policy prior ACD uses verbatim.',
    href: 'https://activeinference.github.io/papers/process_theory.pdf',
  },
  {
    src: 'JOSS · 2022',
    title: 'pymdp',
    desc: 'Heins et al. — the community’s reference implementation. ACD runs the vanilla fixed-point VI Agent with utility, state info gain and parameter info gain all enabled.',
    href: 'https://joss.theoj.org/papers/10.21105/joss.04098',
  },
  {
    src: 'Ann. Math. Stat. · 1956',
    title: 'Lindley: a measure of information provided by an experiment',
    desc: 'The seventy-year-old ancestor: choose the experiment maximising expected information gain. The EFE epistemic term is this quantity.',
    href: 'https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-27/issue-4/On-a-Measure-of-the-Information-Provided-by-an-Experiment/10.1214/aoms/1177728069.full',
  },
  {
    src: 'Neural Comput · 1992',
    title: 'MacKay: information-based objective functions',
    desc: 'Active data selection by expected information gain — the machine-learning branch of the same lineage, cited by the paper as its statistical grounding.',
    href: 'https://direct.mit.edu/neco/article/4/4/590/5648/Information-Based-Objective-Functions-for-Active',
  },
  {
    src: 'arXiv · 2021',
    title: 'Sajid et al.: EFE as BOED + expected utility',
    desc: 'Flat preferences reduce EFE to Lindley design; no ambiguity reduces it to expected utility. The deflationary reading — and why ACD embraces it.',
    href: 'https://arxiv.org/abs/2110.04074',
  },
];

export default function Page() {
  return (
    <>
      <section className="band dark" style={{ paddingTop: '4.5rem' }}>
        <div className="reading">
          <Kicker>Part II · ~20 minutes</Kicker>
          <h2 className="sec">The bridge: circuit discovery as active inference</h2>
          <p className="lede">
            Part I ended with a selection problem: thousands of candidate causal experiments, a
            budget of twenty, and a field that spends it non-adaptively. This part makes one move —{' '}
            <em>the interpretability researcher is an agent, so model the researcher</em> — and then
            builds the machinery in full, at the level of detail you would demand of any pymdp
            paper. Nothing is hidden: every matrix below is the one that shipped.
          </p>
        </div>
      </section>

      <section className="band">
        <div className="reading">
          <Kicker>II.1 · The conceptual move</Kicker>
          <h2 className="sec">Interpretability is perception with a scalpel</h2>
          <p>
            What does a mechanistic interpretability researcher actually do? They maintain beliefs
            about hidden structure — “layer 25 feature 4717 might be a name mover” — that they can
            never observe directly. They choose experiments: ablate it, patch it, steer it. They
            receive noisy observations — a KL divergence, an activation, a connectivity pattern —
            and update. And they stop when further experiments would not change their beliefs. That
            is not <em>like</em> a perception–action loop; it <em>is</em> one, with the transformer
            playing the role of the environment and the intervention API playing the role of the
            sensorium&rsquo;s motor interface.
          </p>
          <p>
            The field already runs this loop — in the researcher&rsquo;s head, or in a greedy
            script. ACD&rsquo;s claim is architectural: make the loop explicit, give it a generative
            model, and let Expected Free Energy allocate the budget. The exploration–exploitation
            schedule that every heuristic hand-tunes (uncertainty bonuses, decay rates,
            layer priors — the paper&rsquo;s bandit baseline needs all three) then <em>falls out</em>{' '}
            of the objective. And because the epistemic term of EFE is exactly Lindley&rsquo;s
            expected information gain, this is simultaneously a seventy-year-old statistical recipe —
            Bayesian optimal experimental design — wearing its biological formulation. The paper is
            explicit about this dual citizenship, and so should the talk be: for this audience, the
            deflationary reading (“EFE = BOED + preferences”) is not an attack, it is the point of
            contact.
          </p>
          <Fig wide caption={
            <>The loop, concretely. Left: the pymdp agent — beliefs over three factors per candidate,
            EFE over the joint (feature, action) space, Boltzmann selection, Dirichlet learning.
            Right: the environment is the transformer itself, probed through{' '}
            <code>feature_intervention</code>, answering in discretised KL. One full cycle ≈ 30 ms
            of GPU time.</>
          }>
            <LoopSVG />
          </Fig>
          <Callout title="What is genuinely non-standard here" tone="violet">
            Three things, flagged before anyone asks. (1) The <strong>environment is real and
            deterministic</strong> — a frozen network — so “hidden states” are epistemic states of
            the analyst, not stochastic latent causes. (2) The agent&rsquo;s <strong>policy space is
            the joint (candidate × intervention-type) space</strong>, re-instantiated per step —
            closer to Bayesian experimental design than to navigation POMDPs. (3){' '}
            <strong>Planning depth is one</strong>: this is myopic EFE, and Part V discusses exactly
            what that concedes (single-step EFE ≈ salience + preference; no sophisticated
            inference). The claim is not “this is the deepest possible active inference agent”; it
            is “even the shallowest one already organises the problem.”
          </Callout>
        </div>
      </section>

      <section className="band soft">
        <div className="reading">
          <Kicker>II.2 · The dictionary</Kicker>
          <h2 className="sec">Every object, mapped</h2>
          <p>
            The fastest way to evaluate a claimed active-inference application is to demand the
            dictionary: what, exactly, realises each object of the formalism? Here it is, complete.
            Notation follows Parr, Pezzulo &amp; Friston (2022).
          </p>
          <DictionarySVG />
          <p>
            Hidden factors: <strong>s₀ importance</strong> ∈ {'{'}negligible, low, moderate, high{'}'},{' '}
            <strong>s₁ layer role</strong> ∈ {'{'}early, middle, late{'}'} (network depth in thirds
            — the discretisation that Part III convicts for the Llama multi-step failure), and{' '}
            <strong>s₂ causal influence</strong> ∈ {'{'}weak, moderate, strong{'}'}: 36 joint states
            per candidate. Observation modalities: <strong>o₀ KL magnitude</strong> in four bins
            (thresholds 10⁻⁴, 10⁻³, 10⁻²), <strong>o₁ activation magnitude</strong> in four bins,{' '}
            <strong>o₂ graph connectivity</strong> in three. Actions: ablation, activation patching,
            feature steering — the three levers <code>feature_intervention</code> exposes.
          </p>
          <Deep title="Deep dive: the prior observation trick (κ = 0.01)">
            <p>
              Before any intervention, the agent seeds each candidate&rsquo;s belief from graph
              metadata: normalised graph importance imp(i) ∈ [0,1] is scaled by κ = 0.01 and
              discretised through the <em>same</em> KL bins the agent will later observe from real
              probes. The scaling is not arbitrary: observed ablation KL on these models lives in
              roughly [10⁻⁴, 10⁻²], so κ maps the top of the importance range onto the top of the
              empirical KL range. Without it, every candidate saturates the “large KL” bin and the
              prior is uninformative. This is the agent&rsquo;s empirical prior — attribution as
              prior, intervention as likelihood — and it is also an answerable critique: the prior
              and the likelihood are not independent sources of evidence. Part V, Q17.
            </p>
          </Deep>
        </div>
      </section>

      <section className="band dark2">
        <div className="reading">
          <Kicker>II.3 · The generative model, exactly as shipped</Kicker>
          <h2 className="sec">A, B, C, D — inspect them</h2>
          <p>
            This widget renders <code>results/generative_model.json</code> from the repository —
            the actual matrices, not an illustration. The B tab is the one to linger on: it encodes
            the paper&rsquo;s one deliberate piece of intervention semantics.
          </p>
          <GenModelInspector />
          <h3 className="sub">The B-matrix commitment</h3>
          <p>
            Factor 0&rsquo;s transitions are action-conditioned: ablation&rsquo;s matrix has
            diagonal 0.50 (25% mass up, 25% down — broad revision possible), patching 0.70,
            steering 0.90 (near-identity). Rows are symmetric around the current state, deliberately:
            symmetric transitions keep expected <em>utility</em> roughly action-invariant, so what
            differentiates actions in the EFE is their <strong>entropy</strong> — and hence their
            epistemic value. The design commitment is only the monotone ordering
          </p>
          <Eq tex={String.raw`H(\mathbf{B}_{\text{ablate}}) \;>\; H(\mathbf{B}_{\text{patch}}) \;>\; H(\mathbf{B}_{\text{steer}}),`} />
          <p>
            mirroring the operational semantics: deleting a feature licenses the largest belief
            revision; rescaling one licenses the smallest. The specific values (0.50/0.70/0.90) are
            the simplest round numbers realising the ordering. This is a <em>prior</em>, stated as
            such — and it is falsifiable within the paper: the emergent behaviour it predicts
            (ablate while uncertain, steer once confident) is what the action traces in Part III
            show on Gemma, and interestingly <em>fail</em> to show on Llama, where observed KL
            statistics push the agent to keep ablating. The behaviour is model-sensitive, which is
            what you want from an inference, not a schedule.
          </p>
        </div>
      </section>

      <section className="band">
        <div className="reading">
          <Kicker>II.4 · The objective</Kicker>
          <h2 className="sec">The per-step EFE, term by term</h2>
          <p>The paper&rsquo;s Equation 6, specialised from the canonical policy EFE with π = a single (feature i, action u) pair:</p>
          <Eq tex={String.raw`G(i,u) \;=\; \underbrace{\mathbb{E}_{Q}\!\left[\log Q(s_{\tau}\,|\,\pi)\;-\;\log Q(s_{\tau}\,|\,o_{\tau},\pi)\right]}_{-\,\text{state information gain (salience)}} \;+\; \underbrace{\mathbb{E}_{Q}\!\left[\log P(o_{\tau})\;-\;\log Q(o_{\tau}\,|\,\pi)\right]}_{-\,\text{pragmatic value}}`} />
          <p>
            with <M tex={String.raw`(i^{*},u^{*}) = \arg\min_{(i,u)} G(i,u)`} /> softened through{' '}
            <M tex={String.raw`P(\pi)\propto e^{-\gamma G(\pi)}`} />, γ = 16. Three implementation
            facts matter to this audience, and all three are checkable in{' '}
            <code>src/active_inference/pomdp_agent.py</code>:
          </p>
          <p>
            <strong>First — the novelty term is on.</strong> The agent runs pymdp with{' '}
            <code>use_utility=True</code>, <code>use_states_info_gain=True</code>, and{' '}
            <code>use_param_info_gain=True</code>, with Dirichlet concentration pA initialised at
            10·A. Since the likelihood is being learned online, EFE here carries the third,
            frequently-forgotten term of the Da Costa decomposition — expected information gain
            about the <em>parameters</em> of A (novelty), computed via the SPM{' '}
            <M tex={String.raw`W = \tfrac{1}{2}\big(\mathrm{diag}(\mathbf{a}^{-1}) - (\mathbf{a}_{0})^{-1}\big)`} />-style
            count-sensitivity. The agent is curious about features whose <em>observation mapping</em>{' '}
            is under-evidenced, not just features whose state is uncertain.
          </p>
          <p>
            <strong>Second — where the risk lives.</strong> pymdp&rsquo;s carve-up is utility over
            observations plus state info gain; by the standard entropy cancellation this equals
            −(risk over observations + ambiguity) exactly. C is a per-modality log-preference
            vector, monotone in KL bin and activation bin (inspect the C tab above). So
            “preferring high KL” is implemented as risk-minimisation toward a preference
            distribution over <em>observations</em> — the orthodox discrete-state construction, and
            the honest place to admit that the discovery goal enters by fiat. What blunts the
            “relabelled reward” attack is the paper&rsquo;s own ablation geometry: rows of B are
            symmetric precisely so that utility differences across <em>actions</em> stay small and
            the epistemic terms do the differentiating. Selection pressure between actions is
            informational by construction; selection pressure between <em>features</em> mixes both,
            as any experimental-design objective must.
          </p>
          <p>
            <strong>Third — the update loop is real variational inference.</strong> Per step:
            vanilla fixed-point VI infers <M tex={String.raw`q(s_{0},s_{1},s_{2}\,|\,o_{0},o_{1},o_{2})`} />;
            the Dirichlet update
          </p>
          <Eq tex={String.raw`p_{A_{m}}(o_{m},\,s_{0},s_{1},s_{2}) \;\mathrel{+}=\; \eta\; q(s_{0})\,q(s_{1})\,q(s_{2}), \qquad \eta = 1`} />
          <p>
            sculpts the likelihood; and convergence is declared when the rolling belief-KL between
            successive posteriors drops below 0.01. Part III shows this learning is not
            decorative: the per-step L1 drift of the learned KL-likelihood halves over the
            20-step budget on Gemma IOI — the observation model measurably stabilises as evidence
            accumulates.
          </p>
          <Deep title="Deep dive: what a single step costs, computationally">
            <p>
              Per candidate set (~40–60 features after per-layer capping), the agent evaluates G
              over ≤180 (feature, action) pairs, each a handful of 36-state matrix-vector products —
              microseconds. The GPU-side intervention dominates at ~30 ms. Total agent overhead per
              20-step episode is negligible against one 18 s attribution pass. Scaling pressure is
              therefore entirely in the candidate set size and the budget, not in the EFE machinery
              — until you deepen the policy horizon, where vanilla enumeration goes as
              |feature × action|^depth and sophisticated-inference-style tree search becomes the
              relevant upgrade path (Part V, Q5).
            </p>
          </Deep>
        </div>
      </section>

      <section className="band soft">
        <div className="reading">
          <Kicker>II.5 · Lineage</Kicker>
          <h2 className="sec">Where this sits in your literature</h2>
          <p>
            The paper is deliberate about its ancestry: Lindley (1956) defined experiment value as
            expected information gain; MacKay (1992) brought it to learning systems; the modern
            BOED and active-causal-discovery literature (Tigas et al. 2022; Rainforth et al. 2024)
            selects intervention targets and values against a posterior over causal structures.
            EFE&rsquo;s epistemic term is provably this same quantity — mutual information between
            states and observations (Friston et al. 2015, Eq. 4) — and with flat preferences,
            discrete EFE <em>is</em> Lindley design (Sajid et al. 2021). What active inference adds
            is not a new estimator but an architecture: preferences and information gain in one
            currency, likelihood learning with its own curiosity term, and action semantics carried
            by the transition model. ACD is, to our knowledge, the first time this architecture has
            been pointed at the internals of another neural network — the nearest neighbours
            (MAIA&rsquo;s LLM experiment loops; ACDC&rsquo;s exhaustive sweeps) have either the
            adaptivity or the formalism, never both.
          </p>
          <LinkCards cards={CANON_CARDS} />
          <Callout title="A symmetry worth savouring" tone="amber">
            The environment being probed is itself a prediction machine. An explicit generative
            model (36 states, hand-specifiable) is being used to reverse-engineer an implicit one
            (2.6B parameters, opaque) — surprise minimisation turned inward on an artefact of
            surprise minimisation. The paper published in <em>Symmetry</em>; the pun writes itself,
            but the methodological point is serious: interpretability tools and cognitive theories
            are converging on the same loop, and this audience owns the theory of that loop.
          </Callout>
          <div style={{ marginTop: '2.2rem', display: 'flex', gap: '0.8rem', flexWrap: 'wrap' }}>
            <Link
              href="/results"
              style={{ fontFamily: 'var(--grotesk)', fontWeight: 600, fontSize: '.9rem', textDecoration: 'none', background: 'var(--teal)', color: '#fff', padding: '11px 22px', borderRadius: 999 }}
            >
              Part III: what actually happened →
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}

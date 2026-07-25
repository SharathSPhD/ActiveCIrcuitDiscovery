'use client';

import Link from 'next/link';
import Deck, { Slide, Reveal } from '../../components/Deck';
import { Eq, M, LinkCards, Callout } from '../../components/Prose';
import { LoopSVG, DictionarySVG } from '../../components/DiagramsPart2';
import GenModelInspector from '../../components/GenModelInspector';

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

export default function BridgeDeck() {
  return (
    <Deck part="Part II · ~20 min" title="The bridge">
      {/* 1 — title */}
      <Slide
        steps={1}
        notes={
          <>
            <p>
              Part I ended with a selection problem: thousands of candidate causal experiments, a
              budget of twenty, and a field that spends it non-adaptively. This part makes one move
              — <em>the interpretability researcher is an agent, so model the researcher</em> — and
              then builds the machinery in full, at the level of detail you would demand of any
              pymdp paper. Nothing is hidden: every matrix in this deck is the one that shipped.
            </p>
          </>
        }
      >
        <p className="kicker">Part II · ~20 minutes</p>
        <h1>
          The bridge:
          <br />
          <span className="accent">circuit discovery as active inference</span>
        </h1>
        <Reveal at={1}>
          <div className="take teal">
            One move: <strong>the interpretability researcher is an agent — so model the
            researcher.</strong>
          </div>
        </Reveal>
      </Slide>

      {/* 2 — the researcher's loop */}
      <Slide
        steps={4}
        notes={
          <>
            <p>
              What does a mechanistic interpretability researcher actually do? They maintain
              beliefs about hidden structure — “layer 25 feature 4717 might be a name mover” —
              that they can never observe directly. They choose experiments: ablate it, patch it,
              steer it. They receive noisy observations — a KL divergence, an activation, a
              connectivity pattern — and update. And they stop when further experiments would not
              change their beliefs. That is not <em>like</em> a perception–action loop; it{' '}
              <em>is</em> one, with the transformer playing the role of the environment and the
              intervention API playing the role of the sensorium&rsquo;s motor interface.
            </p>
            <p>
              The field already runs this loop — in the researcher&rsquo;s head, or in a greedy
              script. ACD&rsquo;s claim is architectural: make the loop explicit, give it a
              generative model, and let Expected Free Energy allocate the budget. The
              exploration–exploitation schedule that every heuristic hand-tunes (uncertainty
              bonuses, decay rates, layer priors — the bandit baseline needs all three) then{' '}
              <em>falls out</em> of the objective. And because the epistemic term of EFE is exactly
              Lindley&rsquo;s expected information gain, this is simultaneously a seventy-year-old
              statistical recipe — Bayesian optimal experimental design — wearing its biological
              formulation. The paper is explicit about this dual citizenship: for this audience,
              the deflationary reading (“EFE = BOED + preferences”) is not an attack, it is the
              point of contact.
            </p>
          </>
        }
      >
        <p className="kicker">II.1 · The conceptual move</p>
        <h1>What does a circuit researcher actually do?</h1>
        <ul className="pts">
          <Reveal at={1}>
            <li>
              Holds <strong>beliefs</strong> about hidden structure — “L25 F4717 might be a name
              mover” — never observed directly
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              Chooses <strong>experiments</strong> — ablate, patch, steer
            </li>
          </Reveal>
          <Reveal at={3}>
            <li>
              Gets noisy <strong>observations</strong> — a KL, an activation — and updates; stops
              when beliefs settle
            </li>
          </Reveal>
        </ul>
        <Reveal at={4}>
          <div className="take teal">
            That is not <em>like</em> a perception–action loop. <strong>It is one.</strong>
          </div>
        </Reveal>
      </Slide>

      {/* 3 — the loop diagram */}
      <Slide
        steps={1}
        notes={
          <>
            <p>
              The loop, concretely. Left: the pymdp agent — beliefs over three factors per
              candidate, EFE over the joint (feature, action) space, Boltzmann selection, Dirichlet
              learning. Right: the environment is the transformer itself, probed through{' '}
              <code>feature_intervention</code>, answering in discretised KL. One full cycle ≈ 30 ms
              of GPU time.
            </p>
          </>
        }
      >
        <p className="kicker">II.1 · The loop, concretely</p>
        <div className="fig-panel" style={{ maxWidth: 1050 }}>
          <LoopSVG />
        </div>
        <Reveal at={1}>
          <div className="stats" style={{ marginTop: '1rem' }}>
            <div className="stat">
              <div className="v">pymdp</div>
              <div className="k">the agent — full EFE, Dirichlet learning</div>
            </div>
            <div className="stat">
              <div className="v">Gemma-2-2B</div>
              <div className="k">the environment — probed via feature_intervention</div>
            </div>
            <div className="stat">
              <div className="v">≈ 30 ms</div>
              <div className="k">one full perception–action cycle</div>
            </div>
          </div>
        </Reveal>
      </Slide>

      {/* 4 — what is non-standard */}
      <Slide
        steps={3}
        notes={
          <>
            <p>
              Three things flagged before anyone asks. (1) The <strong>environment is real and
              deterministic</strong> — a frozen network — so “hidden states” are epistemic states
              of the analyst, not stochastic latent causes. (2) The agent&rsquo;s{' '}
              <strong>policy space is the joint (candidate × intervention-type) space</strong>,
              re-instantiated per step — closer to Bayesian experimental design than to navigation
              POMDPs. (3) <strong>Planning depth is one</strong>: this is myopic EFE, and Part V
              discusses exactly what that concedes (single-step EFE ≈ salience + preference; no
              sophisticated inference). The claim is not “this is the deepest possible active
              inference agent”; it is “even the shallowest one already organises the problem.”
            </p>
          </>
        }
      >
        <p className="kicker">II.1 · On the table before you ask</p>
        <h1>Three non-standard choices</h1>
        <ul className="pts">
          <Reveal at={1}>
            <li>
              The environment is <strong>real and deterministic</strong> — hidden states are the{' '}
              <em>analyst&rsquo;s</em> epistemic states
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              Policy space = <strong>joint (feature × intervention)</strong>, re-instantiated per
              step — closer to BOED than to navigation POMDPs
            </li>
          </Reveal>
          <Reveal at={3}>
            <li>
              <strong>Planning depth = 1</strong> — myopic EFE, and Part V owns what that concedes
            </li>
          </Reveal>
        </ul>
        <Reveal at={3}>
          <div className="take">
            The claim is not “the deepest possible agent.” It is:{' '}
            <strong>even the shallowest one already organises the problem.</strong>
          </div>
        </Reveal>
      </Slide>

      {/* 5 — the dictionary */}
      <Slide
        steps={2}
        notes={
          <>
            <p>
              The fastest way to evaluate a claimed active-inference application is to demand the
              dictionary: what, exactly, realises each object of the formalism? Notation follows
              Parr, Pezzulo &amp; Friston (2022).
            </p>
            <p>
              Hidden factors: <strong>s₀ importance</strong> ∈ {'{'}negligible, low, moderate,
              high{'}'}, <strong>s₁ layer role</strong> ∈ {'{'}early, middle, late{'}'} (network
              depth in thirds — the discretisation that Part III convicts for the Llama multi-step
              failure), and <strong>s₂ causal influence</strong> ∈ {'{'}weak, moderate, strong{'}'}:
              36 joint states per candidate. Observation modalities: <strong>o₀ KL magnitude</strong>{' '}
              in four bins (thresholds 10⁻⁴, 10⁻³, 10⁻²), <strong>o₁ activation magnitude</strong>{' '}
              in four bins, <strong>o₂ graph connectivity</strong> in three. Actions: ablation,
              activation patching, feature steering — the three levers{' '}
              <code>feature_intervention</code> exposes.
            </p>
            <h3>The prior observation trick (κ = 0.01)</h3>
            <p>
              Before any intervention, the agent seeds each candidate&rsquo;s belief from graph
              metadata: normalised graph importance imp(i) ∈ [0,1] is scaled by κ = 0.01 and
              discretised through the <em>same</em> KL bins the agent will later observe from real
              probes. The scaling is not arbitrary: observed ablation KL on these models lives in
              roughly [10⁻⁴, 10⁻²], so κ maps the top of the importance range onto the top of the
              empirical KL range. Without it, every candidate saturates the “large KL” bin and the
              prior is uninformative. This is the agent&rsquo;s empirical prior — attribution as
              prior, intervention as likelihood — and it invites a fair critique: the prior and the
              likelihood are not independent sources of evidence. Part V, Q17.
            </p>
          </>
        }
      >
        <p className="kicker">II.2 · The dictionary</p>
        <h1>
          Every object, <span className="accent">mapped</span>
        </h1>
        <div className="fig-panel" style={{ maxWidth: 1050 }}>
          <DictionarySVG />
        </div>
        <ul className="pts compact" style={{ marginTop: '0.8rem' }}>
          <Reveal at={1}>
            <li>
              <strong>36 hidden states</strong> per candidate: importance (4) × layer role (3) ×
              causal influence (3)
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              <strong>3 observation modalities</strong> (KL, activation, connectivity) ·{' '}
              <strong>3 actions</strong> (ablate, patch, steer)
            </li>
          </Reveal>
        </ul>
      </Slide>

      {/* 6 — generative model inspector */}
      <Slide
        lazy
        notes={
          <>
            <p>
              This widget renders <code>results/generative_model.json</code> from the repository —
              the actual matrices that shipped, not an illustration. The B tab is the one to linger
              on: it encodes the paper&rsquo;s one deliberate piece of intervention semantics.
            </p>
          </>
        }
      >
        <p className="kicker">II.3 · The generative model, exactly as shipped</p>
        <h2>
          A, B, C, D — <span className="dim">these are the actual matrices. Inspect them.</span>
        </h2>
        <GenModelInspector />
      </Slide>

      {/* 7 — B-matrix commitment */}
      <Slide
        steps={3}
        notes={
          <>
            <p>
              Factor 0&rsquo;s transitions are action-conditioned: ablation&rsquo;s matrix has
              diagonal 0.50 (25% mass up, 25% down — broad revision possible), patching 0.70,
              steering 0.90 (near-identity). Rows are symmetric around the current state,
              deliberately: symmetric transitions keep expected <em>utility</em> roughly
              action-invariant, so what differentiates actions in the EFE is their{' '}
              <strong>entropy</strong> — and hence their epistemic value. The specific values
              (0.50/0.70/0.90) are the simplest round numbers realising the ordering.
            </p>
            <p>
              This is a <em>prior</em>, stated as such — and it is falsifiable within the paper:
              the emergent behaviour it predicts (ablate while uncertain, steer once confident) is
              what the action traces in Part III show on Gemma, and interestingly <em>fail</em> to
              show on Llama, where observed KL statistics push the agent to keep ablating. The
              behaviour is model-sensitive, which is what you want from an inference, not a
              schedule.
            </p>
          </>
        }
      >
        <p className="kicker">II.3 · The one piece of intervention semantics</p>
        <h1>The B-matrix commitment</h1>
        <ul className="pts">
          <Reveal at={1}>
            <li>
              Deleting a feature licenses the <strong>largest belief revision</strong>; rescaling
              one, the smallest
            </li>
          </Reveal>
        </ul>
        <Reveal at={1}>
          <Eq tex={String.raw`H(\mathbf{B}_{\text{ablate}}) \;>\; H(\mathbf{B}_{\text{patch}}) \;>\; H(\mathbf{B}_{\text{steer}})`} />
        </Reveal>
        <ul className="pts">
          <Reveal at={2}>
            <li>
              Rows symmetric → utility stays action-invariant → actions differ by{' '}
              <strong>epistemic value only</strong>
            </li>
          </Reveal>
        </ul>
        <Reveal at={3}>
          <div className="take violet">
            It is a <strong>prior, and it is falsifiable</strong> — it predicts: ablate while
            uncertain, steer once confident. Part III tests exactly that.
          </div>
        </Reveal>
      </Slide>

      {/* 8 — the EFE */}
      <Slide
        steps={2}
        notes={
          <>
            <p>
              The paper&rsquo;s Equation 6, specialised from the canonical policy EFE with π = a
              single (feature i, action u) pair. Selection is{' '}
              <M tex={String.raw`(i^{*},u^{*}) = \arg\min_{(i,u)} G(i,u)`} /> softened through{' '}
              <M tex={String.raw`P(\pi)\propto e^{-\gamma G(\pi)}`} />, γ = 16.
            </p>
          </>
        }
      >
        <p className="kicker">II.4 · The objective</p>
        <h1>One score per (feature, action)</h1>
        <Eq tex={String.raw`G(i,u) \;=\; \underbrace{\mathbb{E}_{Q}\!\left[\log Q(s_{\tau}\,|\,\pi)\;-\;\log Q(s_{\tau}\,|\,o_{\tau},\pi)\right]}_{-\,\text{state information gain}} \;+\; \underbrace{\mathbb{E}_{Q}\!\left[\log P(o_{\tau})\;-\;\log Q(o_{\tau}\,|\,\pi)\right]}_{-\,\text{pragmatic value}}`} />
        <ul className="pts" style={{ marginTop: '0.8rem' }}>
          <Reveal at={1}>
            <li>
              <strong>Epistemic</strong>: which probe teaches me most about this feature?
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              <strong>Pragmatic</strong>: which probe finds the high-KL structure I am{' '}
              <em>looking for</em>?
            </li>
          </Reveal>
        </ul>
        <Reveal at={2}>
          <p className="big dim" style={{ marginTop: '0.6rem' }}>
            <M tex={String.raw`P(\pi)\propto e^{-\gamma G(\pi)}`} />, γ = 16 — your process-theory
            softmax, verbatim.
          </p>
        </Reveal>
      </Slide>

      {/* 9 — three implementation facts */}
      <Slide
        steps={3}
        notes={
          <>
            <p>
              <strong>First — the novelty term is on.</strong> The agent runs pymdp with{' '}
              <code>use_utility=True</code>, <code>use_states_info_gain=True</code>, and{' '}
              <code>use_param_info_gain=True</code>, with Dirichlet concentration pA initialised at
              10·A. Since the likelihood is being learned online, EFE carries the third,
              frequently-forgotten term of the Da Costa decomposition — expected information gain
              about the <em>parameters</em> of A (novelty), computed via the SPM{' '}
              <M tex={String.raw`W = \tfrac{1}{2}\big(\mathrm{diag}(\mathbf{a}^{-1}) - (\mathbf{a}_{0})^{-1}\big)`} />-style
              count-sensitivity. The agent is curious about features whose observation mapping is
              under-evidenced, not just features whose state is uncertain.
            </p>
            <p>
              <strong>Second — where the risk lives.</strong> pymdp&rsquo;s carve-up is utility
              over observations plus state info gain; by the standard entropy cancellation this
              equals −(risk over observations + ambiguity) exactly. C is a per-modality
              log-preference vector, monotone in KL bin and activation bin. So “preferring high
              KL” is implemented as risk-minimisation toward a preference distribution over{' '}
              <em>observations</em> — the orthodox discrete-state construction, and the place where
              the discovery goal enters by fiat. What blunts the “relabelled reward” attack is the
              ablation geometry: rows of B are symmetric precisely so that utility differences
              across <em>actions</em> stay small and the epistemic terms do the differentiating.
            </p>
            <p>
              <strong>Third — the update loop is real variational inference.</strong> Per step:
              vanilla fixed-point VI infers{' '}
              <M tex={String.raw`q(s_{0},s_{1},s_{2}\,|\,o_{0},o_{1},o_{2})`} />; the Dirichlet
              update{' '}
              <M tex={String.raw`p_{A_{m}} \mathrel{+}= \eta\, q(s_{0})\,q(s_{1})\,q(s_{2})`} />,
              η = 1, sculpts the likelihood; convergence is declared when the rolling belief-KL
              between successive posteriors drops below 0.01. Part III shows this learning is not
              decorative: the per-step L1 drift of the learned KL-likelihood halves over the
              20-step budget on Gemma IOI.
            </p>
            <h3>What a single step costs</h3>
            <p>
              Per candidate set (~40–60 features after per-layer capping), the agent evaluates G
              over ≤180 (feature, action) pairs, each a handful of 36-state matrix-vector products
              — microseconds. The GPU-side intervention dominates at ~30 ms. Total agent overhead
              per 20-step episode is negligible against one 18 s attribution pass. Scaling pressure
              is entirely in the candidate set size and the budget — until you deepen the policy
              horizon, where enumeration goes as |feature × action|^depth and
              sophisticated-inference-style tree search becomes the upgrade path (Part V, Q5).
            </p>
          </>
        }
      >
        <p className="kicker">II.4 · Checkable in pomdp_agent.py</p>
        <h1>Three facts for the pymdp readers</h1>
        <ul className="pts">
          <Reveal at={1}>
            <li>
              <strong>The novelty term is on</strong> — pA = 10·A, so the agent is curious about
              under-evidenced <em>likelihood parameters</em>, not just uncertain states
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              <strong>The discovery goal enters as C</strong> — risk-minimisation toward preferred{' '}
              <em>observations</em> (high KL), the orthodox construction
            </li>
          </Reveal>
          <Reveal at={3}>
            <li>
              <strong>The updates are real VI</strong> — fixed-point posteriors, Dirichlet learning
              of A (η = 1), convergence at rolling belief-KL &lt; 0.01
            </li>
          </Reveal>
        </ul>
      </Slide>

      {/* 10 — lineage */}
      <Slide
        steps={2}
        notes={
          <>
            <p>
              The paper is deliberate about its ancestry: Lindley (1956) defined experiment value
              as expected information gain; MacKay (1992) brought it to learning systems; the
              modern BOED and active-causal-discovery literature (Tigas et al. 2022; Rainforth et
              al. 2024) selects intervention targets and values against a posterior over causal
              structures. EFE&rsquo;s epistemic term is provably this same quantity — mutual
              information between states and observations (Friston et al. 2015, Eq. 4) — and with
              flat preferences, discrete EFE <em>is</em> Lindley design (Sajid et al. 2021). What
              active inference adds is not a new estimator but an architecture: preferences and
              information gain in one currency, likelihood learning with its own curiosity term,
              and action semantics carried by the transition model. ACD is, to my knowledge, the
              first time this architecture has been pointed at the internals of another neural
              network — the nearest neighbours (MAIA&rsquo;s LLM experiment loops; ACDC&rsquo;s
              exhaustive sweeps) have either the adaptivity or the formalism, never both.
            </p>
            <LinkCards cards={CANON_CARDS} />
          </>
        }
      >
        <p className="kicker">II.5 · Lineage</p>
        <h1>
          Seventy years old,
          <br />
          <span className="accent">wearing your formulation</span>
        </h1>
        <ul className="pts">
          <Reveal at={1}>
            <li>
              Lindley 1956 → MacKay 1992 → modern BOED: <strong>experiment value = expected
              information gain</strong>
            </li>
          </Reveal>
          <Reveal at={1}>
            <li>
              Flat preferences reduce discrete EFE <em>exactly</em> to Lindley design (Sajid 2021)
              — I embrace the deflationary reading
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              What active inference adds is <strong>architecture</strong>: preferences + info gain
              in one currency, likelihood learning with its own curiosity, action semantics in B
            </li>
          </Reveal>
        </ul>
      </Slide>

      {/* 11 — symmetry + handoff */}
      <Slide
        steps={1}
        notes={
          <>
            <Callout title="A symmetry worth savouring" tone="amber">
              The environment being probed is itself a prediction machine. An explicit generative
              model (36 states, hand-specifiable) is being used to reverse-engineer an implicit one
              (2.6B parameters, opaque) — surprise minimisation turned inward on an artefact of
              surprise minimisation. The paper appeared in <em>Symmetry</em>; the pun writes
              itself, but the methodological point is serious: interpretability tools and cognitive
              theories are converging on the same loop, and this audience owns the theory of that
              loop.
            </Callout>
          </>
        }
      >
        <p className="kicker">II.5 · One symmetry before the numbers</p>
        <h1>
          A generative model,
          <br />
          <span className="accent">reverse-engineering a generative model</span>
        </h1>
        <p className="big dim">
          36 explicit states, probing 2.6B implicit parameters — surprise minimisation turned
          inward on an artefact of surprise minimisation.
        </p>
        <Reveal at={1}>
          <div className="take" style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
            <span>Did it work? Every number, next.</span>
            <Link
              href="/results"
              style={{
                fontFamily: 'var(--grotesk)',
                fontWeight: 600,
                fontSize: '.9rem',
                textDecoration: 'none',
                background: 'var(--teal)',
                color: '#fff',
                padding: '10px 20px',
                borderRadius: 999,
                whiteSpace: 'nowrap',
              }}
            >
              Part III: what actually happened →
            </Link>
          </div>
        </Reveal>
      </Slide>
    </Deck>
  );
}

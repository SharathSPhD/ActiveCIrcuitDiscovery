'use client';

import Link from 'next/link';
import Deck, { Slide, Reveal, NextLead } from '../../components/Deck';
import { Eq, M, LinkCards, Callout } from '../../components/Prose';
import { LoopSVG, DictionarySVG } from '../../components/DiagramsPart2';
import {
  TwentyQSVG, PhysicianSVG, StateCubeSVG, BLeversSVG,
  ExploreExploitSVG, TimelineSVG,
} from '../../components/TeachBridge';
import GenModelInspector from '../../components/GenModelInspector';
import EFEIntuition from '../../components/EFEIntuition';
import { ShieldSVG, ForagingSVG, DualEngineSVG, ToolkitSVG } from '../../components/TeachStory';

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
    title: 'Lindley: a measure of the information provided by an experiment',
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
    src: 'Neural Comput · 2021',
    title: 'Millidge, Tschantz & Buckley: Whence the Expected Free Energy?',
    desc: 'EFE is not the expectation of the VFE — it is a chosen objective. The standing foundational question; the deflationary BOED reading here is agnostic on it.',
    href: 'https://arxiv.org/abs/2004.08128',
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
    <Deck part="Chapter II · The bridge" title="Circuit discovery as active inference">
      {/* 1 — chapter title */}
      <Slide
        title="The turn"
        brief="Friston's principle, the dark-room problem, and the one move: model the researcher"
        steps={2}
        notes={
          <>
            <p>
              Chapter I ended with a selection problem: thousands of candidate causal experiments,
              a budget of twenty, and a field that spends the budget non-adaptively. This chapter
              makes one move — <em>the interpretability researcher is an agent, so model the
              researcher</em> — and then builds the machinery in full, at the level of detail any
              pymdp paper would be held to. Every matrix shown is the one that shipped in the
              code.
            </p>
            <p>
              The framework supplying the objective is Karl Friston&rsquo;s free energy principle,
              often described in thermodynamic language — organisms as shields against surprise.
              Treat that as the poster, not the theorem: the free energy in this talk is{' '}
              <em>variational</em> — an information-theoretic bound, not heat — and nothing in the
              results depends on the physics analogy. What is taken from the framework is precise
              and modest: a single objective that prices information and preference in one
              currency.
            </p>
            <p>
              The dark-room problem, spelled out: if agents merely minimised surprise, the optimal
              policy would be to retreat somewhere dark and quiet and never move — sensory input
              would match the prediction of nothingness forever. Biological agents do not do
              this, and the framework&rsquo;s resolution is Expected Free Energy: the objective
              scores <em>future</em> actions by the information they are expected to bring in,
              not just the surprise they avoid. That epistemic drive is precisely what a
              budget-constrained experiment chooser needs.
            </p>
            <p>
              The route: two everyday intuitions first (a party game and a physician), then the
              foraging reframe, the three-engine machine, the loop in full, the generative model
              piece by piece, the toolkit of actions, the objective, and finally where this
              paradigm sits and a falsifiable prediction that Chapter III will test.
            </p>
          </>
        }
      >
        <p className="kicker">Chapter II · the turn</p>
        <h1>
          Circuit discovery
          <br />
          <span className="accent">as active inference</span>
        </h1>
        <div className="fig-panel" style={{ maxWidth: 960 }}>
          <ShieldSVG />
        </div>
        <Reveal at={1}>
          <div className="take violet">
            The classic objection: a surprise-minimiser should hide in a dark, quiet room forever
            — every prediction perfectly confirmed. <strong>Why do organisms explore?</strong> The
            answer is the crown jewel of the framework: <strong>Expected Free Energy</strong> —
            and it is exactly the objective this problem needs.
          </div>
        </Reveal>
        <Reveal at={2}>
          <div className="take teal">
            One move only: <strong>the researcher is an agent — so model the researcher</strong>,
            with that objective.
          </div>
        </Reveal>
        <NextLead>Before any mathematics — a party game everyone already knows.</NextLead>
      </Slide>

      {/* 2 — 20 questions */}
      <Slide
        title="Intuition 1: Twenty Questions"
        brief="A good experiment is one whose answer you cannot predict — expected information gain"
        steps={2}
        notes={
          <>
            <p>
              Twenty Questions contains the entire theory of experiment selection. With 1,000
              candidate animals, “does it live in water?” splits the possibilities roughly in half
              — whatever the answer, about half the candidates are eliminated, a guaranteed 1 bit
              of information. “Is it a blue whale?” could win instantly, but with probability
              1/1000; the expected information gain is nearly zero. Skilled players ask
              distribution-halving questions early and specific questions only once the
              possibilities have narrowed.
            </p>
            <p>
              The principle, stated once and used all chapter:{' '}
              <strong>a good experiment is one whose outcome you cannot predict</strong> — because
              a predictable outcome, once observed, teaches nothing. One refinement keeps it
              honest: the outcome must be unpredictable <em>to you</em>, not intrinsically random
              — formally, expected information gain is expected surprise <em>minus</em> expected
              ambiguity, so a noisy measurement channel legitimately suppresses it. (That
              distinction returns in Chapter III: noisier observation statistics on Llama are
              exactly why the agent behaves differently there.) This quantity was formalised by
              Lindley in 1956 and is precisely the epistemic term of Expected Free Energy (EFE) —
              the objective this chapter builds.
            </p>
          </>
        }
      >
        <p className="kicker">II.1 · Intuition one — a game</p>
        <h1>Twenty Questions</h1>
        <div className="fig-panel" style={{ maxWidth: 1000 }}>
          <TwentyQSVG />
        </div>
        <Reveal at={1}>
          <div className="take teal">
            A good experiment is one <strong>whose answer you cannot predict</strong> — unpredictable{' '}
            <em>to you</em>, not intrinsically noisy. A static-filled channel is unpredictable and
            teaches nothing.
          </div>
        </Reveal>
        <Reveal at={2}>
          <NextLead>
            But pure curiosity is not enough — sometimes the search has a goal. Enter the
            physician.
          </NextLead>
        </Reveal>
      </Slide>

      {/* 3 — the physician */}
      <Slide
        title="Intuition 2: the physician"
        brief="Beliefs over diagnoses, tests chosen by informativeness AND stakes, updates, a stopping rule"
        steps={2}
        notes={
          <>
            <p>
              A physician facing a difficult case holds a <em>differential</em> — a probability
              distribution over diagnoses. Tests differ in informativeness and cost: a blood panel
              is cheap and weak, a CT with contrast is expensive and sharp. Each result updates the
              differential, and testing stops when the treatment decision is clear.
            </p>
            <p>
              Two pulls act on every test choice, and this is the crucial addition beyond Twenty
              Questions: <strong>learn the most</strong> (information), and{' '}
              <strong>find what matters for the patient</strong> (preference). A test that is
              informative about something clinically irrelevant loses to a slightly less
              informative test that bears on the treatment. Any objective for experiment selection
              needs both terms — and Expected Free Energy is exactly the sum of those two terms in
              one currency.
            </p>
          </>
        }
      >
        <p className="kicker">II.1 · Intuition two — a professional</p>
        <h1>The physician&rsquo;s loop</h1>
        <div className="fig-panel" style={{ maxWidth: 1020 }}>
          <PhysicianSVG />
        </div>
        <Reveal at={1}>
          <div className="take">
            This is the seesaw the whole chapter balances:{' '}
            <strong>G = <span style={{ color: 'var(--teal-bright)' }}>epistemic</span> (explore) +{' '}
            <span style={{ color: 'var(--amber)' }}>pragmatic</span> (exploit)</strong> — formalised
            a few slides ahead.
          </div>
        </Reveal>
        <Reveal at={2}>
          <NextLead>Now watch the circuit researcher do exactly the same thing.</NextLead>
        </Reveal>
      </Slide>

      {/* 4 — the researcher is that agent */}
      <Slide
        title="Epistemic foraging"
        brief="The researcher recast as an agent following the scent of uncertainty through the graph"
        steps={4}
        notes={
          <>
            <p>
              What does a mechanistic-interpretability researcher actually do? They hold beliefs
              about hidden structure — “layer 25 feature 4717 might be a name mover” — that can
              never be observed directly. They choose experiments: ablate it, patch it, steer it.
              They receive noisy observations — a KL divergence, an activation, a connectivity
              pattern — and update. They stop when further experiments would not change their
              beliefs. That is not <em>like</em> a perception–action loop; it <em>is</em> one,
              with the transformer as the environment and the intervention API as the motor
              interface. One caution before the foraging image does any work: the network is
              frozen and fully deterministic — nothing in the environment is random. What is dark
              here is the <em>map</em>, not the territory; “foraging” describes the agent&rsquo;s
              uncertainty about a fixed structure, and the mathematics treats it exactly that way
              (the hidden states are states of knowledge, not hidden causes in the machine).
            </p>
            <p>
              The field already runs this loop — in researchers&rsquo; heads, or in greedy
              scripts. ACD&rsquo;s claim is architectural: make the loop explicit, give it a
              generative model, and let Expected Free Energy allocate the budget. The
              exploration–exploitation schedule that heuristics hand-tune (uncertainty bonuses,
              decay rates, layer priors — the paper&rsquo;s bandit baseline needs all three) then
              falls out of the objective.
            </p>
          </>
        }
      >
        <p className="kicker">II.1 · The move, made</p>
        <h1>
          Epistemic <span className="accent">foraging</span>
        </h1>
        <div className="fig-panel" style={{ maxWidth: 1000, marginBottom: '0.6rem' }}>
          <ForagingSVG />
        </div>
        <ul className="pts compact">
          <Reveal at={1}>
            <li>
              Holds <strong>beliefs</strong> about hidden structure — “L25 F4717 might be a name
              mover” — never directly observable
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              Chooses <strong>experiments</strong> — ablate, patch, steer — under a hard budget
            </li>
          </Reveal>
          <Reveal at={3}>
            <li>
              Receives noisy <strong>observations</strong> — a KL, an activation — updates, and
              stops when beliefs settle
            </li>
          </Reveal>
        </ul>
        <Reveal at={4}>
          <div className="take teal">
            That is not <em>like</em> a perception–action loop. <strong>It is one.</strong> The
            transformer is the environment; the intervention API is the sensorimotor interface.
          </div>
        </Reveal>
        <NextLead>First, the machine at arm&rsquo;s length — three engines, one loop.</NextLead>
      </Slide>


      {/* NEW — beat 8: the dual-engine machine */}
      <Slide
        title="The machine: three engines, one loop"
        brief="Attribution backend → active inference agent → intervention engine — the systems view"
        steps={1}
        notes={
          <>
            <p>
              The systems view before the zoom-in. Three engines: the{' '}
              <strong>attribution backend</strong> (Anthropic&rsquo;s circuit-tracer running Edge
              Attribution Patching) builds the graph — the map; the{' '}
              <strong>active inference agent</strong> (vanilla pymdp, minimising Expected Free
              Energy) selects which (feature, lever) to test next; the{' '}
              <strong>intervention engine</strong> (the feature API) executes the probe on the real
              network and returns the observation that updates beliefs. Each engine already
              existed. The contribution of the paper is the loop that connects them.
            </p>
          </>
        }
      >
        <p className="kicker">II.2 · The systems view</p>
        <h1>
          Three engines, <span className="accent">one loop</span>
        </h1>
        <div className="fig-panel" style={{ maxWidth: 1020 }}>
          <DualEngineSVG />
        </div>
        <Reveal at={1}>
          <div className="take teal">
            Each engine already existed. <strong>The contribution is the loop.</strong>
          </div>
        </Reveal>
        <NextLead>Zoom into the middle engine — every box, filled in.</NextLead>
      </Slide>

      {/* 5 — the mapped loop */}
      <Slide
        title="The loop, fully mapped"
        brief="pymdp agent on the left, Gemma-2-2B on the right — one cycle ≈ 30 ms"
        steps={1}
        notes={
          <>
            <p>
              The loop, concretely. Left: the pymdp agent — beliefs over three factors per
              candidate, EFE over the joint (feature, action) space, Boltzmann selection,
              Dirichlet learning of the likelihood. Right: the environment is the transformer
              itself, probed through <code>feature_intervention</code>, answering in discretised
              KL — the KL ruler from Chapter I, chopped into four bins so a discrete agent can
              count outcomes. One full perception–action cycle costs about 30 ms of GPU time.
              Vocabulary for the chapter: a <strong>POMDP</strong> (partially observable Markov
              decision process) is the standard formal shell for an agent that acts on hidden
              states it can only observe noisily; <strong>pymdp</strong> is the community&rsquo;s
              reference Python implementation of discrete active inference; <strong>ACD</strong>{' '}
              (Active Circuit Discovery) is the paper&rsquo;s system.
            </p>
            <Callout title="Three non-standard properties, flagged up front" tone="violet">
              (1) The environment is <strong>real and deterministic</strong> — a frozen network —
              so “hidden states” are epistemic states of the analysis, not stochastic latent
              causes. (2) The policy space is the <strong>joint (candidate × intervention-type)
              space</strong>, re-instantiated per step — closer to Bayesian experimental design
              than to navigation POMDPs. (3) <strong>Planning depth is one</strong> — myopic EFE;
              Chapter V&rsquo;s question bank covers exactly what that concedes. The claim is not
              “the deepest possible agent”; it is “even the shallowest one already organises the
              problem.”
            </Callout>
          </>
        }
      >
        <p className="kicker">II.2 · Boxes filled in</p>
        <div className="fig-panel" style={{ maxWidth: 1060 }}>
          <LoopSVG />
        </div>
        <Reveal at={1}>
          <div className="stats" style={{ marginTop: '0.8rem' }}>
            <div className="stat">
              <div className="v">pymdp</div>
              <div className="k">the community&rsquo;s standard open-source active-inference agent library — used vanilla, nothing custom</div>
            </div>
            <div className="stat">
              <div className="v">Gemma-2-2B</div>
              <div className="k">the environment — a frozen, deterministic network</div>
            </div>
            <div className="stat">
              <div className="v">≈ 30 ms</div>
              <div className="k">one full perception–action cycle</div>
            </div>
          </div>
        </Reveal>
        <NextLead>Any claimed “X as active inference” owes a dictionary. Here it is, complete.</NextLead>
      </Slide>

      {/* 6 — the dictionary */}
      <Slide
        title="The dictionary, complete"
        brief="Every object of the formalism mapped to its realisation — nothing hand-waved"
        steps={1}
        notes={
          <>
            <p>
              The fastest way to evaluate a claimed active-inference application is to demand the
              dictionary: what, exactly, realises each object of the formalism? Notation follows
              Parr, Pezzulo &amp; Friston (2022). One row deserves special attention before the
              matrices appear: the <em>prior observation trick</em>. Before any intervention, each
              candidate&rsquo;s belief is seeded from graph metadata — normalised graph importance
              scaled by κ = 0.01 and discretised through the same KL bins later used for real
              probes. The scaling maps the top of the importance range onto the top of the
              empirical KL range (ablation KLs on these models live in roughly [10⁻⁴, 10⁻²]);
              without it every candidate saturates the top bin and the prior is uninformative.
              Attribution as prior, intervention as likelihood — with two fair critiques kept
              attached: the prior and likelihood are not independent evidence sources, and
              strictly these are <em>pseudo-observations</em> pushed through the shared
              likelihood rather than a prior in the D sense. The same seeding plausibly explains
              why beliefs collapse fast on Gemma (the abrupt handoff in Chapter III).
            </p>
          </>
        }
      >
        <p className="kicker">II.3 · Modeling the causal topology</p>
        <h1>
          Every object, <span className="accent">mapped</span>
        </h1>
        <div className="cols cols-60">
          <div className="fig-panel">
            <DictionarySVG />
          </div>
          <div>
            <Reveal at={1}>
              <div className="fig-panel">
                <StateCubeSVG />
              </div>
              <ul className="pts compact" style={{ marginTop: '0.5rem' }}>
                <li>
                  <strong>importance</strong>: how much this feature matters for <em>this
                  prompt&rsquo;s output</em> · <strong>causal influence</strong>: how strongly it
                  drives <em>other features downstream</em> — related, not the same
                </li>
                <li>
                  <strong>layer role</strong>: the layer <em>index</em> is known; what&rsquo;s
                  inferred is the causal <em>role</em> its depth-band plays for this prompt
                </li>
              </ul>
            </Reveal>
          </div>
        </div>
        <NextLead>Words are cheap. The next slide is the actual shipped model — interactive.</NextLead>
      </Slide>

      {/* 7 — inspector */}
      <Slide
        title="A, B, C, D — the shipped matrices"
        brief="Interactive: inspect the exact generative model from the repository"
        lazy
        notes={
          <>
            <p>
              This widget renders <code>results/generative_model.json</code> from the repository —
              the actual matrices that shipped, not an illustration. The A tab shows the
              likelihood mapping hidden importance to observable KL bins; B shows the
              action-conditioned transitions; C the log-preferences over observations (monotone in
              KL bin); D the sparse prior over importance. The B tab is the one to linger on: it
              encodes the paper&rsquo;s single deliberate piece of intervention semantics, unpacked
              on the next slide.
            </p>
          </>
        }
      >
        <p className="kicker">II.3 · Nothing hidden</p>
        <h2>
          The generative model, exactly as shipped — <span className="dim">click through it</span>
        </h2>
        <ul className="pts compact" style={{ marginBottom: '0.7rem' }}>
          <li>
            <strong>A</strong> likelihood: hidden importance → which KL bin you&rsquo;ll see ·{' '}
            <strong>B</strong> transitions: what each lever lets beliefs do ·{' '}
            <strong>C</strong> preferences: which observations count as success ·{' '}
            <strong>D</strong> prior: where beliefs start
          </li>
        </ul>
        <GenModelInspector />
        <NextLead>One question before that B tab makes sense: why three levers at all?</NextLead>
      </Slide>


      {/* NEW — beat 10: the toolkit of epistemic actions */}
      <Slide
        title="The toolkit of epistemic actions"
        brief="Redundancy means one probe type can't map a circuit: necessity / sufficiency / capacity"
        steps={1}
        notes={
          <>
            <p>
              Why does the action space need three levers? Because circuits contain{' '}
              <strong>redundancy</strong> — Chapter I&rsquo;s IOI circuit has <em>backup name
              movers</em> that take over when the primaries are knocked out. A feature can be
              sufficient but not necessary; ablating it shows nothing while it still matters.
              Mapping true causal topology therefore needs three different questions:{' '}
              <strong>ablation tests necessity</strong> (sever the node — does the behaviour
              survive?), <strong>patching tests sufficiency</strong> (swap a corrupted value for a
              clean one — does the behaviour appear?), and <strong>steering tests capacity</strong>{' '}
              (amplify the outgoing signal — what can this feature drive?). These are the same
              three levers from Chapter I&rsquo;s instrument slide — now with their epistemic
              roles named.
            </p>
          </>
        }
      >
        <p className="kicker">II.4 · Why three levers</p>
        <h1>
          The toolkit of <span className="accent">epistemic actions</span>
        </h1>
        <div className="fig-panel" style={{ maxWidth: 1000 }}>
          <ToolkitSVG />
        </div>
        <Reveal at={1}>
          <div className="take">
            Chapter I&rsquo;s backup name movers are the proof: <strong>redundancy defeats any
            single probe type.</strong>
          </div>
        </Reveal>
        <NextLead>How does a discrete agent know these levers differ? One matrix.</NextLead>
      </Slide>

      {/* 8 — B matrix */}
      <Slide
        title="The B-matrix commitment"
        brief="Ablate licenses broad belief revision, steer almost none — a falsifiable prior"
        steps={3}
        notes={
          <>
            <p>
              Factor 0&rsquo;s transitions are action-conditioned: ablation&rsquo;s matrix has
              diagonal 0.50 (25% mass up, 25% down — broad revision possible), patching 0.70,
              steering 0.90 (near-identity). Rows are symmetric around the current state,
              deliberately: symmetric transitions keep expected <em>utility</em> roughly
              action-invariant, so what differentiates actions inside the EFE is their{' '}
              <strong>entropy</strong> — their epistemic value. The specific numbers are the
              simplest round values realising the ordering; only the ordering is the commitment.
            </p>
            <p>
              Worth pre-empting: the principled encoding of “ablation is more informative” would
              be action-conditioned <em>likelihoods</em> A(o|s,u); belief-diffusing B(u) is the
              tractable surrogate available in vanilla pymdp, and yes — a wider B partly{' '}
              <em>manufactures</em> the epistemic value it then harvests. That is exactly why the
              commitment is stated as an ordering and tested rather than assumed: the Llama
              reversal in Chapter III is evidence the ordering does inferential work rather than
              dictating behaviour.
            </p>
            <p>
              And it is falsifiable within the paper: it predicts <em>ablate while uncertain,
              steer once confident</em>. Chapter III&rsquo;s action traces show exactly that on
              Gemma — and, instructively, <em>not</em> on Llama, where observed KL statistics keep
              the agent ablating. Model-sensitive behaviour is what distinguishes an inference
              from a schedule.
            </p>
          </>
        }
      >
        <p className="kicker">II.4 · The one designed piece</p>
        <h1>Three levers, three licenses</h1>
        <Reveal at={1}>
          <div className="fig-panel" style={{ maxWidth: 1000 }}>
            <BLeversSVG />
          </div>
        </Reveal>
        <Reveal at={2}>
          <Eq tex={String.raw`\bar{H}[\mathbf{B}(\cdot\mid s,\text{ablate})] \;>\; \bar{H}[\mathbf{B}(\cdot\mid s,\text{patch})] \;>\; \bar{H}[\mathbf{B}(\cdot\mid s,\text{steer})]`} />
          <p className="big dim" style={{ marginTop: '0.4rem' }}>
            The network never changes — B describes how far each probe <em>permits beliefs to
            move</em>. H̄ = mean row entropy: wider spread, more revision licensed.
          </p>
        </Reveal>
        <Reveal at={3}>
          <div className="take violet">
            A <strong>prior, and falsifiable</strong>: it predicts <em>ablate while uncertain,
            steer once confident</em>. Chapter III tests exactly this.
          </div>
        </Reveal>
        <NextLead>Now the objective that turns beliefs + levers into a choice.</NextLead>
      </Slide>

      {/* 9 — EFE equation */}
      <Slide
        title="The objective: one score per experiment"
        brief="G(i,u) = epistemic value + pragmatic value — the game and the physician, formalised"
        steps={3}
        notes={
          <>
            <p>
              The paper&rsquo;s Equation 6, specialised from the canonical policy EFE with π = a
              single (feature i, action u) pair, shown here in the epistemic + pragmatic carve-up:
              G = −(state information gain) − (expected log-preference over observations). The
              equivalent risk + ambiguity form is
              G = D<sub>KL</sub>[Q(o|π) ‖ P(o)] + expected ambiguity; the two are related by the
              standard entropy identity. Selection is{' '}
              <M tex={String.raw`(i^{*},u^{*}) = \arg\min_{(i,u)} G(i,u)`} /> softened through{' '}
              <M tex={String.raw`P(\pi)\propto e^{-\gamma G(\pi)}`} />, γ = 16, fixed (the
              process-theory treatment updates γ itself via a precision prior; that mechanism is
              off here).
            </p>
            <p>
              The two terms are the two intuitions of this chapter. The epistemic term is Twenty
              Questions: expected reduction in uncertainty about the feature&rsquo;s hidden state.
              The pragmatic term is the physician&rsquo;s stakes: a preference (C) for observing
              large causal effects — high KL bins — because finding load-bearing structure is what
              the search is <em>for</em>.
            </p>
          </>
        }
      >
        <p className="kicker">II.5 · The objective</p>
        <h1>One score per (feature, lever)</h1>
        <Reveal at={1}>
          <p className="big">
            In plain words first:&ensp;
            <strong style={{ color: 'var(--teal-bright)' }}>−G = (what the probe would teach)</strong>{' '}
            + <strong style={{ color: 'var(--amber)' }}>(how much you&rsquo;d like what it finds)</strong>.
            &ensp;Pick the minimum G.
          </p>
        </Reveal>
        <Reveal at={2}>
          <Eq tex={String.raw`G(i,u) \;=\; -\underbrace{\mathbb{E}_{Q}\!\left[\log Q(s_{\tau}\,|\,o_{\tau},\pi)\;-\;\log Q(s_{\tau}\,|\,\pi)\right]}_{\text{epistemic value (information gain)}} \;-\; \underbrace{\mathbb{E}_{Q}\!\left[\log P(o_{\tau})\right]}_{\text{pragmatic value (expected log-preference)}}`} />
          <p className="dim" style={{ fontFamily: 'var(--grotesk)', fontSize: 'clamp(.8rem,1.2vw,.98rem)', maxWidth: '75ch' }}>
            Q = the agent&rsquo;s current beliefs · P(o) ∝ exp(C), the preferences read as
            “probabilities the agent wants to be true” · π = the single probe (i, u) under
            consideration (planning depth 1) · τ = the moment after the probe
          </p>
        </Reveal>
        <ul className="pts compact" style={{ marginTop: '0.5rem' }}>
          <Reveal at={2}>
            <li>
              <strong>Epistemic</strong> = Twenty Questions: which probe&rsquo;s answer is hardest
              to predict (given a clean channel)?
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              <strong>Pragmatic</strong> = the physician&rsquo;s stakes: which probe finds the{' '}
              <em>load-bearing</em> structure?
            </li>
          </Reveal>
        </ul>
        <Reveal at={3}>
          <p className="big dim">
            <M tex={String.raw`P(\pi)\propto e^{-\gamma G(\pi)}`} />, γ = 16 — inverse
            temperature, fixed at the SPM/pymdp stock value (precision learning off). Provenance,
            stated plainly: the same quantity neuroscience uses to model action selection — and,
            older and plainer, Lindley&rsquo;s 1956 information gain plus a preference term.
          </p>
        </Reveal>
        <NextLead>Equations describe; the next slide lets the trade-off be felt.</NextLead>
      </Slide>

      {/* 10 — interactive EFE */}
      <Slide
        title="Feel the trade-off"
        brief="Interactive: two sliders describe a feature; watch the chosen lever flip"
        lazy
        notes={
          <>
            <p>
              The widget uses toy numbers with the honest shape of the system: each action&rsquo;s
              epistemic capacity follows its B-matrix entropy (ablate &gt; patch &gt; steer), and
              its pragmatic pull follows the observed effect scale (steering produces by far the
              largest KLs). Two experiments to try live: (1) set uncertainty high and expected
              effect low — ablation wins, on epistemic value; (2) drag uncertainty down, as if
              evidence has accumulated over several probes, and raise expected effect — the choice
              flips to steering, on pragmatic value. That flip, repeated over 20 steps, is the
              exploration→exploitation handoff — and nothing schedules it.
            </p>
          </>
        }
      >
        <p className="kicker">II.5 · Hands on</p>
        <h2>
          Two sliders describe one feature — <span className="dim">the objective picks the lever</span>
        </h2>
        <EFEIntuition />
        <NextLead>Played over 20 steps, this trade-off writes a schedule nobody coded.</NextLead>
      </Slide>

      {/* 11 — explore/exploit prediction */}
      <Slide
        title="The prediction: a schedule nobody wrote"
        brief="Epistemic value decays, pragmatic value takes over — ablate era, then steer era"
        steps={2}
        notes={
          <>
            <p>
              Early in an episode, beliefs are flat: every candidate is uncertain, epistemic value
              dominates, and ablation — the widest-transition, most-informative lever — wins the
              argmin. As evidence accumulates, beliefs concentrate and the epistemic differences
              between levers collapse; among near-equal scores, the choice tips to the lever that
              best preserves sharpened beliefs and carries the residual preference weight —
              steering. The mechanism, stated carefully: only the <em>order</em> (ablate era →
              steer era) is predicted; the timing of the handoff is not a parameter — it falls
              wherever beliefs happen to sharpen, which depends on the environment&rsquo;s actual
              KL statistics and on the κ-seeded prior. (On Gemma the handoff is in fact abrupt —
              roughly one opening ablation — consistent with the informative prior collapsing
              beliefs quickly; Chapter III shows the trace.)
            </p>
            <p>
              This is the chapter&rsquo;s falsifiable output. On a model where evidence
              accumulates cleanly (Gemma), the handoff should appear. On a model with noisier
              observation statistics (Llama), the agent should rationally keep ablating.
              Chapter III checks both.
            </p>
          </>
        }
      >
        <p className="kicker">II.5 · What the objective predicts</p>
        <h1>
          A schedule <span className="accent">nobody wrote</span>
        </h1>
        <div className="fig-panel" style={{ maxWidth: 1000 }}>
          <ExploreExploitSVG />
        </div>
        <ul className="pts compact" style={{ marginTop: '0.7rem' }}>
          <Reveal at={1}>
            <li>
              Every bandit needs this schedule <strong>hand-tuned</strong> (bonuses, decay rates,
              priors) — here it <em>falls out of G</em>
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              And it should be <strong>model-sensitive</strong>: clean evidence → handoff; noisy
              evidence → keep ablating
            </li>
          </Reveal>
        </ul>
        <NextLead>Three engine-room facts for the pymdp readers, then the lineage.</NextLead>
      </Slide>

      {/* 12 — implementation facts */}
      <Slide
        title="Engine room: three checkable facts"
        brief="Novelty term on; goals enter as C; real fixed-point VI with Dirichlet learning"
        steps={3}
        notes={
          <>
            <p>
              <strong>First — the novelty term is on.</strong> The agent runs pymdp with{' '}
              <code>use_utility=True</code>, <code>use_states_info_gain=True</code>,{' '}
              <code>use_param_info_gain=True</code>, with Dirichlet concentration pA initialised
              at 10·A. Since the likelihood is learned online, EFE carries the third,
              often-forgotten term of the Da Costa decomposition — expected information gain about
              the <em>parameters</em> of A. The agent is curious about features whose observation
              mapping is under-evidenced, not just features whose state is uncertain.
            </p>
            <p>
              <strong>Second — where the goal lives.</strong> pymdp&rsquo;s carve-up is utility
              over observations plus state info gain; by the standard entropy cancellation this
              equals −(risk + ambiguity) exactly. C is a per-modality log-preference vector,
              monotone in KL bin. “Preferring high KL” is risk-minimisation toward a preference
              distribution over observations — the orthodox discrete-state construction, and the
              honest place to note that the discovery goal enters by fiat. What blunts the
              “relabelled reward” objection is the B-matrix geometry: symmetric rows keep utility
              differences across <em>actions</em> small, so action selection pressure is
              informational by construction.
            </p>
            <p>
              <strong>Third — the update loop is real variational inference.</strong> Per step:
              mean-field fixed-point iteration (minimising variational free energy F) infers the
              factorised marginals <M tex={String.raw`q(s_{0})\,q(s_{1})\,q(s_{2})`} /> given the
              observations; the Dirichlet update deposits count mass in the row of the{' '}
              <em>observed</em> outcome only,{' '}
              <M tex={String.raw`p_{A_{m}}[o_{m}] \mathrel{+}= \eta\; q(s_{0})\otimes q(s_{1})\otimes q(s_{2})`} />,
              η = 1 (pymdp&rsquo;s <code>update_obs_likelihood_dirichlet</code>); convergence is
              declared when the rolling belief-KL between successive posteriors drops below 0.01 —
              a threshold, note, not part of G: vanilla EFE has no principled stopping story, and
              the paper does not claim one. Chapter III shows the learning is
              real: the per-step L1 drift of the learned likelihood halves over the budget.
            </p>
            <p>
              Cost note: per candidate set (~40–60 features after per-layer capping), evaluating G
              over ≤180 (feature, action) pairs is microseconds of 36-state linear algebra; the
              GPU intervention dominates at ~30 ms. All checkable in{' '}
              <code>src/active_inference/pomdp_agent.py</code>.
            </p>
          </>
        }
      >
        <p className="kicker">II.6 · Checkable in pomdp_agent.py</p>
        <h1>Three facts from the engine room</h1>
        <p className="big dim" style={{ marginBottom: '0.6rem' }}>
          If none of these words land, the takeaway is one line:{' '}
          <strong style={{ color: 'var(--teal-bright)' }}>the agent is the textbook one — nothing
          bespoke.</strong>
        </p>
        <ul className="pts">
          <Reveal at={1}>
            <li>
              <strong>The novelty term is on</strong> — pA = 10·A, so curiosity extends to
              under-evidenced <em>likelihood parameters</em>, not just uncertain states
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              <strong>The goal enters as C</strong> — risk-minimisation toward preferred{' '}
              <em>observations</em> (high KL): the orthodox construction
            </li>
          </Reveal>
          <Reveal at={3}>
            <li>
              <strong>The updates are real VI</strong> — fixed-point posteriors, Dirichlet
              learning of A (η = 1), convergence at rolling belief-KL &lt; 0.01
            </li>
          </Reveal>
        </ul>
        <NextLead>So where does this sit among the field&rsquo;s methods?</NextLead>
      </Slide>


      {/* NEW — beat 11: a new paradigm for selection */}
      <Slide
        title="A new paradigm for selection"
        brief="ACDC exhaustive · EAP static single-pass · ACD adaptive and budgeted — not a replacement, a policy"
        steps={2}
        notes={
          <>
            <p>
              Where this sits, stated as a table. ACDC: exhaustive causal search — faithful,
              static, 𝒪(E). EAP: gradient attribution — one pass, static, and the strongest
              ranking available. ACD: POMDP-EFE selection — adaptive, learning online, and
              budgeted at B = 20. The disclaimer that keeps Chapter III honest belongs here
              in advance: <strong>ACD is not a replacement for EAP&rsquo;s graph
              construction</strong> — it is a policy for spending the verification budget{' '}
              <em>on the graph EAP builds</em>. So the fair fight in the next chapter is between{' '}
              <em>selectors</em>, and EAP fights on its home turf.
            </p>
          </>
        }
      >
        <p className="kicker">II.7 · Where this sits</p>
        <h1>
          A new paradigm <span className="accent">for selection</span>
        </h1>
        <div className="tbl-wrap">
          <table className="tbl">
            <thead>
              <tr><th>Method</th><th>Approach</th><th>Adaptivity</th><th>Cost</th></tr>
            </thead>
            <tbody>
              <tr><td>ACDC</td><td>exhaustive edge pruning</td><td>static</td><td className="num">𝒪(E)</td></tr>
              <tr><td>EAP</td><td>gradient attribution</td><td>static</td><td className="num">single pass</td></tr>
              <tr className="hl"><td><strong>ACD</strong></td><td>POMDP–EFE selection</td><td><strong>online learning</strong></td><td className="num"><strong>budgeted · B = 20</strong></td></tr>
            </tbody>
          </table>
        </div>
        <Reveal at={1}>
          <div className="take">
            <strong>Not a replacement for EAP&rsquo;s graphs</strong> — a policy for spending the
            verification budget on the graph EAP builds.
          </div>
        </Reveal>
        <Reveal at={2}>
          <div className="take teal">
            Which pre-arms the evidence chapter: the fair fight is between <em>selectors</em> — on
            EAP&rsquo;s home turf.
          </div>
        </Reveal>
        <NextLead>None of the machinery is new — that lineage is the strength.</NextLead>
      </Slide>

      {/* 13 — lineage */}
      <Slide
        title="Seventy years of lineage"
        brief="Lindley → MacKay → EFE → BOED → ACD: the same quantity, pointed somewhere new"
        steps={2}
        notes={
          <>
            <p>
              Lindley (1956) defined the value of an experiment as expected information gain;
              MacKay (1992) brought it to learning systems; the modern BOED and
              active-causal-discovery literature (Tigas et al. 2022; Rainforth et al. 2024)
              selects interventions against a posterior over causal structures. EFE&rsquo;s
              epistemic term is provably this same quantity — mutual information between states
              and observations (Friston et al. 2015) — and with flat preferences, discrete EFE{' '}
              <em>is</em> Lindley design (Sajid et al. 2021). The deflationary reading, embraced:
              for this audience, “EFE = BOED + preferences” is the point of contact, not an
              attack. One sharper standing question deserves naming: EFE is a <em>choice</em> of
              objective, not a theorem derived from the variational free energy (Millidge,
              Tschantz &amp; Buckley 2021, “Whence the Expected Free Energy?”) — and the
              deflationary BOED reading used here is deliberately agnostic on that debate.
              Catechism for completeness: perception minimises F (variational free energy, the
              evidence bound); action minimises G (expected free energy, the objective above).
            </p>
            <p>
              What active inference adds is not a new estimator but an <strong>architecture</strong>:
              preferences and information gain in one currency, likelihood learning with its own
              curiosity term, action semantics carried by the transition model. ACD is, to the
              paper&rsquo;s knowledge, the first time this architecture has been pointed at the
              internals of another neural network — the nearest neighbours (MAIA&rsquo;s LLM
              experiment loops, ACDC&rsquo;s exhaustive sweeps) have either the adaptivity or the
              formalism, never both.
            </p>
            <LinkCards cards={CANON_CARDS} />
          </>
        }
      >
        <p className="kicker">II.8 · Lineage</p>
        <h1>
          Old machinery,
          <br />
          <span className="accent">new environment</span>
        </h1>
        <div className="fig-panel" style={{ maxWidth: 1020 }}>
          <TimelineSVG />
        </div>
        <ul className="pts compact" style={{ marginTop: '0.7rem' }}>
          <Reveal at={1}>
            <li>
              Flat preferences reduce discrete EFE <em>exactly</em> to Lindley design (Sajid
              2021) — the deflationary reading, embraced
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              What active inference adds is <strong>architecture</strong>: both values in one
              currency, curiosity about the likelihood itself, action semantics in B
            </li>
          </Reveal>
        </ul>
        <NextLead>One symmetry to savour — then the evidence.</NextLead>
      </Slide>

      {/* 14 — symmetry + handoff */}
      <Slide
        title="Recap → the evidence"
        brief="A 36-state model reverse-engineering a 2.6B-parameter one — and a prediction to test"
        steps={2}
        notes={
          <>
            <Callout title="A symmetry worth savouring" tone="amber">
              The environment being probed is itself a prediction machine. An explicit generative
              model (36 states, fully inspectable) is being used to reverse-engineer an implicit
              one (2.6 billion parameters, opaque) — surprise minimisation turned inward on an
              artefact of surprise minimisation. The paper appeared in <em>Symmetry</em>; the pun
              writes itself, but the methodological point is serious: interpretability tools and
              cognitive theories are converging on the same loop.
            </Callout>
          </>
        }
      >
        <p className="kicker">II.9 · One symmetry</p>
        <h1>
          A generative model,
          <br />
          <span className="accent">reverse-engineering a generative model</span>
        </h1>
        <p className="big dim">
          36 explicit states probing 2.6 billion implicit parameters — surprise minimisation
          turned inward on an artefact of surprise minimisation.
        </p>
        <Reveal at={1}>
          <div className="stats">
            <div className="stat">
              <div className="v">1 prediction</div>
              <div className="k">ablate while uncertain → steer once confident</div>
            </div>
            <div className="stat">
              <div className="v violet">0 schedules</div>
              <div className="k">coded anywhere in the agent</div>
            </div>
          </div>
        </Reveal>
        <Reveal at={2}>
          <div className="take" style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
            <span>Did the prediction survive contact with the machine? Every number, next.</span>
            <Link
              href="/results"
              style={{
                fontFamily: 'var(--grotesk)', fontWeight: 600, fontSize: '.9rem', textDecoration: 'none',
                background: 'var(--teal)', color: '#fff', padding: '10px 20px', borderRadius: 999, whiteSpace: 'nowrap',
              }}
            >
              Chapter III: the evidence →
            </Link>
          </div>
        </Reveal>
      </Slide>
    </Deck>
  );
}

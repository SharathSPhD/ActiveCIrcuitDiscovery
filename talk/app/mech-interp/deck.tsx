'use client';

import Link from 'next/link';
import Deck, { Slide, Reveal, NextLead } from '../../components/Deck';
import { Eq, M, LinkCards, Callout } from '../../components/Prose';
import { SuperpositionSVG, TranscoderSVG, PipelineSVG, BudgetSVG } from '../../components/DiagramsPart1';
import {
  NextTokenSVG, StackSVG, PolysemanticSVG, DictionarySAESVG, SteeringDialSVG,
  IOICircuitSVG, LeversSVG, KLRulerSVG, CostScalesSVG, StaticVsAdaptiveSVG, LadderSVG,
} from '../../components/TeachCore';
import GraphPeek from '../../components/GraphPeek';

const ANTHROPIC_CARDS = [
  {
    src: 'Distill · Mar 2020',
    title: 'Zoom In: An Introduction to Circuits',
    desc: 'Olah et al. — features are directions, circuits are subgraphs of features and weights, and analogous circuits recur across models. The founding claims of the field.',
    href: 'https://distill.pub/2020/circuits/zoom-in/',
  },
  {
    src: 'Anthropic · Sep 2022',
    title: 'Toy Models of Superposition',
    desc: 'Why networks pack more features than neurons: sparsity-controlled phase changes, features stored as near-orthogonal directions, interference tolerated.',
    href: 'https://transformer-circuits.pub/2022/toy_model/index.html',
  },
  {
    src: 'Anthropic · Oct 2023',
    title: 'Towards Monosemanticity',
    desc: 'Sparse autoencoders decompose a 512-neuron MLP into thousands of interpretable features — Arabic script, DNA, base64 — where single neurons were hopeless mixtures.',
    href: 'https://transformer-circuits.pub/2023/monosemantic-features/index.html',
  },
  {
    src: 'Anthropic · May 2024',
    title: 'Scaling Monosemanticity',
    desc: '34M features from Claude 3 Sonnet: Golden Gate Bridge, code-bug, sycophancy, deception-adjacent features — and clamping them changes behaviour.',
    href: 'https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html',
  },
  {
    src: 'Anthropic · May 2024',
    title: 'Mapping the Mind of an LLM',
    desc: 'The accessible companion: feature geometry mirrors conceptual similarity — Alcatraz and the Golden State Warriors live near the Golden Gate Bridge.',
    href: 'https://www.anthropic.com/news/mapping-mind-language-model',
  },
  {
    src: 'Anthropic · May 2024',
    title: 'Golden Gate Claude',
    desc: 'One feature clamped to 10× its maximum activation, live on claude.ai for 24 hours. Ask it anything; it answers as the bridge. Steering made public.',
    href: 'https://www.anthropic.com/news/golden-gate-claude',
  },
];

const GRAPH_CARDS = [
  {
    src: 'Transformer Circuits · Mar 2025',
    title: 'Circuit Tracing: Methods',
    desc: 'Cross-layer transcoders, replacement models, error nodes, and attribution graphs whose every edge is an exact linear direct effect — the machinery ACD consumes.',
    href: 'https://transformer-circuits.pub/2025/attribution-graphs/methods.html',
  },
  {
    src: 'Transformer Circuits · Mar 2025',
    title: 'On the Biology of a Large Language Model',
    desc: 'Attribution graphs applied to Claude 3.5 Haiku: multi-step reasoning through “Texas”, poetry planned ahead of the line, a jailbreak dissected mid-sentence.',
    href: 'https://transformer-circuits.pub/2025/attribution-graphs/biology.html',
  },
  {
    src: 'Anthropic · May 2025',
    title: 'Open-sourcing circuit tracing',
    desc: 'circuit-tracer: attribution graphs, pruning, and feature interventions for Gemma-2-2B and Llama-3.2-1B — the released library ACD builds directly on.',
    href: 'https://www.anthropic.com/research/open-source-circuit-tracing',
  },
  {
    src: 'Neuronpedia',
    title: 'Interactive graphs & features',
    desc: 'Browse GemmaScope transcoder features and generate attribution graphs for Gemma-2-2B in the browser. The live embed in this chapter runs here.',
    href: 'https://www.neuronpedia.org/gemma-2-2b/graph',
  },
];

export default function MechDeck() {
  return (
    <Deck part="Chapter I · Inside the black box" title="Mechanistic interpretability, from zero">
      {/* 1 — chapter title + roadmap */}
      <Slide
        title="The climb ahead"
        brief="Chapter roadmap: from what a model does, to the experiment-selection problem"
        steps={1}
        notes={
          <>
            <p>
              This chapter assumes nothing about transformers. It climbs five rungs: what a
              language model actually does; why single neurons fail as units of analysis and what
              replaces them (features); how features wire into circuits; how circuit tracing turns
              a prompt into a causal map; and the open problem all of it exposes — deciding which
              causal experiment to run next under a budget. That last rung is where active
              inference enters, in Chapter II.
            </p>
            <p>
              The material follows one thread of work, most of it from Anthropic&rsquo;s
              interpretability team, because that thread produced the open tooling the ACD paper
              builds on. Reference cards with links appear in the notes of the relevant slides.
            </p>
          </>
        }
      >
        <p className="kicker">Chapter I · no prerequisites</p>
        <h1>
          Inside the <span className="accent">black box</span>
        </h1>
        <div className="fig-panel" style={{ maxWidth: 980 }}>
          <LadderSVG stage={5} />
        </div>
        <Reveal at={1}>
          <div className="take teal">
            Five rungs, each one earned. The chapter ends at a problem — <strong>and the problem
            is the point.</strong>
          </div>
        </Reveal>
        <NextLead>Start at the very bottom: what does a language model actually do?</NextLead>
      </Slide>

      {/* 2 — next-token prediction */}
      <Slide
        title="One job: predict the next word"
        brief="An LLM outputs a probability for every possible next token — everything else follows"
        steps={2}
        notes={
          <>
            <p>
              A large language model does exactly one thing: given a sequence of tokens, it
              outputs a probability distribution over every possible next token. All of its
              apparent abilities — answering, coding, translating — are this one operation applied
              repeatedly. The sentence on the slide is deliberately chosen: it is the exact test
              sentence used throughout the rest of the talk (the “IOI” task). The model strongly
              prefers “Mary”, because “John” already appeared as the giver — to get this right, it
              must have tracked who is who.
            </p>
            <p>
              This slide also quietly sets up the measuring instrument for everything that
              follows: since the model&rsquo;s entire output is a probability distribution, any
              change made inside the model can be measured by how far that distribution moves.
            </p>
          </>
        }
      >
        <p className="kicker">I.1 · Ground zero</p>
        <h1>A language model does one thing</h1>
        <div className="fig-panel" style={{ maxWidth: 980 }}>
          <NextTokenSVG />
        </div>
        <ul className="pts compact" style={{ marginTop: '0.7rem' }}>
          <Reveal at={1}>
            <li>
              Read tokens in → put a <strong>probability on every possible next token</strong>.
              That is the whole job.
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              To put 72% on <strong>Mary</strong>, something inside must have tracked{' '}
              <em>who already gave the bag</em> — remember this sentence, it returns all talk
            </li>
          </Reveal>
        </ul>
        <NextLead>So what is the “something inside”?</NextLead>
      </Slide>

      {/* 3 — the stack */}
      <Slide
        title="2.6 billion unlabeled numbers"
        brief="The transformer stack: attention moves information, MLPs transform it — nothing is labeled"
        steps={2}
        notes={
          <>
            <p>
              Between input and output sits a stack of layers — 26 in Gemma-2-2B, the model used
              throughout this talk. Each token carries a running vector (its position on the
              “residual stream”); attention blocks move information between tokens, MLP blocks
              transform each token&rsquo;s vector in place. The computation is fully visible — every
              one of the 2.6 billion weights can be printed — and fully opaque, because none of
              those numbers is labeled with what it means.
            </p>
            <p>
              Interpretability&rsquo;s founding constraint: unlike biology, there is perfect
              observability and perfect intervenability — any number can be read or overwritten at
              will. What is missing is not access. It is <em>vocabulary</em>.
            </p>
          </>
        }
      >
        <p className="kicker">I.1 · The machine between</p>
        <h1>
          Fully visible.
          <br />
          <span className="accent">Totally opaque.</span>
        </h1>
        <div className="fig-panel" style={{ maxWidth: 980 }}>
          <StackSVG />
        </div>
        <ul className="pts compact" style={{ marginTop: '0.7rem' }}>
          <Reveal at={1}>
            <li>
              Every weight can be read, every activation overwritten — <strong>perfect access,
              zero labels</strong>
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              The missing thing is not data. It is a <em>vocabulary</em> for what the numbers mean
            </li>
          </Reveal>
        </ul>
        <NextLead>The obvious first guess for that vocabulary: the neuron.</NextLead>
      </Slide>

      {/* 4 — neurons lie */}
      <Slide
        title="Neurons lie"
        brief="A single unit fires for cat faces, car fronts, and cat legs — polysemanticity"
        steps={2}
        notes={
          <>
            <p>
              The natural unit of analysis — one neuron, one concept — fails empirically. The
              founding document of the circuits agenda, <em>Zoom In</em> (Olah et al., Distill
              2020), documents a single InceptionV1 unit responding to cat faces, fronts of cars,
              and cat legs: three unrelated concepts sharing one number. This is{' '}
              <strong>polysemanticity</strong>, and it is the norm, not the exception. A typical
              language-model neuron responds to “a mixture of academic citations, English
              dialogue, HTTP requests, and Korean text.”
            </p>
            <p>
              If units mix unrelated concepts, any analysis built on units compounds that
              ambiguity at every layer. Before circuits can be found, the unit-of-analysis
              question has to be answered properly.
            </p>
          </>
        }
      >
        <p className="kicker">I.2 · The wrong unit</p>
        <h1>Neurons lie.</h1>
        <div className="cols cols-60">
          <div className="fig-panel">
            <PolysemanticSVG />
          </div>
          <div>
            <ul className="pts compact">
              <Reveal at={1}>
                <li>
                  One unit, <strong>three unrelated concepts</strong> — and this is typical, not
                  cherry-picked
                </li>
              </Reveal>
              <Reveal at={2}>
                <li>
                  Build an explanation on mixed units and the ambiguity{' '}
                  <em>compounds at every layer</em>
                </li>
              </Reveal>
            </ul>
            <Reveal at={2}>
              <div className="take">
                Before finding circuits, the field had to answer:{' '}
                <strong>what is the right unit?</strong>
              </div>
            </Reveal>
          </div>
        </div>
        <NextLead>Why would a network scramble its own concepts? It has a good reason.</NextLead>
      </Slide>

      {/* 5 — superposition */}
      <Slide
        title="Superposition"
        brief="More concepts than neurons → concepts stored as directions, every neuron a mixture"
        steps={3}
        notes={
          <>
            <p>
              The explanation — confirmed in toy models (Elhage et al. 2022) — is{' '}
              <strong>superposition</strong>. A network wants to track far more concepts than it
              has neurons. An analogy: a closet with 100 hooks and 10,000 items of clothing. The
              trick is to hang items <em>between</em> hooks — as long as most items are rarely
              needed at the same time (sparsity), collisions stay rare and the interference is
              worth the capacity. Concepts get stored as <em>directions</em> in activation space,
              nearly orthogonal but not axis-aligned; any single neuron (axis) then reads a
              mixture. Polysemanticity is what superposition looks like when you insist on reading
              the neuron basis.
            </p>
            <Callout title="For active inference readers" tone="violet">
              Superposition is a claim about the state space of the system to be explained: the
              hidden causes are linearly readable but not axis-aligned, so any method inspecting
              native coordinates is doing inference under the wrong parameterisation. The
              field&rsquo;s answer was to learn a better one.
            </Callout>
          </>
        }
      >
        <p className="kicker">I.2 · The packing trick</p>
        <h1>
          More concepts
          <br />
          than neurons
        </h1>
        <div className="cols cols-60">
          <div className="fig-panel light">
            <SuperpositionSVG />
          </div>
          <div>
            <ul className="pts compact">
              <Reveal at={1}>
                <li>
                  A closet with 100 hooks and 10,000 items: hang things{' '}
                  <strong>between the hooks</strong>
                </li>
              </Reveal>
              <Reveal at={2}>
                <li>
                  Works because concepts are <em>sparse</em> — rarely needed at the same moment
                </li>
              </Reveal>
              <Reveal at={3}>
                <li>
                  So concepts live as <strong>directions</strong>, not neurons — and every neuron
                  reads a mixture
                </li>
              </Reveal>
            </ul>
          </div>
        </div>
        <Reveal at={3}>
          <div className="take violet">
            The concepts are still there — recoverable, linear —{' '}
            <strong>just written in the wrong basis.</strong>
          </div>
        </Reveal>
        <NextLead>If the basis is wrong, learn a better one.</NextLead>
      </Slide>

      {/* 6 — dictionary learning */}
      <Slide
        title="Dictionary learning"
        brief="Sparse autoencoders rewrite the mixture as a few named, monosemantic features"
        steps={3}
        notes={
          <>
            <p>
              <em>Towards Monosemanticity</em> (2023) made the fix concrete: train a{' '}
              <strong>sparse autoencoder</strong> — reconstruction loss plus an L1 sparsity
              penalty, with an overcomplete latent basis — on billions of activations. From a
              512-neuron MLP it recovered thousands of features that behave monosemantically:
              dedicated features for Arabic script, DNA sequences, base64 strings, Hebrew.
            </p>
            <p>
              <em>Scaling Monosemanticity</em> (May 2024) ran the same recipe on a production
              model — up to <strong>34 million features</strong> from a middle layer of Claude 3
              Sonnet — and found features for the Golden Gate Bridge (firing across languages{' '}
              <em>and images</em>), code bugs, sycophantic praise, secrecy, deception-adjacent
              behaviour.
            </p>
            <LinkCards cards={ANTHROPIC_CARDS.slice(0, 4)} />
          </>
        }
      >
        <p className="kicker">I.2 · The fix</p>
        <h1>
          Un-mixing the signal:
          <br />
          <span className="accent">dictionary learning</span>
        </h1>
        <div className="fig-panel" style={{ maxWidth: 980 }}>
          <DictionarySAESVG />
        </div>
        <div className="stats" style={{ marginTop: '0.7rem' }}>
          <Reveal at={1}>
            <div className="stat">
              <div className="v">512 → 1000s</div>
              <div className="k">neurons in → monosemantic features out (2023)</div>
            </div>
          </Reveal>
          <Reveal at={2}>
            <div className="stat">
              <div className="v">34M</div>
              <div className="k">features recovered from Claude 3 Sonnet (2024)</div>
            </div>
          </Reveal>
        </div>
        <Reveal at={3}>
          <NextLead>Named features are a nice story. How do you prove a feature is real?</NextLead>
        </Reveal>
      </Slide>

      {/* 7 — Golden Gate */}
      <Slide
        title="Proof by steering: Golden Gate Claude"
        brief="Clamp one feature to 10× and a frontier model insists it is a bridge — features are causal"
        steps={2}
        notes={
          <>
            <p>
              The proof was shipped, not argued. In May 2024, Anthropic clamped a single feature —
              the Golden Gate Bridge feature — to roughly 10× its maximum activation and put the
              model live on claude.ai for 24 hours. Asked anything, it answered as the bridge
              (“my physical form is the iconic bridge itself…”). Clamp a scam-email feature
              instead, and the model writes the scam it would otherwise refuse.
            </p>
            <p>
              Two lessons carry forward. Features are <strong>causal handles</strong>, not just
              correlates — moving one moves behaviour. And the team&rsquo;s own caution: finding a
              feature is not the same as knowing what the model will do with it. The steering
              benchmark in Chapter III is a small-model, quantified descendant of exactly this
              demonstration — with the random-feature control the original never needed.
            </p>
            <LinkCards cards={ANTHROPIC_CARDS.slice(4)} />
          </>
        }
      >
        <p className="kicker">I.2 · The demonstration</p>
        <h1>Proof by steering</h1>
        <div className="fig-panel" style={{ maxWidth: 980 }}>
          <SteeringDialSVG />
        </div>
        <ul className="pts compact" style={{ marginTop: '0.7rem' }}>
          <Reveal at={1}>
            <li>
              Features are <strong>causal handles</strong> — turn one, and behaviour turns with it
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              Chapter III runs this exact experiment, quantified, <em>with the control this demo
              never had</em>
            </li>
          </Reveal>
        </ul>
        <NextLead>These features are not hypothetical — one can be touched right now.</NextLead>
      </Slide>

      {/* 8 — live feature */}
      <Slide
        title="Touch one feature, live"
        brief="A real Gemma-2-2B transcoder feature on Neuronpedia — type into it"
        lazy
        notes={
          <>
            <p>
              This is a live Neuronpedia dashboard for feature 16369 of the layer-20 GemmaScope
              transcoder (16k dictionary) for <strong>Gemma-2-2B</strong> — the very dictionary the
              ACD agent&rsquo;s candidate features come from (GemmaScope, DeepMind 2024: 400+
              JumpReLU SAEs and per-layer transcoders across every layer of Gemma-2 2B/9B, &gt;30M
              features). It shows top activating examples, an auto-generated explanation, and a
              test box: type any sentence and watch the feature respond.
            </p>
            <p>
              Explore further at{' '}
              <a href="https://www.neuronpedia.org/gemma-2-2b" target="_blank" rel="noopener">
                neuronpedia.org/gemma-2-2b
              </a>
              . This is what one node of the attribution graph <em>is</em>.
            </p>
          </>
        }
      >
        <p className="kicker">I.2 · Not hypothetical</p>
        <h2>
          A real Gemma-2-2B feature — <span className="dim">type a sentence into it</span>
        </h2>
        <iframe
          src="https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-transcoder-16k/16369?embed=true&embedexplanation=true&embedplots=true&embedtest=true"
          title="Neuronpedia — Gemma-2-2B layer-20 transcoder feature"
          style={{
            width: '100%',
            height: 'min(56vh, 540px)',
            border: '1px solid var(--navy-hairline)',
            borderRadius: 12,
            background: '#fff',
          }}
        />
        <NextLead>A vocabulary of features is not yet an explanation. Concepts need wiring.</NextLead>
      </Slide>

      {/* 9 — IOI circuit */}
      <Slide
        title="Circuits: the IOI algorithm"
        brief="26 attention heads implement detect-the-duplicate → suppress it → move the other name"
        steps={3}
        notes={
          <>
            <p>
              Features are a vocabulary; explanation needs <em>circuits</em> — which features feed
              which to produce the behaviour. The canonical existence proof is the{' '}
              <strong>IOI circuit</strong> (Wang et al. 2022), and it explains the exact sentence
              from the start of this chapter. GPT-2 small answers “…John gave the bag to __”
              through 26 attention heads in seven functional classes: duplicate-token heads notice
              “John” appears twice; S-inhibition heads suppress the duplicate at the answer
              position; name-mover heads copy “Mary” into the output; and backup name movers take
              over when the primaries are ablated.
            </p>
            <p>
              Every arrow was established causally, by <strong>path patching</strong>: swap in
              activations from a corrupted prompt along one path at a time and measure the effect
              on the output. The same paper fixed the field&rsquo;s evaluation vocabulary —
              faithfulness, completeness, minimality. The catch: this one circuit took
              person-months of manual work. Chapter III finds the same late-layer name-mover
              signature in Gemma, at transcoder-feature resolution.
            </p>
          </>
        }
      >
        <p className="kicker">I.3 · From vocabulary to algorithm</p>
        <h1>
          The sentence, <span className="accent">explained by a circuit</span>
        </h1>
        <div className="fig-panel" style={{ maxWidth: 1000 }}>
          <IOICircuitSVG />
        </div>
        <ul className="pts compact" style={{ marginTop: '0.7rem' }}>
          <Reveal at={1}>
            <li>
              A real algorithm, in the weights: <strong>detect the duplicate → suppress it → move
              the other name</strong>
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              Every arrow proven by a <strong>causal experiment</strong> (path patching), one path
              at a time
            </li>
          </Reveal>
          <Reveal at={3}>
            <li>
              The cost: <em>person-months</em> — for one behaviour, in a small model
            </li>
          </Reveal>
        </ul>
        <NextLead>Causal experiments need a measuring instrument. Here it is.</NextLead>
      </Slide>

      {/* 10 — levers + KL */}
      <Slide
        title="Interventions, and KL as the ruler"
        brief="Ablate / patch / steer a feature; measure how far the output distribution moves"
        steps={2}
        notes={
          <>
            <p>
              An intervention overwrites one feature&rsquo;s activation and lets the network run:
              set it to zero (<strong>ablation</strong> — what breaks without it?), set it to its
              value on a different prompt (<strong>activation patching</strong> — what does it
              carry?), or multiply it far beyond its normal range (<strong>feature
              steering</strong> — what does it amplify?).
            </p>
            <p>
              The measuring instrument comes from the first slide of the chapter: the
              model&rsquo;s output is a probability distribution, so the effect of any
              intervention is <strong>KL divergence</strong> between the clean and intervened
              next-token distributions. One number per experiment: large KL means the feature was
              load-bearing for this prompt; tiny KL means the model barely noticed. KL is the
              effect measure for everything in Chapters III and IV.
            </p>
          </>
        }
      >
        <p className="kicker">I.3 · The instrument</p>
        <h1>
          Three levers, <span className="accent">one ruler</span>
        </h1>
        <Reveal at={1}>
          <div className="fig-panel" style={{ maxWidth: 980 }}>
            <LeversSVG />
          </div>
        </Reveal>
        <Reveal at={2}>
          <div className="fig-panel" style={{ maxWidth: 980, marginTop: '0.7rem' }}>
            <KLRulerSVG />
          </div>
        </Reveal>
        <NextLead>Manual circuit-finding took months. The field automated it — at a price.</NextLead>
      </Slide>

      {/* 11 — ACDC vs EAP */}
      <Slide
        title="Attribution got cheap. Verification didn't."
        brief="EAP scores every edge from 3 passes — strong, but correlational; probes stay the ground truth"
        steps={3}
        notes={
          <>
            <p>
              <strong>ACDC</strong> (Conmy et al. 2023) automated circuit-finding faithfully:
              corrupt every edge, keep those whose corruption moves the output — but at one
              forward pass per candidate edge, tens of thousands of passes for GPT-2-small-sized
              graphs. <strong>Edge Attribution Patching</strong> (Syed et al. 2023) collapsed all
              of that into a first-order Taylor approximation: scores for <em>every</em> edge from
              two forward passes and one backward pass — and it recovers ground-truth circuits{' '}
              <em>better</em> than the exhaustive method it approximates.
            </p>
            <p>
              What EAP misses: it is a linearisation, and inherits linearisation pathologies —
              zero-gradient saturation (addressed by integrated-gradients variants), sign-blindness
              to suppressive components (ACDC famously drops IOI&rsquo;s <em>negative</em> name
              movers), dependence on the corruption distribution. And the faithfulness-metric
              edifice is fragile: Miller et al. (2024) show published circuit faithfulness scores
              swing wildly under seemingly minor ablation-scheme choices. The consequence:
              attribution gives cheap, noisy, <em>correlational</em> rankings — and real causal
              verification remains the expensive, trusted signal.
            </p>
          </>
        }
      >
        <p className="kicker">I.4 · Automation</p>
        <h1>
          Attribution got cheap.
          <br />
          <span className="accent">Verification didn&rsquo;t.</span>
        </h1>
        <ul className="pts compact">
          <Reveal at={1}>
            <li>
              <strong>ACDC</strong>: causally prune every edge — faithful, but tens of thousands
              of passes
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              <strong>EAP</strong>: every edge scored from <em>2 forward + 1 backward pass</em> —
              a Taylor approximation that beats the method it approximates
            </li>
          </Reveal>
        </ul>
        <Reveal at={2}>
          <Eq tex={String.raw`\Delta L \;\approx\; (e_{\text{corrupt}} - e_{\text{clean}})^{\top}\, \nabla_{e} L\big|_{\text{clean}}`} />
        </Reveal>
        <Reveal at={3}>
          <div className="take">
            Hold onto this: <strong>EAP is embarrassingly strong</strong> — in Chapter III it is
            the baseline the agent does not beat, and the paper says so in its abstract.
          </div>
        </Reveal>
        <NextLead>One more tool upgrade, and the full causal map is buildable.</NextLead>
      </Slide>

      {/* 12 — transcoders */}
      <Slide
        title="Transcoders: replace, don't describe"
        brief="Swap each MLP for a sparse legible imitation — feature-to-feature causality becomes linear algebra"
        steps={2}
        notes={
          <>
            <p>
              SAEs describe <em>what is represented</em> at a point; circuits need to trace
              computation <em>through</em> the MLP nonlinearity. A <strong>transcoder</strong>{' '}
              (Dunefsky, Chlenski &amp; Nanda 2024) learns a sparse approximation of the
              MLP&rsquo;s <em>input→output function</em> — it does not describe the MLP, it{' '}
              <em>replaces</em> it with a wide, sparse, interpretable one. The payoff is
              structural: attribution between transcoder features factorises into an
              input-invariant term (“virtual weights”, computable once from weights alone) times
              an input-dependent scalar (the upstream activation). Feature-to-feature causal
              structure through MLPs becomes linear algebra.
            </p>
            <Callout title="Terminology for the rest of the talk">
              A <strong>feature</strong> from here on is a transcoder latent, identified by{' '}
              <code>(layer, token position, feature index)</code>. Gemma-2-2B carries a 16k-entry
              GemmaScope transcoder at each of its 26 layers; Llama-3.2-1B uses community
              transcoders across its 16 layers. On a typical IOI prompt, ~12,600 Gemma features
              are active; ~2,200 survive influence pruning.
            </Callout>
          </>
        }
      >
        <p className="kicker">I.4 · The last tool</p>
        <div className="cols">
          <div>
            <h1>
              Replace the MLP,
              <br />
              <span className="accent">don&rsquo;t describe it</span>
            </h1>
            <ul className="pts compact">
              <Reveal at={1}>
                <li>
                  A transcoder <strong>imitates the MLP&rsquo;s function</strong> with a wide,
                  sparse, legible dictionary
                </li>
              </Reveal>
              <Reveal at={2}>
                <li>
                  Feature→feature causality becomes <strong>linear algebra</strong> (virtual
                  weights × activation)
                </li>
              </Reveal>
            </ul>
            <Reveal at={2}>
              <div className="take teal">
                From here on, a <strong>feature</strong> ={' '}
                <code>(layer, position, index)</code> of a transcoder latent.
              </div>
            </Reveal>
          </div>
          <div className="fig-panel light">
            <TranscoderSVG />
          </div>
        </div>
        <NextLead>Now assemble everything into the object the agent will work on.</NextLead>
      </Slide>

      {/* 13 — attribution graphs, real one */}
      <Slide
        title="An attribution graph — a real one"
        brief="The actual pruned IOI graph for Gemma-2-2B: 110 nodes, exact linear edges"
        steps={2}
        notes={
          <>
            <p>
              The 2025 <em>Circuit Tracing</em> pipeline (Ameisen et al.): swap every MLP for its
              transcoder (a <strong>replacement model</strong>); for one specific prompt, freeze
              attention patterns and layer-norm denominators at their actual values and add error
              nodes → a <strong>local replacement model</strong> that reproduces the original
              model&rsquo;s output on that prompt <em>exactly</em>, and is <em>linear</em> in
              feature activations. In that linear system, every edge of the attribution graph is
              an exact direct effect — not an estimate. Prune to the influential subgraph (~10×
              smaller, retaining ~80% of explained behaviour) and the result is what the slide
              shows: a per-prompt causal hypothesis.
            </p>
            <p>
              The companion <em>Biology</em> paper demonstrates the machinery at frontier scale —
              Claude 3.5 Haiku computing “capital of the state containing Dallas” through internal
              Texas features, planning rhymes ahead of the line — with its own caveat repeated
              throughout: graphs are <em>hypotheses</em>, requiring intervention to validate.
            </p>
            <LinkCards cards={GRAPH_CARDS.slice(0, 2)} />
          </>
        }
      >
        <p className="kicker">I.5 · The causal map</p>
        <h1>
          This is an <span className="accent">attribution graph</span>
        </h1>
        <div className="fig-panel" style={{ maxWidth: 1050 }}>
          <GraphPeek />
        </div>
        <ul className="pts compact" style={{ marginTop: '0.6rem' }}>
          <Reveal at={1}>
            <li>
              Not a sketch — <strong>the actual pruned graph</strong> for the chapter&rsquo;s
              sentence, on Gemma-2-2B
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              Every edge an <strong>exact linear direct effect</strong> — and the whole thing is
              still a <em>hypothesis</em> until probed
            </li>
          </Reveal>
        </ul>
        <NextLead>The machinery is open-source — and its price list shapes everything.</NextLead>
      </Slide>

      {/* 14 — pipeline + economics */}
      <Slide
        title="The pipeline and its price list"
        brief="circuit-tracer: 18 s for a graph of thousands, 30 ms per causal probe"
        steps={2}
        notes={
          <>
            <p>
              In May 2025 the machinery went public as <code>circuit-tracer</code> (Anthropic
              Fellows with Decode Research): per-layer transcoders, attribution, pruning, and a
              clean causal API. One attribution graph costs ~18 s on Gemma-2-2B; one{' '}
              <code>feature_intervention</code> call — ablate, patch, or steer — costs ~30 ms,
              with the intervention propagating through the real network. KL between clean and
              intervened next-token distributions is the effect measure.
            </p>
            <pre>{`from circuit_tracer import ReplacementModel, attribute

model = ReplacementModel.from_pretrained(
    "google/gemma-2-2b", transcoder_set="gemma")

graph = attribute(prompt, model)          # hypothesis  (~18 s)

logits, _ = model.feature_intervention(   # evidence    (~30 ms)
    prompt, [(layer, pos, feature_idx, value)])`}</pre>
            <LinkCards cards={GRAPH_CARDS.slice(2)} />
          </>
        }
      >
        <p className="kicker">I.5 · Open-source, with a price list</p>
        <h1>
          Hypotheses are cheap.
          <br />
          <span className="accent">Evidence is budgeted.</span>
        </h1>
        <Reveal at={1}>
          <div className="fig-panel" style={{ maxWidth: 1020 }}>
            <PipelineSVG />
          </div>
        </Reveal>
        <Reveal at={2}>
          <div className="fig-panel" style={{ maxWidth: 1020, marginTop: '0.7rem' }}>
            <CostScalesSVG />
          </div>
        </Reveal>
        <NextLead>Cheap maps, budgeted probes — a decision problem is forming.</NextLead>
      </Slide>

      {/* 15 — the open problem */}
      <Slide
        title="The open problem"
        brief="6,600 possible experiments, 20 affordable — and every method spends them non-adaptively"
        steps={3}
        notes={
          <>
            <p>
              The economics, side by side: a 2,200-feature candidate set × 3 intervention types is
              a 6,600-arm experiment space; a realistic verification budget is ~20 probes per
              prompt. Every existing automated method spends that budget{' '}
              <em>non-adaptively</em>: ACDC sweeps everything; EAP ranks once and walks the list;
              even the impressive LLM-agent systems (MAIA, 2024) choose experiments by
              prompt-engineered heuristics — no belief state, no information-theoretic objective,
              no notion of what an observation is worth.
            </p>
          </>
        }
      >
        <p className="kicker">I.6 · Where the chapter lands</p>
        <h1>
          Who decides which
          <br />
          <span className="accent">experiment runs next?</span>
        </h1>
        <div className="cols cols-60">
          <div>
            <div className="stats">
              <Reveal at={1}>
                <div className="stat">
                  <div className="v">6,600</div>
                  <div className="k">possible experiments (2,200 features × 3 levers)</div>
                </div>
              </Reveal>
              <Reveal at={1}>
                <div className="stat">
                  <div className="v amber">20</div>
                  <div className="k">probes the budget affords</div>
                </div>
              </Reveal>
            </div>
            <Reveal at={2}>
              <div className="fig-panel light">
                <BudgetSVG />
              </div>
            </Reveal>
          </div>
          <Reveal at={3}>
            <div className="fig-panel">
              <StaticVsAdaptiveSVG />
            </div>
          </Reveal>
        </div>
        <NextLead>Choosing informative experiments under uncertainty has a seventy-year-old theory…</NextLead>
      </Slide>

      {/* 16 — recap + handoff */}
      <Slide
        title="Chapter I recap → the bridge"
        brief="All five rungs climbed; the epistemic term of EFE is exactly what the problem needs"
        steps={2}
        notes={
          <>
            <p>
              Sequential selection of maximally informative experiments under uncertainty has a
              statistical home: Lindley&rsquo;s 1956 expected information gain, MacKay&rsquo;s
              1992 information-based objectives, the modern Bayesian experimental-design
              literature and its active-causal-discovery branch (Tigas et al. 2022). And it has a
              biological formulation this audience knows better than anyone:{' '}
              <strong>Expected Free Energy</strong>, whose epistemic term <em>is</em> expected
              information gain (Friston et al. 2015), and whose pragmatic term supplies what pure
              information gain lacks — a preference for experiments that find what the search is{' '}
              <em>for</em>. To the paper&rsquo;s knowledge, nobody had put that machinery in
              charge of a mechanistic-interpretability budget before.
            </p>
          </>
        }
      >
        <p className="kicker">I.6 · Recap</p>
        <h1>The problem, precisely</h1>
        <div className="fig-panel" style={{ maxWidth: 980 }}>
          <LadderSVG stage={5} />
        </div>
        <Reveal at={1}>
          <div className="big" style={{ margin: '1rem 0 0.6rem' }}>
            <M tex={String.raw`\mathbb{E}_{Q}\,D_{\mathrm{KL}}[Q(s\mid o,\pi)\,\|\,Q(s\mid\pi)] = I(s;o\mid\pi)`} />
          </div>
          <p className="big dim">
            The epistemic term of Expected Free Energy <strong style={{ color: 'var(--teal-bright)' }}>is</strong>{' '}
            expected information gain — the exact currency this problem is priced in.
          </p>
        </Reveal>
        <Reveal at={2}>
          <div className="take teal" style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
            <span>
              Chapter II builds the bridge: the researcher becomes the agent.
            </span>
            <Link
              href="/active-inference"
              style={{
                fontFamily: 'var(--grotesk)', fontWeight: 600, fontSize: '.9rem', textDecoration: 'none',
                background: 'var(--teal)', color: '#fff', padding: '10px 20px', borderRadius: 999, whiteSpace: 'nowrap',
              }}
            >
              Chapter II →
            </Link>
          </div>
        </Reveal>
      </Slide>
    </Deck>
  );
}

'use client';

import Link from 'next/link';
import Deck, { Slide, Reveal } from '../../components/Deck';
import { Eq, M, LinkCards, Callout } from '../../components/Prose';
import { SuperpositionSVG, TranscoderSVG, PipelineSVG, BudgetSVG } from '../../components/DiagramsPart1';

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
    desc: 'Browse GemmaScope transcoder features and generate attribution graphs for Gemma-2-2B in the browser. The live embed in this deck runs here.',
    href: 'https://www.neuronpedia.org/gemma-2-2b/graph',
  },
];

export default function MechDeck() {
  return (
    <Deck part="Part I · ~20 min" title="Mechanistic interpretability">
      {/* 1 — title */}
      <Slide
        steps={1}
        notes={
          <>
            <p>
              Most of this room models brains, not transformers. In twenty minutes I want to hand
              you the exact objects the ACD agent acts on — transcoder features, attribution graphs,
              causal interventions — and one open problem in this field that should feel like home:
              adaptive experiment selection under a budget.
            </p>
            <p>
              Everything follows one thread of work, most of it from Anthropic&rsquo;s
              interpretability team, because that thread produced the open tooling the paper runs
              on. Every claim links to its source (see the reference cards in the notes of the
              relevant slides).
            </p>
          </>
        }
      >
        <p className="kicker">Part I · ~20 minutes</p>
        <h1>
          Mechanistic interpretability,
          <br />
          <span className="accent">from neurons to attribution graphs</span>
        </h1>
        <p className="big dim">
          What I need you to have in hand before the bridge: features, graphs, interventions.
        </p>
        <Reveal at={1}>
          <div className="take teal">
            And one open problem that is secretly yours:{' '}
            <strong>which causal experiment do I run next?</strong>
          </div>
        </Reveal>
      </Slide>

      {/* 2 — Zoom In claims */}
      <Slide
        steps={3}
        notes={
          <>
            <p>
              The founding document of the circuits agenda is <em>Zoom In</em> (Olah et al., Distill
              2020). It stakes three claims: <strong>features</strong> — not neurons — are the
              fundamental units of neural networks, and they correspond to <em>directions</em> in
              activation space; features compose into <strong>circuits</strong>, computational
              subgraphs of features connected by weights; and both are <strong>universal</strong> —
              analogous curve detectors appear in AlexNet, InceptionV1, VGG19 and ResNet, learned
              independently. The evidence standard it set matters: curve detectors in InceptionV1
              were established by seven independent lines of evidence, down to reimplementing the
              weights by hand.
            </p>
            <p>
              The obstacle it named also matters: <strong>polysemanticity</strong>. A single
              InceptionV1 unit fires for cat faces, fronts of cars, and cat legs. If units mix
              unrelated concepts, circuit analysis compounds ambiguity at every layer.
            </p>
          </>
        }
      >
        <p className="kicker">I.1 · The unit of analysis</p>
        <h1>Neurons lie.</h1>
        <ul className="pts">
          <Reveal at={1}>
            <li>
              <strong>Features</strong> are the units — and they are <em>directions</em> in
              activation space, not neurons <span className="dim">(Zoom In, 2020)</span>
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              Features wire into <strong>circuits</strong> — computational subgraphs that recur
              across models
            </li>
          </Reveal>
          <Reveal at={3}>
            <li>
              The obstacle: <strong>polysemanticity</strong> — one unit fires for cat faces, car
              fronts, <em>and</em> cat legs
            </li>
          </Reveal>
        </ul>
      </Slide>

      {/* 3 — superposition */}
      <Slide
        steps={2}
        notes={
          <>
            <p>
              The explanation for polysemanticity — confirmed in toy models in 2022 — is{' '}
              <strong>superposition</strong>: a network that wants to represent more features than
              it has dimensions stores them as nearly-orthogonal directions and tolerates the
              interference, because feature sparsity makes collisions rare. Polysemanticity is what
              superposition looks like when you insist on reading the neuron basis.
            </p>
            <Callout title="For this audience" tone="violet">
              Superposition is a claim about the <em>state space</em> of the system you want to
              explain. The hidden causes are linearly readable but not axis-aligned, so any method
              that inspects native coordinates — single neurons, single MLP units — is doing
              inference under the wrong parameterisation. The field&rsquo;s answer was to learn a
              better one.
            </Callout>
          </>
        }
      >
        <p className="kicker">I.1 · Superposition</p>
        <div className="cols">
          <div>
            <h1>
              More features
              <br />
              than neurons
            </h1>
            <ul className="pts">
              <Reveal at={1}>
                <li>
                  Sparse features → packed as <strong>near-orthogonal directions</strong>,
                  interference tolerated
                </li>
              </Reveal>
            </ul>
            <Reveal at={2}>
              <div className="take violet">
                For you: the hidden causes are linearly readable but{' '}
                <strong>not axis-aligned</strong> — the neuron basis is the wrong parameterisation.
              </div>
            </Reveal>
          </div>
          <div className="fig-panel">
            <SuperpositionSVG />
          </div>
        </div>
      </Slide>

      {/* 4 — SAEs */}
      <Slide
        steps={3}
        notes={
          <>
            <p>
              <em>Towards Monosemanticity</em> (2023) made the fix concrete. Train a{' '}
              <strong>sparse autoencoder</strong> (SAE) — reconstruction loss plus an L1 sparsity
              penalty — on billions of MLP activations, with an overcomplete latent basis. From a
              512-neuron MLP it recovered thousands of features that behave monosemantically:
              dedicated features for Arabic script, DNA sequences, base64 strings, Hebrew — where a
              typical <em>neuron</em> of the same model responds to “a mixture of academic
              citations, English dialogue, HTTP requests, and Korean text.”
            </p>
            <p>
              <em>Scaling Monosemanticity</em> (May 2024) ran the same recipe on a production model
              — SAEs of 1M, 4M and <strong>34M features</strong> on a middle-layer residual stream
              of Claude 3 Sonnet — and found features for Golden Gate Bridge (firing across
              languages <em>and images</em>), code bugs, sycophantic praise, secrecy,
              deception-adjacent behaviour.
            </p>
            <LinkCards cards={ANTHROPIC_CARDS.slice(0, 4)} />
          </>
        }
      >
        <p className="kicker">I.2 · Dictionary learning</p>
        <h1>
          Sparse autoencoders:
          <br />
          <span className="accent">buying back monosemanticity</span>
        </h1>
        <div className="stats">
          <Reveal at={1}>
            <div className="stat">
              <div className="v">512 → 1000s</div>
              <div className="k">neurons in, monosemantic features out (2023)</div>
            </div>
          </Reveal>
          <Reveal at={2}>
            <div className="stat">
              <div className="v">34M</div>
              <div className="k">features from Claude 3 Sonnet (2024)</div>
            </div>
          </Reveal>
        </div>
        <Reveal at={3}>
          <ul className="pts">
            <li>
              Dedicated features for Arabic script, DNA, base64, code bugs, sycophancy,{' '}
              <strong>the Golden Gate Bridge</strong> — across languages <em>and images</em>
            </li>
          </ul>
        </Reveal>
      </Slide>

      {/* 5 — Golden Gate Claude */}
      <Slide
        steps={2}
        notes={
          <>
            <p>
              Anthropic made the causal point unforgettable by shipping it:{' '}
              <a href="https://www.anthropic.com/news/golden-gate-claude" target="_blank" rel="noopener">
                Golden Gate Claude
              </a>{' '}
              was live on claude.ai for 24 hours in May 2024 — a single feature clamped to ~10× its
              maximum activation, a frontier model cheerfully insisting it is a suspension bridge.
              It remains the cleanest public demonstration that these learned directions are
              steering wheels, not just correlates. Clamp a scam-email feature and the model writes
              the scam it would otherwise refuse.
            </p>
            <p>
              Two cautions from the team itself: finding a feature is not the same as knowing what
              the model will do with it. ACD&rsquo;s steering benchmark (Part III) is a small-model,
              quantified descendant of exactly this experiment — five landmark concepts on
              Gemma-2-2B, with a random-feature control the original demo never needed.
            </p>
            <LinkCards cards={ANTHROPIC_CARDS.slice(4)} />
          </>
        }
      >
        <p className="kicker">I.2 · The demonstration</p>
        <h1>Golden Gate Claude</h1>
        <ul className="pts">
          <li>
            One feature clamped to <strong>10×</strong> max activation → the model{' '}
            <em>answers as the bridge</em>, live for 24 h
          </li>
          <Reveal at={1}>
            <li>
              Features are <strong>causal handles</strong>, not correlates
            </li>
          </Reveal>
        </ul>
        <Reveal at={2}>
          <div className="take">
            My steering benchmark in Part III is this experiment, quantified — with the
            random-feature control this demo never needed.
          </div>
        </Reveal>
      </Slide>

      {/* 6 — live feature */}
      <Slide
        lazy
        notes={
          <>
            <p>
              This is a live Neuronpedia dashboard for feature 16369 of the layer-20 GemmaScope
              transcoder (16k dictionary) for <strong>Gemma-2-2B</strong> — the very dictionary
              ACD&rsquo;s candidate features come from (GemmaScope, DeepMind 2024: 400+ JumpReLU
              SAEs and per-layer transcoders across every layer of Gemma-2 2B/9B, &gt;30M
              features). Activations, auto-interp explanation, and a test box.
            </p>
            <p>
              During the talk: type a sentence, watch the feature respond. Explore more at{' '}
              <a href="https://www.neuronpedia.org/gemma-2-2b" target="_blank" rel="noopener">
                neuronpedia.org/gemma-2-2b
              </a>
              . This is what one node of the attribution graph <em>is</em>.
            </p>
          </>
        }
      >
        <p className="kicker">I.2 · Touch one</p>
        <h2>
          A live Gemma-2-2B transcoder feature — <span className="dim">type into it</span>
        </h2>
        <iframe
          src="https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-transcoder-16k/16369?embed=true&embedexplanation=true&embedplots=true&embedtest=true"
          title="Neuronpedia — Gemma-2-2B layer-20 transcoder feature"
          style={{
            width: '100%',
            height: 'min(58vh, 560px)',
            border: '1px solid var(--navy-hairline)',
            borderRadius: 12,
            background: '#fff',
          }}
        />
      </Slide>

      {/* 7 — circuits / IOI */}
      <Slide
        steps={3}
        notes={
          <>
            <p>
              Features alone are a vocabulary; explanation needs <em>circuits</em> — who feeds whom
              to produce the behaviour. The canonical existence proof is the{' '}
              <strong>IOI circuit</strong> (Wang et al. 2022): for prompts like{' '}
              <em>“When John and Mary went to the store, John gave the bag to __”</em>, GPT-2 small
              routes the answer through 26 attention heads in seven functional classes —
              duplicate-token heads detect the repeated “John”, S-inhibition heads suppress it,
              name-mover heads copy “Mary” into the output, and backup name movers take over when
              the primaries are ablated. Established by <strong>path patching</strong>: swap in
              activations from a corrupted prompt along one path at a time and measure the effect.
              The paper also fixed the field&rsquo;s evaluation vocabulary — faithfulness,
              completeness, minimality.
            </p>
            <p>
              ACD&rsquo;s IOI benchmark is this task, transplanted to transcoder features on Gemma
              and Llama; the paper finds the same late-layer name-mover signature (its top causal
              features sit in layers 24–25 of 26 — Part III).
            </p>
          </>
        }
      >
        <p className="kicker">I.3 · From features to circuits</p>
        <h1>
          The IOI circuit — <span className="accent">proof that circuits exist</span>
        </h1>
        <p className="big dim">“When John and Mary went to the store, John gave the bag to __”</p>
        <ul className="pts">
          <Reveal at={1}>
            <li>
              GPT-2 small answers through <strong>26 attention heads</strong> in 7 functional
              classes — detect the duplicate, suppress it, move the other name
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              Established by <strong>path patching</strong> — one causal experiment per path
            </li>
          </Reveal>
          <Reveal at={3}>
            <li>
              Cost: <em>person-months of manual work</em> — this is the pain the field automated
            </li>
          </Reveal>
        </ul>
      </Slide>

      {/* 8 — ACDC vs EAP */}
      <Slide
        steps={3}
        notes={
          <>
            <p>
              <strong>ACDC</strong> (Conmy et al. 2023) walks the computational graph in reverse
              topological order and prunes every edge whose corruption changes output KL by less
              than a threshold: faithful, but one forward pass per candidate edge — tens of
              thousands of passes on GPT-2-small-sized graphs.{' '}
              <strong>Edge Attribution Patching</strong> (Syed et al. 2023) replaces all of that
              with a first-order Taylor approximation — scores for <em>every</em> edge from two
              forward passes and one backward pass, and it recovers ground-truth circuits{' '}
              <em>better</em> than the exhaustive method it approximates.
            </p>
            <h3>What EAP misses — and why verification stays expensive</h3>
            <p>
              EAP is a linearisation, and it inherits linearisation pathologies: zero-gradient
              saturation (fixed by integrated-gradients variants, EAP-IG), sign-blindness of
              KL-based variants to suppressive components (ACDC famously drops the IOI{' '}
              <em>negative</em> name movers), and dependence on the corruption distribution. Worse,
              the whole faithfulness-metric edifice is fragile: Miller et al. (2024) show published
              circuit faithfulness scores swing wildly under zero- vs mean- vs resample-ablation
              and other seemingly minor choices. The practical consequence: attribution gives you
              cheap, noisy, <em>correlational</em> rankings, and real causal verification —
              actually running interventions — remains the expensive, trusted signal. That gap
              between cheap hypothesis and costly experiment is precisely where a Bayesian
              experiment-selection agent earns its keep.
            </p>
          </>
        }
      >
        <p className="kicker">I.3 · Automation</p>
        <h1>
          Attribution got cheap.
          <br />
          <span className="accent">Verification didn&rsquo;t.</span>
        </h1>
        <ul className="pts">
          <Reveal at={1}>
            <li>
              <strong>ACDC</strong>: prune every edge causally — tens of thousands of forward
              passes
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              <strong>EAP</strong>: all edge scores from <em>2 forward + 1 backward pass</em> — a
              Taylor approximation that beats the exhaustive method
            </li>
          </Reveal>
        </ul>
        <Reveal at={2}>
          <Eq tex={String.raw`\Delta L \;\approx\; (e_{\text{corrupt}} - e_{\text{clean}})^{\top}\, \nabla_{e} L\big|_{\text{clean}}`} />
        </Reveal>
        <Reveal at={3}>
          <div className="take">
            Remember this: <strong>EAP is embarrassingly strong.</strong> In Part III it is the
            baseline my agent does not beat — and the paper says so in the abstract.
          </div>
        </Reveal>
      </Slide>

      {/* 9 — transcoders */}
      <Slide
        steps={2}
        notes={
          <>
            <p>
              SAEs describe <em>what is represented</em> at a point in the network. But circuits
              need to trace computation <em>through</em> the MLP nonlinearity, and there SAEs force
              you to differentiate through the original MLP per-input. A <strong>transcoder</strong>{' '}
              (Dunefsky, Chlenski &amp; Nanda 2024) instead learns a sparse approximation of the
              MLP&rsquo;s <em>input→output function</em> — it doesn&rsquo;t describe the MLP, it{' '}
              <em>replaces</em> it with a wide, sparse, interpretable one. The payoff is structural:
              attribution between transcoder features factorises into an{' '}
              <strong>input-invariant</strong> term (“virtual weights”, the downstream encoder
              dotted with the upstream decoder — computable once from weights alone) times an{' '}
              <strong>input-dependent</strong> scalar (the upstream feature&rsquo;s activation).
              Feature-to-feature causal structure through MLPs becomes linear algebra. Follow-up
              work (Paulo et al. 2025) argues transcoder latents are also simply more interpretable
              than SAE latents, and the 2025 Anthropic stack is built on transcoders end to end.
            </p>
            <Callout title="Terminology for the rest of the talk">
              A <strong>feature</strong> from here on is a transcoder latent, identified by the
              triple <code>(layer, token position, feature index)</code>. Gemma-2-2B carries a
              16k-entry GemmaScope transcoder at each of its 26 layers; Llama-3.2-1B uses community
              transcoders (<code>mntss/transcoder-Llama-3.2-1B</code>) across its 16 layers. On a
              typical IOI prompt, ~12,600 Gemma features are active; ~2,200 survive influence
              pruning. That set is the ACD agent&rsquo;s world.
            </Callout>
          </>
        }
      >
        <p className="kicker">I.4 · Transcoders</p>
        <div className="cols">
          <div>
            <h1>
              Swap the MLP
              <br />
              for a <span className="accent">sparse, legible one</span>
            </h1>
            <ul className="pts">
              <Reveal at={1}>
                <li>
                  Transcoder = learned sparse replacement of the MLP&rsquo;s{' '}
                  <strong>input→output function</strong>
                </li>
              </Reveal>
              <Reveal at={2}>
                <li>
                  Feature-to-feature causal structure becomes <strong>linear algebra</strong>{' '}
                  (virtual weights × activation)
                </li>
              </Reveal>
            </ul>
            <Reveal at={2}>
              <div className="take teal">
                From here on, a <strong>feature</strong> = <code>(layer, position, index)</code> of a
                transcoder latent.
              </div>
            </Reveal>
          </div>
          <div className="fig-panel">
            <TranscoderSVG />
          </div>
        </div>
      </Slide>

      {/* 10 — attribution graphs */}
      <Slide
        steps={2}
        notes={
          <>
            <p>
              The 2025 methods paper (<em>Circuit Tracing</em>, Ameisen et al.) assembles all of the
              above into a pipeline. Swap every MLP for its transcoder → a{' '}
              <strong>replacement model</strong>. For a specific prompt, freeze attention patterns
              and layer-norm denominators at their actual values and add <strong>error nodes</strong>{' '}
              (true MLP output minus transcoder reconstruction) → a <strong>local replacement
              model</strong> that reproduces the original model&rsquo;s output on that prompt{' '}
              <em>exactly</em>, and — crucially — is <em>linear</em> in feature activations. In that
              linear system, every edge of the <strong>attribution graph</strong> (token embeddings
              → features → features → logits) is an exact direct effect, not an estimate. Prune to
              the influential subgraph (~10× smaller, retaining ~80% of explained behaviour) and
              you have a per-prompt causal hypothesis about the computation.
            </p>
            <p>
              The companion <em>Biology</em> paper demonstrates this machinery at frontier scale:
              Claude 3.5 Haiku computing “capital of the state containing Dallas” <em>through</em>{' '}
              internal Texas features (swap in California features and it says Sacramento),
              planning rhyme words before writing the line, assembling a jailbreak acronym
              letter-by-letter without representing the word. Its own coverage caveats: satisfying
              circuit insight on roughly a quarter of prompts attempted; graphs are hypotheses
              requiring intervention validation.
            </p>
            <LinkCards cards={GRAPH_CARDS.slice(0, 2)} />
          </>
        }
      >
        <p className="kicker">I.5 · Attribution graphs</p>
        <h1>
          A per-prompt <span className="accent">causal hypothesis</span>
        </h1>
        <div className="fig-panel" style={{ maxWidth: 980 }}>
          <PipelineSVG />
        </div>
        <ul className="pts compact" style={{ marginTop: '0.9rem' }}>
          <Reveal at={1}>
            <li>
              Local replacement model: exact on this prompt, <strong>linear</strong> in features →
              every graph edge is an <em>exact direct effect</em>
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              Pruned ~10× → the influential subgraph is a <strong>hypothesis</strong>; interventions
              are the test
            </li>
          </Reveal>
        </ul>
      </Slide>

      {/* 11 — circuit-tracer API */}
      <Slide
        steps={1}
        notes={
          <>
            <p>
              In May 2025 the machinery went public: <code>circuit-tracer</code> (built by
              Anthropic Fellows with Decode Research) ships per-layer transcoders, attribution,
              pruning and a clean causal API. One attribution graph costs ~18 s on Gemma-2-2B; one{' '}
              <code>feature_intervention</code> call — ablate (value 0), patch (reference value),
              or steer (multiple of the clean activation) — costs ~30 ms, with the intervention
              propagating through the real network. ACD uses the graph for hypotheses and the
              interventions for evidence. KL divergence between clean and intervened next-token
              distributions is the effect measure throughout.
            </p>
            <LinkCards cards={GRAPH_CARDS.slice(2)} />
          </>
        }
      >
        <p className="kicker">I.5 · The tooling this all runs on</p>
        <h2>
          <code>circuit-tracer</code> — open-sourced May 2025
        </h2>
        <pre>{`from circuit_tracer import ReplacementModel, attribute

model = ReplacementModel.from_pretrained(
    "google/gemma-2-2b", transcoder_set="gemma")

graph = attribute(prompt, model)          # hypothesis  (~18 s)

logits, _ = model.feature_intervention(   # evidence    (~30 ms)
    prompt, [(layer, pos, feature_idx, value)])`}</pre>
        <Reveal at={1}>
          <div className="stats">
            <div className="stat">
              <div className="v">18 s</div>
              <div className="k">one attribution graph — cheap, correlational</div>
            </div>
            <div className="stat">
              <div className="v">30 ms</div>
              <div className="k">one causal intervention — the trusted signal</div>
            </div>
          </div>
        </Reveal>
      </Slide>

      {/* 12 — the open problem */}
      <Slide
        steps={3}
        notes={
          <>
            <p>
              Put the economics side by side. Attribution: one 18-second pass yields importance
              scores for thousands of features — cheap, correlational, known-fragile. Causal
              verification: 30 ms per feature per intervention type, but the trusted currency of
              the field — and on any real auditing task you cannot afford it exhaustively. A
              2,200-feature candidate set × 3 intervention types is a 6,600-arm experiment space; a
              realistic verification budget might be 20 probes. Every existing automated method
              spends that budget <em>non-adaptively</em>: ACDC sweeps everything, EAP ranks once
              and walks the list, and even the impressive LLM-agent systems (MAIA, 2024) choose
              experiments by prompt-engineered heuristics with no belief state, no
              information-theoretic objective, no notion of what an observation is worth.
            </p>
          </>
        }
      >
        <p className="kicker">I.6 · The open problem</p>
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
                  <div className="k">candidate experiments (2,200 features × 3 interventions)</div>
                </div>
              </Reveal>
              <Reveal at={1}>
                <div className="stat">
                  <div className="v amber">20</div>
                  <div className="k">probes you can actually afford</div>
                </div>
              </Reveal>
            </div>
            <ul className="pts">
              <Reveal at={2}>
                <li>
                  Every existing method spends it <strong>non-adaptively</strong> — sweep
                  everything, or rank once and walk the list
                </li>
              </Reveal>
              <Reveal at={3}>
                <li>
                  No belief state. No objective. <em>No notion of what an observation is worth.</em>
                </li>
              </Reveal>
            </ul>
          </div>
          <div className="fig-panel">
            <BudgetSVG />
          </div>
        </div>
      </Slide>

      {/* 13 — handoff */}
      <Slide
        steps={1}
        notes={
          <>
            <p>
              Sequential selection of maximally informative experiments under uncertainty has a
              seventy-year-old statistical home — Lindley&rsquo;s 1956 expected information gain,
              MacKay&rsquo;s 1992 information-based objective functions, the modern Bayesian
              experimental-design literature and its active-causal-discovery branch (Tigas et al.
              2022). And it has a biological formulation this room knows better than anyone:{' '}
              <strong>Expected Free Energy</strong>, whose epistemic term <em>is</em> expected
              information gain (Friston et al. 2015), and whose pragmatic term supplies exactly
              what pure information gain lacks: a preference for experiments that find what you are{' '}
              <em>looking for</em>. To my knowledge, nobody had put that machinery in charge of a
              mechanistic interpretability budget before this paper.
            </p>
          </>
        }
      >
        <p className="kicker">I.6 · Where the answer lives</p>
        <h1>
          You already have
          <br />
          the objective for this.
        </h1>
        <div className="big" style={{ margin: '0.6rem 0 1rem' }}>
          <M tex={String.raw`\mathbb{E}_{Q}\,D_{\mathrm{KL}}[Q(s\mid o,\pi)\,\|\,Q(s\mid\pi)] = I(s;o\mid\pi)`} />
        </div>
        <ul className="pts">
          <li>
            EFE&rsquo;s epistemic term <strong>is</strong> Lindley&rsquo;s 1956 expected
            information gain
          </li>
          <li>
            Its pragmatic term adds what pure info-gain lacks: <em>find what you are looking for</em>
          </li>
        </ul>
        <Reveal at={1}>
          <div className="take teal" style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
            <span>
              Nobody had pointed this machinery at an interpretability budget. So I did.
            </span>
            <Link
              href="/active-inference"
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
              Part II: the bridge →
            </Link>
          </div>
        </Reveal>
      </Slide>
    </Deck>
  );
}

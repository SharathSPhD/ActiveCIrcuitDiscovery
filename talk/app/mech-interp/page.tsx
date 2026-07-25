import type { Metadata } from 'next';
import Link from 'next/link';
import { Kicker, Callout, Deep, LinkCards, Fig, Eq, M } from '../../components/Prose';
import { SuperpositionSVG, TranscoderSVG, PipelineSVG, BudgetSVG } from '../../components/DiagramsPart1';

export const metadata: Metadata = {
  title: 'Part I — Mechanistic Interpretability, from Neurons to Attribution Graphs',
  description:
    'Superposition, sparse autoencoders, Golden Gate Claude, transcoders, and circuit tracing — the exact substrate Active Circuit Discovery operates on.',
};

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
    desc: 'Browse GemmaScope transcoder features and generate attribution graphs for Gemma-2-2B in the browser. The demo embedded below runs here.',
    href: 'https://www.neuronpedia.org/gemma-2-2b/graph',
  },
];

export default function Page() {
  return (
    <>
      <section className="band dark" style={{ paddingTop: '4.5rem' }}>
        <div className="reading">
          <Kicker>Part I · ~20 minutes</Kicker>
          <h2 className="sec">Mechanistic interpretability, from neurons to attribution graphs</h2>
          <p className="lede">
            This part owes the room a debt: most of you model brains, not transformers. The goal is
            to hand you, in twenty minutes, the exact objects the ACD agent acts on — transcoder
            features, attribution graphs, causal interventions — and the one open problem in this
            field that should feel like home: <em>adaptive experiment selection under a budget</em>.
          </p>
          <p>
            Everything here follows one thread of work, most of it from Anthropic&rsquo;s
            interpretability team, because that thread produced the open tooling the paper runs on.
            Each claim links to its source; the cards are worth opening later.
          </p>
        </div>
      </section>

      <section className="band">
        <div className="reading">
          <Kicker>I.1 · The unit-of-analysis problem</Kicker>
          <h2 className="sec">Neurons lie. Features are directions.</h2>
          <p>
            The founding document of the circuits agenda is <em>Zoom In</em> (Olah et al., Distill
            2020). It stakes three claims: <strong>features</strong> — not neurons — are the
            fundamental units of neural networks, and they correspond to <em>directions</em> in
            activation space; features compose into <strong>circuits</strong>, computational
            subgraphs of features connected by weights; and both are <strong>universal</strong> —
            analogous curve detectors appear in AlexNet, InceptionV1, VGG19 and ResNet, learned
            independently. The evidence standard it set matters: curve detectors in InceptionV1 were
            established by seven independent lines of evidence, down to reimplementing the weights
            by hand.
          </p>
          <p>
            The obstacle it named also matters: <strong>polysemanticity</strong>. A single
            InceptionV1 unit fires for cat faces, fronts of cars, and cat legs. If units mix
            unrelated concepts, circuit analysis compounds ambiguity at every layer. The proposed
            explanation — confirmed in toy models two years later — is{' '}
            <strong>superposition</strong>: a network that wants to represent more features than it
            has dimensions stores them as nearly-orthogonal directions and tolerates the
            interference, because feature sparsity makes collisions rare. Polysemanticity is what
            superposition looks like when you insist on reading the neuron basis.
          </p>
          <Fig caption={
            <>Superposition, schematically. Left: with dense features, a 2-D activation space holds two
            clean basis-aligned features. Right: sparse features let the network pack five directions
            into the same plane; any single neuron (axis) now reads a mixture. The features are still
            there — as directions — but no neuron is monosemantic. (After Elhage et al. 2022.)</>
          }>
            <SuperpositionSVG />
          </Fig>
          <Callout title="Why you should care" tone="violet">
            For this audience: superposition is a claim about the <em>state space</em> of the system
            you want to explain. The hidden causes are linearly readable but not axis-aligned, so any
            method that inspects native coordinates — single neurons, single MLP units — is doing
            inference under the wrong parameterisation. The field&rsquo;s answer was to learn a better
            one.
          </Callout>
        </div>
      </section>

      <section className="band soft">
        <div className="reading">
          <Kicker>I.2 · Dictionary learning</Kicker>
          <h2 className="sec">Sparse autoencoders: buying back monosemanticity</h2>
          <p>
            <em>Towards Monosemanticity</em> (2023) made the fix concrete. Train a{' '}
            <strong>sparse autoencoder</strong> (SAE) — reconstruction loss plus an L1 sparsity
            penalty — on billions of MLP activations, with an overcomplete latent basis. From a
            512-neuron MLP it recovered thousands of features that behave monosemantically:
            dedicated features for Arabic script, DNA sequences, base64 strings, Hebrew — where a
            typical <em>neuron</em> of the same model responds to “a mixture of academic citations,
            English dialogue, HTTP requests, and Korean text.”
          </p>
          <p>
            <em>Scaling Monosemanticity</em> (May 2024) ran the same recipe on a production model —
            SAEs of 1M, 4M and <strong>34M features</strong> on a middle-layer residual stream of
            Claude 3 Sonnet — and found features for Golden Gate Bridge (firing across languages{' '}
            <em>and images</em>), code bugs, sycophantic praise, secrecy, deception-adjacent
            behaviour. Two findings matter for everything downstream. First, features are{' '}
            <strong>causal handles</strong>: clamp the Golden Gate feature to ~10× its maximum
            activation and the model self-identifies as the bridge; clamp a scam-email feature and
            the model writes the scam it would otherwise refuse. Second, the team&rsquo;s own
            caution: finding a feature is not the same as knowing what the model will do with it.
          </p>
          <p>
            Anthropic made the first point unforgettable by shipping it:{' '}
            <a href="https://www.anthropic.com/news/golden-gate-claude" target="_blank" rel="noopener">
              Golden Gate Claude
            </a>{' '}
            was live on claude.ai for 24 hours in May 2024 — a single feature clamped, a
            frontier model cheerfully insisting it is a suspension bridge. It remains the cleanest
            public demonstration that these learned directions are steering wheels, not just
            correlates. ACD&rsquo;s steering benchmark (Part III) is a small-model, quantified
            descendant of exactly this experiment — five landmark concepts on Gemma-2-2B, with a
            random-feature control the original demo never needed.
          </p>
          <LinkCards cards={ANTHROPIC_CARDS} />
          <h3 className="sub">Touch one: a live GemmaScope transcoder feature</h3>
          <p>
            Below is a live Neuronpedia dashboard for a transcoder feature of{' '}
            <strong>Gemma-2-2B</strong> — the very dictionary ACD&rsquo;s candidate features come
            from (GemmaScope, DeepMind 2024: 400+ JumpReLU SAEs and per-layer transcoders across
            every layer of Gemma-2 2B/9B, &gt;30M features). Activations, auto-interp explanation,
            and a test box you can type into. This is what one node of the attribution graph{' '}
            <em>is</em>.
          </p>
          <Fig
            panel={false}
            caption={
              <>
                Live embed: feature 16369 of the layer-20 GemmaScope transcoder (16k dictionary) for
                Gemma-2-2B, via Neuronpedia. During the talk: type a sentence, watch the feature
                respond. Explore more at{' '}
                <a href="https://www.neuronpedia.org/gemma-2-2b" target="_blank" rel="noopener">
                  neuronpedia.org/gemma-2-2b
                </a>.
              </>
            }
          >
            <iframe
              src="https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-transcoder-16k/16369?embed=true&embedexplanation=true&embedplots=true&embedtest=true"
              title="Neuronpedia — Gemma-2-2B layer-20 transcoder feature"
              style={{ width: '100%', height: 540, border: '1px solid rgba(27,26,22,.15)', borderRadius: 12, background: '#fff' }}
              loading="lazy"
            />
          </Fig>
        </div>
      </section>

      <section className="band">
        <div className="reading">
          <Kicker>I.3 · From features to circuits</Kicker>
          <h2 className="sec">Circuits, and the cost of finding them</h2>
          <p>
            Features alone are a vocabulary; explanation needs <em>circuits</em> — who feeds whom to
            produce the behaviour. The canonical existence proof is the{' '}
            <strong>IOI circuit</strong> (Wang et al. 2022): for prompts like{' '}
            <em>“When John and Mary went to the store, John gave the bag to&nbsp;__”</em>, GPT-2
            small routes the answer through 26 attention heads in seven functional classes —
            duplicate-token heads detect the repeated “John”, S-inhibition heads suppress it,
            name-mover heads copy “Mary” into the output, and backup name movers take over when the
            primaries are ablated. Established by <strong>path patching</strong>: swap in
            activations from a corrupted prompt along one path at a time and measure the effect.
            The paper also fixed the field&rsquo;s evaluation vocabulary — faithfulness,
            completeness, minimality. ACD&rsquo;s IOI benchmark is this task, transplanted to
            transcoder features on Gemma and Llama; the paper finds the same late-layer name-mover
            signature (its top causal features sit in layers 24–25 of 26 — Part III).
          </p>
          <p>
            Manual circuit analysis took person-months, so the field automated it — and here the
            compute economics that motivate ACD become visible. <strong>ACDC</strong> (Conmy et al.
            2023) walks the computational graph in reverse topological order and prunes every edge
            whose corruption changes output KL by less than a threshold: faithful, but one forward
            pass per candidate edge — tens of thousands of passes on GPT-2-small-sized graphs.{' '}
            <strong>Edge Attribution Patching</strong> (Syed et al. 2023) replaces all of that with
            a first-order Taylor approximation:
          </p>
          <Eq tex={String.raw`\Delta L \;\approx\; (e_{\text{corrupt}} - e_{\text{clean}})^{\top}\, \nabla_{e} L\big|_{\text{clean}}`} />
          <p>
            — scores for <em>every</em> edge from two forward passes and one backward pass, and it
            recovers ground-truth circuits <em>better</em> than the exhaustive method it
            approximates. Remember that result: in Part III, a direct EAP ranking is the baseline
            the ACD agent fails to beat, and the paper says so in the abstract. EAP being
            embarrassingly strong is not an ACD-specific inconvenience; it is the field&rsquo;s
            recurring punchline.
          </p>
          <Deep title="Deep dive: what EAP misses — and why verification stays expensive">
            <p>
              EAP is a linearisation, and it inherits linearisation pathologies: zero-gradient
              saturation (fixed by integrated-gradients variants, EAP-IG), sign-blindness of
              KL-based variants to suppressive components (ACDC famously drops the IOI{' '}
              <em>negative</em> name movers), and dependence on the corruption distribution. Worse,
              the whole faithfulness-metric edifice is fragile: Miller et al. (2024) show published
              circuit faithfulness scores swing wildly under zero- vs mean- vs resample-ablation and
              other seemingly minor choices. The practical consequence: attribution gives you cheap,
              noisy, <em>correlational</em> rankings, and real causal verification — actually
              running interventions — remains the expensive, trusted signal. That gap between cheap
              hypothesis and costly experiment is precisely where a Bayesian experiment-selection
              agent earns its keep.
            </p>
          </Deep>
        </div>
      </section>

      <section className="band soft">
        <div className="reading">
          <Kicker>I.4 · Transcoders</Kicker>
          <h2 className="sec">Why ACD&rsquo;s features are transcoder features</h2>
          <p>
            SAEs describe <em>what is represented</em> at a point in the network. But circuits need
            to trace computation <em>through</em> the MLP nonlinearity, and there SAEs force you to
            differentiate through the original MLP per-input. A <strong>transcoder</strong>{' '}
            (Dunefsky, Chlenski &amp; Nanda 2024) instead learns a sparse approximation of the MLP&rsquo;s{' '}
            <em>input→output function</em> — it doesn&rsquo;t describe the MLP, it{' '}
            <em>replaces</em> it with a wide, sparse, interpretable one. The payoff is structural:
            attribution between transcoder features factorises into an{' '}
            <strong>input-invariant</strong> term (“virtual weights”, the downstream encoder dotted
            with the upstream decoder — computable once from weights alone) times an{' '}
            <strong>input-dependent</strong> scalar (the upstream feature&rsquo;s activation).
            Feature-to-feature causal structure through MLPs becomes linear algebra. Follow-up work
            (Paulo et al. 2025) argues transcoder latents are also simply more interpretable than
            SAE latents, and the 2025 Anthropic stack is built on transcoders end to end.
          </p>
          <Fig caption={
            <>The transcoder move. The MLP is swapped for a wide sparse dictionary trained to imitate
            its input→output map. Features become causal computational units: interventions on them
            propagate through the real network via the replacement wiring.</>
          }>
            <TranscoderSVG />
          </Fig>
          <Callout title="Terminology for the rest of the talk">
            A <strong>feature</strong> from here on is a transcoder latent, identified by the triple{' '}
            <code>(layer, token position, feature index)</code>. Gemma-2-2B carries a 16k-entry
            GemmaScope transcoder at each of its 26 layers; Llama-3.2-1B uses community transcoders
            (<code>mntss/transcoder-Llama-3.2-1B</code>) across its 16 layers. On a typical IOI
            prompt, ~12,600 Gemma features are active; ~2,200 survive influence pruning. That set is
            the ACD agent&rsquo;s world.
          </Callout>
        </div>
      </section>

      <section className="band dark2">
        <div className="reading">
          <Kicker>I.5 · Attribution graphs</Kicker>
          <h2 className="sec">Circuit tracing: the substrate ACD consumes</h2>
          <p>
            The 2025 methods paper (<em>Circuit Tracing</em>, Ameisen et al.) assembles all of the
            above into a pipeline. Swap every MLP for its transcoder → a{' '}
            <strong>replacement model</strong>. For a specific prompt, freeze attention patterns and
            layer-norm denominators at their actual values and add <strong>error nodes</strong>{' '}
            (true MLP output minus transcoder reconstruction) → a <strong>local replacement
            model</strong> that reproduces the original model&rsquo;s output on that prompt{' '}
            <em>exactly</em>, and — crucially — is <em>linear</em> in feature activations. In that
            linear system, every edge of the <strong>attribution graph</strong> (token embeddings →
            features → features → logits) is an exact direct effect, not an estimate. Prune to the
            influential subgraph (~10× smaller, retaining ~80% of explained behaviour) and you have
            a per-prompt causal hypothesis about the computation.
          </p>
          <Fig wide caption={
            <>The full substrate, prompt to intervention. Attribution (via EAP over the local
            replacement model) is cheap and yields the graph; <code>feature_intervention</code> is
            the trusted causal probe. ACD sits between them, deciding which probe to spend next.</>
          }>
            <PipelineSVG />
          </Fig>
          <p>
            The companion <em>Biology</em> paper is the demonstration that this machinery finds
            real structure at frontier scale: Claude 3.5 Haiku computing “capital of the state
            containing Dallas” <em>through</em> internal Texas features (swap in California
            features and it says Sacramento), planning rhyme words before writing the line,
            assembling a jailbreak acronym letter-by-letter without representing the word. And it is
            honest about coverage — satisfying circuit insight on roughly a quarter of prompts
            attempted, graphs as hypotheses requiring intervention validation, published diagrams as
            “highly distilled, subjectively determined simplifications.”
          </p>
          <p>
            In May 2025 the machinery went public: <code>circuit-tracer</code> (built by Anthropic
            Fellows with Decode Research) ships per-layer transcoders, attribution, pruning and —
            the part everything in this talk hangs on — a clean causal API:
          </p>
          <pre>{`from circuit_tracer import ReplacementModel, attribute

model = ReplacementModel.from_pretrained(
    "google/gemma-2-2b", transcoder_set="gemma")

graph = attribute(prompt, model)          # EAP attribution graph (~18 s)

logits, _ = model.feature_intervention(   # causally correct intervention (~30 ms)
    prompt, [(layer, pos, feature_idx, value)])`}</pre>
          <p>
            One attribution graph costs ~18 s on Gemma-2-2B; one <code>feature_intervention</code>{' '}
            call — ablate (value 0), patch (reference value), or steer (multiple of the clean
            activation) — costs ~30 ms, with the intervention propagating through the real network.
            ACD uses the graph for hypotheses and the interventions for evidence. KL divergence
            between clean and intervened next-token distributions is the effect measure throughout.
          </p>
          <LinkCards cards={GRAPH_CARDS} />
        </div>
      </section>

      <section className="band">
        <div className="reading">
          <Kicker>I.6 · The open problem</Kicker>
          <h2 className="sec">Who decides which experiment to run next?</h2>
          <p>
            Put the economics side by side. Attribution: one 18-second pass yields importance
            scores for thousands of features — cheap, correlational, known-fragile. Causal
            verification: 30 ms per feature per intervention type, but the honest currency of the
            field — and on any real auditing task you cannot afford it exhaustively. A 2,200-feature
            candidate set × 3 intervention types is a 6,600-arm experiment space; a realistic
            verification budget might be 20 probes. Every existing automated method spends that
            budget <em>non-adaptively</em>: ACDC sweeps everything, EAP ranks once and walks the
            list, and even the impressive LLM-agent systems (MAIA, 2024) choose experiments by
            prompt-engineered heuristics with no belief state, no information-theoretic objective,
            no notion of what an observation is worth.
          </p>
          <Fig caption={
            <>The selection problem. A pruned candidate set (nodes, sized by graph importance) and a
            budget of 20 causal probes. Static rankings walk a fixed list; an adaptive agent updates
            beliefs after each observation and re-plans. The question of this talk: what should the
            objective of that agent be?</>
          }>
            <BudgetSVG />
          </Fig>
          <p>
            Sequential selection of maximally informative experiments under uncertainty has a
            seventy-year-old statistical home — Lindley&rsquo;s 1956 expected information gain,
            MacKay&rsquo;s 1992 information-based objective functions, the modern Bayesian
            experimental-design literature and its active-causal-discovery branch (Tigas et al.
            2022). And it has a biological formulation this room knows better than anyone:{' '}
            <strong>Expected Free Energy</strong>, whose epistemic term <em>is</em> expected
            information gain (<M tex={String.raw`\mathbb{E}_{Q}\,D_{\mathrm{KL}}[Q(s\mid o,\pi)\,\|\,Q(s\mid\pi)] = I(s;o\mid\pi)`} />{' '}
            — Friston et al. 2015), and whose pragmatic term supplies exactly what pure information
            gain lacks: a preference for experiments that find what you are <em>looking for</em>.
            To our knowledge, nobody had put that machinery in charge of a mechanistic
            interpretability budget before this paper. That is the bridge Part II builds.
          </p>
          <div style={{ marginTop: '2.2rem' }}>
            <Link
              href="/active-inference"
              style={{ fontFamily: 'var(--grotesk)', fontWeight: 600, fontSize: '.9rem', textDecoration: 'none', background: 'var(--teal)', color: '#fff', padding: '11px 22px', borderRadius: 999 }}
            >
              Part II: the bridge →
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}

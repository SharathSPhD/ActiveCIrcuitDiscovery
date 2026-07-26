'use client';

import Link from 'next/link';
import Deck, { Slide, Reveal, NextLead, useSlideStep } from '../../components/Deck';
import { Eq, M, LinkCards, Callout } from '../../components/Prose';
import { SuperpositionSVG, TranscoderSVG, PipelineSVG } from '../../components/DiagramsPart1';
import {
  NextTokenSVG, StackSVG, PolysemanticSVG, DictionarySAESVG, SteeringDialSVG,
  IOICircuitSVG, LeversSVG, KLRulerSVG, CostScalesSVG, StaticVsAdaptiveSVG, LadderSVG,
} from '../../components/TeachCore';
import GraphPeek from '../../components/GraphPeek';
import { EntangledSVG, HighwaySVG, ExplosionSVG } from '../../components/TeachStory';

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

function LadderBuild() {
  // advance the ladder rung-by-rung with the clicker (steps 0–4 → stages 1–5)
  const step = useSlideStep();
  return <LadderSVG stage={Math.min(step + 1, 5)} />;
}

export default function MechDeck() {
  return (
    <Deck part="Chapter I · Inside the black box" title="Mechanistic interpretability, from zero">
      {/* 1 — chapter title + roadmap */}
      <Slide
        title="The climb ahead"
        brief="Chapter roadmap: from what a model does, to the experiment-selection problem"
        steps={5}
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
          <LadderBuild />
        </div>
        <Reveal at={5}>
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
              sentence used throughout the rest of the talk (the “indirect-object
              identification”, or IOI, task). Strictly, models read <em>tokens</em>, not words —
              the distinction matters nowhere in this talk. The model strongly
              prefers “Mary”, because “John” already appeared as the giver — to get this right, it
              must have tracked who is who.
            </p>
            <p>
              (The probability numbers on this slide and the KL slide are illustrative, chosen to
              make the pattern legible — the qualitative story matches Gemma-2-2B&rsquo;s real
              behaviour on this prompt.) This slide also quietly sets up the measuring instrument
              for everything that follows: since the model&rsquo;s entire output is a probability distribution, any
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
              Read <strong>tokens</strong> in (roughly: words) → put a probability on every
              possible next token. That is the whole job.
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              To put 72% on <strong>Mary</strong>, something inside must have tracked{' '}
              <em>who already gave the bag</em> — remember this sentence: it returns throughout the talk
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
        <p className="big dim" style={{ maxWidth: '52ch' }}>
          Picture a codebase of 2.6 billion lines of alien code: no comments, no variable names,
          no documentation — <strong style={{ color: 'var(--cream)' }}>and you are responsible
          for debugging it.</strong>
        </p>
        <div className="fig-panel" style={{ maxWidth: 980 }}>
          <StackSVG />
        </div>
        <ul className="pts compact" style={{ marginTop: '0.7rem' }}>
          <Reveal at={1}>
            <li>
              Two kinds of numbers: <strong>weights</strong> — 2.6B learned settings, fixed — and{' '}
              <strong>activations</strong> — the momentary values that flow when a sentence goes
              in. A <strong>neuron</strong> = one slot in that flow.
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              Every weight readable, every activation overwritable — <strong>perfect access, zero
              labels</strong>. The missing thing is a <em>vocabulary</em>.
            </li>
          </Reveal>
        </ul>
        <NextLead>The ideal would be simple: one concept, one pathway. Reality is not that.</NextLead>
      </Slide>


      {/* NEW — beat 2: the entangled mess */}
      <Slide
        title="The entangled mess"
        brief="The ideal (one concept, one pathway) vs the reality (everything smeared) — the chapter's villain"
        steps={1}
        notes={
          <>
            <p>
              The picture everyone wants is on the left: one concept, one pathway, a clear causal
              link — a network you could read like a wiring diagram. The reality is on the right:
              every concept is smeared across hundreds of neurons, and every neuron participates
              in hundreds of concepts. A causal map cannot be built if the individual nodes cannot
              be isolated. The next three slides are, in order: the evidence that units really do
              mix (a single neuron caught in the act), the explanation for why the network
              tangles itself on purpose (superposition), and the fix (dictionary learning).
            </p>
          </>
        }
      >
        <p className="kicker">I.2 · The villain of the chapter</p>
        <h1>
          The <span className="accent">entangled mess</span>
        </h1>
        <div className="fig-panel" style={{ maxWidth: 1000 }}>
          <EntangledSVG />
        </div>
        <Reveal at={1}>
          <div className="take">
            No isolated nodes → no causal map. The next three slides: the <strong>evidence</strong>,
            the <strong>explanation</strong>, and the <strong>fix</strong>.
          </div>
        </Reveal>
        <NextLead>First, the evidence — one neuron, caught in the act.</NextLead>
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
        <p className="kicker">I.2 · The mess, measured</p>
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
        <p className="kicker">I.2 · Why the network tangles itself</p>
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
                  A “direction” = a fixed <strong>combination of neurons</strong> — the concept
                  isn&rsquo;t at hook #7, it&rsquo;s “30% of hook 7 + 70% of hook 8”
                </li>
              </Reveal>
            </ul>
          </div>
        </div>
        <Reveal at={3}>
          <div className="take violet">
            The concepts are still there — recoverable, linear —{' '}
            <strong>just read out in the wrong coordinates.</strong> Reading one neuron at a time
            is why you see gibberish.
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
              dedicated features for Arabic script, DNA sequences, base64 strings, Hebrew — where
              one neuron of that same (one-layer study) model responds to “a mixture of academic
              citations, English dialogue, HTTP requests, and Korean text.”
            </p>
            <p>
              Caveat the field itself carries: dictionaries are not unique. Features split and
              merge as dictionary size changes, training seed matters, and whether SAE features
              are the <em>right</em> decomposition remains actively contested — one reason the
              causal probes of the next slides, not the dictionary itself, carry the evidential
              weight in this talk.
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
          <ul className="pts compact">
            <li>
              The names aren&rsquo;t built in: read the thousands of snippets where a feature
              fires hardest, and <strong>the pattern names itself</strong> — you&rsquo;ll do
              exactly this on a live feature in two slides
            </li>
          </ul>
        </Reveal>
        <NextLead>Named features are a nice story. How do you prove a feature is real?</NextLead>
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
              Strictly: steering shows the direction is <em>causally potent</em>, not that it is
              the model&rsquo;s own unit — <strong>the random-direction control in Chapter III
              exists for exactly this reason</strong>
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
              ACD agent&rsquo;s candidate features come from. (Gemma Scope, DeepMind 2024, shipped
              400+ JumpReLU SAEs across every layer of Gemma-2 2B/9B; the per-layer{' '}
              <em>transcoders</em> used by circuit-tracer and this demo were the follow-up
              release, 2025.) It shows top activating examples, an auto-generated explanation, and a
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
        <p className="kicker">I.2 · Not hypothetical — the Gemma experiment</p>
        <h2>
          A real Gemma-2-2B feature — <span className="dim">type a sentence into it</span>
        </h2>
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', margin: '0.2rem 0 0.7rem', fontFamily: 'var(--grotesk)', fontSize: 'clamp(.72rem,1vw,.85rem)' }}>
          <span className="dim" style={{ alignSelf: 'center' }}>try:</span>
          {[
            'The Golden Gate Bridge was closed for repairs.',
            'She walked across the old suspension bridge at dawn.',
            'The committee approved the budget without debate.',
          ].map((t) => (
            <span key={t} style={{ border: '1px solid var(--navy-hairline)', borderRadius: 999, padding: '4px 12px', background: 'var(--navy-panel)', color: 'var(--cream)' }}>
              “{t}”
            </span>
          ))}
          <span className="dim" style={{ alignSelf: 'center' }}>— watch which ones light it up, and how hard</span>
        </div>
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
        <p className="dim" style={{ fontFamily: 'var(--grotesk)', fontSize: 'clamp(.74rem,1vw,.88rem)', margin: '0.5rem 0 0' }}>
          This is also where feature <em>names</em> come from: the dashboard&rsquo;s top
          activating snippets are the evidence, the label is read off them. Explore the whole
          dictionary at{' '}
          <a href="https://www.neuronpedia.org/gemma-2-2b" target="_blank" rel="noopener">neuronpedia.org/gemma-2-2b</a>.
        </p>
        <NextLead>A vocabulary of features is not yet an explanation. Concepts need wiring.</NextLead>
      </Slide>


      {/* NEW — beat 3: the residual stream highway */}
      <Slide
        title="The highway"
        brief="Discard the layer-cake: one shared communication channel, with read/write ramps"
        steps={2}
        notes={
          <>
            <p>
              Before circuits can make sense, one mental-model upgrade: discard the textbook
              picture of a stack of layers each transforming the whole signal. The working picture
              is a <strong>highway</strong> — the residual stream, one continuous communication
              channel per token running the full depth of the model. Attention heads and MLPs are
              not stages the signal passes through; they are <strong>off-ramps and on-ramps</strong>:
              each one <em>reads</em> a little information off the stream, computes something, and{' '}
              <em>writes</em> its contribution back. Most of the signal flows straight through,
              untouched.
            </p>
            <p>
              This picture is what makes “circuits” a natural idea: a circuit is a chain of
              components that read what earlier components wrote — traffic on the shared highway.
              It is also why features (directions in the stream) are the right unit: they are the
              cargo the ramps read and write.
            </p>
          </>
        }
      >
        <p className="kicker">I.3 · A mental-model upgrade</p>
        <h1>
          The <span className="accent">highway</span>
        </h1>
        <div className="fig-panel" style={{ maxWidth: 1020 }}>
          <HighwaySVG />
        </div>
        <ul className="pts compact" style={{ marginTop: '0.6rem' }}>
          <Reveal at={1}>
            <li>
              Not a stack of stages — <strong>one shared stream</strong>, with components that{' '}
              <em>read from</em> and <em>write to</em> it
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              A <strong>circuit</strong> = a chain of ramps where later ones read what earlier
              ones wrote
            </li>
          </Reveal>
        </ul>
        <NextLead>Here is one real chain, fully worked out.</NextLead>
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
              Every arrow proven by a <strong>causal experiment</strong> — one path swapped at a
              time (exactly what “patching” means is the next slide)
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
        <p className="big dim" style={{ marginBottom: '0.5rem' }}>
          Three different causal questions:&ensp;
          <strong style={{ color: 'var(--amber)' }}>ablate → necessity</strong> ·{' '}
          <strong style={{ color: '#9fb6e8' }}>patch → sufficiency</strong> ·{' '}
          <strong style={{ color: 'var(--violet-soft)' }}>steer → capacity</strong>
        </p>
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
              greedily prune edges one at a time, keeping any whose removal shifts the output —
              but at one forward pass per candidate edge, tens of thousands of passes for
              GPT-2-small-sized graphs. Attribution patching (Nanda 2022–23) replaced patching
              with a first-order Taylor approximation; <strong>Edge Attribution Patching</strong>{' '}
              (Syed et al. 2023) extended it to edges: scores for <em>every</em> edge from two
              forward passes and one backward pass — and empirically those cheap scores recover
              the manually-found reference circuits <em>better</em> than ACDC&rsquo;s expensive
              greedy search. (Strictly, EAP linearises activation patching, not ACDC — the
              surprise is that the linearisation out-performs the search.)
            </p>
            <p>
              What the cheap methods miss: EAP is a linearisation and inherits its pathologies —
              zero-gradient saturation (addressed by integrated-gradients variants, EAP-IG, Hanna
              et al. 2024) and dependence on the corruption distribution. A separate,
              metric-shaped failure: with KL as the metric, both ACDC and EAP are blind to the{' '}
              <em>sign</em> of suppressive components — which is how ACDC famously drops
              IOI&rsquo;s negative name movers (a signed metric like logit-diff sees them). And the faithfulness-metric
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
              “Edges” = the <strong>connections between components</strong> — the arrows in the
              IOI diagram. <strong>ACDC</strong> greedily prunes them one at a time, causally —
              faithful, but tens of thousands of passes
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              <strong>EAP</strong>: every edge scored from <em>2 forward + 1 backward pass</em> —
              and the cheap scores recover the manually-found circuits better than ACDC&rsquo;s
              expensive search (Syed et al. 2023)
            </li>
          </Reveal>
        </ul>
        <Reveal at={2}>
          <Eq tex={String.raw`\Delta L \;\approx\; (e_{\text{corrupt}} - e_{\text{clean}})^{\top}\, \nabla_{e} L\big|_{\text{clean}}`} />
          <p className="dim" style={{ fontFamily: 'var(--grotesk)', fontSize: 'clamp(.8rem,1.2vw,.98rem)' }}>
            In words: predicted damage ≈ (how much the edge would change) × (how sensitive the
            output is to it) — a calculus prediction, not an actual experiment.
          </p>
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
              computation <em>through</em> the MLP nonlinearity. Why describing is not enough: an
              SAE is a snapshot at one point — it cannot say how a feature <em>causes</em> the
              features downstream, because the MLP&rsquo;s tangle sits between them. A transcoder
              swaps the tangle for something traceable. A <strong>transcoder</strong>{' '}
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
              an exact direct effect — not an estimate. The exactness is conditional: attention
              patterns and layer-norm denominators are frozen at their observed values, and error
              nodes absorb a substantial fraction of the computation the transcoders do not
              reconstruct — which bounds how much of the behaviour the interpretable graph can
              explain. Prune to the influential subgraph (~10×
              smaller, retaining ~80% of the influence mass) and the result is what the slide
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
              Every dot = a <strong>feature</strong> from this chapter; every line = “this
              feature&rsquo;s activity <strong>directly feeds</strong> that one” — the actual
              pruned graph for the chapter&rsquo;s sentence
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              Edges are <strong>exact linear direct effects</strong> — exact <em>given</em> frozen
              attention, frozen normalisation, and error nodes carrying the unexplained remainder.
              The whole graph is a <em>hypothesis</em> until probed
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
              prompt. Why only 20, when a probe is 30 ms? Because 30 ms is the 2B-model toy price:
              probe cost grows with model scale, audits run over many prompts × many behaviours,
              and each result still needs interpreting — B = 20 is the paper&rsquo;s protocol,
              standing in for every regime where causal experiments are genuinely scarce. How the
              field spends such budgets today: the attribution methods are non-adaptive — ACDC
              sweeps everything, EAP ranks once and walks the list; the LLM-agent systems (MAIA,
              Shaham &amp; Schwettmann et al. 2024) <em>are</em> adaptive — hypothesise,
              experiment, revise — but heuristic: no explicit belief state, no quantitative
              information objective, no budget-awareness. And one more absence, easy to miss: the
              attribution graph contains false positives and false negatives, yet its ranking
              carries zero concept of its own uncertainty — nothing in it can say “highly
              confident about this node, guessing about that one.” A belief state is exactly the
              thing that can.
            </p>
          </>
        }
      >
        <p className="kicker">I.6 · The wall</p>
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
                  <div className="k">probes the protocol affords — 30 ms is the 2B toy price; real audits are many prompts × behaviours × scale</div>
                </div>
              </Reveal>
            </div>
            <Reveal at={2}>
              <div className="fig-panel">
                <ExplosionSVG />
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
          <p className="big" style={{ margin: '1rem 0 0.4rem' }}>
            In plain words: <strong style={{ color: 'var(--teal-bright)' }}>pick the experiment
            you expect to learn the most from.</strong> Formally:
          </p>
          <div className="big" style={{ margin: '0 0 0.6rem' }}>
            <M tex={String.raw`\mathbb{E}_{Q}\,D_{\mathrm{KL}}[Q(s\mid o,\pi)\,\|\,Q(s\mid\pi)] = I(s;o\mid\pi)`} />
          </div>
          <p className="big dim">
            The epistemic term of <strong>Expected Free Energy</strong> (EFE — Chapter II&rsquo;s
            objective) <strong style={{ color: 'var(--teal-bright)' }}>is</strong> expected
            information gain. <span className="dim">(s = hidden states, o = observations, π = the
            chosen experiment.)</span>
          </p>
        </Reveal>
        <Reveal at={2}>
          <div className="take teal" style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
            <span>
              What if circuit discovery were treated not as a sorting algorithm — but as an{' '}
              <strong>agent foraging for information in an environment it hasn&rsquo;t mapped?</strong>
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

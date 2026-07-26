'use client';

/* eslint-disable @next/next/no-img-element */
import Link from 'next/link';
import Deck, { Slide, Reveal, NextLead } from '../components/Deck';
import { TwoFieldsSVG } from '../components/TeachCore';

const PARTS = [
  { href: '/mech-interp', num: 'I', title: 'Inside the black box', desc: 'Mechanistic interpretability from zero — features, circuits, graphs, and the experiment-selection problem', time: '16 slides' },
  { href: '/active-inference', num: 'II', title: 'The bridge', desc: 'Circuit discovery as active inference — intuitions first, then the full shipped machinery', time: '14 slides' },
  { href: '/results', num: 'III', title: 'The evidence', desc: 'Every result, in a protective order — including the control, the null, and the diagnosed failure', time: '14 slides' },
  { href: '/demo', num: 'IV', title: 'The live laboratory', desc: 'Real POMDP episodes and live feature steering on a DGX Spark — replayable when offline', time: 'interactive' },
  { href: '/qa', num: 'V', title: 'Hard questions', desc: 'The questions an expert audience will ask, each with the concession and the reply', time: '39 Q&As' },
];

export default function HomeDeck() {
  return (
    <Deck part="Active Circuit Discovery" title="Opening & agenda">
      {/* 1 — hook + paper */}
      <Slide
        title="The opening hook"
        brief="Seventy years of experiment-selection theory meets a new environment: a transformer's insides"
        steps={2}
        notes={
          <>
            <p>
              The hook, in full: the active inference community has spent decades formalising how
              agents choose informative actions — the theory that explains where an eye saccades
              next. Mechanistic interpretability has just built something unprecedented: an
              environment with perfect observability, perfect intervenability, real scientific
              stakes — and no principled way to decide which experiment to run next. This talk is
              about pointing the first machinery at the second problem.
            </p>
            <p>
              <em>Active Circuit Discovery</em> (Symmetry 18(6):1043, 2026) puts a pymdp POMDP
              agent — full Expected Free Energy, learned likelihood, action-conditioned
              transitions — in charge of choosing causal interventions on transcoder features
              inside Gemma-2-2B and Llama-3.2-1B, on top of Anthropic&rsquo;s circuit-tracer
              attribution graphs. All experiments ran on an NVIDIA DGX Spark (GB10, 128 GB unified
              memory); code, notebooks, Docker images and raw JSON results are public.
            </p>
          </>
        }
      >
        <div className="cols cols-60">
          <div>
            <p className="kicker">A talk for the Active Inference community</p>
            <h1 style={{ maxWidth: '24ch' }}>
              Circuit discovery is
              <br />
              an act of inference.
              <br />
              <span className="accent">So run it as one.</span>
            </h1>
            <Reveal at={1}>
              <p className="big dim" style={{ maxWidth: '46ch' }}>
                The theory that explains where an eye looks next — put in charge of causal
                experiments inside a language model.
              </p>
            </Reveal>
            <Reveal at={2}>
              <p className="dim" style={{ fontFamily: 'var(--grotesk)', fontSize: 'clamp(.8rem,1.1vw,.95rem)' }}>
                Sathish, S. · <em>Symmetry</em> 18(6):1043 · 2026 · experiments on NVIDIA DGX
                Spark (GB10) · code, data and this application are public
              </p>
            </Reveal>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <img
              src="/paper-page1.png"
              alt="First page of the Active Circuit Discovery paper in Symmetry"
              style={{
                maxHeight: 'min(62vh, 560px)', width: 'auto', maxWidth: '100%',
                borderRadius: 10, border: '1px solid var(--navy-hairline)',
                boxShadow: '0 18px 50px rgba(0,0,0,.5)', transform: 'rotate(1.2deg)',
              }}
            />
          </div>
        </div>
        <NextLead>Why this audience, specifically? Because two fields are asking one question.</NextLead>
      </Slide>

      {/* 2 — two fields */}
      <Slide
        title="Two fields, one question"
        brief="Where should the eye look next / which feature should be probed next — the same problem"
        steps={1}
        notes={
          <>
            <p>
              Active inference formalises how an agent chooses actions to resolve uncertainty
              about hidden states — the mathematics behind saccade selection, foraging, and active
              sensing. Mechanistic interpretability faces a problem with identical structure:
              thousands of candidate causal probes into a network&rsquo;s hidden computation, a
              tiny budget, and the need to choose the most informative one next. The same
              question, asked about two very different systems — and until this paper, the two
              literatures had not been formally connected on this problem.
            </p>
          </>
        }
      >
        <p className="kicker">The claim of the talk</p>
        <h1>Two fields, one question</h1>
        <div className="fig-panel" style={{ maxWidth: 1000 }}>
          <TwoFieldsSVG />
        </div>
        <Reveal at={1}>
          <div className="take teal">
            One field owns the <strong>theory</strong> of that question. The other just built the
            perfect <strong>laboratory</strong> for it.
          </div>
        </Reveal>
        <NextLead>What this talk promises to leave behind.</NextLead>
      </Slide>

      {/* 3 — learning outcomes */}
      <Slide
        title="What you will take away"
        brief="Four concrete abilities by the end — for novice and expert alike"
        steps={4}
        notes={
          <>
            <p>
              The talk is built so that no interpretability background is required — Chapter I
              starts at “a model predicts the next word” — while the active inference machinery is
              treated at the level of detail a pymdp author would demand. Each chapter opens with
              a contents map (press T anywhere), every slide has speaker notes (press N), and the
              live laboratory in Chapter IV runs real episodes when the GPU is reachable, or
              replays the paper&rsquo;s recorded runs identically when it is not.
            </p>
          </>
        }
      >
        <p className="kicker">Promises, up front</p>
        <h1>By the end, you can…</h1>
        <ul className="pts">
          <Reveal at={1}>
            <li>
              <strong>Read an attribution graph</strong> — and say what its nodes, edges and
              pruning mean
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              <strong>Map every POMDP object</strong> — A, B, C, D, EFE — onto a concrete
              interpretability operation
            </li>
          </Reveal>
          <Reveal at={3}>
            <li>
              <strong>Judge the results</strong> — including the control, the null, and the
              failure the paper prints about itself
            </li>
          </Reveal>
          <Reveal at={4}>
            <li>
              <strong>Run the system</strong> — live episodes and live steering, in the browser,
              against a real GPU
            </li>
          </Reveal>
        </ul>
        <NextLead>Equally important: what this work does not claim.</NextLead>
      </Slide>

      {/* 4 — scope */}
      <Slide
        title="Scope: what this is, and is not"
        brief="An intervention-selection layer; no SOTA claim; EAP stays unbeaten"
        steps={3}
        notes={
          <>
            <p>
              ACD is an <strong>intervention-selection layer</strong> over existing
              attribution-graph tooling. It consumes a pruned candidate set of transcoder features
              from Anthropic&rsquo;s <code>circuit-tracer</code> and decides, under a budget of 20
              interventions per prompt, which feature to test next and with which intervention
              type. It is <em>not</em> a whole-circuit discovery method, and the paper claims no
              state of the art. The claims that survive the paper&rsquo;s own statistics: the
              agent significantly beats random selection on Gemma IOI (+43.5% relative, paired
              permutation p = 0.031); it is competitive with bandit and greedy baselines; a direct
              EAP ranking remains the strongest static selector; and steering-driven KL
              amplification is a property of the action, not evidence of better discovery.
            </p>
            <p>
              The object of interest for this audience is not the benchmark score. It is that the
              exploration–exploitation schedule <em>emerged</em> from a generative model — with no
              hand-tuned schedule anywhere — and emerged <em>differently</em> on different
              architectures, tracking their observation statistics. The same EFE that explains
              saccades allocated a causal-experiment budget inside a transformer.
            </p>
          </>
        }
      >
        <p className="kicker">Scope, stated before anything else</p>
        <h1>
          What this is — <span className="accent">and is not</span>
        </h1>
        <ul className="pts">
          <Reveal at={1}>
            <li>
              An <strong>intervention-selection layer</strong>: 20 causal probes per prompt,
              allocated by Expected Free Energy
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              <em>Not</em> a whole-circuit method · <em>no</em> state-of-the-art claim ·
              EAP&rsquo;s static ranking <strong>stays unbeaten</strong> — stated in the abstract
            </li>
          </Reveal>
          <Reveal at={3}>
            <li>
              The interesting object: an explore→exploit schedule that{' '}
              <strong>emerged from a generative model</strong> — differently on each architecture
            </li>
          </Reveal>
        </ul>
        <NextLead>The route through the material.</NextLead>
      </Slide>

      {/* 5 — agenda */}
      <Slide
        title="The route"
        brief="Five chapters — each opens with its own contents map (press T)"
        steps={1}
        notes={
          <>
            <p>
              The ordering is deliberate. Interpretability first, because the bridge to active
              inference only lands once the intervention-selection problem is concrete. Then the
              bridge, then the evidence, then the machine running live, then the question bank.
              Navigation: arrow keys or space to advance; T opens the chapter contents anywhere;
              N opens speaker notes on any slide; the URL hash bookmarks any slide.
            </p>
          </>
        }
      >
        <p className="kicker">The route</p>
        <h1>Five chapters</h1>
        <div className="partgrid" style={{ marginTop: '1rem' }}>
          {PARTS.map((p) => (
            <Link key={p.href} href={p.href} className="partcard">
              <span className="pc-num">CHAPTER {p.num}</span>
              <span className="pc-title" style={{ fontSize: 'clamp(1rem, 1.5vw, 1.25rem)' }}>{p.title}</span>
              <span className="pc-desc">{p.desc}</span>
              <span className="pc-time">{p.time}</span>
            </Link>
          ))}
        </div>
        <Reveal at={1}>
          <div className="take teal" style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
            <span>
              Press <strong>T</strong> in any chapter for its contents · <strong>N</strong> for
              notes on any slide.
            </span>
            <Link
              href="/mech-interp"
              style={{
                fontFamily: 'var(--grotesk)', fontWeight: 600, fontSize: '.9rem', textDecoration: 'none',
                background: 'var(--teal)', color: '#fff', padding: '10px 20px', borderRadius: 999, whiteSpace: 'nowrap',
              }}
            >
              Begin Chapter I →
            </Link>
          </div>
        </Reveal>
      </Slide>
    </Deck>
  );
}

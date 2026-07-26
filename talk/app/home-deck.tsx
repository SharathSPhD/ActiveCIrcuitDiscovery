'use client';

/* eslint-disable @next/next/no-img-element */
import Link from 'next/link';
import Deck, { Slide, Reveal, NextLead } from '../components/Deck';
import { TwoFieldsSVG } from '../components/TeachCore';
import { VisionSVG } from '../components/TeachStory';

const C = { cream: '#ede8dc', soft: '#b9b4a6', teal: '#4fd8ce', amber: '#f0a24b', violet: '#b49bf0', blue: '#9fb6e8', panel: '#10151f', line: '#1e2836' };
const GF = 'var(--grotesk)';

/* four take-away tiles with small pictograms */
function PromisesSVG() {
  const tiles: [string, string, JSX.Element][] = [
    ['read a graph', C.blue, (
      <g key="g1">{[[0,18],[14,4],[28,12],[42,0],[34,22]].map(([x,y],i)=>(<circle key={i} cx={x-21+0} cy={y-11} r={3.6} fill={C.blue} opacity={0.9} />))}
        <line x1={-21} y1={7} x2={-7} y2={-7} stroke={C.blue} strokeWidth={1.4} opacity={.6}/>
        <line x1={-7} y1={-7} x2={7} y2={1} stroke={C.blue} strokeWidth={1.4} opacity={.6}/>
        <line x1={7} y1={1} x2={21} y2={-11} stroke={C.blue} strokeWidth={1.4} opacity={.6}/></g>)],
    ['map the POMDP', C.teal, (
      <g key="g2"><rect x={-22} y={-14} width={20} height={13} rx={3} fill="none" stroke={C.teal} strokeWidth={1.6}/>
        <rect x={2} y={-14} width={20} height={13} rx={3} fill="none" stroke={C.teal} strokeWidth={1.6}/>
        <rect x={-22} y={2} width={20} height={13} rx={3} fill="none" stroke={C.teal} strokeWidth={1.6}/>
        <rect x={2} y={2} width={20} height={13} rx={3} fill="none" stroke={C.teal} strokeWidth={1.6}/>
        <text x={-12} y={-4} fontSize={9} fontFamily={GF} fill={C.teal} textAnchor="middle">A</text>
        <text x={12} y={-4} fontSize={9} fontFamily={GF} fill={C.teal} textAnchor="middle">B</text>
        <text x={-12} y={12} fontSize={9} fontFamily={GF} fill={C.teal} textAnchor="middle">C</text>
        <text x={12} y={12} fontSize={9} fontFamily={GF} fill={C.teal} textAnchor="middle">D</text></g>)],
    ['judge the results', C.amber, (
      <g key="g3"><line x1={0} y1={-16} x2={0} y2={14} stroke={C.amber} strokeWidth={2}/>
        <line x1={-18} y1={-8} x2={18} y2={-8} stroke={C.amber} strokeWidth={2}/>
        <path d="M -18 -8 L -24 4 A 7 7 0 0 0 -12 4 Z" fill="none" stroke={C.amber} strokeWidth={1.5}/>
        <path d="M 18 -8 L 12 4 A 7 7 0 0 0 24 4 Z" fill="none" stroke={C.amber} strokeWidth={1.5}/></g>)],
    ['run the system', C.violet, (
      <g key="g4"><circle cx={0} cy={0} r={16} fill="none" stroke={C.violet} strokeWidth={1.8}/>
        <path d="M -4 -8 L 9 0 L -4 8 Z" fill={C.violet}/></g>)],
  ];
  return (
    <svg viewBox="0 0 860 130" style={{ width: '100%', height: 'auto' }} role="img" aria-label="Four take-aways: read a graph, map the POMDP, judge the results, run the system">
      {tiles.map(([label, color, icon], i) => {
        const x = 40 + i * 200;
        return (
          <g key={label} transform={`translate(${x + 90}, 52)`}>
            <rect x={-90} y={-44} width={180} height={88} rx={13} fill={C.panel} stroke={color} strokeWidth={1.3} />
            <g transform="translate(0,-8)">{icon}</g>
            <text x={0} y={32} fontSize={12.5} fontFamily={GF} fontWeight={700} fill={color} textAnchor="middle">{label}</text>
          </g>
        );
      })}
    </svg>
  );
}

/* scope: inside vs outside the claim */
function ScopeSVG() {
  return (
    <svg viewBox="0 0 860 180" style={{ width: '100%', height: 'auto' }} role="img" aria-label="In scope: choosing 20 probes well. Out of scope: whole-circuit discovery, beating EAP">
      <rect x={30} y={24} width={380} height={130} rx={14} fill="rgba(79,216,206,.07)" stroke={C.teal} strokeWidth={1.4} />
      <text x={220} y={52} fontSize={13} fontFamily={GF} fontWeight={700} fill={C.teal} textAnchor="middle">what it is</text>
      <text x={220} y={80} fontSize={12} fontFamily={GF} fill={C.cream} textAnchor="middle">a selection layer: spend 20 causal probes</text>
      <text x={220} y={98} fontSize={12} fontFamily={GF} fill={C.cream} textAnchor="middle">per prompt, as informatively as possible</text>
      <text x={220} y={128} fontSize={11} fontFamily={GF} fill={C.soft} textAnchor="middle">an emergent, model-sensitive explore→exploit policy</text>
      <rect x={450} y={24} width={380} height={130} rx={14} fill="rgba(224,86,122,.05)" stroke="#e0567a" strokeWidth={1.4} />
      <text x={640} y={52} fontSize={13} fontFamily={GF} fontWeight={700} fill="#e0567a" textAnchor="middle">what it is not</text>
      <text x={640} y={80} fontSize={12} fontFamily={GF} fill={C.cream} textAnchor="middle">not a whole-circuit discovery method</text>
      <text x={640} y={98} fontSize={12} fontFamily={GF} fill={C.cream} textAnchor="middle">no state-of-the-art claim — EAP stays unbeaten</text>
      <text x={640} y={128} fontSize={11} fontFamily={GF} fill={C.soft} textAnchor="middle">both stated in the paper&rsquo;s own abstract</text>
    </svg>
  );
}

/* the route: five chapter stops on a path */
function RouteSVG() {
  const stops: [string, string][] = [
    ['I', 'black box'], ['II', 'the bridge'], ['III', 'evidence'], ['IV', 'live lab'], ['V', 'questions'],
  ];
  return (
    <svg viewBox="0 0 860 110" style={{ width: '100%', height: 'auto' }} role="img" aria-label="The route through five chapters">
      <path d="M 60 62 C 200 20, 300 96, 430 58 C 560 22, 660 92, 800 52" fill="none" stroke={C.line} strokeWidth={2.5} />
      {stops.map(([n, label], i) => {
        const pts = [[60, 62], [245, 57], [430, 58], [615, 56], [800, 52]];
        const [x, y] = pts[i];
        return (
          <g key={n}>
            <circle cx={x} cy={y} r={15} fill={C.panel} stroke={C.teal} strokeWidth={1.6} />
            <text x={x} y={y + 4} fontSize={11} fontFamily={GF} fontWeight={700} fill={C.teal} textAnchor="middle">{n}</text>
            <text x={x} y={y + 34} fontSize={10.5} fontFamily={GF} fill={C.soft} textAnchor="middle">{label}</text>
          </g>
        );
      })}
    </svg>
  );
}

const PARTS = [
  { href: '/mech-interp', num: 'I', title: 'Inside the black box', desc: 'Mechanistic interpretability from zero — features, circuits, graphs, and the experiment-selection problem', time: '18 slides' },
  { href: '/active-inference', num: 'II', title: 'The bridge', desc: 'Circuit discovery as active inference — intuitions first, then the full shipped machinery', time: '17 slides' },
  { href: '/results', num: 'III', title: 'The evidence', desc: 'Every result, in a protective order — including the control, the null, the diagnosed failure, and the closing vision', time: '16 slides' },
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
        <div className="cols">
          <div>
            <p className="kicker">Mechanistic interpretability meets active inference · delivered by Dr Sharath Sathish</p>
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
                Talk delivered by <strong style={{ color: 'var(--cream)' }}>Dr Sharath
                Sathish</strong> · <em>Symmetry</em> 18(6):1043 · 2026 · experiments on NVIDIA
                DGX Spark (GB10) · code, data and this application are public ·{' '}
                <Link href="/author" style={{ color: 'var(--teal-bright)' }}>about the author →</Link>
              </p>
            </Reveal>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <a
              href="https://www.mdpi.com/2073-8994/18/6/1043"
              target="_blank"
              rel="noopener"
              title="Open the paper on mdpi.com"
              style={{
                display: 'block', textDecoration: 'none', background: '#fff', borderRadius: 12,
                border: '1px solid var(--navy-hairline)', boxShadow: '0 18px 50px rgba(0,0,0,.5)',
                transform: 'rotate(1.2deg)', overflow: 'hidden', maxWidth: '100%',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, padding: '8px 14px', background: '#f4f6f8', borderBottom: '1px solid #dde3e8' }}>
                <span style={{ fontFamily: 'var(--grotesk)', fontSize: '.72rem', color: '#2f6f8f', fontWeight: 700 }}>
                  mdpi.com · Symmetry 18(6):1043 · Open Access
                </span>
                <span style={{ fontFamily: 'var(--grotesk)', fontSize: '.68rem', color: '#4a5b68' }}>
                  open on MDPI ↗
                </span>
              </div>
              <img
                src="/paper-mdpi.png"
                alt="The published article on mdpi.com — Active Circuit Discovery, Symmetry 18(6):1043 — click to open"
                style={{ maxWidth: 'min(46vw, 640px)', width: '100%', height: 'auto', display: 'block' }}
              />
            </a>
          </div>
        </div>
        <Reveal at={2}>
          <div className="fig-panel" style={{ maxWidth: 1020, marginTop: '0.8rem', padding: '0.6rem' }}>
            <VisionSVG resolved={false} compact />
            <p className="dim" style={{ fontFamily: 'var(--grotesk)', fontSize: 'clamp(.72rem,1vw,.86rem)', margin: '0.4rem 0 0.2rem', textAlign: 'center' }}>
              somewhere in this machinery is a causal explanation — this talk is about how to
              choose the path to it (the image returns, resolved, at the close)
            </p>
          </div>
        </Reveal>
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
              The four promises track the narrative arc: the mess and the wall (Chapter I), the
              turn and the machine (Chapter II), the result and its limits (Chapter III), and the
              vision at the close. The talk is built so that no interpretability background is
              required — Chapter I starts at “a model predicts the next word” — while the active
              inference machinery is treated at the level of detail a pymdp author would demand. Each chapter opens with
              a contents map (press T anywhere), every slide has speaker notes (press N), and the
              live laboratory in Chapter IV runs real episodes when the GPU is reachable, or
              replays the paper&rsquo;s recorded runs identically when it is not.
            </p>
          </>
        }
      >
        <p className="kicker">Promises, up front</p>
        <h1>By the end, you can…</h1>
        <div className="fig-panel" style={{ maxWidth: 980, marginBottom: '0.8rem' }}>
          <PromisesSVG />
        </div>
        <ul className="pts compact">
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
        <div className="fig-panel" style={{ maxWidth: 980, marginBottom: '0.8rem' }}>
          <ScopeSVG />
        </div>
        <ul className="pts compact">
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
        <div className="fig-panel" style={{ maxWidth: 1020, margin: '0.4rem 0 0.8rem' }}>
          <RouteSVG />
        </div>
        <div className="partgrid" style={{ marginTop: '0.4rem' }}>
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

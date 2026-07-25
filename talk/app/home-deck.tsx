'use client';

import Link from 'next/link';
import Deck, { Slide, Reveal } from '../components/Deck';
import HeroCircuit from '../components/HeroCircuit';

const PARTS = [
  { href: '/mech-interp', num: 'I', title: 'Mech interp, from neurons to graphs', time: '~20 min' },
  { href: '/active-inference', num: 'II', title: 'The bridge: discovery as inference', time: '~20 min' },
  { href: '/results', num: 'III', title: 'Results, in full', time: '~15 min' },
  { href: '/demo', num: 'IV', title: 'Live demo on the DGX Spark', time: '~10 min' },
  { href: '/qa', num: 'V', title: 'Q&A pocket', time: 'discussion' },
];

export default function HomeDeck() {
  return (
    <Deck part="Active Circuit Discovery" title="Title & agenda">
      {/* 1 — title */}
      <Slide
        steps={1}
        notes={
          <>
            <p>
              <em>Active Circuit Discovery</em> puts a pymdp POMDP agent — full Expected Free
              Energy, learned likelihood, action-conditioned transitions — in charge of choosing
              causal interventions on transcoder features inside Gemma-2-2B and Llama-3.2-1B, on
              top of Anthropic&rsquo;s circuit-tracer attribution graphs.
            </p>
            <p>
              Paper: Sathish, S. — <em>Active Circuit Discovery</em>, Symmetry 18(6):1043 (2026).
              All experiments on NVIDIA DGX Spark (GB10, 128 GB unified memory). Code, notebooks
              (free-tier T4), Docker images and raw JSON results are public.
            </p>
          </>
        }
      >
        <div style={{ position: 'absolute', inset: 0, opacity: 0.5, pointerEvents: 'none' }}>
          <HeroCircuit />
          <div className="hero-veil" />
        </div>
        <div style={{ position: 'relative' }}>
          <p className="kicker">A talk for the Active Inference community · Sharath Sathish</p>
          <h1 style={{ maxWidth: '26ch' }}>
            Circuit discovery is an act of inference.
            <br />
            <span className="accent">So I ran it as one.</span>
          </h1>
          <Reveal at={1}>
            <p className="big dim" style={{ maxWidth: '52ch' }}>
              A pymdp agent — full EFE, learned likelihood — choosing causal interventions inside
              Gemma-2-2B, on Anthropic&rsquo;s attribution graphs.
            </p>
            <p className="dim" style={{ fontFamily: 'var(--grotesk)', fontSize: 'clamp(.8rem,1.1vw,.95rem)' }}>
              Symmetry 18(6):1043 · 2026 · experiments on NVIDIA DGX Spark (GB10)
            </p>
          </Reveal>
        </div>
      </Slide>

      {/* 2 — agenda */}
      <Slide
        steps={1}
        notes={
          <>
            <p>
              The ordering is deliberate. Mechanistic interpretability first, because the
              connection to active inference only lands once the intervention-selection problem is
              concrete. Then the bridge, then the evidence, then the machine running live, then the
              discussion.
            </p>
          </>
        }
      >
        <p className="kicker">Agenda · 60–75 minutes</p>
        <h1>Five parts</h1>
        <div className="partgrid" style={{ marginTop: '1.2rem' }}>
          {PARTS.map((p) => (
            <Link key={p.href} href={p.href} className="partcard">
              <span className="pc-num">PART {p.num}</span>
              <span className="pc-title" style={{ fontSize: 'clamp(1rem, 1.6vw, 1.3rem)' }}>
                {p.title}
              </span>
              <span className="pc-time">{p.time}</span>
            </Link>
          ))}
        </div>
        <Reveal at={1}>
          <div className="take teal">
            The order matters: the bridge only lands once the{' '}
            <strong>intervention-selection problem</strong> is concrete.
          </div>
        </Reveal>
      </Slide>

      {/* 3 — scope up front */}
      <Slide
        steps={3}
        notes={
          <>
            <p>
              ACD is an <strong>intervention-selection layer</strong> over existing
              attribution-graph tooling. It consumes a pruned candidate set of transcoder features
              from Anthropic&rsquo;s <code>circuit-tracer</code> and decides, under a budget of 20
              interventions per prompt, which feature to test next and with which intervention
              type — ablation, activation patching, or feature steering. It is <em>not</em> a
              whole-circuit discovery method, and the paper claims no state of the art. The claims
              that survive the paper&rsquo;s own statistics: the agent significantly beats random
              selection on Gemma IOI (+43.5% relative, paired permutation p = 0.031), it is
              competitive with bandit and greedy baselines, a direct EAP ranking remains the
              strongest static selector, and steering-driven KL amplification is a property of the
              action, not evidence of better discovery.
            </p>
            <p>
              The interesting object for this audience is not the benchmark score. It is that the
              exploration–exploitation schedule <em>emerged</em> from a generative model — ablation
              first for epistemic value, steering later for pragmatic value, on Gemma; a
              conservative ablation-heavy policy on Llama — with no hand-tuned schedule anywhere.
              The same EFE that explains saccades allocated a causal-experiment budget inside a
              transformer.
            </p>
          </>
        }
      >
        <p className="kicker">Scope, stated up front</p>
        <h1>
          What this is — <span className="accent">and is not</span>
        </h1>
        <ul className="pts">
          <Reveal at={1}>
            <li>
              An <strong>intervention-selection layer</strong>: 20 causal probes per prompt, chosen
              by EFE over circuit-tracer&rsquo;s candidates
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              <em>Not</em> a whole-circuit method, <em>no</em> state-of-the-art claim — EAP&rsquo;s
              static ranking stays unbeaten
            </li>
          </Reveal>
          <Reveal at={3}>
            <li>
              The point: the explore→exploit schedule <strong>emerged from a generative model</strong>{' '}
              — the EFE that explains saccades, allocating a causal budget inside a transformer
            </li>
          </Reveal>
        </ul>
        <Reveal at={3}>
          <div className="take" style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
            <span>Let&rsquo;s build it up properly.</span>
            <Link
              href="/mech-interp"
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
              Part I →
            </Link>
          </div>
        </Reveal>
      </Slide>
    </Deck>
  );
}

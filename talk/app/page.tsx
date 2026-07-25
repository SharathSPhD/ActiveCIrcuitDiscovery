import Link from 'next/link';
import { Kicker, Callout } from '../components/Prose';
import HeroCircuit from '../components/HeroCircuit';

const PARTS = [
  {
    href: '/mech-interp',
    num: 'PART I',
    title: 'Mechanistic interpretability, from neurons to attribution graphs',
    desc: 'The Anthropic lineage — superposition, sparse autoencoders, Golden Gate Claude, transcoders, circuit tracing — built up to the exact substrate ACD operates on, and the open problem it targets: who decides which causal experiment to run next?',
    time: '~20 min of the talk',
  },
  {
    href: '/active-inference',
    num: 'PART II',
    title: 'The bridge: circuit discovery as active inference',
    desc: 'The conceptual move first, then the full machinery: the POMDP generative model factor by factor, the exact per-step EFE, Dirichlet learning of the likelihood, and an honest account of which choices are canonical and which are design decisions.',
    time: '~20 min of the talk',
  },
  {
    href: '/results',
    num: 'PART III',
    title: 'Results, in full and without varnish',
    desc: 'Every table from the paper plus the per-step dynamics behind them: bounded oracle efficiency, the RCK decomposition, the steering selectivity null, the Llama multi-step failure and its diagnosis, sensitivity sweeps, and the hypothesis scorecard.',
    time: '~15 min of the talk',
  },
  {
    href: '/demo',
    num: 'PART IV',
    title: 'Live demo: watch the agent discover',
    desc: 'A real POMDP episode on real transcoder features — streamed live from a DGX Spark GB10 through a Cloudflare tunnel when the GPU is up, replayed from recorded runs when it isn’t. Beliefs, EFE landscape, interventions, KL, updates.',
    time: '~10 min of the talk',
  },
  {
    href: '/qa',
    num: 'PART V',
    title: 'The Q&A shield',
    desc: 'Forty hard questions this audience will actually ask — from “single-step EFE is just salience” to “EAP beats you, so what?” — each with the honest concession and the strong reply, pre-written.',
    time: 'for the discussion',
  },
];

export default function Home() {
  return (
    <>
      <section className="hero">
        <HeroCircuit />
        <div className="hero-veil" />
        <div className="hero-inner">
          <p className="kicker" style={{ color: 'var(--teal-bright)' }}>
            An expert talk for the Active Inference community
          </p>
          <h1>Circuit discovery is an act of inference. So run it as one.</h1>
          <p className="hero-sub">
            <em>Active Circuit Discovery</em> puts a pymdp POMDP agent — full Expected Free Energy,
            learned likelihood, action-conditioned transitions — in charge of choosing causal
            interventions on transcoder features inside Gemma-2-2B and Llama-3.2-1B, on top of
            Anthropic&rsquo;s circuit-tracer attribution graphs. This site is the long-form version of
            the talk: deeper than the paper, honest about every result, and ending in a live run.
          </p>
          <div style={{ display: 'flex', gap: '0.7rem', marginTop: '1.8rem', flexWrap: 'wrap' }}>
            <Link
              href="/mech-interp"
              style={{
                fontFamily: 'var(--grotesk)',
                fontSize: '.85rem',
                fontWeight: 600,
                textDecoration: 'none',
                background: 'var(--teal)',
                color: '#fff',
                padding: '10px 20px',
                borderRadius: 999,
              }}
            >
              Start with Part I →
            </Link>
            <Link
              href="/demo"
              style={{
                fontFamily: 'var(--grotesk)',
                fontSize: '.85rem',
                fontWeight: 600,
                textDecoration: 'none',
                border: '1px solid var(--teal-bright)',
                color: 'var(--teal-bright)',
                padding: '10px 20px',
                borderRadius: 999,
              }}
            >
              Jump to the live demo
            </Link>
          </div>
        </div>
      </section>

      <section className="band dark2">
        <div className="wide">
          <Kicker>The talk in five parts</Kicker>
          <h2 className="sec" style={{ maxWidth: '22ch' }}>
            Built for an audience that knows EFE better than we do
          </h2>
          <p className="lede" style={{ maxWidth: '65ch' }}>
            The ordering is deliberate. Mechanistic interpretability first, because the connection
            to active inference only lands once the intervention-selection problem is concrete.
            Then the bridge, then the evidence, then the machine running live, then the fight.
          </p>
          <div className="partgrid">
            {PARTS.map((p) => (
              <Link key={p.href} href={p.href} className="partcard">
                <span className="pc-num">{p.num}</span>
                <span className="pc-title">{p.title}</span>
                <span className="pc-desc">{p.desc}</span>
                <span className="pc-time">{p.time}</span>
              </Link>
            ))}
          </div>
        </div>
      </section>

      <section className="band">
        <div className="reading">
          <Kicker>What the paper is — and is not</Kicker>
          <h2 className="sec">Scope, stated up front</h2>
          <p>
            ACD is an <strong>intervention-selection layer</strong> over existing attribution-graph
            tooling. It consumes a pruned candidate set of transcoder features from Anthropic&rsquo;s{' '}
            <code>circuit-tracer</code> and decides, under a budget of 20 interventions per prompt,
            which feature to test next and with which intervention type — ablation, activation
            patching, or feature steering. It is <em>not</em> a whole-circuit discovery method, and
            the paper claims no state of the art. The claims that survive the paper&rsquo;s own
            statistics: the agent significantly beats random selection on Gemma IOI (+43.5%
            relative, paired permutation p&nbsp;=&nbsp;0.031), it is competitive with bandit and
            greedy baselines, a direct EAP ranking remains the strongest static selector, and
            steering-driven KL amplification is a property of the action, not evidence of better
            discovery. Part III walks every number.
          </p>
          <Callout title="Why this audience" tone="violet">
            The interesting object for active inference researchers is not the benchmark score. It
            is that the exploration–exploitation schedule <em>emerged</em> from a generative model —
            ablation first for epistemic value, steering later for pragmatic value, on Gemma; a
            conservative ablation-heavy policy on Llama — with no hand-tuned schedule anywhere. The
            same EFE that explains saccades allocated a causal-experiment budget inside a
            transformer. Whether that framing survives your scrutiny is what Part V is for.
          </Callout>
          <p style={{ fontFamily: 'var(--grotesk)', fontSize: '.85rem', color: 'var(--ink-faint)' }}>
            Paper: Sathish, S. — <em>Active Circuit Discovery</em>, Symmetry 18(6):1043 (2026) ·
            all experiments on NVIDIA DGX Spark (GB10, 128 GB unified memory) · code, notebooks
            (free-tier T4), Docker images and raw JSON results are public.
          </p>
        </div>
      </section>
    </>
  );
}

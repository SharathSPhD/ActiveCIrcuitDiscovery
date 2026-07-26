import type { Metadata } from 'next';
import Link from 'next/link';
import { Kicker, Deep } from '../../components/Prose';
import DemoApp from '../../components/DemoApp';
import SteeringLab from '../../components/SteeringLab';

export const metadata: Metadata = {
  title: 'Chapter IV — The Live Laboratory',
  description:
    'Real POMDP episodes and live feature steering on a DGX Spark, streamed to the browser — with recorded-run replay when the GPU is offline.',
};

const G = 'var(--grotesk)';

function DemoPipelineSVG() {
  const box = (x: number, w: number, title: string, sub: string, color: string) => (
    <g>
      <rect x={x} y={26} width={w} height={58} rx={11} fill={`${color}18`} stroke={color} strokeWidth={1.4} />
      <text x={x + w / 2} y={50} fontSize={12.5} fontFamily={G} fontWeight={600} fill={color} textAnchor="middle">{title}</text>
      <text x={x + w / 2} y={70} fontSize={10} fontFamily={G} fill="#b9b4a6" textAnchor="middle">{sub}</text>
    </g>
  );
  const arr = (x: number) => (
    <text x={x} y={62} fontSize={15} fill="#4fd8ce" textAnchor="middle">→</text>
  );
  return (
    <svg viewBox="0 0 1080 100" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="What pressing run does: prompt, cached graph, then twenty EFE-chosen interventions streamed back">
      {box(6, 130, 'prompt', 'canned = precomputed', '#ede8dc')}
      {arr(148)}
      {box(160, 170, 'attribution graph', '~48 candidates (18 s if new)', '#9fb6e8')}
      {arr(342)}
      {box(354, 320, '20 × [ EFE → intervene → observe → update ]', 'one real causal probe per step, ~30 ms each', '#4fd8ce')}
      {arr(686)}
      {box(698, 170, 'SSE stream', 'every step, live to this page', '#b49bf0')}
      {arr(880)}
      {box(892, 180, 'convergence', 'rolling belief-KL < 0.01', '#f0a24b')}
    </svg>
  );
}

export default function Page() {
  return (
    <>
      <section className="band dark" style={{ paddingTop: '4rem', paddingBottom: '1.4rem' }}>
        <div className="wide">
          <Kicker>Chapter IV · the live laboratory</Kicker>
          <h2 className="sec" style={{ fontSize: 'clamp(2rem, 4.4vw, 3.2rem)', maxWidth: '22ch' }}>
            Everything so far, <span style={{ color: 'var(--teal-bright)' }}>running</span>
          </h2>
          <p className="lede" style={{ maxWidth: '64ch', fontFamily: G, fontSize: 'clamp(1rem, 1.7vw, 1.3rem)' }}>
            This page is the actual system from the paper — the pymdp agent choosing real
            (feature, action) pairs inside Gemma-2-2B on an NVIDIA DGX Spark. When the badge
            reads <strong>LIVE</strong>, every number is computed as it appears. When the GPU is
            offline, the identical interface replays the paper&rsquo;s recorded runs. Same story
            either way; only the badge changes.
          </p>
          <div className="fig-panel" style={{ marginTop: '1.2rem' }}>
            <div style={{ fontFamily: G, fontSize: '.7rem', letterSpacing: '.12em', textTransform: 'uppercase', color: 'var(--cream-soft)', marginBottom: 6 }}>
              what pressing “run” actually does
            </div>
            <DemoPipelineSVG />
          </div>
        </div>
      </section>

      <section className="band dark2" style={{ paddingTop: '1.4rem' }}>
        <div className="wide">
          <h3 className="sub" style={{ color: 'var(--cream)', marginTop: 0 }}>Experiment 1 · A full discovery episode</h3>
          <p style={{ fontFamily: G, fontSize: '.9rem', color: 'var(--cream-soft)', maxWidth: '75ch', marginTop: '0.3rem' }}>
            Four things to watch, in order: <strong style={{ color: 'var(--amber)' }}>①</strong> step 1
            is almost always an ablation — with flat beliefs, the widest lever has the highest
            epistemic value. <strong style={{ color: 'var(--violet-soft)' }}>②</strong> on Gemma,
            the action column flips to steering around step 3–5 as beliefs concentrate — the
            explore→exploit handoff from Chapters II–III, happening live (the Llama replay never
            commits to that flip: 70% ablations). <strong style={{ color: 'var(--teal-bright)' }}>③</strong>{' '}
            belief entropy falls monotonically until the convergence flag at rolling
            belief-KL &lt; 0.01. <strong style={{ color: 'var(--rose)' }}>④</strong> the KL race:
            the agent&rsquo;s bar dwarfing the ablation-only baselines is the RCK amplification —
            a property of the steering lever, not a discovery claim.
          </p>
          <DemoApp />

          <h3 className="sub" style={{ color: 'var(--cream)', marginTop: '2.4rem' }}>Experiment 2 · The steering laboratory</h3>
          <p style={{ fontFamily: G, fontSize: '.9rem', color: 'var(--cream-soft)', maxWidth: '75ch', marginTop: '0.3rem' }}>
            The Golden Gate experiment from Chapter I, runnable: pick a prompt, pick one of the
            graph&rsquo;s top-10 features, and sweep its activation from 0× to 10×. Watch KL rise
            with dose and the top tokens drift toward the feature&rsquo;s concept — and recall
            from Chapter III that circuit-selected features beat random controls on{' '}
            <em>magnitude</em>, but not significantly on <em>selectivity</em>.
          </p>
          <SteeringLab />

          <Deep title="How the plumbing works">
            <p>
              Browser → Vercel edge → <code>/api/dgx/*</code> proxy (tunnel URL and API key live
              server-side) → Cloudflare Worker (permanent URL, current tunnel held in KV) →
              quick tunnel → FastAPI on the DGX Spark → one GPU lock, SSE stream back. Latency
              budget: graph build 18 s (precomputed for the canned prompts), agent EFE evaluation
              ~0.3 s per step over ~48×3 candidate-action pairs, each{' '}
              <code>feature_intervention</code> ~30 ms. A full 20-intervention episode streams in
              well under half a minute; free-form prompts pay the one-off graph build with a
              visible status line. Backend code ships in the repository under{' '}
              <code>dgx-server/</code>.
            </p>
          </Deep>
          <div style={{ marginTop: '1.6rem' }}>
            <Link
              href="/qa"
              style={{
                fontFamily: G, fontWeight: 600, fontSize: '.9rem', textDecoration: 'none',
                background: 'var(--teal)', color: '#fff', padding: '11px 22px', borderRadius: 999,
              }}
            >
              Chapter V: hard questions →
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}

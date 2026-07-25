import type { Metadata } from 'next';
import Link from 'next/link';
import { Kicker, Callout, Deep } from '../../components/Prose';
import DemoApp from '../../components/DemoApp';

export const metadata: Metadata = {
  title: 'Part IV — Live Demo: Watch the Agent Discover',
  description:
    'A real POMDP episode over real transcoder features, streamed from a DGX Spark through a Cloudflare tunnel — with recorded-run replay when the GPU is offline.',
};

export default function Page() {
  return (
    <>
      <section className="band dark" style={{ paddingTop: '4rem', paddingBottom: '1.6rem' }}>
        <div className="wide">
          <Kicker>Part IV · ~10 minutes</Kicker>
          <h2 className="sec" style={{ fontSize: 'clamp(2rem, 4.4vw, 3.2rem)', maxWidth: '20ch' }}>
            Watch the agent <span style={{ color: 'var(--teal-bright)' }}>discover</span>
          </h2>
          <p className="lede" style={{ maxWidth: '62ch', fontFamily: 'var(--grotesk)', fontSize: 'clamp(1.05rem, 1.8vw, 1.35rem)' }}>
            This is the actual system — the paper&rsquo;s pymdp agent choosing real
            (feature, action) pairs inside Gemma-2-2B, live from the DGX Spark when the badge
            reads <strong>LIVE</strong>, replaying the paper&rsquo;s recorded runs when it
            doesn&rsquo;t. Same story either way; only the badge changes.
          </p>
        </div>
      </section>

      <section className="band dark2" style={{ paddingTop: '1.6rem' }}>
        <div className="wide">
          <DemoApp />
          <Deep title="My narration track while it runs">
            <p>
              <strong>Step 1 is almost always an ablation</strong> — the B-matrix gives ablation
              the highest transition entropy, so when beliefs are flat its epistemic value
              dominates. Watch the action column after step 3–4 on Gemma: the agent flips to
              steering as importance beliefs concentrate — pragmatic value takes over, exactly the
              exploration→exploitation handoff EFE predicts. On the Llama replay it never makes
              that flip with confidence: 70% ablations. Point at the entropy panel: monotone
              decline, and the convergence flag when the rolling belief-KL drops under 0.01. Then
              the race panel: the agent&rsquo;s multi-action cumulative KL dwarfing the
              ablation-only baselines is <em>not</em> a discovery-efficiency claim — it is the RCK
              amplification from Part III, now happening in front of you.
            </p>
          </Deep>
          <Deep title="How the plumbing works">
            <p>
              Browser → Vercel edge → <code>/api/dgx/*</code> proxy (tunnel URL and API key live
              server-side) → Cloudflare tunnel → FastAPI on the DGX → one GPU lock, SSE stream
              back. Latency budget: graph build 18 s (precomputed for canned prompts), agent EFE
              evaluation ~0.3 s per step over ~48×3 candidate-action pairs, each{' '}
              <code>feature_intervention</code> ~30 ms. A full 20-intervention episode streams in
              well under half a minute. Free-form prompts pay the one-off graph build with a
              visible timer. Backend code ships in the repo under <code>dgx-server/</code>.
            </p>
          </Deep>
          <div style={{ marginTop: '1.6rem' }}>
            <Link
              href="/qa"
              style={{
                fontFamily: 'var(--grotesk)',
                fontWeight: 600,
                fontSize: '.9rem',
                textDecoration: 'none',
                background: 'var(--teal)',
                color: '#fff',
                padding: '11px 22px',
                borderRadius: 999,
              }}
            >
              Part V: the Q&amp;A pocket →
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}

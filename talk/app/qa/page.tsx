import type { Metadata } from 'next';
import { Kicker, Callout } from '../../components/Prose';
import QABrowser from '../../components/QABrowser';

export const metadata: Metadata = {
  title: 'Part V — The Q&A Pocket',
  description:
    'Thirty-nine hard questions the active inference community will actually ask — each with the concession up front and the strong reply.',
};

export default function Page() {
  return (
    <section className="band dark2" style={{ paddingTop: '4rem', minHeight: '100vh' }}>
      <div className="reading" style={{ maxWidth: 860 }}>
        <Kicker>Part V · my pocket for the discussion</Kicker>
        <h2 className="sec" style={{ fontSize: 'clamp(2rem, 4.4vw, 3rem)' }}>
          The Q&amp;A pocket
        </h2>
        <p className="lede" style={{ fontFamily: 'var(--grotesk)' }}>
          Not softballs — the questions Friston, Parr, Da Costa, Heins, Millidge, Tschantz, Sajid
          and the pymdp authors would ask, plus the interpretability-side attacks and the
          statistics objections, aimed at the paper&rsquo;s actual weak points. Each answer opens
          with the concession, then lands the strongest defensible reply. I don&rsquo;t present
          this part; it&rsquo;s here for when the room pushes.
        </p>
        <Callout title="My rules of engagement" tone="amber">
          Concede fast, concede specifically, then land the one thing the concession does not
          touch. Never defend RCK as efficiency, never claim EAP is beaten, never say
          &ldquo;first ever&rdquo; without &ldquo;to our knowledge&rdquo;. If a question exceeds
          the paper — structure learning, deep policies, continuous states — the answer is
          collaboration bait, not bluffing: the machinery is one file of pymdp and the repo is MIT.
        </Callout>
        <QABrowser />
      </div>
    </section>
  );
}

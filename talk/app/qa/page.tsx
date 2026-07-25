import type { Metadata } from 'next';
import { Kicker, Callout } from '../../components/Prose';
import QABrowser from '../../components/QABrowser';

export const metadata: Metadata = {
  title: 'Part V — The Q&A Shield',
  description:
    'Thirty-nine hard questions the active inference community will actually ask — each with the honest concession and the strong reply.',
};

export default function Page() {
  return (
    <section className="band dark2" style={{ paddingTop: '4.5rem', minHeight: '100vh' }}>
      <div className="reading" style={{ maxWidth: 860 }}>
        <Kicker>Part V · for the discussion</Kicker>
        <h2 className="sec">The Q&amp;A shield</h2>
        <p className="lede">
          These are not softballs. They are the questions Friston, Parr, Da Costa, Heins, Millidge,
          Tschantz, Sajid and the pymdp authors would ask, plus the interpretability-side attacks
          and the statistics objections — reconstructed from their published critiques (Whence the
          EFE?, the BOED reductions, the faithfulness-metrics literature) and aimed at this
          paper&rsquo;s actual weak points. Each answer opens with the honest concession, because
          this audience rewards nothing else, then gives the strongest defensible reply.
        </p>
        <Callout title="Rules of engagement for the live Q&A" tone="amber">
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

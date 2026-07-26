import type { Metadata } from 'next';
import { Kicker, Callout } from '../../components/Prose';
import QABrowser from '../../components/QABrowser';

export const metadata: Metadata = {
  title: 'Chapter V — Hard Questions',
  description:
    'Thirty-nine questions an expert audience will ask about Active Circuit Discovery — each answered with the concession first, then the strongest defensible reply.',
};

export default function Page() {
  return (
    <section className="band dark2" style={{ paddingTop: '4rem', minHeight: '100vh' }}>
      <div className="reading" style={{ maxWidth: 860 }}>
        <Kicker>Chapter V · for the discussion</Kicker>
        <h2 className="sec" style={{ fontSize: 'clamp(2rem, 4.4vw, 3rem)' }}>
          Hard questions
        </h2>
        <p className="lede" style={{ fontFamily: 'var(--grotesk)' }}>
          Not softballs. These are the questions the active inference community&rsquo;s sharpest
          readers would ask — reconstructed from the published critiques (“Whence the EFE?”, the
          BOED reductions, the faithfulness-metrics literature) and aimed at the paper&rsquo;s
          actual weak points — plus the interpretability-side attacks and the statistics
          objections. Each answer opens with the concession, then gives the strongest defensible
          reply. This chapter is not presented; it backs the discussion.
        </p>
        <Callout title="How the answers are built" tone="amber">
          Concede fast, concede specifically, then land the one thing the concession does not
          touch. RCK is never defended as efficiency; EAP is never claimed beaten; “first” never
          appears without “to our knowledge”. Where a question exceeds the paper — structure
          learning, deep policies, continuous states — the answer points at the open repository
          rather than bluffing: the machinery is one file of pymdp, and the code is MIT-licensed.
        </Callout>
        <QABrowser />
      </div>
    </section>
  );
}

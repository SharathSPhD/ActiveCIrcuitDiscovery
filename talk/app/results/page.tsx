import type { Metadata } from 'next';
import ResultsDeck from './deck';

export const metadata: Metadata = {
  title: 'Part III — Results, in Full',
  description:
    'Bounded oracle efficiency, the RCK decomposition, the steering selectivity null, the Llama multi-step failure and its diagnosis, sensitivity sweeps, and the hypothesis scorecard.',
};

export default function Page() {
  return <ResultsDeck />;
}

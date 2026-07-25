import type { Metadata } from 'next';
import MechDeck from './deck';

export const metadata: Metadata = {
  title: 'Part I — Mechanistic Interpretability, from Neurons to Attribution Graphs',
  description:
    'Superposition, sparse autoencoders, Golden Gate Claude, transcoders, and circuit tracing — the exact substrate Active Circuit Discovery operates on.',
};

export default function Page() {
  return <MechDeck />;
}

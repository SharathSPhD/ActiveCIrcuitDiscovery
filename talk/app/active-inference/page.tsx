import type { Metadata } from 'next';
import BridgeDeck from './deck';

export const metadata: Metadata = {
  title: 'Part II — The Bridge: Circuit Discovery as Active Inference',
  description:
    'The conceptual mapping and the full POMDP machinery: generative model, per-step EFE, Dirichlet learning, and which choices are canonical vs design decisions.',
};

export default function Page() {
  return <BridgeDeck />;
}

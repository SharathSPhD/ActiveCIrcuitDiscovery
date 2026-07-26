import type { Metadata } from 'next';
import Link from 'next/link';
import { Kicker } from '../../components/Prose';

export const metadata: Metadata = {
  title: 'About the Author — Dr Sharath Sathish',
  description:
    'Dr Sharath Sathish — author of Active Circuit Discovery (Symmetry, 2026). Publications, active-inference research thread, and TechNektar.',
};

const G = 'var(--grotesk)';

type Pub = {
  title: string;
  venue: string;
  year: string;
  cites?: string;
  tag?: 'active inference' | 'this talk' | 'interpretability' | 'world models' | 'epistemic AI';
};

const PUBS: Pub[] = [
  { title: 'Active Circuit Discovery: A Multi-Action POMDP Agent for Causal Feature Identification in Transformer Attribution Graphs', venue: 'Symmetry 18(6):1043', year: '2026', tag: 'this talk' },
  { title: 'Centrifugal Compressor Design and Surge Simulation for Active Inference Based Control', venue: 'ASME Turbo Expo, V009T19A011', year: '2024', cites: '1', tag: 'active inference' },
  { title: 'Refusal as a Broken Symmetry: Mechanistic Interpretability of Output-Policy Capture Across Jailbreak, Hypnosis, and Vaśīkaraṇa', venue: 'Preprints', year: '2026', tag: 'interpretability' },
  { title: 'Recognition-Gated Workspace Steering: Pratyabhijñā as an Engineering Specification for Language Model Control', venue: 'Preprints', year: '2026', tag: 'interpretability' },
  { title: 'Pramana: Fine-Tuning Large Language Models for Epistemic Reasoning through Navya-Nyaya', venue: 'arXiv:2604.04937', year: '2026', cites: '1', tag: 'epistemic AI' },
  { title: 'Pratyakṣa: A Context-Engineering System for Long-Context, Hallucination-Resistant Agentic AI', venue: 'preprint', year: '—', tag: 'epistemic AI' },
  { title: 'DreamPrice: A Causal DreamerV3 World Model for Offline Retail Pricing', venue: 'preprint', year: '2026', tag: 'world models' },
  { title: 'Deep Reinforcement Learning for Autonomous Control of Supercritical CO2 Brayton Cycles in Steel Industry Waste Heat Recovery', venue: 'preprint', year: '2026' },
  { title: 'Analysis of a 10 MW recompression supercritical carbon dioxide cycle for tropical climatic conditions', venue: 'Applied Thermal Engineering 186', year: '2021', cites: '22' },
  { title: 'Equation of state based analytical formulation for optimization of sCO2 Brayton cycle', venue: 'J. Supercritical Fluids 177', year: '2021', cites: '16' },
  { title: 'Design of 20 kW Turbomachinery for Closed Loop Supercritical CO2 Brayton Test Loop Facility', venue: 'ASME Turbo Expo', year: '2019', cites: '14' },
  { title: 'Novel Approaches for sCO2 Axial Turbine Design', venue: 'ASME Turbo Expo', year: '2019', cites: '10' },
  { title: 'Simple Recuperated s-CO2 Cycle Revisited: Optimization of Operating Parameters for Maximum Cycle Efficiency', venue: 'ASME Turbo Expo', year: '2019', cites: '9' },
  { title: 'Wind Tunnel Testing and Modeling Implications of an Advanced Turbine Cascade', venue: 'arXiv:2407.11210', year: '2024', cites: '1' },
];

const TAG_COLOR: Record<string, string> = {
  'this talk': 'var(--teal-bright)',
  'active inference': 'var(--teal-bright)',
  interpretability: 'var(--violet-soft)',
  'world models': 'var(--amber)',
  'epistemic AI': 'var(--amber)',
};

function PubRow({ p }: { p: Pub }) {
  const hot = p.tag === 'this talk' || p.tag === 'active inference';
  return (
    <div
      style={{
        display: 'flex', gap: '0.9rem', alignItems: 'baseline', padding: '0.65rem 0.9rem',
        borderRadius: 10, marginBottom: 6,
        background: hot ? 'rgba(79,216,206,.07)' : 'transparent',
        border: hot ? '1px solid rgba(79,216,206,.35)' : '1px solid transparent',
      }}
    >
      <span style={{ fontFamily: 'var(--mono)', fontSize: '.7rem', color: 'var(--cream-soft)', minWidth: 38 }}>{p.year}</span>
      <span style={{ flex: 1 }}>
        <span style={{ fontFamily: G, fontSize: '.92rem', fontWeight: hot ? 650 : 500, color: 'var(--cream)' }}>{p.title}</span>
        <span style={{ fontFamily: G, fontSize: '.76rem', color: 'var(--cream-soft)' }}> · {p.venue}{p.cites ? ` · ${p.cites} citations` : ''}</span>
      </span>
      {p.tag && (
        <span style={{
          fontFamily: 'var(--mono)', fontSize: '.6rem', letterSpacing: '.1em', textTransform: 'uppercase',
          color: TAG_COLOR[p.tag], border: `1px solid ${TAG_COLOR[p.tag]}`, borderRadius: 999, padding: '2px 9px', whiteSpace: 'nowrap',
        }}>
          {p.tag}
        </span>
      )}
    </div>
  );
}

export default function Page() {
  return (
    <section className="band dark2" style={{ paddingTop: '4rem', minHeight: '100vh' }}>
      <div className="wide">
        <Kicker>About the author</Kicker>
        <h2 className="sec" style={{ fontSize: 'clamp(2rem, 4.4vw, 3rem)', maxWidth: '24ch' }}>
          Talk delivered by <span style={{ color: 'var(--teal-bright)' }}>Dr Sharath Sathish</span>
        </h2>
        <p className="lede" style={{ fontFamily: G, maxWidth: '70ch' }}>
          Senior Data Scientist (bp plc, London) and founder of TechNektar — twenty-plus years
          spanning aerospace and energy turbomachinery to frontier AI research. The thread running
          through the recent work is the one this talk walks: agents with explicit generative
          models — active inference, world models, epistemic reasoning — pointed at real
          engineering systems, from supercritical-CO₂ compressors to the insides of language
          models.
        </p>

        {/* Scholar */}
        <h3 className="sub" style={{ color: 'var(--cream)' }}>Publications</h3>
        <p style={{ fontFamily: G, fontSize: '.85rem', color: 'var(--cream-soft)', maxWidth: '75ch', marginTop: '0.2rem' }}>
          Highlighted rows are the <strong style={{ color: 'var(--teal-bright)' }}>active
          inference</strong> thread — the same machinery in this talk, first applied to compressor
          surge control, then to circuit discovery. Full record on{' '}
          <a href="https://scholar.google.com/citations?user=dcyu5ucAAAAJ&hl=en" target="_blank" rel="noopener">
            Google Scholar
          </a>{' '}
          (84 citations · Google blocks embedding its pages, so the profile opens in a new tab).
        </p>
        <div style={{ border: '1px solid var(--navy-hairline)', borderRadius: 14, background: 'var(--navy-panel)', padding: '0.8rem 0.7rem', margin: '0.8rem 0 0' }}>
          {PUBS.map((p) => <PubRow key={p.title} p={p} />)}
          <div style={{ padding: '0.5rem 0.9rem' }}>
            <a
              href="https://scholar.google.com/citations?user=dcyu5ucAAAAJ&hl=en"
              target="_blank" rel="noopener"
              style={{ fontFamily: G, fontSize: '.82rem', fontWeight: 600 }}
            >
              All publications on Google Scholar ↗
            </a>
          </div>
        </div>

        {/* Sites */}
        <h3 className="sub" style={{ color: 'var(--cream)', marginTop: '2.6rem' }}>TechNektar — research to product</h3>
        <p style={{ fontFamily: G, fontSize: '.85rem', color: 'var(--cream-soft)', maxWidth: '75ch', marginTop: '0.2rem' }}>
          Two live windows: the research-and-engineering studio (mechanistic interpretability, RL
          control, causal inference, world models) and the portfolio. Both are the real sites,
          embedded.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '1rem', marginTop: '0.8rem' }}>
          <div style={{ border: '1px solid var(--navy-hairline)', borderRadius: 14, overflow: 'hidden', background: '#fff' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 14px', background: 'var(--navy-panel)', borderBottom: '1px solid var(--navy-hairline)' }}>
              <span style={{ fontFamily: G, fontSize: '.75rem', fontWeight: 700, color: 'var(--amber)' }}>technektar.com</span>
              <a href="https://technektar.com" target="_blank" rel="noopener" style={{ fontFamily: G, fontSize: '.7rem' }}>open ↗</a>
            </div>
            <iframe
              src="https://technektar.com"
              title="TechNektar — where frontier research becomes product"
              style={{ width: '100%', height: 520, border: 'none', display: 'block' }}
              loading="lazy"
            />
          </div>
          <div style={{ border: '1px solid var(--navy-hairline)', borderRadius: 14, overflow: 'hidden', background: '#fff' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 14px', background: 'var(--navy-panel)', borderBottom: '1px solid var(--navy-hairline)' }}>
              <span style={{ fontFamily: G, fontSize: '.75rem', fontWeight: 700, color: 'var(--teal-bright)' }}>technektar.dev</span>
              <a href="https://technektar.dev" target="_blank" rel="noopener" style={{ fontFamily: G, fontSize: '.7rem' }}>open ↗</a>
            </div>
            <iframe
              src="https://technektar.dev"
              title="Dr Sharath Sathish — portfolio"
              style={{ width: '100%', height: 520, border: 'none', display: 'block' }}
              loading="lazy"
            />
          </div>
        </div>

        <div style={{ margin: '2.2rem 0 1rem' }}>
          <Link
            href="/"
            style={{ fontFamily: G, fontWeight: 600, fontSize: '.9rem', textDecoration: 'none', background: 'var(--teal)', color: '#fff', padding: '11px 22px', borderRadius: 999 }}
          >
            ← Back to the talk
          </Link>
        </div>
      </div>
    </section>
  );
}

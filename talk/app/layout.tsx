import type { Metadata } from 'next';
import './globals.css';
import 'katex/dist/katex.min.css';
import TopNav from '../components/TopNav';
import Progress from '../components/Progress';

export const metadata: Metadata = {
  title: 'Active Circuit Discovery — An Expert Talk for the Active Inference Community',
  description:
    'Circuit discovery in large language models as active inference: mechanistic interpretability foundations, the EFE bridge, full results, a live DGX Spark demo, and an expert Q&A — companion site for the Active Circuit Discovery paper (Symmetry, 2026).',
  openGraph: {
    title: 'Active Circuit Discovery — Expert Talk',
    description:
      'Mechanistic interpretability meets Expected Free Energy: the full technical story of the ACD paper, with a live demo.',
    type: 'article',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400;1,6..72,500&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <Progress />
        <TopNav />
        <main>{children}</main>
        <footer className="footer">
          <div className="wide" style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', justifyContent: 'space-between' }}>
            <div>
              Sharath Sathish — <em>Active Circuit Discovery: Active Inference-guided interventions for
              mechanistic interpretability</em>, Symmetry 18(6):1043, 2026.
            </div>
            <div style={{ display: 'flex', gap: '1.2rem' }}>
              <a href="https://www.mdpi.com/2073-8994/18/6/1043" target="_blank" rel="noopener">Paper</a>
              <a href="https://github.com/SharathSPhD/ActiveCIrcuitDiscovery" target="_blank" rel="noopener">Code &amp; data</a>
              <a href="https://sharathsphd.github.io/ActiveCIrcuitDiscovery/" target="_blank" rel="noopener">Story version</a>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}

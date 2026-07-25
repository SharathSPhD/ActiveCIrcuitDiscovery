import katex from 'katex';

/** Server-rendered KaTeX. Use <Eq> for display math, <M> for inline. */
export function Eq({ children, tex }: { children?: string; tex?: string }) {
  const src = tex ?? children ?? '';
  const html = katex.renderToString(src, { displayMode: true, throwOnError: false });
  return <div className="eq-display" dangerouslySetInnerHTML={{ __html: html }} />;
}

export function M({ children, tex }: { children?: string; tex?: string }) {
  const src = tex ?? children ?? '';
  const html = katex.renderToString(src, { displayMode: false, throwOnError: false });
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

export function Kicker({ children }: { children: React.ReactNode }) {
  return <p className="kicker">{children}</p>;
}

export function Callout({
  title,
  tone = 'teal',
  children,
}: {
  title: string;
  tone?: 'teal' | 'violet' | 'amber' | 'rose';
  children: React.ReactNode;
}) {
  return (
    <div className={`callout ${tone === 'teal' ? '' : tone}`}>
      <div className="co-title">{title}</div>
      {children}
    </div>
  );
}

export function Deep({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className="deep">
      <summary>
        <span className="dd-sign">+</span> {title}
      </summary>
      <div className="dd-body">{children}</div>
    </details>
  );
}

export interface CardDef {
  src: string;
  title: string;
  desc: string;
  href: string;
}

export function LinkCards({ cards }: { cards: CardDef[] }) {
  return (
    <div className="linkcards">
      {cards.map((c) => (
        <a key={c.href} className="linkcard" href={c.href} target="_blank" rel="noopener">
          <div className="lc-src">{c.src}</div>
          <div className="lc-title">{c.title}</div>
          <div className="lc-desc">{c.desc}</div>
        </a>
      ))}
    </div>
  );
}

export function Fig({
  caption,
  children,
  panel = true,
  wide = false,
}: {
  caption: React.ReactNode;
  children: React.ReactNode;
  panel?: boolean;
  wide?: boolean;
}) {
  return (
    <figure className="fig" style={wide ? { maxWidth: 'var(--maxw-wide)' } : undefined}>
      {panel ? <div className="fig-panel">{children}</div> : children}
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

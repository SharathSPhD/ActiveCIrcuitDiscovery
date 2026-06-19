import { useEffect, useRef, useState } from 'preact/hooks';

/**
 * Generic animated flowchart engine used to recreate the paper's diagrams
 * (architecture, Active-Inference loop, per-step decision flow) as crisp,
 * native SVG with flowing data-particles along the edges.
 *
 * Coordinates are authored in a fixed viewBox; the SVG scales responsively.
 */

export interface FNode {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  label: string; // use \n for line breaks
  group?: string;
  shape?: 'rect' | 'round' | 'diamond';
  emphasis?: boolean;
}
export interface FEdge {
  from: string;
  to: string;
  /** anchor sides; if omitted, inferred from geometry */
  fromSide?: Side;
  toSide?: Side;
  label?: string;
  feedback?: boolean; // dashed, secondary color, no forward particle
  curve?: number; // horizontal/vertical bow for routing around
}
export interface FGroup {
  id: string;
  label: string;
  color: string;
  bbox: { x: number; y: number; w: number; h: number };
}
type Side = 'top' | 'bottom' | 'left' | 'right';

interface Props {
  nodes: FNode[];
  edges: FEdge[];
  groups?: FGroup[];
  viewW: number;
  viewH: number;
  ariaLabel: string;
  caption?: string;
  testid?: string;
}

const GROUP_COLOR: Record<string, string> = {
  backend: '#22d3ee',
  agent: '#a855f7',
  genmodel: '#f59e0b',
  io: '#34d399',
  neutral: '#8b98ac',
};

function anchor(n: FNode, side: Side): [number, number] {
  switch (side) {
    case 'top': return [n.x + n.w / 2, n.y];
    case 'bottom': return [n.x + n.w / 2, n.y + n.h];
    case 'left': return [n.x, n.y + n.h / 2];
    case 'right': return [n.x + n.w, n.y + n.h / 2];
  }
}
function inferSides(a: FNode, b: FNode): [Side, Side] {
  const dx = b.x + b.w / 2 - (a.x + a.w / 2);
  const dy = b.y + b.h / 2 - (a.y + a.h / 2);
  if (Math.abs(dy) >= Math.abs(dx)) return dy >= 0 ? ['bottom', 'top'] : ['top', 'bottom'];
  return dx >= 0 ? ['right', 'left'] : ['left', 'right'];
}

export default function FlowDiagram({ nodes, edges, groups = [], viewW, viewH, ariaLabel, caption, testid }: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const [reduce, setReduce] = useState(false);
  const [shown, setShown] = useState(false);
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));

  useEffect(() => {
    if (typeof window === 'undefined') return;
    setReduce(window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    if (!ref.current) return;
    const io = new IntersectionObserver(
      (es) => es.forEach((e) => e.isIntersecting && setShown(true)),
      { threshold: 0.05, rootMargin: '0px 0px -10% 0px' }
    );
    io.observe(ref.current);
    // Fallback: never leave the diagram hidden if the observer misses (e.g. very tall SVG).
    const t = setTimeout(() => setShown(true), 1600);
    return () => { io.disconnect(); clearTimeout(t); };
  }, []);

  const edgePath = (e: FEdge): string => {
    const a = byId[e.from];
    const b = byId[e.to];
    if (!a || !b) return '';
    const [fs, ts] = e.fromSide && e.toSide ? [e.fromSide, e.toSide] : inferSides(a, b);
    const [x1, y1] = anchor(a, fs);
    const [x2, y2] = anchor(b, ts);
    if (e.curve) {
      // route with a sideways bow (used for feedback loops)
      const mx = (x1 + x2) / 2 + e.curve;
      return `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
    }
    // smooth orthogonal-ish cubic
    const dx = x2 - x1;
    const dy = y2 - y1;
    const c = 0.45;
    if (fs === 'bottom' || fs === 'top') return `M ${x1} ${y1} C ${x1} ${y1 + dy * c}, ${x2} ${y2 - dy * c}, ${x2} ${y2}`;
    return `M ${x1} ${y1} C ${x1 + dx * c} ${y1}, ${x2 - dx * c} ${y2}, ${x2} ${y2}`;
  };

  return (
    <figure class="flow" data-testid={testid}>
      <svg
        ref={ref}
        viewBox={`0 0 ${viewW} ${viewH}`}
        role="img"
        aria-label={ariaLabel}
        class={shown ? 'flow-svg shown' : 'flow-svg'}
        preserveAspectRatio="xMidYMid meet"
      >
        <defs>
          <marker id="fd-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-soft)" />
          </marker>
          <filter id="fd-glow" x="-40%" y="-40%" width="180%" height="180%">
            <feGaussianBlur stdDeviation="2.2" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {/* group containers */}
        {groups.map((g) => (
          <g>
            <rect x={g.bbox.x} y={g.bbox.y} width={g.bbox.w} height={g.bbox.h} rx="14"
                  fill={`${g.color}0d`} stroke={`${g.color}55`} stroke-dasharray="5 4" />
            <text x={g.bbox.x + 12} y={g.bbox.y + 20} fill={g.color} font-size="13" font-weight="600">{g.label}</text>
          </g>
        ))}

        {/* edges */}
        {edges.map((e, i) => {
          const d = edgePath(e);
          const color = e.feedback ? 'var(--violet)' : 'var(--ink-soft)';
          return (
            <g key={`e${i}`}>
              <path id={`fd-${testid}-${i}`} d={d} fill="none" stroke={color}
                    stroke-width="1.6" stroke-opacity={e.feedback ? 0.7 : 0.85}
                    stroke-dasharray={e.feedback ? '5 4' : undefined}
                    marker-end="url(#fd-arrow)" />
              {!e.feedback && !reduce && shown && (
                <circle r="3" fill="var(--cyan)" filter="url(#fd-glow)">
                  <animateMotion dur={`${2 + (i % 3) * 0.4}s`} repeatCount="indefinite" begin={`${(i % 5) * 0.25}s`}>
                    <mpath href={`#fd-${testid}-${i}`} />
                  </animateMotion>
                </circle>
              )}
              {e.label && (() => {
                const a = byId[e.from], b = byId[e.to];
                const mx = ((a.x + a.w / 2) + (b.x + b.w / 2)) / 2;
                const my = ((a.y + a.h / 2) + (b.y + b.h / 2)) / 2;
                return <text x={mx} y={my - 4} fill="var(--ink-faint)" font-size="10.5" text-anchor="middle">{e.label}</text>;
              })()}
            </g>
          );
        })}

        {/* nodes */}
        {nodes.map((n, i) => {
          const color = GROUP_COLOR[n.group ?? 'neutral'] ?? GROUP_COLOR.neutral;
          const lines = n.label.split('\n');
          const cx = n.x + n.w / 2;
          return (
            <g class="fd-node" style={`--d:${i * 60}ms`} key={n.id}>
              {n.shape === 'diamond' ? (
                <polygon points={`${cx},${n.y} ${n.x + n.w},${n.y + n.h / 2} ${cx},${n.y + n.h} ${n.x},${n.y + n.h / 2}`}
                         fill="var(--bg-panel)" stroke={color} stroke-width={n.emphasis ? 2 : 1.4} />
              ) : (
                <rect x={n.x} y={n.y} width={n.w} height={n.h} rx={n.shape === 'round' ? n.h / 2 : 9}
                      fill="var(--bg-panel)" stroke={color} stroke-width={n.emphasis ? 2.2 : 1.4}
                      filter={n.emphasis ? 'url(#fd-glow)' : undefined} />
              )}
              <text x={cx} y={n.y + n.h / 2} fill="var(--ink)" font-size="11.5"
                    text-anchor="middle" dominant-baseline="middle">
                {lines.map((ln, li) => (
                  <tspan x={cx} dy={li === 0 ? `${-(lines.length - 1) * 0.6}em` : '1.2em'}>{ln}</tspan>
                ))}
              </text>
            </g>
          );
        })}
      </svg>
      {caption && <figcaption>{caption}</figcaption>}

      <style>{`
        .flow { margin: 1.6rem 0; }
        .flow-svg { width: 100%; height: auto; display: block; }
        figcaption { margin-top: 0.6rem; font-size: 0.82rem; color: var(--ink-faint); text-align: center; }
        .fd-node { opacity: 0; transform: translateY(8px); }
        .flow-svg.shown .fd-node { opacity: 1; transform: none;
          transition: opacity 0.5s ease var(--d), transform 0.5s ease var(--d); }
        @media (prefers-reduced-motion: reduce) {
          .fd-node { opacity: 1 !important; transform: none !important; transition: none !important; }
        }
      `}</style>
    </figure>
  );
}

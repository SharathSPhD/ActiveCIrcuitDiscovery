/* A real attribution graph, drawn from data: graph_ioi_gemma.json (110 pruned nodes,
   420 strongest edges) for the IOI prompt on Gemma-2-2B. Server-rendered SVG. */

import graph from '../data/graph_ioi_gemma.json';

const G = 'var(--grotesk)';
const M = 'var(--mono)';

type Node = { id: string; layer: number; ctx: number; influence: number; logit: boolean };
type Link = { s: string; t: string; w: number };

export default function GraphPeek() {
  const nodes = (graph as any).nodes as Node[];
  const links = (graph as any).links as Link[];
  const tokens = (graph as any).prompt_tokens as string[];
  const maxL = Math.max(...nodes.map((n) => n.layer));
  const nCtx = tokens.length;

  const W = 960, H = 430, padL = 56, padR = 20, padT = 24, padB = 66;
  const X = (ctx: number) => padL + (ctx / Math.max(1, nCtx - 1)) * (W - padL - padR);
  const Y = (layer: number) => H - padB - (layer / maxL) * (H - padT - padB);
  const pos = new Map(nodes.map((n) => [n.id, { x: X(n.ctx), y: Y(n.layer) }]));
  const maxW = Math.max(...links.map((l) => Math.abs(l.w)));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="A real pruned attribution graph for the IOI prompt: feature nodes by token position and layer, connected by exact linear edges">
      {/* edges */}
      {links.slice(0, 260).map((l, i) => {
        const s = pos.get(l.s), t = pos.get(l.t);
        if (!s || !t) return null;
        return (
          <line key={i} x1={s.x} y1={s.y} x2={t.x} y2={t.y}
            stroke={l.w >= 0 ? '#4fd8ce' : '#e0567a'}
            strokeWidth={0.5 + (Math.abs(l.w) / maxW) * 1.6}
            opacity={0.14 + (Math.abs(l.w) / maxW) * 0.3} />
        );
      })}
      {/* nodes */}
      {nodes.map((n) => (
        <circle key={n.id} cx={X(n.ctx)} cy={Y(n.layer)} r={2.5 + n.influence * 6.5}
          fill={n.logit ? '#f0a24b' : '#9fb6e8'} opacity={0.55 + n.influence * 0.45}>
          <title>{`${n.id} · layer ${n.layer} · influence ${n.influence.toFixed(2)}`}</title>
        </circle>
      ))}
      {/* axes */}
      <text x={14} y={Y(maxL) + 4} fontSize={10.5} fontFamily={M} fill="#b9b4a6">L{maxL}</text>
      <text x={14} y={Y(0) + 4} fontSize={10.5} fontFamily={M} fill="#b9b4a6">L0</text>
      <text x={16} y={(Y(0) + Y(maxL)) / 2} fontSize={10.5} fontFamily={G} fill="#b9b4a6"
        transform={`rotate(-90 16 ${(Y(0) + Y(maxL)) / 2})`} textAnchor="middle">layer →</text>
      {tokens.map((t, i) => (
        <text key={i} x={X(i)} y={H - padB + 18} fontSize={9.5} fontFamily={M} fill="#b9b4a6"
          textAnchor="end" transform={`rotate(-38 ${X(i)} ${H - padB + 18})`}>
          {t.replace('<bos>', '⟨bos⟩')}
        </text>
      ))}
      <text x={padL} y={H - 8} fontSize={11.5} fontFamily={G} fill="#b9b4a6">
        110 pruned feature nodes · 420 strongest of the exact linear edges · node size = graph influence · amber = output logits
      </text>
    </svg>
  );
}

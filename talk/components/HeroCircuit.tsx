/** Decorative animated attribution-graph backdrop for the hero (pure SVG, server-renderable). */
const NODES = [
  { x: 120, y: 300, r: 13, c: '#4fd8ce' },
  { x: 260, y: 180, r: 10, c: '#7fc9de' },
  { x: 300, y: 380, r: 12, c: '#9fb6e8' },
  { x: 470, y: 120, r: 9, c: '#7fc9de' },
  { x: 520, y: 280, r: 15, c: '#b49bf0' },
  { x: 620, y: 420, r: 10, c: '#9fb6e8' },
  { x: 740, y: 200, r: 12, c: '#b49bf0' },
  { x: 860, y: 330, r: 16, c: '#4fd8ce' },
  { x: 990, y: 240, r: 9, c: '#f0a24b' },
  { x: 1080, y: 380, r: 11, c: '#b49bf0' },
];
const EDGES: [number, number][] = [
  [0, 1], [0, 2], [1, 3], [1, 4], [2, 4], [3, 6], [4, 5], [4, 6], [5, 7], [6, 7], [7, 8], [7, 9], [8, 9],
];

export default function HeroCircuit() {
  return (
    <svg
      viewBox="0 0 1200 520"
      preserveAspectRatio="xMidYMid slice"
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: 0.55 }}
      aria-hidden
    >
      <g stroke="#2a3a4d" strokeWidth={1.5}>
        {EDGES.map(([a, b], i) => (
          <line key={`b${i}`} x1={NODES[a].x} y1={NODES[a].y} x2={NODES[b].x} y2={NODES[b].y} />
        ))}
      </g>
      <g strokeWidth={1.8} strokeDasharray="6 14" opacity={0.8}>
        {EDGES.map(([a, b], i) => (
          <line
            key={`f${i}`}
            className="flow-edge"
            x1={NODES[a].x}
            y1={NODES[a].y}
            x2={NODES[b].x}
            y2={NODES[b].y}
            stroke={i % 3 === 0 ? '#4fd8ce' : i % 3 === 1 ? '#b49bf0' : '#f0a24b'}
            style={{ animationDelay: `${(i * 0.35) % 2.4}s` }}
          />
        ))}
      </g>
      <g>
        {NODES.map((n, i) => (
          <circle key={i} cx={n.x} cy={n.y} r={n.r} fill={n.c} opacity={0.9} />
        ))}
      </g>
    </svg>
  );
}

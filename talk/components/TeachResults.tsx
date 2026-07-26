/* Teaching visuals for Part III — protocol and failure diagnosis. Dark-slide styled. */

const G = 'var(--grotesk)';
const M = 'var(--mono)';
const CREAM = '#ede8dc';
const SOFT = '#b9b4a6';
const TEAL = '#4fd8ce';
const AMBER = '#f0a24b';
const VIOLET = '#b49bf0';
const BLUE = '#9fb6e8';
const ROSE = '#e0567a';
const PANEL = '#10151f';
const LINE = '#1e2836';

/* ---------- the oracle race: how the scoring works ---------- */
export function OracleRaceSVG() {
  const lanes: [string, number, string][] = [
    ['oracle (knows every answer)', 1.0, SOFT],
    ['EAP static ranking', 0.919, AMBER],
    ['POMDP agent', 0.82, TEAL],
    ['bandit', 0.744, BLUE],
    ['random', 0.571, ROSE],
  ];
  return (
    <svg viewBox="0 0 880 300" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="All methods run the same ablation-only race against an oracle that knows every KL in advance; scores are percent of the oracle">
      <text x={40} y={34} fontSize={13} fontFamily={G} fill={CREAM}>
        the race: same 20 ablations, same prompts — scored as % of an oracle that already knows every KL
      </text>
      {lanes.map(([name, v, c], i) => {
        const y = 62 + i * 42;
        return (
          <g key={name as string}>
            <text x={244} y={y + 14} fontSize={12} fontFamily={G} fill={c as string} textAnchor="end">{name}</text>
            <rect x={260} y={y} width={520} height={20} rx={6} fill="rgba(237,232,220,.07)" />
            <rect x={260} y={y} width={520 * (v as number)} height={20} rx={6} fill={c as string} opacity={i === 0 ? 0.55 : 0.9} />
            <text x={266 + 520 * (v as number)} y={y + 15} fontSize={11.5} fontFamily={M} fill={CREAM}>
              {((v as number) * 100).toFixed(1)}%
            </text>
          </g>
        );
      })}
      <line x1={780} y1={56} x2={780} y2={270} stroke={SOFT} strokeWidth={1.6} strokeDasharray="5 4" />
      <text x={786} y={286} fontSize={11} fontFamily={G} fill={SOFT} textAnchor="end">100% — cannot be exceeded (by construction)</text>
    </svg>
  );
}

/* ---------- the Llama failure: a state space too coarse to see ---------- */
export function LayerBinsSVG() {
  const nL = 16;
  const massLayer = 3; // early-layer causal mass
  const seg = (x0: number, w: number, label: string, c: string, y: number) => (
    <g>
      <rect x={x0} y={y} width={w} height={30} rx={7} fill={`${c}22`} stroke={c} strokeWidth={1.3} />
      <text x={x0 + w / 2} y={y + 20} fontSize={11.5} fontFamily={G} fill={c} textAnchor="middle">{label}</text>
    </g>
  );
  const LX = (i: number) => 60 + i * 47;
  return (
    <svg viewBox="0 0 880 330" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="Sixteen layers collapsed into three role bins hide where the causal mass is; six bins reveal it">
      {/* layer strip */}
      {Array.from({ length: nL }).map((_, i) => (
        <g key={i}>
          <rect x={LX(i)} y={40} width={40} height={26} rx={6} fill={i === massLayer ? AMBER : PANEL}
            stroke={i === massLayer ? AMBER : LINE} strokeWidth={i === massLayer ? 1.8 : 1} />
          <text x={LX(i) + 20} y={58} fontSize={10.5} fontFamily={M} fill={i === massLayer ? '#080b11' : SOFT} textAnchor="middle">L{i}</text>
        </g>
      ))}
      <text x={LX(massLayer) + 20} y={30} fontSize={11} fontFamily={G} fontWeight={700} fill={AMBER} textAnchor="middle">the causal mass lives here</text>

      {/* 3 bins */}
      <text x={60} y={106} fontSize={12.5} fontFamily={G} fontWeight={600} fill={ROSE}>3-bin belief state (as shipped)</text>
      {seg(60, 247, 'early = L0–L5 … all one state', ROSE, 116)}
      {seg(313, 247, 'middle', SOFT, 116)}
      {seg(566, 246, 'late', SOFT, 116)}
      <text x={60} y={172} fontSize={11.5} fontFamily={G} fill={SOFT}>
        L3’s signal is averaged with five other layers — the agent literally cannot represent “the action is at L3”
      </text>

      {/* 6 bins */}
      <text x={60} y={212} fontSize={12.5} fontFamily={G} fontWeight={600} fill={TEAL}>6-bin belief state (one flag changed)</text>
      {Array.from({ length: 6 }).map((_, i) => seg(60 + i * 126, 120, `bin ${i + 1}`, i === 1 ? AMBER : TEAL, 222))}
      <text x={60} y={278} fontSize={11.5} fontFamily={G} fill={SOFT}>
        now the early-layer mass falls in its own bin — efficiency quadruples, 9.3% → 37.8%, with the objective untouched
      </text>
      <text x={440} y={314} fontSize={13} fontFamily={G} fontWeight={600} fill={CREAM} textAnchor="middle">
        the failure was in the model’s state space, not the objective — and refining the state space is what fixed it
      </text>
    </svg>
  );
}

/* ---------- results-chapter roadmap ---------- */
export function ResultsMapSVG({ stage = 6 }: { stage?: number }) {
  const stops: [string, string][] = [
    ['protocol', 'how scoring works'],
    ['headline', 'agent vs baselines'],
    ['the control', 'RCK ≠ discovery'],
    ['steering', 'a kept null'],
    ['the failure', 'diagnosed in model terms'],
    ['verdict', 'what survives'],
  ];
  return (
    <svg viewBox="0 0 880 120" style={{ width: '100%', height: 'auto' }} role="img" aria-label="Results chapter roadmap">
      <line x1={50} y1={46} x2={840} y2={46} stroke={LINE} strokeWidth={2} />
      {stops.map(([t, s], i) => {
        const x = 70 + i * 150;
        const on = i < stage;
        return (
          <g key={t} opacity={on ? 1 : 0.4}>
            <circle cx={x} cy={46} r={8} fill={on ? TEAL : PANEL} stroke={on ? TEAL : SOFT} strokeWidth={1.5} />
            <text x={x} y={78} fontSize={12} fontFamily={G} fontWeight={700} fill={on ? CREAM : SOFT} textAnchor="middle">{t}</text>
            <text x={x} y={96} fontSize={10} fontFamily={G} fill={SOFT} textAnchor="middle">{s}</text>
          </g>
        );
      })}
    </svg>
  );
}

/* Theme-styled explanatory SVG diagrams for Part I (server-rendered, no deps). */

const G = 'var(--grotesk)';

export function SuperpositionSVG() {
  const dirs = [12, 60, 118, 165, 210].map((deg) => (deg * Math.PI) / 180);
  return (
    <svg viewBox="0 0 720 300" className="chart-svg" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="Dense features align with neuron axes; sparse features pack many directions into the same space">
      {/* Left: 2 features, axis aligned */}
      <g transform="translate(170,160)">
        <line x1={-120} y1={0} x2={130} y2={0} stroke="#8a8471" strokeWidth={1} />
        <line x1={0} y1={110} x2={0} y2={-120} stroke="#8a8471" strokeWidth={1} />
        <line x1={0} y1={0} x2={110} y2={0} stroke="#0e7c86" strokeWidth={4} strokeLinecap="round" />
        <line x1={0} y1={0} x2={0} y2={-100} stroke="#6d28d9" strokeWidth={4} strokeLinecap="round" />
        <circle cx={110} cy={0} r={6} fill="#0e7c86" />
        <circle cx={0} cy={-100} r={6} fill="#6d28d9" />
        <text x={104} y={22} fontSize={12} fontFamily={G} fill="#4a4639">feature A = neuron 1</text>
        <text x={8} y={-104} fontSize={12} fontFamily={G} fill="#4a4639">feature B = neuron 2</text>
        <text x={-118} y={132} fontSize={13} fontFamily={G} fontWeight={600} fill="#1b1a16">
          dense features → axis-aligned
        </text>
      </g>
      {/* Right: 5 features superposed */}
      <g transform="translate(540,160)">
        <line x1={-130} y1={0} x2={140} y2={0} stroke="#8a8471" strokeWidth={1} />
        <line x1={0} y1={110} x2={0} y2={-120} stroke="#8a8471" strokeWidth={1} />
        {dirs.map((a, i) => {
          const colors = ['#0e7c86', '#6d28d9', '#f0a24b', '#e0567a', '#4f8adb'];
          const x = Math.cos(a) * 105;
          const y = -Math.sin(a) * 105;
          return (
            <g key={i}>
              <line x1={0} y1={0} x2={x} y2={y} stroke={colors[i]} strokeWidth={3.5} strokeLinecap="round" opacity={0.9} />
              <circle cx={x} cy={y} r={5.5} fill={colors[i]} />
            </g>
          );
        })}
        <text x={-128} y={132} fontSize={13} fontFamily={G} fontWeight={600} fill="#1b1a16">
          sparse features → superposition
        </text>
        <text x={16} y={26} fontSize={11.5} fontFamily={G} fill="#6f6a58">every neuron reads a mixture</text>
      </g>
    </svg>
  );
}

export function TranscoderSVG() {
  return (
    <svg viewBox="0 0 760 320" className="chart-svg" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="Transcoder replaces the MLP with a wide sparse dictionary imitating its input-output map">
      <defs>
        <marker id="tc-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#0e7c86" />
        </marker>
      </defs>
      {/* residual stream */}
      <line x1={40} y1={160} x2={720} y2={160} stroke="#b6ae99" strokeWidth={10} strokeLinecap="round" opacity={0.55} />
      <text x={40} y={140} fontSize={12} fontFamily={G} fill="#6f6a58">residual stream</text>
      {/* original MLP above */}
      <g>
        <rect x={250} y={40} width={180} height={58} rx={10} fill="#1b1a16" opacity={0.85} />
        <text x={340} y={64} fontSize={13} fontFamily={G} fontWeight={600} fill="#ede8dc" textAnchor="middle">dense MLP</text>
        <text x={340} y={84} fontSize={11} fontFamily={G} fill="#b9b4a6" textAnchor="middle">polysemantic neurons</text>
        <path d="M270 160 C 270 120, 280 98, 296 98" fill="none" stroke="#8a8471" strokeWidth={2} />
        <path d="M384 98 C 400 98, 410 120, 410 160" fill="none" stroke="#8a8471" strokeWidth={2} />
        <text x={472} y={70} fontSize={12} fontFamily={G} fill="#e0567a">✕ replaced</text>
        <line x1={250} y1={40} x2={430} y2={98} stroke="#e0567a" strokeWidth={2} opacity={0.7} />
        <line x1={430} y1={40} x2={250} y2={98} stroke="#e0567a" strokeWidth={2} opacity={0.7} />
      </g>
      {/* transcoder below */}
      <g>
        <rect x={205} y={210} width={270} height={82} rx={12} fill="#0e7c86" opacity={0.12} stroke="#0e7c86" strokeWidth={1.5} />
        <text x={340} y={232} fontSize={13} fontFamily={G} fontWeight={600} fill="#0e7c86" textAnchor="middle">
          transcoder (sparse, 16k wide)
        </text>
        {Array.from({ length: 15 }).map((_, i) => (
          <circle key={i} cx={232 + i * 15.5} cy={258} r={5}
            fill={[2, 6, 11].includes(i) ? '#0e7c86' : 'none'}
            stroke="#0e7c86" strokeWidth={1.4} opacity={[2, 6, 11].includes(i) ? 1 : 0.45} />
        ))}
        <text x={340} y={284} fontSize={11} fontFamily={G} fill="#4a4639" textAnchor="middle">
          few active features imitate the MLP&rsquo;s input→output map
        </text>
        <path d="M270 160 C 270 190, 278 210, 292 210" fill="none" stroke="#0e7c86" strokeWidth={2.5} markerEnd="url(#tc-arrow)" />
        <path d="M388 210 C 402 210, 410 190, 410 160" fill="none" stroke="#0e7c86" strokeWidth={2.5} markerEnd="url(#tc-arrow)" />
      </g>
      {/* virtual weights note */}
      <g>
        <rect x={520} y={210} width={210} height={82} rx={12} fill="none" stroke="#6d28d9" strokeWidth={1.4} strokeDasharray="5 4" />
        <text x={625} y={236} fontSize={12} fontFamily={G} fontWeight={600} fill="#6d28d9" textAnchor="middle">feature→feature edges</text>
        <text x={625} y={256} fontSize={11} fontFamily={G} fill="#4a4639" textAnchor="middle">enc·dec “virtual weight”</text>
        <text x={625} y={274} fontSize={11} fontFamily={G} fill="#4a4639" textAnchor="middle">× upstream activation</text>
      </g>
    </svg>
  );
}

export function PipelineSVG() {
  const box = (x: number, y: number, w: number, title: string, sub: string, color: string) => (
    <g>
      <rect x={x} y={y} width={w} height={64} rx={11} fill={color} opacity={0.13} stroke={color} strokeWidth={1.5} />
      <text x={x + w / 2} y={y + 27} fontSize={13} fontFamily={G} fontWeight={600} fill={color} textAnchor="middle">{title}</text>
      <text x={x + w / 2} y={y + 47} fontSize={10.5} fontFamily={G} fill="currentColor" opacity={0.75} textAnchor="middle">{sub}</text>
    </g>
  );
  const arrow = (x1: number, x2: number, y = 92) => (
    <line x1={x1} y1={y} x2={x2} y2={y} stroke="#4fd8ce" strokeWidth={2.5} markerEnd="url(#pl-arrow)" />
  );
  return (
    <svg viewBox="0 0 1080 230" className="chart-svg" style={{ width: '100%', height: 'auto', color: 'var(--cream)' }} role="img"
      aria-label="Pipeline from prompt to attribution graph to interventions">
      <defs>
        <marker id="pl-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#4fd8ce" />
        </marker>
      </defs>
      {box(10, 60, 150, 'prompt', '“…gave the bag to”', '#ede8dc')}
      {arrow(162, 190)}
      {box(192, 60, 200, 'local replacement model', 'transcoders + frozen attn + error nodes', '#7fc9de')}
      {arrow(394, 422)}
      {box(424, 60, 190, 'attribution graph', 'EAP · exact linear edges · ~18 s', '#9fb6e8')}
      {arrow(616, 644)}
      {box(646, 60, 180, 'pruned candidates', '~2,200 features (80% influence)', '#b49bf0')}
      {arrow(828, 856)}
      {box(858, 60, 210, 'feature_intervention', 'ablate / patch / steer · ~30 ms · KL', '#f0a24b')}
      <text x={540} y={170} fontSize={12.5} fontFamily={G} fill="#b9b4a6" textAnchor="middle">
        cheap, correlational hypothesis space ← — — — — — — — — — → expensive, trusted causal evidence
      </text>
      <text x={540} y={196} fontSize={12.5} fontFamily={G} fontWeight={600} fill="#4fd8ce" textAnchor="middle">
        ACD lives in the gap: which of ~6,600 (feature × action) probes deserves each of your 20 shots?
      </text>
    </svg>
  );
}

export function BudgetSVG() {
  // candidate nodes with importance sizing; a subset probed
  const nodes = [
    [70, 60, 5], [140, 110, 9], [210, 50, 6], [280, 130, 13], [350, 70, 7],
    [420, 140, 5], [490, 60, 10], [560, 120, 6], [630, 55, 8], [700, 135, 16],
    [120, 190, 6], [230, 205, 8], [340, 195, 5], [450, 210, 11], [560, 195, 7], [670, 210, 6],
  ] as const;
  const probed = new Set([3, 9, 6, 13]);
  return (
    <svg viewBox="0 0 780 280" className="chart-svg" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="A large candidate set with only a few probes affordable">
      {nodes.map(([x, y, r], i) => (
        <g key={i}>
          <circle cx={x} cy={y} r={r} fill={probed.has(i) ? '#0e7c86' : '#9fb6e8'} opacity={probed.has(i) ? 1 : 0.55} />
          {probed.has(i) && (
            <circle cx={x} cy={y} r={r + 6} fill="none" stroke="#0e7c86" strokeWidth={1.6} strokeDasharray="3 3" />
          )}
        </g>
      ))}
      <text x={70} y={252} fontSize={12.5} fontFamily={G} fill="#4a4639">
        ~2,200 candidates × 3 intervention types
      </text>
      <text x={520} y={252} fontSize={12.5} fontFamily={G} fontWeight={600} fill="#0e7c86">
        budget: 20 causal probes per prompt
      </text>
    </svg>
  );
}

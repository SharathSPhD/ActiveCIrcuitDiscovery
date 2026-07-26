/* Story visuals for the narrative arc (mess → highway → wall → turn → forager →
   machine → toolkit → contrast → vision). Gold-pathway-on-dark-machinery aesthetic,
   server-renderable SVG, no deps. */

const G = 'var(--grotesk)';
const M = 'var(--mono)';
const CREAM = '#ede8dc';
const SOFT = '#b9b4a6';
const TEAL = '#4fd8ce';
const GOLD = '#d9a441';
const AMBER = '#f0a24b';
const VIOLET = '#b49bf0';
const BLUE = '#9fb6e8';
const ROSE = '#e0567a';
const PANEL = '#10151f';
const LINE = '#1e2836';

/* ---------- beat 2 · THE MESS: ideal vs entangled reality ---------- */
export function EntangledSVG() {
  const rnd = (i: number) => ((i * 9301 + 49297) % 233280) / 233280;
  return (
    <svg viewBox="0 0 900 320" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="The ideal: one concept, one pathway, a clear causal link. The reality: every concept smeared across hundreds of neurons — an unreadable causal map.">
      {/* ideal */}
      <text x={225} y={34} fontSize={14} fontFamily={G} fontWeight={700} fill={TEAL} textAnchor="middle">THE IDEAL</text>
      {[0, 1, 2, 3].map((i) => {
        const y = 66 + i * 56;
        return (
          <g key={i}>
            <rect x={60} y={y} width={54} height={34} rx={7} fill={PANEL} stroke={TEAL} strokeWidth={1.3} />
            <text x={87} y={y + 22} fontSize={10.5} fontFamily={M} fill={CREAM} textAnchor="middle">c{i + 1}</text>
            <line x1={118} y1={y + 17} x2={330} y2={y + 17} stroke={TEAL} strokeWidth={2.4} markerEnd="url(#ent-a)" />
            <rect x={334} y={y} width={54} height={34} rx={7} fill="rgba(79,216,206,.1)" stroke={TEAL} strokeWidth={1.3} />
          </g>
        );
      })}
      <text x={225} y={300} fontSize={11.5} fontFamily={G} fill={SOFT} textAnchor="middle">one concept → one pathway → a clear causal link</text>

      {/* reality */}
      <text x={675} y={34} fontSize={14} fontFamily={G} fontWeight={700} fill={ROSE} textAnchor="middle">THE REALITY: SUPERPOSITION</text>
      {(() => {
        const L = [0, 1, 2, 3].map((i) => ({ x: 510, y: 83 + i * 56 }));
        const R = [0, 1, 2, 3].map((i) => ({ x: 838, y: 83 + i * 56 }));
        const lines: JSX.Element[] = [];
        let k = 0;
        for (const a of L) for (const b of R) {
          const bend = (rnd(k) - 0.5) * 90;
          lines.push(
            <path key={k} d={`M ${a.x + 8} ${a.y} C ${640 + bend} ${a.y + bend}, ${700 - bend} ${b.y - bend}, ${b.x - 8} ${b.y}`}
              fill="none" stroke={k % 5 === 0 ? GOLD : SOFT} strokeWidth={k % 5 === 0 ? 1.8 : 0.9}
              opacity={k % 5 === 0 ? 0.85 : 0.3} />
          );
          k++;
        }
        return (
          <>
            {lines}
            {L.map((p, i) => (
              <g key={`l${i}`}>
                <rect x={p.x - 46} y={p.y - 17} width={54} height={34} rx={7} fill={PANEL} stroke={ROSE} strokeWidth={1.2} />
                <text x={p.x - 19} y={p.y + 5} fontSize={10.5} fontFamily={M} fill={CREAM} textAnchor="middle">c{i + 1}</text>
              </g>
            ))}
            {R.map((p, i) => (
              <rect key={`r${i}`} x={p.x - 8} y={p.y - 17} width={54} height={34} rx={7} fill="rgba(224,86,122,.07)" stroke={ROSE} strokeWidth={1.2} />
            ))}
          </>
        );
      })()}
      <text x={675} y={300} fontSize={11.5} fontFamily={G} fill={SOFT} textAnchor="middle">every concept smeared across hundreds of neurons — the causal map is unreadable</text>
      <defs>
        <marker id="ent-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill={TEAL} />
        </marker>
      </defs>
    </svg>
  );
}

/* ---------- beat 3 · THE HIGHWAY: residual stream with on/off ramps ---------- */
export function HighwaySVG() {
  const ramp = (x: number, up: boolean, kind: 'read' | 'write') => {
    const y0 = up ? 150 : 194;
    const y1 = up ? 78 : 268;
    const c = kind === 'read' ? BLUE : GOLD;
    return (
      <g>
        <path d={`M ${x} ${y0} C ${x + 16} ${up ? y0 - 26 : y0 + 26}, ${x + 24} ${up ? y1 + 22 : y1 - 22}, ${x + 34} ${y1}`}
          fill="none" stroke={c} strokeWidth={4} strokeLinecap="round"
          markerEnd={kind === 'read' ? 'url(#hw-b)' : undefined} opacity={0.9} />
        {kind === 'write' && (
          <path d={`M ${x + 44} ${y1} C ${x + 54} ${up ? y1 + 22 : y1 - 22}, ${x + 62} ${up ? y0 - 26 : y0 + 26}, ${x + 78} ${y0}`}
            fill="none" stroke={GOLD} strokeWidth={4} strokeLinecap="round" markerEnd="url(#hw-g)" opacity={0.9} />
        )}
      </g>
    );
  };
  return (
    <svg viewBox="0 0 940 340" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="The residual stream as a highway: attention heads and MLPs are off-ramps that read from the stream and on-ramps that write back to it.">
      {/* highway */}
      <rect x={20} y={150} width={900} height={44} rx={10} fill="#151b26" stroke={LINE} strokeWidth={1.5} />
      {Array.from({ length: 18 }).map((_, i) => (
        <rect key={i} x={40 + i * 50} y={169} width={24} height={5} rx={2.5} fill={GOLD} opacity={0.55} />
      ))}
      <text x={470} y={177} fontSize={14} fontFamily={G} fontWeight={700} fill={CREAM} textAnchor="middle" style={{ paintOrder: 'stroke' }} stroke="#151b26" strokeWidth={5}>
        THE RESIDUAL STREAM
      </text>
      <path d="M 880 172 L 912 172" stroke={GOLD} strokeWidth={4} markerEnd="url(#hw-g)" />

      {/* attention above */}
      {[80, 300, 560, 760].map((x, i) => (
        <g key={`a${i}`}>
          <rect x={x} y={30} width={104} height={48} rx={10} fill="rgba(159,182,232,.08)" stroke={BLUE} strokeWidth={1.4} />
          <text x={x + 52} y={50} fontSize={11.5} fontFamily={G} fontWeight={600} fill={BLUE} textAnchor="middle">attention</text>
          <text x={x + 52} y={66} fontSize={9.5} fontFamily={G} fill={SOFT} textAnchor="middle">moves info between tokens</text>
          {ramp(x + 10, true, 'read')}
          {ramp(x + 10, true, 'write')}
        </g>
      ))}
      {/* MLPs below */}
      {[190, 430, 660].map((x, i) => (
        <g key={`m${i}`}>
          <rect x={x} y={268} width={104} height={48} rx={10} fill="rgba(180,155,240,.08)" stroke={VIOLET} strokeWidth={1.4} />
          <text x={x + 52} y={288} fontSize={11.5} fontFamily={G} fontWeight={600} fill={VIOLET} textAnchor="middle">MLP</text>
          <text x={x + 52} y={304} fontSize={9.5} fontFamily={G} fill={SOFT} textAnchor="middle">transforms in place</text>
          {ramp(x + 10, false, 'read')}
          {ramp(x + 10, false, 'write')}
        </g>
      ))}
      <text x={30} y={135} fontSize={11} fontFamily={G} fill={BLUE}>off-ramps: read from the stream</text>
      <text x={30} y={222} fontSize={11} fontFamily={G} fill={GOLD}>on-ramps: write back to it</text>
      <defs>
        <marker id="hw-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill={GOLD} />
        </marker>
        <marker id="hw-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill={BLUE} />
        </marker>
      </defs>
    </svg>
  );
}

/* ---------- beat 4 · THE WALL: combinatorial explosion vs flat budget ---------- */
export function ExplosionSVG() {
  const N = 60;
  const pts = Array.from({ length: N }, (_, i) => {
    const x = 90 + (i / (N - 1)) * 740;
    const y = 268 - Math.pow(i / (N - 1), 3.2) * 228;
    return `${i ? 'L' : 'M'}${x},${y}`;
  }).join(' ');
  return (
    <svg viewBox="0 0 900 330" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="Exhaustive causal mapping cost explodes with graph size; the practical budget is a flat line at 20 interventions.">
      <line x1={90} y1={40} x2={90} y2={272} stroke={LINE} strokeWidth={1.6} />
      <line x1={90} y1={272} x2={850} y2={272} stroke={LINE} strokeWidth={1.6} />
      <text x={52} y={160} fontSize={11.5} fontFamily={G} fill={SOFT} textAnchor="middle" transform="rotate(-90 52 160)">
        computational cost (interventions)
      </text>
      <text x={470} y={306} fontSize={11.5} fontFamily={G} fill={SOFT} textAnchor="middle">graph size (nodes, layers, edges)</text>
      {/* explosion curve */}
      <path d={pts} fill="none" stroke={CREAM} strokeWidth={4.5} strokeLinecap="round" />
      <text x={430} y={96} fontSize={13.5} fontFamily={G} fontWeight={700} fill={CREAM}>exhaustive causal mapping — 𝒪(E)</text>
      <text x={430} y={116} fontSize={11} fontFamily={G} fill={SOFT}>one experiment per edge: ACDC-style sweeps</text>
      {/* budget line */}
      <line x1={90} y1={252} x2={850} y2={252} stroke={GOLD} strokeWidth={5} strokeLinecap="round" />
      <text x={620} y={242} fontSize={13.5} fontFamily={G} fontWeight={700} fill={GOLD}>the practical budget: 20 interventions</text>
      {/* gap marker */}
      <line x1={780} y1={60} x2={780} y2={246} stroke={ROSE} strokeWidth={1.4} strokeDasharray="5 4" />
      <text x={772} y={150} fontSize={11.5} fontFamily={G} fill={ROSE} textAnchor="end">the wall: everything</text>
      <text x={772} y={166} fontSize={11.5} fontFamily={G} fill={ROSE} textAnchor="end">above the gold line</text>
      <text x={772} y={182} fontSize={11.5} fontFamily={G} fill={ROSE} textAnchor="end">is unaffordable</text>
    </svg>
  );
}

/* ---------- beat 5 · THE TURN: variational shield (hedged FEP) ---------- */
export function ShieldSVG() {
  const bell = Array.from({ length: 80 }, (_, i) => {
    const t = (i / 79) * 6 - 3;
    const x = 470 + t * 110;
    const y = 250 - Math.exp(-t * t) * 180;
    return `${i ? 'L' : 'M'}${x},${y}`;
  }).join(' ');
  return (
    <svg viewBox="0 0 940 300" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="The free energy principle: systems that persist keep themselves inside a tight distribution of expected states — surprise minimised, uncertainty in the tails.">
      <line x1={80} y1={250} x2={860} y2={250} stroke={LINE} strokeWidth={1.6} />
      <path d={`${bell} L 800 250 L 140 250 Z`} fill="rgba(217,164,65,.14)" stroke="none" />
      <path d={bell} fill="none" stroke={GOLD} strokeWidth={3.5} />
      <text x={470} y={52} fontSize={13} fontFamily={G} fontWeight={700} fill={GOLD} textAnchor="middle">expected states — surprise minimised</text>
      <text x={150} y={272} fontSize={11.5} fontFamily={G} fill={SOFT}>uncertainty</text>
      <text x={790} y={272} fontSize={11.5} fontFamily={G} fill={SOFT} textAnchor="end">uncertainty</text>
      <text x={130} y={110} fontSize={12.5} fontFamily={G} fill={CREAM}>Friston&rsquo;s free energy principle:</text>
      <text x={130} y={130} fontSize={12.5} fontFamily={G} fill={CREAM}>systems that persist act, perceive and learn</text>
      <text x={130} y={150} fontSize={12.5} fontFamily={G} fill={CREAM}>by minimising a bound on surprise</text>
      <text x={130} y={182} fontSize={11} fontFamily={G} fill={SOFT}>(variational — an information bound, not heat;</text>
      <text x={130} y={198} fontSize={11} fontFamily={G} fill={SOFT}>the poster, not the theorem — nothing here rests on physics)</text>
    </svg>
  );
}

/* ---------- beat 7 · THE FORAGER: flashlight cone through dark circuitry ---------- */
export function ForagingSVG() {
  const rnd = (i: number) => ((i * 7919 + 104729) % 15485863) / 15485863;
  const nodes = Array.from({ length: 90 }, (_, i) => ({
    x: 120 + rnd(i) * 780,
    y: 40 + rnd(i + 137) * 250,
    r: 1.6 + rnd(i + 51) * 2.6,
  }));
  const inCone = (n: { x: number; y: number }) => {
    const dx = n.x - 120, dy = n.y - 165;
    const ang = Math.atan2(dy, dx);
    return Math.abs(ang) < 0.42 && dx > 0 && dx < 572;
  };
  return (
    <svg viewBox="0 0 940 330" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="An agent's flashlight cone sweeps a dark field of circuitry, lighting the features its beliefs say are most uncertain and most promising.">
      <rect x={0} y={0} width={940} height={330} rx={14} fill="#0a0e15" />
      {/* cone */}
      <path d="M 120 165 L 690 45 L 690 285 Z" fill="rgba(217,164,65,.13)" />
      <path d="M 120 165 L 690 45" stroke="rgba(217,164,65,.5)" strokeWidth={1.2} />
      <path d="M 120 165 L 690 285" stroke="rgba(217,164,65,.5)" strokeWidth={1.2} />
      {/* dark circuitry */}
      {nodes.map((n, i) => {
        const lit = inCone(n);
        return (
          <g key={i}>
            {i % 3 === 0 && (
              <line x1={n.x} y1={n.y} x2={nodes[(i + 7) % nodes.length].x} y2={nodes[(i + 7) % nodes.length].y}
                stroke={lit ? GOLD : '#1d2634'} strokeWidth={0.7} opacity={lit ? 0.5 : 0.6} />
            )}
            <circle cx={n.x} cy={n.y} r={n.r} fill={lit ? GOLD : '#2a3547'} opacity={lit ? 0.95 : 0.8} />
          </g>
        );
      })}
      {/* agent */}
      <rect x={96} y={149} width={32} height={32} rx={8} fill={GOLD} />
      <rect x={104} y={157} width={16} height={16} rx={4} fill="#0a0e15" />
      <text x={112} y={208} fontSize={11.5} fontFamily={G} fontWeight={700} fill={GOLD} textAnchor="middle">the agent</text>
      <text x={700} y={70} fontSize={12} fontFamily={G} fill={GOLD}>the beam goes where</text>
      <text x={700} y={88} fontSize={12} fontFamily={G} fill={GOLD}>uncertainty is highest</text>
      <text x={700} y={112} fontSize={10.5} fontFamily={G} fill={SOFT}>(the map is dark — the territory</text>
      <text x={700} y={128} fontSize={10.5} fontFamily={G} fill={SOFT}>is frozen and deterministic)</text>
    </svg>
  );
}

/* ---------- beat 8 · THE MACHINE: dual-engine architecture ---------- */
export function DualEngineSVG() {
  const engine = (x: number, w: number, num: string, title: string, sub1: string, sub2: string, c: string, gold = false) => (
    <g>
      <rect x={x} y={64} width={w} height={150} rx={14} fill={gold ? 'rgba(217,164,65,.09)' : PANEL} stroke={c} strokeWidth={gold ? 2 : 1.4} />
      <text x={x + w / 2} y={50} fontSize={11.5} fontFamily={G} fontWeight={700} fill={c} textAnchor="middle">{num}</text>
      <text x={x + w / 2} y={104} fontSize={13.5} fontFamily={G} fontWeight={700} fill={c} textAnchor="middle">{title}</text>
      <text x={x + w / 2} y={132} fontSize={11} fontFamily={G} fill={CREAM} textAnchor="middle">{sub1}</text>
      <text x={x + w / 2} y={150} fontSize={11} fontFamily={G} fill={SOFT} textAnchor="middle">{sub2}</text>
    </g>
  );
  return (
    <svg viewBox="0 0 940 300" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="Three engines: attribution backend generates the graph; the active inference agent selects targets; the intervention engine executes and returns observations that update beliefs.">
      {engine(30, 260, '1 · ATTRIBUTION BACKEND', 'circuit-tracer / EAP', 'builds the attribution graph', '~18 s per prompt · the map', BLUE)}
      {engine(340, 260, '2 · ACTIVE INFERENCE AGENT', 'pymdp POMDP', 'minimises Expected Free Energy', 'chooses (feature, lever) each step', GOLD, true)}
      {engine(650, 260, '3 · INTERVENTION ENGINE', 'feature_intervention', 'executes transcoder-level actions', '~30 ms per probe · the experiment', VIOLET)}
      {/* forward arrows */}
      <line x1={292} y1={120} x2={336} y2={120} stroke={BLUE} strokeWidth={3} markerEnd="url(#de-b)" />
      <text x={314} y={108} fontSize={9.5} fontFamily={G} fill={BLUE} textAnchor="middle">graph</text>
      <line x1={602} y1={120} x2={646} y2={120} stroke={GOLD} strokeWidth={3} markerEnd="url(#de-g)" />
      <text x={624} y={108} fontSize={9.5} fontFamily={G} fill={GOLD} textAnchor="middle">action</text>
      {/* belief update return */}
      <path d="M 780 216 C 780 262, 470 262, 470 218" fill="none" stroke={VIOLET} strokeWidth={3} markerEnd="url(#de-v)" />
      <text x={625} y={278} fontSize={11} fontFamily={G} fill={VIOLET} textAnchor="middle">observation (KL) → belief update, every single step</text>
      <text x={470} y={26} fontSize={12} fontFamily={G} fill={SOFT} textAnchor="middle">
        each engine already existed — the contribution is the loop that connects them
      </text>
      <defs>
        <marker id="de-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill={BLUE} /></marker>
        <marker id="de-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill={GOLD} /></marker>
        <marker id="de-v" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill={VIOLET} /></marker>
      </defs>
    </svg>
  );
}

/* ---------- beat 10 · THE TOOLKIT: necessity / sufficiency / capacity ---------- */
export function ToolkitSVG() {
  const panel = (x: number, title: string, q: string, what: string, c: string, icon: JSX.Element) => (
    <g>
      <rect x={x} y={54} width={272} height={186} rx={13} fill={PANEL} stroke={c} strokeWidth={1.5} strokeDasharray="none" />
      <text x={x + 136} y={84} fontSize={15} fontFamily={G} fontWeight={700} fill={c} textAnchor="middle">{title}</text>
      <text x={x + 136} y={104} fontSize={11.5} fontFamily={G} fontWeight={600} fill={CREAM} textAnchor="middle">{q}</text>
      <g transform={`translate(${x + 136}, 160)`}>{icon}</g>
      <text x={x + 136} y={222} fontSize={10.5} fontFamily={G} fill={SOFT} textAnchor="middle">{what}</text>
    </g>
  );
  return (
    <svg viewBox="0 0 900 300" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="Ablation tests necessity, patching tests sufficiency, steering tests capacity — circuits contain redundancy, so all three are needed.">
      <text x={450} y={30} fontSize={12.5} fontFamily={G} fill={CREAM} textAnchor="middle">
        circuits contain redundancy — backup circuits take over when primaries fail — so one kind of probe cannot map them
      </text>
      {panel(20, 'ablation', 'tests NECESSITY', 'sever the node: output → 0', AMBER, (
        <g>
          <line x1={-50} y1={0} x2={-12} y2={0} stroke={AMBER} strokeWidth={3} />
          <line x1={12} y1={0} x2={50} y2={0} stroke={AMBER} strokeWidth={3} strokeDasharray="4 4" opacity={0.5} />
          <line x1={-12} y1={-14} x2={12} y2={14} stroke={ROSE} strokeWidth={3.4} />
          <circle cx={0} cy={0} r={9} fill="none" stroke={AMBER} strokeWidth={2} />
          <text x={62} y={5} fontSize={13} fontFamily={M} fill={AMBER}>∅</text>
        </g>
      ))}
      {panel(314, 'patching', 'tests SUFFICIENCY', 'swap in a clean value from another run', BLUE, (
        <g>
          <circle cx={-30} cy={-16} r={11} fill="#3d4d63" />
          <circle cx={30} cy={-16} r={11} fill={BLUE} />
          <path d="M -30 -2 C -30 18, 22 18, 28 -4" fill="none" stroke={BLUE} strokeWidth={2.4} markerEnd="url(#tk-b)" strokeDasharray="4 3" />
          <text x={-30} y={-34} fontSize={9.5} fontFamily={G} fill={SOFT} textAnchor="middle">corrupted</text>
          <text x={30} y={-34} fontSize={9.5} fontFamily={G} fill={BLUE} textAnchor="middle">clean</text>
        </g>
      ))}
      {panel(608, 'steering', 'tests CAPACITY', 'amplify the outgoing signal ×N', VIOLET, (
        <g>
          <circle cx={-44} cy={0} r={9} fill={VIOLET} />
          {Array.from({ length: 3 }).map((_, i) => (
            <path key={i} d={`M ${-30 + i * 26} 0 q 9 ${-14 - i * 7} 18 0 q 9 ${14 + i * 7} 18 0`} fill="none" stroke={VIOLET} strokeWidth={2.2} opacity={0.85} />
          ))}
          <text x={52} y={-22} fontSize={11} fontFamily={M} fill={VIOLET}>×N</text>
        </g>
      ))}
      <text x={450} y={282} fontSize={12.5} fontFamily={G} fontWeight={600} fill={CREAM} textAnchor="middle">
        three different causal questions — which is why the agent&rsquo;s action space has three levers
      </text>
      <defs>
        <marker id="tk-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill={BLUE} /></marker>
      </defs>
    </svg>
  );
}

/* ---------- beat 14 · THE CONTRAST: RL loop vs belief-driven information seeking ---------- */
export function RLvsAISVG() {
  return (
    <svg viewBox="0 0 940 310" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="Standard RL circulates action and reward and needs bolted-on exploration heuristics; active inference carries explicit beliefs and values information natively.">
      {/* RL */}
      <text x={235} y={34} fontSize={14} fontFamily={G} fontWeight={700} fill={SOFT} textAnchor="middle">standard RL under uncertainty</text>
      {[
        ['agent', 60, 120], ['action', 235, 60], ['reward', 235, 186],
      ].map(([t, x, y]) => (
        <g key={t as string}>
          <rect x={(x as number)} y={(y as number)} width={120} height={44} rx={10} fill={PANEL} stroke={SOFT} strokeWidth={1.3} />
          <text x={(x as number) + 60} y={(y as number) + 27} fontSize={12.5} fontFamily={G} fontWeight={600} fill={CREAM} textAnchor="middle">{t}</text>
        </g>
      ))}
      <path d="M 150 120 C 170 90, 200 82, 231 82" fill="none" stroke={SOFT} strokeWidth={2.4} markerEnd="url(#rv-s)" />
      <path d="M 355 90 C 400 110, 400 170, 359 200" fill="none" stroke={SOFT} strokeWidth={2.4} markerEnd="url(#rv-s)" />
      <path d="M 231 208 C 190 208, 160 180, 148 160" fill="none" stroke={SOFT} strokeWidth={2.4} markerEnd="url(#rv-s)" />
      <text x={235} y={262} fontSize={11.5} fontFamily={G} fill={ROSE} textAnchor="middle">no native reason to explore — needs bolted-on</text>
      <text x={235} y={279} fontSize={11.5} fontFamily={G} fill={ROSE} textAnchor="middle">bonuses, decay rates, priors (the bandit needed all three)</text>

      {/* AI */}
      <text x={700} y={34} fontSize={14} fontFamily={G} fontWeight={700} fill={GOLD} textAnchor="middle">active inference</text>
      <circle cx={615} cy={140} r={62} fill="none" stroke={GOLD} strokeWidth={3} />
      <circle cx={785} cy={140} r={62} fill="none" stroke={GOLD} strokeWidth={3} />
      <text x={615} y={132} fontSize={11.5} fontFamily={G} fontWeight={600} fill={CREAM} textAnchor="middle">explicit</text>
      <text x={615} y={148} fontSize={11.5} fontFamily={G} fontWeight={600} fill={CREAM} textAnchor="middle">belief updating</text>
      <text x={785} y={132} fontSize={11.5} fontFamily={G} fontWeight={600} fill={CREAM} textAnchor="middle">value of</text>
      <text x={785} y={148} fontSize={11.5} fontFamily={G} fontWeight={600} fill={CREAM} textAnchor="middle">information</text>
      <path d="M 664 96 C 690 78, 710 78, 736 96" fill="none" stroke={GOLD} strokeWidth={2.4} markerEnd="url(#rv-g)" />
      <path d="M 736 184 C 710 202, 690 202, 664 184" fill="none" stroke={GOLD} strokeWidth={2.4} markerEnd="url(#rv-g)" />
      <text x={700} y={262} fontSize={11.5} fontFamily={G} fill={TEAL} textAnchor="middle">curiosity is in the objective itself — the epistemic term</text>
      <text x={700} y={279} fontSize={11.5} fontFamily={G} fill={TEAL} textAnchor="middle">prices the unknown, with nothing bolted on</text>
      <defs>
        <marker id="rv-s" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill={SOFT} /></marker>
        <marker id="rv-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill={GOLD} /></marker>
      </defs>
    </svg>
  );
}

/* ---------- beat 15 · THE VISION: golden inference pathway resolved ---------- */
export function VisionSVG({ resolved = true, compact = false }: { resolved?: boolean; compact?: boolean } = {}) {
  const rnd = (i: number) => ((i * 2654435761) % 4294967296) / 4294967296;
  const cols = [140, 280, 420, 560, 700, 840];
  const nodes: { x: number; y: number }[][] = cols.map((x, c) =>
    Array.from({ length: 7 }, (_, i) => ({ x, y: 60 + i * 36 + (rnd(c * 13 + i) - 0.5) * 10 }))
  );
  const path = [3, 4, 2, 3, 1, 2]; // chosen node index per column
  return (
    <svg viewBox={compact ? '0 40 940 250' : '0 0 940 330'} style={{ width: '100%', height: 'auto' }} role="img"
      aria-label={resolved
        ? 'A dense network with one golden pathway resolved through it — the chosen path of budgeted, informative experiments.'
        : 'A dense, dark network — somewhere inside it is a causal explanation waiting to be chosen.'}>
      <rect x={0} y={0} width={940} height={330} rx={14} fill="#0a0e15" />
      {/* dense background edges */}
      {nodes.slice(0, -1).map((col, c) =>
        col.map((a, i) =>
          nodes[c + 1].map((b, j) =>
            (i + j + c) % 2 === 0 ? (
              <line key={`${c}-${i}-${j}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#1c2533" strokeWidth={0.8} />
            ) : null
          )
        )
      )}
      {/* golden pathway */}
      {resolved && path.slice(0, -1).map((pi, c) => {
        const a = nodes[c][pi], b = nodes[c + 1][path[c + 1]];
        return <line key={c} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={GOLD} strokeWidth={4.5} strokeLinecap="round" />;
      })}
      {nodes.map((col, c) =>
        col.map((n, i) => (
          <circle key={`${c}-${i}`} cx={n.x} cy={n.y} r={resolved && i === path[c] ? 7 : 4}
            fill={resolved && i === path[c] ? GOLD : '#2a3547'} stroke={resolved && i === path[c] ? '#f5d68a' : 'none'} strokeWidth={1.5} />
        ))
      )}
      {resolved && (
        <>
          <path d={`M 840 ${nodes[5][path[5]].y} L 900 ${nodes[5][path[5]].y}`} stroke={GOLD} strokeWidth={4.5} markerEnd="url(#vi-g)" />
          <text x={906} y={nodes[5][path[5]].y - 12} fontSize={11} fontFamily={G} fill={GOLD} textAnchor="end">causal</text>
          <text x={906} y={nodes[5][path[5]].y + 4} fontSize={11} fontFamily={G} fill={GOLD} textAnchor="end">resolution</text>
        </>
      )}
      {!compact && (
        <text x={470} y={312} fontSize={12.5} fontFamily={G} fill={CREAM} textAnchor="middle">
          {resolved
            ? 'the golden path from the opening slide, resolved — chosen one budgeted, informative experiment at a time'
            : 'somewhere in this machinery is a causal explanation — the talk is about how to choose the path to it'}
        </text>
      )}
      <defs>
        <marker id="vi-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill={GOLD} /></marker>
      </defs>
    </svg>
  );
}

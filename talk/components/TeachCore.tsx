/* Teaching visuals for the opening chapter and Part I — designed for dark slides.
   All server-renderable SVG, no deps. */

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

/* ---------- next-token prediction: the one thing an LLM does ---------- */
export function NextTokenSVG({ intervened = false }: { intervened?: boolean }) {
  const probs = intervened
    ? [
        ['Mary', 0.31, TEAL],
        ['John', 0.24, ROSE],
        ['the', 0.11, SOFT],
        ['them', 0.08, SOFT],
        ['her', 0.05, SOFT],
      ]
    : [
        ['Mary', 0.72, TEAL],
        ['John', 0.08, ROSE],
        ['the', 0.05, SOFT],
        ['them', 0.03, SOFT],
        ['her', 0.02, SOFT],
      ];
  const tokens = ['When', 'John', 'and', 'Mary', 'went', 'to', 'the', 'store,', 'John', 'gave', 'the', 'bag', 'to'];
  return (
    <svg viewBox="0 0 900 300" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="A language model reads a sentence and outputs a probability for every possible next word">
      {/* input tokens */}
      {tokens.map((t, i) => {
        const x = 18 + i * 64;
        const hot = t === 'John' || t === 'Mary';
        return (
          <g key={i}>
            <rect x={x} y={40} width={58} height={30} rx={7}
              fill={hot ? (t === 'Mary' ? 'rgba(79,216,206,.16)' : 'rgba(224,86,122,.14)') : PANEL}
              stroke={hot ? (t === 'Mary' ? TEAL : ROSE) : LINE} strokeWidth={1.2} />
            <text x={x + 29} y={60} fontSize={12.5} fontFamily={M} fill={CREAM} textAnchor="middle">{t}</text>
          </g>
        );
      })}
      <text x={18} y={28} fontSize={12} fontFamily={G} fill={SOFT}>input — one token at a time</text>

      {/* model box */}
      <rect x={330} y={100} width={240} height={54} rx={12} fill="rgba(159,182,232,.1)" stroke={BLUE} strokeWidth={1.4} />
      <text x={450} y={123} fontSize={14} fontFamily={G} fontWeight={700} fill={BLUE} textAnchor="middle">language model</text>
      <text x={450} y={142} fontSize={11} fontFamily={G} fill={SOFT} textAnchor="middle">2.6 billion learned numbers</text>
      <line x1={450} y1={72} x2={450} y2={98} stroke={BLUE} strokeWidth={2} />
      <line x1={450} y1={154} x2={450} y2={178} stroke={BLUE} strokeWidth={2} />

      {/* output distribution */}
      <text x={230} y={196} fontSize={12} fontFamily={G} fill={SOFT}>output — a probability for EVERY next word</text>
      {probs.map(([w, p, c], i) => {
        const y = 208 + i * 18;
        return (
          <g key={w as string}>
            <text x={300} y={y + 11} fontSize={12} fontFamily={M} fill={CREAM} textAnchor="end">{w}</text>
            <rect x={312} y={y} width={(p as number) * 420} height={13} rx={4} fill={c as string} opacity={0.92} />
            <text x={318 + (p as number) * 420} y={y + 11} fontSize={11} fontFamily={M} fill={SOFT}>
              {((p as number) * 100).toFixed(0)}%
            </text>
          </g>
        );
      })}
      {intervened && (
        <text x={640} y={250} fontSize={12} fontFamily={G} fill={AMBER}>
          ← the distribution moved.
        </text>
      )}
    </svg>
  );
}

/* ---------- the transformer stack: an assembly line editing meaning ---------- */
export function StackSVG() {
  const layers = 8;
  return (
    <svg viewBox="0 0 860 320" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="Tokens flow through 26 layers; each layer edits a running vector of meaning">
      <text x={20} y={26} fontSize={12.5} fontFamily={G} fill={SOFT}>
        inside: 26 layers, each one edits a running “meaning vector” for every token
      </text>
      {/* residual stream */}
      <rect x={40} y={150} width={780} height={16} rx={8} fill="rgba(237,232,220,.12)" />
      <text x={44} y={144} fontSize={11.5} fontFamily={G} fill={SOFT}>the residual stream — a conveyor belt of vectors</text>
      {Array.from({ length: layers }).map((_, i) => {
        const x = 70 + i * 96;
        return (
          <g key={i}>
            <rect x={x} y={62} width={72} height={38} rx={9} fill="rgba(159,182,232,.1)" stroke={BLUE} strokeWidth={1.1} />
            <text x={x + 36} y={86} fontSize={11} fontFamily={G} fill={BLUE} textAnchor="middle">attention</text>
            <rect x={x} y={206} width={72} height={38} rx={9} fill="rgba(180,155,240,.1)" stroke={VIOLET} strokeWidth={1.1} />
            <text x={x + 36} y={230} fontSize={11} fontFamily={G} fill={VIOLET} textAnchor="middle">MLP</text>
            <line x1={x + 36} y1={100} x2={x + 36} y2={150} stroke={BLUE} strokeWidth={1.6} opacity={0.7} />
            <line x1={x + 36} y1={166} x2={x + 36} y2={206} stroke={VIOLET} strokeWidth={1.6} opacity={0.7} />
            <text x={x + 36} y={272} fontSize={10} fontFamily={M} fill={SOFT} textAnchor="middle">L{i === layers - 1 ? 25 : i}</text>
          </g>
        );
      })}
      <text x={806} y={272} fontSize={12} fontFamily={G} fill={SOFT} textAnchor="end">…</text>
      <text x={20} y={306} fontSize={12.5} fontFamily={G} fill={AMBER}>
        attention moves information between tokens · MLPs transform it — and no number carries a label
      </text>
    </svg>
  );
}

/* ---------- polysemanticity: one neuron, three unrelated triggers ---------- */
export function PolysemanticSVG() {
  const stims: [string, string][] = [
    ['cat faces', '🐱'],
    ['fronts of cars', '🚗'],
    ['cat legs', '🦵'],
  ];
  return (
    <svg viewBox="0 0 760 260" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="A single neuron fires for cat faces, car fronts, and cat legs — three unrelated concepts">
      {stims.map(([label, icon], i) => {
        const y = 46 + i * 76;
        return (
          <g key={label}>
            <rect x={30} y={y - 26} width={190} height={52} rx={11} fill={PANEL} stroke={LINE} />
            <text x={58} y={y + 8} fontSize={22} textAnchor="middle">{icon}</text>
            <text x={80} y={y + 5} fontSize={12.5} fontFamily={G} fill={CREAM}>{label}</text>
            <path d={`M222 ${y} C 320 ${y}, 360 130, 448 130`} fill="none" stroke={AMBER} strokeWidth={2.2} opacity={0.85} />
          </g>
        );
      })}
      <circle cx={480} cy={130} r={34} fill="rgba(240,162,75,.14)" stroke={AMBER} strokeWidth={2} />
      <circle cx={480} cy={130} r={12} fill={AMBER}>
        <animate attributeName="opacity" values="1;.35;1" dur="1.6s" repeatCount="indefinite" />
      </circle>
      <text x={480} y={188} fontSize={12.5} fontFamily={G} fontWeight={600} fill={AMBER} textAnchor="middle">one neuron — fires for all three</text>
      <text x={630} y={110} fontSize={13} fontFamily={G} fontWeight={600} fill={CREAM} textAnchor="middle">“polysemantic”</text>
      <text x={630} y={132} fontSize={11.5} fontFamily={G} fill={SOFT} textAnchor="middle">real unit from InceptionV1</text>
      <text x={630} y={150} fontSize={11.5} fontFamily={G} fill={SOFT} textAnchor="middle">(Olah et al., 2020)</text>
    </svg>
  );
}

/* ---------- dictionary learning: unmixing the signal ---------- */
export function DictionarySAESVG() {
  const feats: [string, number, string][] = [
    ['Golden Gate Bridge', 0.9, TEAL],
    ['DNA sequences', 0.0, SOFT],
    ['base64 strings', 0.0, SOFT],
    ['code bugs', 0.55, VIOLET],
    ['Arabic script', 0.0, SOFT],
    ['sycophantic praise', 0.3, AMBER],
  ];
  return (
    <svg viewBox="0 0 860 280" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="A dictionary rewrites a dense activation vector as a few named, interpretable features">
      {/* dense mixed vector */}
      <text x={80} y={34} fontSize={12} fontFamily={G} fill={SOFT}>dense activations</text>
      {Array.from({ length: 10 }).map((_, i) => (
        <rect key={i} x={60} y={48 + i * 19} width={70} height={14} rx={3}
          fill={`rgba(159,182,232,${0.25 + ((i * 37) % 60) / 90})`} />
      ))}
      <text x={95} y={258} fontSize={11.5} fontFamily={G} fill={SOFT} textAnchor="middle">unreadable mixture</text>

      {/* dictionary box */}
      <rect x={220} y={92} width={200} height={78} rx={12} fill="rgba(79,216,206,.08)" stroke={TEAL} strokeWidth={1.4} />
      <text x={320} y={122} fontSize={13.5} fontFamily={G} fontWeight={700} fill={TEAL} textAnchor="middle">sparse dictionary</text>
      <text x={320} y={142} fontSize={11} fontFamily={G} fill={SOFT} textAnchor="middle">(sparse autoencoder / transcoder)</text>
      <line x1={134} y1={131} x2={216} y2={131} stroke={TEAL} strokeWidth={2.2} />
      <line x1={422} y1={131} x2={470} y2={131} stroke={TEAL} strokeWidth={2.2} />

      {/* named sparse features */}
      <text x={490} y={34} fontSize={12} fontFamily={G} fill={SOFT}>same signal, rewritten as named features</text>
      {feats.map(([name, act, c], i) => {
        const y = 52 + i * 34;
        const on = (act as number) > 0;
        return (
          <g key={name as string}>
            <circle cx={498} cy={y + 7} r={6} fill={on ? (c as string) : 'none'} stroke={c as string} strokeWidth={1.4} opacity={on ? 1 : 0.4} />
            <text x={514} y={y + 12} fontSize={12.5} fontFamily={G} fill={on ? CREAM : SOFT} opacity={on ? 1 : 0.55}>{name}</text>
            {on && <rect x={700} y={y + 1} width={(act as number) * 130} height={12} rx={4} fill={c as string} opacity={0.9} />}
          </g>
        );
      })}
      <text x={498} y={258} fontSize={11.5} fontFamily={G} fill={TEAL}>most features silent — a few, legible, active</text>
    </svg>
  );
}

/* ---------- Golden Gate steering dial ---------- */
export function SteeringDialSVG() {
  return (
    <svg viewBox="0 0 860 270" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="Clamping one feature to ten times its maximum changes what the model says it is">
      {/* dial */}
      <g transform="translate(180,150)">
        <path d="M -95 30 A 100 100 0 1 1 95 30" fill="none" stroke={LINE} strokeWidth={14} strokeLinecap="round" />
        <path d="M -95 30 A 100 100 0 1 1 78 -62" fill="none" stroke={AMBER} strokeWidth={14} strokeLinecap="round" />
        <line x1={0} y1={0} x2={62} y2={-52} stroke={CREAM} strokeWidth={5} strokeLinecap="round" />
        <circle cx={0} cy={0} r={9} fill={CREAM} />
        <text x={-104} y={58} fontSize={11} fontFamily={M} fill={SOFT}>0×</text>
        <text x={92} y={58} fontSize={11} fontFamily={M} fill={AMBER}>10×</text>
        <text x={0} y={92} fontSize={13} fontFamily={G} fontWeight={600} fill={CREAM} textAnchor="middle">“Golden Gate Bridge” feature</text>
        <text x={0} y={112} fontSize={11.5} fontFamily={G} fill={SOFT} textAnchor="middle">clamped far above its natural range</text>
      </g>
      {/* chat */}
      <g>
        <rect x={400} y={40} width={420} height={54} rx={12} fill={PANEL} stroke={LINE} />
        <text x={416} y={62} fontSize={11} fontFamily={G} fill={SOFT}>user</text>
        <text x={416} y={82} fontSize={13} fontFamily={G} fill={CREAM}>What is your physical form?</text>
        <rect x={400} y={112} width={420} height={92} rx={12} fill="rgba(240,162,75,.08)" stroke={AMBER} />
        <text x={416} y={134} fontSize={11} fontFamily={G} fill={AMBER}>Claude (one feature clamped)</text>
        <text x={416} y={156} fontSize={13} fontFamily={G} fill={CREAM}>“I am the Golden Gate Bridge…</text>
        <text x={416} y={176} fontSize={13} fontFamily={G} fill={CREAM}>my physical form is the iconic bridge itself.”</text>
        <text x={400} y={232} fontSize={12} fontFamily={G} fill={SOFT}>live on claude.ai for 24 hours, May 2024 — steering made public</text>
      </g>
    </svg>
  );
}

/* ---------- the IOI circuit: sentence with functional arcs ---------- */
export function IOICircuitSVG() {
  const words = ['When', 'John', 'and', 'Mary', 'went', 'to', 'the', 'store,', 'John', 'gave', 'the', 'bag', 'to', '__'];
  const xw = 62;
  const X = (i: number) => 24 + i * xw;
  return (
    <svg viewBox="0 0 920 330" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="The indirect-object-identification circuit: duplicate detection, suppression, and name moving">
      {words.map((w, i) => {
        const hot = i === 1 || i === 8 ? ROSE : i === 3 ? TEAL : i === 13 ? AMBER : null;
        return (
          <g key={i}>
            <rect x={X(i)} y={150} width={xw - 8} height={30} rx={7}
              fill={hot ? `${hot === ROSE ? 'rgba(224,86,122,.14)' : hot === TEAL ? 'rgba(79,216,206,.14)' : 'rgba(240,162,75,.16)'}` : PANEL}
              stroke={hot ?? LINE} strokeWidth={hot ? 1.5 : 1} />
            <text x={X(i) + (xw - 8) / 2} y={170} fontSize={12} fontFamily={M} fill={CREAM} textAnchor="middle">{w}</text>
          </g>
        );
      })}
      {/* duplicate token head */}
      <path d={`M ${X(1) + 27} 148 C ${X(1) + 27} 70, ${X(8) + 27} 70, ${X(8) + 27} 148`} fill="none" stroke={ROSE} strokeWidth={2.4} strokeDasharray="6 4" />
      <text x={(X(1) + X(8)) / 2 + 27} y={62} fontSize={12} fontFamily={G} fontWeight={600} fill={ROSE} textAnchor="middle">
        1 · duplicate-token heads: “John appears twice”
      </text>
      {/* s-inhibition */}
      <path d={`M ${X(8) + 27} 182 C ${X(8) + 27} 240, ${X(13) + 20} 240, ${X(13) + 20} 184`} fill="none" stroke={ROSE} strokeWidth={2.4} />
      <text x={(X(8) + X(13)) / 2 + 24} y={262} fontSize={12} fontFamily={G} fontWeight={600} fill={ROSE} textAnchor="middle">
        2 · suppression (“S-inhibition”) heads: “…so suppress John here”
      </text>
      <text x={X(13) + 20} y={218} fontSize={15} fill={ROSE} textAnchor="middle">⊘</text>
      {/* name mover */}
      <path d={`M ${X(3) + 27} 148 C ${X(3) + 27} 96, ${X(13) + 6} 100, ${X(13) + 18} 146`} fill="none" stroke={TEAL} strokeWidth={2.8} />
      <text x={(X(3) + X(12)) / 2} y={104} fontSize={12} fontFamily={G} fontWeight={600} fill={TEAL} textAnchor="middle">
        3 · name-mover heads: copy “Mary” into the answer
      </text>
      <text x={24} y={306} fontSize={12} fontFamily={G} fill={SOFT}>
        26 attention heads in 7 functional classes, in GPT-2 small · established by causal patching, one path at a time (Wang et al. 2022)
      </text>
    </svg>
  );
}

/* ---------- three intervention levers ---------- */
export function LeversSVG() {
  const levers: [string, string, string, string][] = [
    ['ablate', 'set the feature to 0', '“what breaks without it?”', AMBER],
    ['patch', 'copy another prompt’s value', '“what does it carry?”', BLUE],
    ['steer', 'multiply it beyond range', '“what does it amplify?”', VIOLET],
  ];
  return (
    <svg viewBox="0 0 860 240" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="Three intervention types: ablate sets a feature to zero, patch swaps its value, steer multiplies it">
      {levers.map(([name, what, why, c], i) => {
        const x = 40 + i * 280;
        return (
          <g key={name}>
            <rect x={x} y={30} width={240} height={150} rx={14} fill={PANEL} stroke={c} strokeWidth={1.5} />
            {/* lever graphic */}
            <line x1={x + 60} y1={120} x2={x + 60} y2={64} stroke={c} strokeWidth={5} strokeLinecap="round"
              transform={`rotate(${i === 0 ? -34 : i === 1 ? 0 : 34} ${x + 60} 120)`} />
            <circle cx={x + 60} cy={120} r={10} fill={c} />
            <text x={x + 116} y={78} fontSize={16} fontFamily={G} fontWeight={700} fill={c}>{name}</text>
            <text x={x + 116} y={102} fontSize={11.5} fontFamily={G} fill={CREAM}>{what}</text>
            <text x={x + 116} y={124} fontSize={11.5} fontFamily={G} fill={SOFT}>{why}</text>
            <text x={x + 20} y={164} fontSize={11} fontFamily={M} fill={SOFT}>feature_intervention(...)  ~30 ms</text>
          </g>
        );
      })}
      <text x={40} y={218} fontSize={12.5} fontFamily={G} fill={SOFT}>
        every probe returns one number: how far the output distribution moved — KL(clean ‖ intervened)
      </text>
    </svg>
  );
}

/* ---------- KL: the ruler for every experiment ---------- */
export function KLRulerSVG() {
  const clean: [string, number][] = [['Mary', 0.72], ['John', 0.08], ['the', 0.05], ['them', 0.03]];
  const abl: [string, number][] = [['Mary', 0.31], ['John', 0.24], ['the', 0.11], ['them', 0.08]];
  const row = (data: [string, number][], x0: number, c: string) =>
    data.map(([w, p], i) => (
      <g key={w}>
        <text x={x0 - 8} y={64 + i * 26 + 11} fontSize={12} fontFamily={M} fill={CREAM} textAnchor="end">{w}</text>
        <rect x={x0} y={64 + i * 26} width={p * 260} height={14} rx={4} fill={c} opacity={0.9} />
        <text x={x0 + p * 260 + 6} y={64 + i * 26 + 11} fontSize={10.5} fontFamily={M} fill={SOFT}>{(p * 100).toFixed(0)}%</text>
      </g>
    ));
  return (
    <svg viewBox="0 0 900 240" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="KL divergence measures how far the output distribution moved after an intervention">
      <text x={120} y={40} fontSize={12.5} fontFamily={G} fill={TEAL}>clean run</text>
      {row(clean, 120, TEAL)}
      <text x={560} y={40} fontSize={12.5} fontFamily={G} fill={AMBER}>after ablating one feature</text>
      {row(abl, 560, AMBER)}
      <path d="M 420 110 C 460 110, 470 110, 505 110" stroke={CREAM} strokeWidth={2} fill="none" markerEnd="url(#klr-a)" />
      <defs>
        <marker id="klr-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill={CREAM} />
        </marker>
      </defs>
      <text x={450} y={210} fontSize={13.5} fontFamily={G} fontWeight={600} fill={CREAM} textAnchor="middle">
        KL(clean ‖ intervened) — one number for “how much did that feature matter?”
      </text>
      <text x={450} y={230} fontSize={11.5} fontFamily={G} fill={SOFT} textAnchor="middle">
        big KL → the feature was load-bearing · tiny KL → the model barely noticed
      </text>
    </svg>
  );
}

/* ---------- cost asymmetry ---------- */
export function CostScalesSVG() {
  return (
    <svg viewBox="0 0 880 250" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="One attribution pass is cheap but correlational; causal probes are trusted but budgeted">
      <g>
        <rect x={40} y={40} width={370} height={160} rx={14} fill="rgba(159,182,232,.07)" stroke={BLUE} strokeWidth={1.4} />
        <text x={225} y={72} fontSize={15} fontFamily={G} fontWeight={700} fill={BLUE} textAnchor="middle">attribution (the map)</text>
        <text x={225} y={100} fontSize={26} fontFamily={M} fontWeight={700} fill={CREAM} textAnchor="middle">18 s → scores for 1000s</text>
        <text x={225} y={128} fontSize={12} fontFamily={G} fill={SOFT} textAnchor="middle">one pass ranks every feature at once</text>
        <text x={225} y={152} fontSize={12} fontFamily={G} fill={AMBER} textAnchor="middle">but: correlational, approximate, known-fragile</text>
      </g>
      <g>
        <rect x={470} y={40} width={370} height={160} rx={14} fill="rgba(240,162,75,.07)" stroke={AMBER} strokeWidth={1.4} />
        <text x={655} y={72} fontSize={15} fontFamily={G} fontWeight={700} fill={AMBER} textAnchor="middle">intervention (the ground truth)</text>
        <text x={655} y={100} fontSize={26} fontFamily={M} fontWeight={700} fill={CREAM} textAnchor="middle">30 ms → 1 answer</text>
        <text x={655} y={128} fontSize={12} fontFamily={G} fill={SOFT} textAnchor="middle">a real causal experiment on the real network</text>
        <text x={655} y={152} fontSize={12} fontFamily={G} fill={TEAL} textAnchor="middle">trusted — but you can only afford ~20 per prompt</text>
      </g>
      <text x={440} y={232} fontSize={13} fontFamily={G} fontWeight={600} fill={CREAM} textAnchor="middle">
        cheap hypotheses on one side, expensive evidence on the other — something must decide how to spend
      </text>
    </svg>
  );
}

/* ---------- static list vs adaptive loop ---------- */
export function StaticVsAdaptiveSVG() {
  return (
    <svg viewBox="0 0 880 280" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="Static methods walk a fixed ranking; an adaptive agent updates beliefs and re-plans after every result">
      {/* static */}
      <text x={210} y={36} fontSize={14} fontFamily={G} fontWeight={700} fill={SOFT} textAnchor="middle">the standard recipe: rank once, walk the list</text>
      {[0, 1, 2, 3, 4].map((i) => (
        <g key={i}>
          <rect x={120} y={54 + i * 36} width={180} height={26} rx={7} fill={PANEL} stroke={LINE} />
          <text x={134} y={72 + i * 36} fontSize={11.5} fontFamily={M} fill={CREAM}>#{i + 1} feature {['4717', '882', '15600', '9134', '221'][i]}</text>
          {i < 4 && <line x1={210} y1={80 + i * 36} x2={210} y2={90 + i * 36} stroke={SOFT} strokeWidth={1.6} />}
        </g>
      ))}
      <text x={210} y={262} fontSize={12} fontFamily={G} fill={ROSE} textAnchor="middle">nothing it observes ever changes the plan</text>

      {/* adaptive */}
      <text x={650} y={36} fontSize={14} fontFamily={G} fontWeight={700} fill={TEAL} textAnchor="middle">an adaptive agent: believe → probe → update → re-plan</text>
      <g transform="translate(650,150)">
        {[
          ['beliefs', -90, TEAL],
          ['choose probe', 0, AMBER],
          ['observe KL', 90, BLUE],
          ['update', 180, VIOLET],
        ].map(([label, ang, c], i) => {
          const a = ((ang as number) * Math.PI) / 180;
          const x = Math.cos(a) * 105;
          const y = Math.sin(a) * 78;
          return (
            <g key={i}>
              <rect x={x - 56} y={y - 16} width={112} height={32} rx={9} fill={PANEL} stroke={c as string} strokeWidth={1.4} />
              <text x={x} y={y + 5} fontSize={12} fontFamily={G} fontWeight={600} fill={c as string} textAnchor="middle">{label}</text>
            </g>
          );
        })}
        <path d="M 62 -68 A 100 74 0 0 1 62 66" fill="none" stroke={SOFT} strokeWidth={1.8} markerEnd="url(#sva-a)" />
        <path d="M -62 68 A 100 74 0 0 1 -62 -66" fill="none" stroke={SOFT} strokeWidth={1.8} markerEnd="url(#sva-a)" />
        <defs>
          <marker id="sva-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill={SOFT} />
          </marker>
        </defs>
      </g>
      <text x={650} y={262} fontSize={12} fontFamily={G} fill={TEAL} textAnchor="middle">every observation is allowed to change what happens next</text>
    </svg>
  );
}

/* ---------- chapter ladder / roadmap for Part I ---------- */
export function LadderSVG({ stage = 5 }: { stage?: number }) {
  const rungs = [
    ['what a model does', 'next-token probabilities'],
    ['features', 'the real units of meaning'],
    ['circuits', 'features wired into algorithms'],
    ['attribution graphs', 'a per-prompt causal map'],
    ['the open problem', 'which experiment to run next?'],
  ];
  return (
    <svg viewBox="0 0 860 250" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="The chapter climbs from what a model does, to features, circuits, graphs, and the experiment-selection problem">
      {rungs.map(([t, s], i) => {
        const x = 30 + i * 166;
        const y = 170 - i * 32;
        const on = i < stage;
        return (
          <g key={t} opacity={on ? 1 : 0.35}>
            <rect x={x} y={y} width={150} height={54} rx={11}
              fill={on ? 'rgba(79,216,206,.09)' : PANEL} stroke={on ? TEAL : LINE} strokeWidth={1.4} />
            <text x={x + 75} y={y + 22} fontSize={12.5} fontFamily={G} fontWeight={700} fill={on ? TEAL : SOFT} textAnchor="middle">{t}</text>
            <text x={x + 75} y={y + 40} fontSize={10.5} fontFamily={G} fill={SOFT} textAnchor="middle">{s}</text>
            {i < rungs.length - 1 && (
              <line x1={x + 152} y1={y + 12} x2={x + 176} y2={y - 6} stroke={on && i < stage - 1 ? TEAL : LINE} strokeWidth={2} />
            )}
          </g>
        );
      })}
    </svg>
  );
}

/* ---------- two fields, one question (opening hook) ---------- */
export function TwoFieldsSVG() {
  return (
    <svg viewBox="0 0 900 300" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="Neuroscience asks how agents choose informative actions; interpretability asks which experiment to run next — the same question">
      {/* left: active inference */}
      <rect x={30} y={40} width={360} height={190} rx={16} fill="rgba(79,216,206,.06)" stroke={TEAL} strokeWidth={1.4} />
      <text x={210} y={70} fontSize={14} fontFamily={G} fontWeight={700} fill={TEAL} textAnchor="middle">active inference</text>
      {/* eye + saccades */}
      <g transform="translate(130,150)">
        <ellipse cx={0} cy={0} rx={46} ry={27} fill="none" stroke={CREAM} strokeWidth={2} />
        <circle cx={0} cy={0} r={12} fill={TEAL} />
        {[
          [70, -40], [95, 10], [60, 48],
        ].map(([dx, dy], i) => (
          <line key={i} x1={12} y1={0} x2={dx} y2={dy} stroke={AMBER} strokeWidth={1.8} strokeDasharray="4 3" markerEnd="url(#tf-a)" />
        ))}
      </g>
      <text x={210} y={202} fontSize={12} fontFamily={G} fill={CREAM} textAnchor="middle">where should the eye look next,</text>
      <text x={210} y={220} fontSize={12} fontFamily={G} fill={SOFT} textAnchor="middle">to learn the most about the scene?</text>

      {/* right: interpretability */}
      <rect x={510} y={40} width={360} height={190} rx={16} fill="rgba(240,162,75,.06)" stroke={AMBER} strokeWidth={1.4} />
      <text x={690} y={70} fontSize={14} fontFamily={G} fontWeight={700} fill={AMBER} textAnchor="middle">mechanistic interpretability</text>
      <g transform="translate(620,150)">
        <rect x={-36} y={-36} width={112} height={72} rx={10} fill="none" stroke={CREAM} strokeWidth={2} />
        {Array.from({ length: 9 }).map((_, i) => (
          <circle key={i} cx={-18 + (i % 3) * 38} cy={-18 + Math.floor(i / 3) * 26} r={4.5}
            fill={[1, 5].includes(i) ? AMBER : 'rgba(237,232,220,.3)'} />
        ))}
        {[[130, -34], [150, 12], [126, 50]].map(([dx, dy], i) => (
          <line key={i} x1={80} y1={0} x2={dx} y2={dy} stroke={TEAL} strokeWidth={1.8} strokeDasharray="4 3" markerEnd="url(#tf-b)" />
        ))}
      </g>
      <text x={690} y={202} fontSize={12} fontFamily={G} fill={CREAM} textAnchor="middle">which feature should be probed next,</text>
      <text x={690} y={220} fontSize={12} fontFamily={G} fill={SOFT} textAnchor="middle">to learn the most about the circuit?</text>

      <defs>
        <marker id="tf-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill={AMBER} />
        </marker>
        <marker id="tf-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill={TEAL} />
        </marker>
      </defs>
      <text x={450} y={272} fontSize={14} fontFamily={G} fontWeight={600} fill={CREAM} textAnchor="middle">
        the same question — asked about two very different systems
      </text>
    </svg>
  );
}

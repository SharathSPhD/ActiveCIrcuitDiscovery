/* Teaching visuals for Part II — building the active-inference intuition. Dark-slide styled. */

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

/* ---------- 20 Questions: expected information gain in everyday form ---------- */
export function TwentyQSVG() {
  return (
    <svg viewBox="0 0 900 320" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="In twenty questions, a good question halves the possibilities; a bad one almost never pays off">
      <text x={30} y={34} fontSize={13.5} fontFamily={G} fill={CREAM}>
        Guess the animal in 20 questions. 1,000 possibilities. Which question do you ask first?
      </text>
      {/* good question */}
      <g>
        <rect x={40} y={64} width={380} height={54} rx={12} fill="rgba(79,216,206,.08)" stroke={TEAL} strokeWidth={1.4} />
        <text x={230} y={87} fontSize={14} fontFamily={G} fontWeight={700} fill={TEAL} textAnchor="middle">“Does it live in water?”</text>
        <text x={230} y={106} fontSize={11} fontFamily={G} fill={SOFT} textAnchor="middle">either answer removes ~half the candidates</text>
        {/* halving bar */}
        <rect x={80} y={140} width={300} height={20} rx={6} fill="rgba(237,232,220,.12)" />
        <rect x={80} y={140} width={150} height={20} rx={6} fill={TEAL} opacity={0.7} />
        <line x1={230} y1={132} x2={230} y2={168} stroke={CREAM} strokeWidth={2} />
        <text x={155} y={180} fontSize={11} fontFamily={M} fill={TEAL} textAnchor="middle">yes → 500 left</text>
        <text x={305} y={180} fontSize={11} fontFamily={M} fill={SOFT} textAnchor="middle">no → 500 left</text>
        <text x={230} y={210} fontSize={12.5} fontFamily={G} fontWeight={600} fill={TEAL} textAnchor="middle">expected information gain: 1.0 bit</text>
        <text x={230} y={230} fontSize={11} fontFamily={G} fill={SOFT} textAnchor="middle">guaranteed to learn a lot, whatever happens</text>
      </g>
      {/* bad question */}
      <g>
        <rect x={480} y={64} width={380} height={54} rx={12} fill="rgba(224,86,122,.07)" stroke={ROSE} strokeWidth={1.4} />
        <text x={670} y={87} fontSize={14} fontFamily={G} fontWeight={700} fill={ROSE} textAnchor="middle">“Is it a blue whale?”</text>
        <text x={670} y={106} fontSize={11} fontFamily={G} fill={SOFT} textAnchor="middle">almost certainly “no” — and “no” teaches almost nothing</text>
        <rect x={520} y={140} width={300} height={20} rx={6} fill="rgba(237,232,220,.12)" />
        <rect x={520} y={140} width={0.4} height={20} fill={ROSE} />
        <line x1={520.7} y1={132} x2={520.7} y2={168} stroke={ROSE} strokeWidth={2} />
        <text x={545} y={180} fontSize={11} fontFamily={M} fill={ROSE}>yes → done! (p ≈ 0.001)</text>
        <text x={730} y={180} fontSize={11} fontFamily={M} fill={SOFT} textAnchor="middle">no → 999 left</text>
        <text x={670} y={210} fontSize={12.5} fontFamily={G} fontWeight={600} fill={ROSE} textAnchor="middle">expected information gain: ≈ 0.01 bits</text>
        <text x={670} y={230} fontSize={11} fontFamily={G} fill={SOFT} textAnchor="middle">a jackpot you will almost never hit</text>
      </g>
      <text x={450} y={286} fontSize={13.5} fontFamily={G} fontWeight={600} fill={CREAM} textAnchor="middle">
        good experimenters do not ask the question they hope is true —
      </text>
      <text x={450} y={306} fontSize={13.5} fontFamily={G} fontWeight={600} fill={TEAL} textAnchor="middle">
        they ask the question whose answer they cannot predict
      </text>
    </svg>
  );
}

/* ---------- the physician: beliefs, tests, updates — with a goal ---------- */
export function PhysicianSVG() {
  const dx: [string, number][] = [
    ['pneumonia', 0.45],
    ['bronchitis', 0.3],
    ['embolism', 0.15],
    ['other', 0.1],
  ];
  return (
    <svg viewBox="0 0 900 330" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="A physician holds beliefs over diagnoses, chooses tests by informativeness and stakes, and updates">
      {/* beliefs */}
      <rect x={30} y={40} width={260} height={220} rx={14} fill={PANEL} stroke={TEAL} strokeWidth={1.3} />
      <text x={160} y={68} fontSize={13} fontFamily={G} fontWeight={700} fill={TEAL} textAnchor="middle">beliefs (differential)</text>
      {dx.map(([d, p], i) => (
        <g key={d}>
          <text x={54} y={100 + i * 40} fontSize={12} fontFamily={G} fill={CREAM}>{d}</text>
          <rect x={54} y={108 + i * 40} width={190} height={12} rx={4} fill="rgba(237,232,220,.12)" />
          <rect x={54} y={108 + i * 40} width={190 * p} height={12} rx={4} fill={TEAL} opacity={0.85} />
        </g>
      ))}
      {/* tests */}
      <rect x={330} y={40} width={260} height={220} rx={14} fill={PANEL} stroke={AMBER} strokeWidth={1.3} />
      <text x={460} y={68} fontSize={13} fontFamily={G} fontWeight={700} fill={AMBER} textAnchor="middle">available tests</text>
      {[
        ['chest X-ray', 'cheap · moderately informative'],
        ['CT + contrast', 'costly · very informative'],
        ['blood panel', 'cheap · weakly informative'],
      ].map(([t, s], i) => (
        <g key={t}>
          <rect x={352} y={86 + i * 54} width={216} height={40} rx={9} fill="rgba(240,162,75,.07)" stroke={LINE} />
          <text x={366} y={103 + i * 54} fontSize={12} fontFamily={G} fontWeight={600} fill={CREAM}>{t}</text>
          <text x={366} y={119 + i * 54} fontSize={10.5} fontFamily={G} fill={SOFT}>{s}</text>
        </g>
      ))}
      <text x={460} y={252} fontSize={11} fontFamily={G} fill={SOFT} textAnchor="middle">the budget is real: time, cost, risk</text>
      {/* update */}
      <rect x={630} y={40} width={240} height={220} rx={14} fill={PANEL} stroke={VIOLET} strokeWidth={1.3} />
      <text x={750} y={68} fontSize={13} fontFamily={G} fontWeight={700} fill={VIOLET} textAnchor="middle">result → update</text>
      <text x={750} y={104} fontSize={12} fontFamily={G} fill={CREAM} textAnchor="middle">X-ray: infiltrate present</text>
      {dx.map(([d], i) => {
        const p2 = [0.78, 0.12, 0.06, 0.04][i];
        return (
          <g key={d}>
            <text x={654} y={136 + i * 30} fontSize={11} fontFamily={G} fill={SOFT}>{d}</text>
            <rect x={740} y={128 + i * 30} width={110 * p2} height={10} rx={3} fill={VIOLET} opacity={0.85} />
          </g>
        );
      })}
      <text x={750} y={252} fontSize={11} fontFamily={G} fill={SOFT} textAnchor="middle">stop when treatment is clear</text>
      {/* arrows */}
      <line x1={292} y1={150} x2={326} y2={150} stroke={SOFT} strokeWidth={2} markerEnd="url(#ph-a)" />
      <line x1={592} y1={150} x2={626} y2={150} stroke={SOFT} strokeWidth={2} markerEnd="url(#ph-a)" />
      <path d="M 750 264 C 750 306, 160 306, 160 264" fill="none" stroke={SOFT} strokeWidth={1.8} strokeDasharray="5 4" markerEnd="url(#ph-a)" />
      <defs>
        <marker id="ph-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill={SOFT} />
        </marker>
      </defs>
      <text x={450} y={318} fontSize={12.5} fontFamily={G} fontWeight={600} fill={CREAM} textAnchor="middle">
        two pulls in every choice: learn the most (information) — and find what matters for the patient (preference)
      </text>
    </svg>
  );
}

/* ---------- the generic perception–action loop ---------- */
export function GenericLoopSVG() {
  return (
    <svg viewBox="0 0 860 260" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="The generic perception-action loop: an agent's model issues actions into a world and receives observations back">
      <rect x={80} y={60} width={280} height={140} rx={16} fill="rgba(79,216,206,.06)" stroke={TEAL} strokeWidth={1.5} />
      <text x={220} y={98} fontSize={15} fontFamily={G} fontWeight={700} fill={TEAL} textAnchor="middle">agent</text>
      <text x={220} y={126} fontSize={12} fontFamily={G} fill={CREAM} textAnchor="middle">beliefs about hidden states</text>
      <text x={220} y={146} fontSize={12} fontFamily={G} fill={CREAM} textAnchor="middle">a generative model of the world</text>
      <text x={220} y={166} fontSize={12} fontFamily={G} fill={SOFT} textAnchor="middle">acts to reduce expected surprise</text>

      <rect x={500} y={60} width={280} height={140} rx={16} fill="rgba(240,162,75,.06)" stroke={AMBER} strokeWidth={1.5} />
      <text x={640} y={98} fontSize={15} fontFamily={G} fontWeight={700} fill={AMBER} textAnchor="middle">world</text>
      <text x={640} y={126} fontSize={12} fontFamily={G} fill={CREAM} textAnchor="middle">hidden states the agent</text>
      <text x={640} y={146} fontSize={12} fontFamily={G} fill={CREAM} textAnchor="middle">can never see directly</text>
      <text x={640} y={166} fontSize={12} fontFamily={G} fill={SOFT} textAnchor="middle">answers only through observations</text>

      <path d="M 362 100 C 420 100, 440 100, 496 100" fill="none" stroke={TEAL} strokeWidth={2.6} markerEnd="url(#gl-a)" />
      <text x={430} y={88} fontSize={12.5} fontFamily={G} fontWeight={600} fill={TEAL} textAnchor="middle">action</text>
      <path d="M 496 165 C 440 165, 420 165, 362 165" fill="none" stroke={AMBER} strokeWidth={2.6} markerEnd="url(#gl-b)" />
      <text x={430} y={188} fontSize={12.5} fontFamily={G} fontWeight={600} fill={AMBER} textAnchor="middle">observation</text>
      <defs>
        <marker id="gl-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill={TEAL} />
        </marker>
        <marker id="gl-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill={AMBER} />
        </marker>
      </defs>
      <text x={430} y={240} fontSize={13} fontFamily={G} fill={SOFT} textAnchor="middle">
        the claim of this chapter: circuit discovery already has exactly this shape
      </text>
    </svg>
  );
}

/* ---------- the 4×3×3 state lattice ---------- */
export function StateCubeSVG() {
  return (
    <svg viewBox="0 0 880 260" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="Per candidate feature: importance times layer role times causal influence equals 36 joint states">
      {/* importance */}
      <g>
        <text x={130} y={40} fontSize={13} fontFamily={G} fontWeight={700} fill={TEAL} textAnchor="middle">importance s₀</text>
        {['negligible', 'low', 'moderate', 'high'].map((l, i) => (
          <g key={l}>
            <rect x={60} y={56 + i * 40} width={140} height={30} rx={8} fill="rgba(79,216,206,.08)" stroke={TEAL} strokeWidth={1.1} />
            <text x={130} y={76 + i * 40} fontSize={12} fontFamily={G} fill={CREAM} textAnchor="middle">{l}</text>
          </g>
        ))}
        <text x={130} y={236} fontSize={12} fontFamily={M} fill={TEAL} textAnchor="middle">4</text>
      </g>
      <text x={242} y={140} fontSize={22} fontFamily={G} fill={SOFT} textAnchor="middle">×</text>
      {/* layer role */}
      <g>
        <text x={352} y={40} fontSize={13} fontFamily={G} fontWeight={700} fill={BLUE} textAnchor="middle">layer role s₁</text>
        {['early', 'middle', 'late'].map((l, i) => (
          <g key={l}>
            <rect x={282} y={68 + i * 44} width={140} height={32} rx={8} fill="rgba(159,182,232,.08)" stroke={BLUE} strokeWidth={1.1} />
            <text x={352} y={89 + i * 44} fontSize={12} fontFamily={G} fill={CREAM} textAnchor="middle">{l}</text>
          </g>
        ))}
        <text x={352} y={236} fontSize={12} fontFamily={M} fill={BLUE} textAnchor="middle">3</text>
      </g>
      <text x={464} y={140} fontSize={22} fontFamily={G} fill={SOFT} textAnchor="middle">×</text>
      {/* causal */}
      <g>
        <text x={574} y={40} fontSize={13} fontFamily={G} fontWeight={700} fill={VIOLET} textAnchor="middle">causal influence s₂</text>
        {['weak', 'moderate', 'strong'].map((l, i) => (
          <g key={l}>
            <rect x={504} y={68 + i * 44} width={140} height={32} rx={8} fill="rgba(180,155,240,.08)" stroke={VIOLET} strokeWidth={1.1} />
            <text x={574} y={89 + i * 44} fontSize={12} fontFamily={G} fill={CREAM} textAnchor="middle">{l}</text>
          </g>
        ))}
        <text x={574} y={236} fontSize={12} fontFamily={M} fill={VIOLET} textAnchor="middle">3</text>
      </g>
      <text x={686} y={140} fontSize={22} fontFamily={G} fill={SOFT} textAnchor="middle">=</text>
      <g>
        <rect x={716} y={96} width={140} height={88} rx={14} fill="rgba(240,162,75,.08)" stroke={AMBER} strokeWidth={1.5} />
        <text x={786} y={134} fontSize={26} fontFamily={M} fontWeight={700} fill={AMBER} textAnchor="middle">36</text>
        <text x={786} y={158} fontSize={11.5} fontFamily={G} fill={SOFT} textAnchor="middle">joint states</text>
        <text x={786} y={174} fontSize={11.5} fontFamily={G} fill={SOFT} textAnchor="middle">per candidate</text>
      </g>
    </svg>
  );
}

/* ---------- B-matrix: three levers, three amounts of belief-license ---------- */
export function BLeversSVG() {
  const bells: [string, number, number, string][] = [
    // label, diagonal, spread-width, color
    ['ablate · diag 0.50', 0.5, 96, AMBER],
    ['patch · diag 0.70', 0.7, 62, BLUE],
    ['steer · diag 0.90', 0.9, 30, VIOLET],
  ];
  return (
    <svg viewBox="0 0 880 300" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="Ablation licenses broad belief revision, patching moderate, steering almost none">
      {bells.map(([label, diag, w, c], i) => {
        const cx = 160 + i * 280;
        return (
          <g key={label as string}>
            <text x={cx} y={44} fontSize={13.5} fontFamily={G} fontWeight={700} fill={c as string} textAnchor="middle">{label}</text>
            {/* transition spread bell */}
            <path
              d={`M ${cx - 110} 190 C ${cx - (w as number)} 190, ${cx - (w as number) * 0.5} ${190 - 110 * (diag as number)}, ${cx} ${190 - 120 * (diag as number)} C ${cx + (w as number) * 0.5} ${190 - 110 * (diag as number)}, ${cx + (w as number)} 190, ${cx + 110} 190`}
              fill="none" stroke={c as string} strokeWidth={2.6}
            />
            <line x1={cx - 118} y1={190} x2={cx + 118} y2={190} stroke={LINE} strokeWidth={1.5} />
            <text x={cx} y={212} fontSize={10.5} fontFamily={G} fill={SOFT} textAnchor="middle">belief about importance, after the probe</text>
            <text x={cx} y={238} fontSize={11.5} fontFamily={G} fill={CREAM} textAnchor="middle">
              {i === 0 ? 'deleting a feature can teach you anything' : i === 1 ? 'swapping its value teaches a moderate amount' : 'rescaling it barely revises beliefs'}
            </text>
          </g>
        );
      })}
      <text x={440} y={282} fontSize={13} fontFamily={G} fontWeight={600} fill={CREAM} textAnchor="middle">
        one committed ordering — H(B) ablate &gt; patch &gt; steer — and it is a falsifiable prior, not a schedule
      </text>
    </svg>
  );
}

/* ---------- explore → exploit: the crossing curves EFE predicts ---------- */
export function ExploreExploitSVG() {
  const N = 20;
  const ep = Array.from({ length: N }, (_, i) => Math.exp(-i / 6));
  const pr = Array.from({ length: N }, (_, i) => 1 - Math.exp(-(i + 2) / 7));
  const X = (i: number) => 70 + (i / (N - 1)) * 700;
  const Y = (v: number) => 210 - v * 150;
  return (
    <svg viewBox="0 0 880 300" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="Epistemic value dominates early steps, pragmatic value later — predicting ablate-then-steer">
      <line x1={70} y1={210} x2={790} y2={210} stroke={LINE} strokeWidth={1.5} />
      <path d={ep.map((v, i) => `${i ? 'L' : 'M'}${X(i)},${Y(v)}`).join(' ')} fill="none" stroke={TEAL} strokeWidth={3} />
      <path d={pr.map((v, i) => `${i ? 'L' : 'M'}${X(i)},${Y(v)}`).join(' ')} fill="none" stroke={AMBER} strokeWidth={3} />
      <text x={140} y={70} fontSize={13} fontFamily={G} fontWeight={700} fill={TEAL}>epistemic value — “I know nothing yet”</text>
      <text x={520} y={70} fontSize={13} fontFamily={G} fontWeight={700} fill={AMBER}>pragmatic value — “now find the big effects”</text>
      {/* crossing region */}
      <line x1={X(6.5)} y1={40} x2={X(6.5)} y2={210} stroke={SOFT} strokeWidth={1.2} strokeDasharray="5 4" />
      <text x={X(6.5)} y={236} fontSize={11.5} fontFamily={G} fill={SOFT} textAnchor="middle">the handoff</text>
      {/* action eras */}
      <rect x={70} y={248} width={X(6.5) - 70} height={16} rx={5} fill={AMBER} opacity={0.35} />
      <rect x={X(6.5)} y={248} width={790 - X(6.5)} height={16} rx={5} fill={VIOLET} opacity={0.4} />
      <text x={(70 + X(6.5)) / 2} y={260} fontSize={10.5} fontFamily={G} fill={CREAM} textAnchor="middle">ablate era</text>
      <text x={(790 + X(6.5)) / 2} y={260} fontSize={10.5} fontFamily={G} fill={CREAM} textAnchor="middle">steer era</text>
      <text x={70} y={290} fontSize={12.5} fontFamily={G} fill={SOFT}>
        intervention step 1 → 20 · no schedule is coded anywhere — if this pattern appears, it emerged from the objective
      </text>
    </svg>
  );
}

/* ---------- lineage timeline ---------- */
export function TimelineSVG() {
  const items: [string, string, string, string][] = [
    ['1956', 'Lindley', 'experiment value = expected information gain', SOFT],
    ['1992', 'MacKay', 'information-based objectives for learning systems', SOFT],
    ['2015', 'Friston et al.', 'EFE: epistemic term = that same information gain', TEAL],
    ['2021', 'Sajid et al.', 'flat preferences ⇒ EFE is Lindley design', TEAL],
    ['2022', 'pymdp · BOED', 'reference tooling on both sides of the bridge', BLUE],
    ['2026', 'ACD', 'the machinery pointed at a transformer’s internals', AMBER],
  ];
  return (
    <svg viewBox="0 0 900 260" style={{ width: '100%', height: 'auto' }} role="img"
      aria-label="A timeline from Lindley 1956 through EFE to ACD 2026">
      <line x1={60} y1={130} x2={850} y2={130} stroke={LINE} strokeWidth={2.5} />
      {items.map(([yr, who, what, c], i) => {
        const x = 80 + i * 150;
        const up = i % 2 === 0;
        return (
          <g key={yr as string}>
            <circle cx={x} cy={130} r={7} fill={c as string} />
            <line x1={x} y1={130} x2={x} y2={up ? 92 : 168} stroke={c as string} strokeWidth={1.4} />
            <text x={x} y={up ? 60 : 196} fontSize={13} fontFamily={M} fontWeight={700} fill={c as string} textAnchor="middle">{yr}</text>
            <text x={x} y={up ? 78 : 214} fontSize={12} fontFamily={G} fontWeight={600} fill={CREAM} textAnchor="middle">{who}</text>
            <text x={x} y={up ? 92 - 62 : 232} fontSize={10} fontFamily={G} fill={SOFT} textAnchor="middle">
              {(what as string).length > 46 ? '' : ''}
            </text>
            {/* wrap what into two lines max */}
            {(() => {
              const wordsArr = (what as string).split(' ');
              const mid = Math.ceil(wordsArr.length / 2);
              const l1 = wordsArr.slice(0, mid).join(' ');
              const l2 = wordsArr.slice(mid).join(' ');
              const base = up ? 20 : 246;
              return (
                <>
                  <text x={x} y={base} fontSize={9.8} fontFamily={G} fill={SOFT} textAnchor="middle">{l1}</text>
                  <text x={x} y={base + 13} fontSize={9.8} fontFamily={G} fill={SOFT} textAnchor="middle">{l2}</text>
                </>
              );
            })()}
          </g>
        );
      })}
    </svg>
  );
}

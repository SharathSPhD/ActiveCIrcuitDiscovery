/* Server-rendered SVG charts computed from the paper's real result JSONs. */
import stats from '../data/stats.json';
import ioiG from '../data/replays/ioi_gemma.json';
import ioiL from '../data/replays/ioi_llama.json';
import msG from '../data/replays/multistep_gemma.json';
import msL from '../data/replays/multistep_llama.json';
import steerG from '../data/steering_gemma.json';
import steerL from '../data/steering_llama.json';

const G = 'var(--grotesk)';
const COLORS: Record<string, string> = {
  oracle: '#8c94a4',
  eap: '#f0a24b',
  ai_abl: '#4fd8ce',
  bandit: '#b49bf0',
  ucb: '#9fb6e8',
  greedy: '#e0567a',
  random: '#6f7a8c',
};
const LABELS: Record<string, string> = {
  oracle: 'Oracle',
  eap: 'EAP',
  ai_abl: 'POMDP-abl',
  bandit: 'Bandit',
  ucb: 'UCB',
  greedy: 'Greedy',
  random: 'Random',
};

export function Legend({ keys }: { keys: string[] }) {
  return (
    <div className="legend-row">
      {keys.map((k) => (
        <span className="li" key={k}>
          <span className="sw" style={{ background: COLORS[k] }} />
          {LABELS[k] ?? k}
        </span>
      ))}
    </div>
  );
}

/* ---------- 1. Bounded oracle efficiency bars with CI whiskers ---------- */
export function EffBars() {
  const tasks: { title: string; data: any }[] = [
    { title: 'Gemma-2-2B · IOI', data: (stats as any).gemma.ioi.methods },
    { title: 'Llama-3.2-1B · IOI', data: (stats as any).llama.ioi.methods },
    { title: 'Gemma-2-2B · Multi-step', data: (stats as any).gemma.multistep.methods },
    { title: 'Llama-3.2-1B · Multi-step', data: (stats as any).llama.multistep.methods },
  ];
  const order = ['eap', 'ai_abl', 'bandit', 'ucb', 'greedy', 'random'];
  const W = 520, H = 190, padL = 86, padR = 46, barH = 17, gap = 7;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.2rem' }}>
      {tasks.map((t) => (
        <div key={t.title}>
          <div style={{ fontFamily: G, fontSize: '.8rem', fontWeight: 600, marginBottom: 4 }}>{t.title}</div>
          <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" style={{ width: '100%', height: 'auto' }}>
            {[0, 25, 50, 75, 100].map((x) => (
              <g key={x}>
                <line x1={padL + (x / 100) * (W - padL - padR)} y1={6} x2={padL + (x / 100) * (W - padL - padR)} y2={H - 22}
                  stroke="currentColor" opacity={x === 100 ? 0.5 : 0.12} strokeDasharray={x === 100 ? '4 3' : undefined} />
                <text x={padL + (x / 100) * (W - padL - padR)} y={H - 8} fontSize={10} textAnchor="middle" fill="currentColor" opacity={0.6}>{x}%</text>
              </g>
            ))}
            {order.map((k, i) => {
              const m = t.data[k]?.oracle_eff;
              if (!m) return null;
              const y = 10 + i * (barH + gap);
              const sx = (v: number) => padL + (Math.max(0, Math.min(v, 105)) / 100) * (W - padL - padR);
              return (
                <g key={k}>
                  <text x={padL - 8} y={y + barH - 5} fontSize={11} textAnchor="end" fill="currentColor" opacity={0.85} fontFamily={G}>{LABELS[k]}</text>
                  <rect x={padL} y={y} width={sx(m.point) - padL} height={barH} rx={3} fill={COLORS[k]} opacity={0.85} />
                  <line x1={sx(m.lo)} y1={y + barH / 2} x2={sx(m.hi)} y2={y + barH / 2} stroke="currentColor" strokeWidth={1.4} opacity={0.75} />
                  <line x1={sx(m.lo)} y1={y + 3} x2={sx(m.lo)} y2={y + barH - 3} stroke="currentColor" strokeWidth={1.4} opacity={0.75} />
                  <line x1={sx(m.hi)} y1={y + 3} x2={sx(m.hi)} y2={y + barH - 3} stroke="currentColor" strokeWidth={1.4} opacity={0.75} />
                  <text x={Math.min(sx(m.point) + 5, W - 2)} y={y + barH - 5} fontSize={10.5} fill="currentColor" opacity={0.9} fontFamily="var(--mono)">
                    {m.point.toFixed(1)}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      ))}
    </div>
  );
}

/* ---------- 2. Cumulative KL curves (mean over prompts) ---------- */
function meanCum(seriesList: number[][]): number[] {
  const B = seriesList[0].length;
  const out: number[] = [];
  let acc = 0;
  for (let t = 0; t < B; t++) {
    const stepMean = seriesList.reduce((s, x) => s + x[t], 0) / seriesList.length;
    acc += stepMean;
    out.push(acc);
  }
  return out;
}

export function CumKLChart({ task = 'ioi' }: { task?: 'ioi' | 'multistep' }) {
  const reps = task === 'ioi' ? [ioiG, ioiL] : [msG, msL];
  const W = 520, H = 250, padL = 60, padB = 30, padT = 14, padR = 14;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.2rem' }}>
      {reps.map((rep: any) => {
        const prompts = rep.prompts as any[];
        const series: Record<string, number[]> = {
          oracle: meanCum(prompts.map((p) => p.baselines.oracle)),
          eap: meanCum(prompts.map((p) => p.baselines.eap)),
          ai_abl: meanCum(prompts.map((p) => p.ai.abl_kls)),
          bandit: meanCum(prompts.map((p) => p.baselines.bandit)),
          greedy: meanCum(prompts.map((p) => p.baselines.greedy)),
        };
        const maxY = Math.max(...series.oracle) * 1.06;
        const B = series.oracle.length;
        const x = (t: number) => padL + (t / (B - 1)) * (W - padL - padR);
        const y = (v: number) => H - padB - (v / maxY) * (H - padB - padT);
        const path = (vals: number[]) => vals.map((v, t) => `${t === 0 ? 'M' : 'L'}${x(t).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
        return (
          <div key={rep.model}>
            <div style={{ fontFamily: G, fontSize: '.8rem', fontWeight: 600, marginBottom: 4 }}>
              {rep.model === 'gemma' ? 'Gemma-2-2B' : 'Llama-3.2-1B'} · mean cumulative ablation KL
            </div>
            <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" style={{ width: '100%', height: 'auto' }}>
              <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="currentColor" opacity={0.35} />
              <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="currentColor" opacity={0.35} />
              {[0.25, 0.5, 0.75, 1].map((f) => (
                <text key={f} x={padL - 6} y={y(maxY * f) + 3} fontSize={9.5} textAnchor="end" fill="currentColor" opacity={0.6} fontFamily="var(--mono)">
                  {(maxY * f).toExponential(1)}
                </text>
              ))}
              {[1, 5, 10, 15, 20].map((s) => (
                <text key={s} x={x(s - 1)} y={H - padB + 14} fontSize={9.5} textAnchor="middle" fill="currentColor" opacity={0.6}>{s}</text>
              ))}
              <text x={(W + padL) / 2} y={H - 2} fontSize={10} textAnchor="middle" fill="currentColor" opacity={0.65} fontFamily={G}>intervention step</text>
              {Object.entries(series).map(([k, vals]) => (
                <path key={k} d={path(vals)} fill="none" stroke={COLORS[k]} strokeWidth={k === 'oracle' ? 2.6 : 2}
                  strokeDasharray={k === 'oracle' ? '6 4' : undefined} opacity={0.95} />
              ))}
            </svg>
          </div>
        );
      })}
    </div>
  );
}

/* ---------- 3. RCK log-scale bars ---------- */
export function RCKChart() {
  const models: { title: string; d: any }[] = [
    { title: 'Gemma-2-2B · IOI', d: (stats as any).gemma.ioi.methods },
    { title: 'Llama-3.2-1B · IOI', d: (stats as any).llama.ioi.methods },
  ];
  const order: [string, string][] = [
    ['ai', 'POMDP (multi)'],
    ['bandit_steer', 'Bandit+steer'],
    ['random_steer', 'Random+steer'],
    ['greedy_steer', 'Greedy+steer'],
    ['random_action', 'Random-action'],
  ];
  const W = 520, H = 175, padL = 118, padR = 50, barH = 17, gap = 8;
  const lo = 20, hi = 3000;
  const sx = (v: number) => padL + ((Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (W - padL - padR);
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.2rem' }}>
      {models.map((m) => (
        <div key={m.title}>
          <div style={{ fontFamily: G, fontSize: '.8rem', fontWeight: 600, marginBottom: 4 }}>{m.title} · RCK (log scale)</div>
          <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" style={{ width: '100%', height: 'auto' }}>
            {[100, 1000].map((v) => (
              <g key={v}>
                <line x1={sx(v)} y1={4} x2={sx(v)} y2={H - 20} stroke={v === 100 ? '#f0a24b' : 'currentColor'} opacity={v === 100 ? 0.8 : 0.15} strokeDasharray="4 3" />
                <text x={sx(v)} y={H - 6} fontSize={10} textAnchor="middle" fill="currentColor" opacity={0.65}>{v}%</text>
              </g>
            ))}
            <text x={sx(100)} y={12} fontSize={9} textAnchor="middle" fill="#f0a24b" fontFamily={G}>ablation oracle</text>
            {order.map(([k, label], i) => {
              const r = m.d[k]?.rck;
              if (!r) return null;
              const yy = 18 + i * (barH + gap);
              return (
                <g key={k}>
                  <text x={padL - 8} y={yy + barH - 5} fontSize={11} textAnchor="end" fill="currentColor" opacity={0.85} fontFamily={G}>{label}</text>
                  <rect x={padL} y={yy} width={Math.max(2, sx(r.point) - padL)} height={barH} rx={3}
                    fill={k === 'ai' ? '#4fd8ce' : '#b49bf0'} opacity={k === 'ai' ? 0.95 : 0.6} />
                  <line x1={sx(r.lo)} y1={yy + barH / 2} x2={sx(Math.min(r.hi, hi))} y2={yy + barH / 2} stroke="currentColor" strokeWidth={1.3} opacity={0.7} />
                  <text x={Math.min(sx(r.point) + 5, W - 4)} y={yy + barH - 5} fontSize={10} fill="currentColor" opacity={0.9} fontFamily="var(--mono)">
                    {Math.round(r.point)}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      ))}
    </div>
  );
}

/* ---------- 4. Steering dose-response ---------- */
export function SteeringChart() {
  const sets: { title: string; d: any }[] = [
    { title: 'Gemma-2-2B', d: steerG },
    { title: 'Llama-3.2-1B', d: steerL },
  ];
  const W = 520, H = 240, padL = 52, padB = 34, padT = 16, padR = 52;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.2rem' }}>
      {sets.map(({ title, d }) => {
        const mults: number[] = d.multipliers;
        const x = (i: number) => padL + (i / (mults.length - 1)) * (W - padL - padR);
        const selC = mults.map((m) => d.top_changes[m.toFixed(1)].changed);
        const ctlC = mults.map((m) => d.control_changes[m.toFixed(1)].changed);
        const yC = (v: number) => H - padB - (v / 12) * (H - padB - padT);
        const path = (vals: number[]) => vals.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${yC(v).toFixed(1)}`).join(' ');
        return (
          <div key={title}>
            <div style={{ fontFamily: G, fontSize: '.8rem', fontWeight: 600, marginBottom: 4 }}>
              {title} · top-1 prediction changes out of 50
            </div>
            <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" style={{ width: '100%', height: 'auto' }}>
              <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="currentColor" opacity={0.35} />
              {[0, 4, 8, 12].map((v) => (
                <g key={v}>
                  <line x1={padL} y1={yC(v)} x2={W - padR} y2={yC(v)} stroke="currentColor" opacity={0.09} />
                  <text x={padL - 6} y={yC(v) + 3} fontSize={10} textAnchor="end" fill="currentColor" opacity={0.6} fontFamily="var(--mono)">{v}</text>
                </g>
              ))}
              {mults.map((m, i) => (
                <text key={m} x={x(i)} y={H - padB + 15} fontSize={10} textAnchor="middle" fill="currentColor" opacity={0.65}>{m}×</text>
              ))}
              <text x={(W + padL) / 2} y={H - 3} fontSize={10} textAnchor="middle" fill="currentColor" opacity={0.65} fontFamily={G}>steering multiplier</text>
              <path d={path(selC)} fill="none" stroke="#4fd8ce" strokeWidth={2.4} />
              <path d={path(ctlC)} fill="none" stroke="#8c94a4" strokeWidth={2.2} strokeDasharray="5 4" />
              {selC.map((v, i) => <circle key={`s${i}`} cx={x(i)} cy={yC(v)} r={3.4} fill="#4fd8ce" />)}
              {ctlC.map((v, i) => <circle key={`c${i}`} cx={x(i)} cy={yC(v)} r={3.2} fill="#8c94a4" />)}
              <text x={W - padR + 6} y={yC(selC[selC.length - 1])} fontSize={10} fill="#4fd8ce" fontFamily={G}>selected</text>
              <text x={W - padR + 6} y={yC(ctlC[ctlC.length - 1]) + 10} fontSize={10} fill="#8c94a4" fontFamily={G}>control</text>
            </svg>
          </div>
        );
      })}
    </div>
  );
}

/* ---------- 5. Action selection strip (per prompt × step) ---------- */
const ACTION_COLOR: Record<string, string> = {
  ablation: '#f0a24b',
  activation_patching: '#9fb6e8',
  feature_steering: '#b49bf0',
};

export function ActionStrip() {
  const sets: { title: string; rep: any }[] = [
    { title: 'Gemma-2-2B · IOI — 5 ablations to start, then 94/100 steering', rep: ioiG },
    { title: 'Llama-3.2-1B · IOI — 70/100 ablation: the agent stays conservative', rep: ioiL },
  ];
  const cell = 21, gapY = 6, padL = 60;
  return (
    <div style={{ display: 'grid', gap: '1.4rem' }}>
      {sets.map(({ title, rep }) => {
        const prompts = rep.prompts as any[];
        const W = padL + 20 * cell + 10;
        const H = prompts.length * (cell + gapY) + 30;
        return (
          <div key={title}>
            <div style={{ fontFamily: G, fontSize: '.8rem', fontWeight: 600, marginBottom: 4 }}>{title}</div>
            <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" style={{ width: '100%', maxWidth: 620, height: 'auto' }}>
              {prompts.map((p, r) =>
                p.ai.actions.map((a: string, c: number) => (
                  <rect key={`${r}-${c}`} x={padL + c * cell} y={r * (cell + gapY)} width={cell - 3} height={cell - 3} rx={4}
                    fill={ACTION_COLOR[a] ?? '#666'} opacity={0.92} />
                ))
              )}
              {prompts.map((_, r) => (
                <text key={r} x={padL - 8} y={r * (cell + gapY) + cell - 8} fontSize={10} textAnchor="end" fill="currentColor" opacity={0.65} fontFamily={G}>
                  P{r + 1}
                </text>
              ))}
              {[1, 5, 10, 15, 20].map((s) => (
                <text key={s} x={padL + (s - 1) * cell + cell / 2} y={H - 10} fontSize={9.5} textAnchor="middle" fill="currentColor" opacity={0.6}>{s}</text>
              ))}
            </svg>
          </div>
        );
      })}
      <div className="legend-row">
        <span className="li"><span className="sw" style={{ background: ACTION_COLOR.ablation }} /> ablation</span>
        <span className="li"><span className="sw" style={{ background: ACTION_COLOR.activation_patching }} /> activation patching</span>
        <span className="li"><span className="sw" style={{ background: ACTION_COLOR.feature_steering }} /> feature steering</span>
      </div>
    </div>
  );
}

/* ---------- 6. A-matrix convergence ---------- */
export function AConvChart() {
  const drift: number[] = (stats as any).gemma.ioi.a_convergence.per_step_mean_drift;
  const W = 520, H = 200, padL = 62, padB = 30, padT = 12, padR = 16;
  const maxY = Math.max(...drift) * 1.1;
  const x = (i: number) => padL + (i / (drift.length - 1)) * (W - padL - padR);
  const y = (v: number) => H - padB - (v / maxY) * (H - padB - padT);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" style={{ width: '100%', maxWidth: 560, height: 'auto' }}>
      <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="currentColor" opacity={0.35} />
      <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="currentColor" opacity={0.35} />
      {[0.5, 1].map((f) => (
        <text key={f} x={padL - 6} y={y(maxY * f) + 3} fontSize={9.5} textAnchor="end" fill="currentColor" opacity={0.6} fontFamily="var(--mono)">
          {(maxY * f).toExponential(1)}
        </text>
      ))}
      <path d={drift.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')} fill="none" stroke="#4fd8ce" strokeWidth={2.4} />
      {drift.map((v, i) => <circle key={i} cx={x(i)} cy={y(v)} r={2.6} fill="#4fd8ce" />)}
      {[1, 5, 10, 15, 19].map((s) => (
        <text key={s} x={x(s - 1)} y={H - padB + 14} fontSize={9.5} textAnchor="middle" fill="currentColor" opacity={0.6}>{s}</text>
      ))}
      <text x={(W + padL) / 2} y={H - 2} fontSize={10} textAnchor="middle" fill="currentColor" opacity={0.65} fontFamily={G}>
        update index · mean L1 drift of learned KL-likelihood, Gemma IOI (halves: 1.03e-3 → 5.0e-4)
      </text>
    </svg>
  );
}

/* ---------- 7. Domain efficiency heat table ---------- */
export function DomainTable() {
  const doms = ['geography', 'math', 'science', 'logic', 'history'];
  const nice: Record<string, string> = { geography: 'Geography', math: 'Mathematics', science: 'Science', logic: 'Logic', history: 'History' };
  const models: ['gemma' | 'llama', string][] = [['gemma', 'Gemma-2-2B'], ['llama', 'Llama-3.2-1B']];
  const cols: [string, string][] = [
    ['eap_oracle_efficiency', 'EAP'],
    ['ai_abl_oracle_efficiency', 'POMDP-abl'],
    ['bandit_oracle_efficiency', 'Bandit'],
    ['ucb_oracle_efficiency', 'UCB'],
  ];
  const cellBg = (v: number) => `rgba(79, 216, 206, ${0.04 + 0.5 * Math.max(0, Math.min(1, (v - 40) / 60))})`;
  return (
    <div className="tbl-wrap">
      <table className="tbl">
        <thead>
          <tr>
            <th>Model</th>
            <th>Domain</th>
            {cols.map(([, l]) => <th key={l} className="num">{l} %</th>)}
            <th>Layer mass (top-10)</th>
          </tr>
        </thead>
        <tbody>
          {models.flatMap(([m, label]) =>
            doms.map((dom, i) => {
              const d = (stats as any)[m].domain.domains[dom];
              const ld = d.layer_distribution;
              const tot = ld.early + ld.mid + ld.late || 1;
              return (
                <tr key={`${m}-${dom}`}>
                  <td>{i === 0 ? label : ''}</td>
                  <td>{nice[dom]}</td>
                  {cols.map(([k]) => (
                    <td key={k} className="num" style={{ background: cellBg(d[k]) }}>{d[k].toFixed(1)}</td>
                  ))}
                  <td style={{ minWidth: 130 }}>
                    <svg viewBox="0 0 120 12" style={{ width: 120, height: 12 }}>
                      <rect x={0} y={2} width={(ld.early / tot) * 120} height={8} fill="#4fd8ce" rx={2} />
                      <rect x={(ld.early / tot) * 120} y={2} width={(ld.mid / tot) * 120} height={8} fill="#9fb6e8" />
                      <rect x={((ld.early + ld.mid) / tot) * 120} y={2} width={(ld.late / tot) * 120} height={8} fill="#b49bf0" rx={2} />
                    </svg>
                  </td>
                </tr>
              );
            })
          )}
        </tbody>
      </table>
      <div className="legend-row">
        <span className="li"><span className="sw" style={{ background: '#4fd8ce' }} /> early layers</span>
        <span className="li"><span className="sw" style={{ background: '#9fb6e8' }} /> middle</span>
        <span className="li"><span className="sw" style={{ background: '#b49bf0' }} /> late</span>
      </div>
    </div>
  );
}

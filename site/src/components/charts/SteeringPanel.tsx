import { useState } from 'preact/hooks';
import { scaleLinear, scaleBand } from 'd3';
import Toggle from '../interactive/Toggle';
import { useChartWidth } from './useChartWidth';
import { steeringDose, steeringHypotheses } from '../../data';
import { MODEL_LABELS, type ModelKey } from '../../utils/theme';

/**
 * Scene 7b — steering dose-response. Prediction-change counts for circuit-
 * selected features vs matched random controls, across the multiplier sweep.
 * They track each other: selectivity (H2b) is not significant.
 */
export default function SteeringPanel() {
  const [model, setModel] = useState<ModelKey>('gemma');
  const { ref, width } = useChartWidth(680);
  const dose = steeringDose(model);
  const hyp = steeringHypotheses(model);

  const W = Math.max(width, 320);
  const padL = 40;
  const padR = 16;
  const padT = 16;
  const padB = 34;
  const H = 280;
  const x = scaleBand<string>().domain(dose.map((d) => d.multiplier.toFixed(1))).range([padL, W - padR]).padding(0.25);
  const inner = scaleBand<string>().domain(['selected', 'control']).range([0, x.bandwidth()]).padding(0.12);
  const maxC = Math.max(...dose.flatMap((d) => [d.selected.changed, d.control.changed]), 1);
  const y = scaleLinear().domain([0, maxC]).range([H - padB, padT]);
  const total = dose[0]?.selected.total ?? 50;

  return (
    <div class="steer" data-testid="steering-panel">
      <div class="controls">
        <Toggle ariaLabel="Model" testid="steer-model" value={model} onChange={(v) => setModel(v as ModelKey)}
          options={[{ value: 'gemma', label: MODEL_LABELS.gemma }, { value: 'llama', label: MODEL_LABELS.llama }]} />
        <div class="legend">
          <span class="leg"><span class="sw" style="background:var(--cyan)" />Circuit-selected</span>
          <span class="leg"><span class="sw" style="background:var(--ink-faint)" />Random control</span>
        </div>
      </div>

      <p class="cap">
        Of {total} steered features, how many flipped the model's next-token prediction —
        {MODEL_LABELS[model]}, by activation multiplier.
      </p>

      <div ref={ref} class="svg-wrap">
        <svg width={W} height={H} role="img" data-testid="steering-chart"
             aria-label={`Steering dose-response for ${MODEL_LABELS[model]}`}>
          {y.ticks(4).map((t) => (
            <g>
              <line x1={padL} x2={W - padR} y1={y(t)} y2={y(t)} stroke="var(--hairline)" />
              <text x={padL - 6} y={y(t)} fill="var(--ink-faint)" font-size="10" text-anchor="end" dominant-baseline="middle">{t}</text>
            </g>
          ))}
          {dose.map((d) => {
            const gx = x(d.multiplier.toFixed(1))!;
            const items: [string, number, string][] = [
              ['selected', d.selected.changed, 'var(--cyan)'],
              ['control', d.control.changed, 'var(--ink-faint)'],
            ];
            return (
              <g data-testid={`steer-group-${d.multiplier}`}>
                {items.map(([role, val, color]) => (
                  <rect x={gx + inner(role)!} y={y(val)} width={inner.bandwidth()} height={y(0) - y(val)}
                        fill={color} rx="2" data-testid={`steer-${role}-${d.multiplier}`} data-count={val} />
                ))}
                <text x={gx + x.bandwidth() / 2} y={H - padB + 14} fill="var(--ink-soft)" font-size="10" text-anchor="middle">×{d.multiplier}</text>
              </g>
            );
          })}
        </svg>
      </div>

      <div class="verdicts">
        <div class="v v-ok" data-testid="h2a-verdict">
          <span class="vc">H2a · manipulability</span>
          <span class="vt">Accepted</span>
          <span class="vp mono">at ×{hyp.h2a.multiplier}: {hyp.h2a.changed}/{hyp.h2a.total} flip · p = {hyp.h2a.p_value.toExponential(1)}</span>
        </div>
        <div class="v v-no" data-testid="h2b-verdict">
          <span class="vc">H2b · selectivity</span>
          <span class="vt">Not supported</span>
          <span class="vp mono">selected {(hyp.h2b.selected_rate * 100).toFixed(0)}% vs control {(hyp.h2b.control_rate * 100).toFixed(0)}% · p = {hyp.h2b.p_value.toFixed(2)}</span>
        </div>
      </div>

      <style>{`
        .steer { margin: 1.4rem 0; }
        .controls { display:flex; gap:1rem; align-items:center; flex-wrap:wrap; margin-bottom:0.7rem; }
        .legend { display:flex; gap:1rem; }
        .leg { display:flex; align-items:center; gap:0.4rem; font-size:0.8rem; color:var(--ink-faint); }
        .sw { width:11px; height:11px; border-radius:3px; }
        .cap { font-size:0.9rem; color:var(--ink-faint); margin:0 0 0.6rem; }
        .svg-wrap { width:100%; }
        .verdicts { display:flex; gap:1rem; flex-wrap:wrap; margin-top:0.9rem; }
        .v { flex:1; min-width:220px; border:1px solid var(--hairline); border-left-width:3px;
             border-radius:var(--radius); padding:0.8rem 1rem; background:var(--bg-panel);
             display:flex; flex-direction:column; gap:0.25rem; }
        .v-ok { border-left-color: var(--accept); }
        .v-no { border-left-color: var(--reject); }
        .vc { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--ink-faint); }
        .vt { font-weight:650; color:var(--ink); }
        .v-ok .vt { color: var(--accept); }
        .v-no .vt { color: var(--reject); }
        .vp { font-size:0.8rem; color:var(--ink-soft); }
      `}</style>
    </div>
  );
}

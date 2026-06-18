import { useState } from 'preact/hooks';
import { scaleLinear } from 'd3';
import Toggle from '../interactive/Toggle';
import { useChartWidth } from './useChartWidth';
import { efficiencyBars, h1, TASK_LABELS, type Task } from '../../data';
import { MODEL_LABELS, type ModelKey } from '../../utils/theme';
import { fmtPct } from '../../utils/math';

/**
 * Scene 6 — "Does it work?" Bounded oracle-efficiency bars with 95% CIs,
 * across model × task, plus the H1 (vs random) significance readout.
 */
export default function EfficiencyExplorer() {
  const [model, setModel] = useState<ModelKey>('gemma');
  const [task, setTask] = useState<Task>('ioi');
  const { ref, width } = useChartWidth(680);

  const bars = efficiencyBars(model, task);
  const rowH = 38;
  const padL = 150;
  const padR = 56;
  const padT = 8;
  const h = bars.length * rowH + padT;
  const x = scaleLinear().domain([0, 100]).range([padL, Math.max(width - padR, padL + 120)]);

  const showH1 = task !== 'domain';
  const h1stat = showH1 ? h1(model, task as 'ioi' | 'multistep') : null;

  return (
    <div class="eff" data-testid="efficiency-explorer">
      <div class="controls">
        <Toggle
          ariaLabel="Model"
          testid="eff-model"
          value={model}
          onChange={(v) => setModel(v as ModelKey)}
          options={[
            { value: 'gemma', label: MODEL_LABELS.gemma },
            { value: 'llama', label: MODEL_LABELS.llama },
          ]}
        />
        <Toggle
          ariaLabel="Task"
          testid="eff-task"
          value={task}
          onChange={(v) => setTask(v as Task)}
          options={[
            { value: 'ioi', label: 'IOI' },
            { value: 'multistep', label: 'Multi-step' },
            { value: 'domain', label: 'Domains' },
          ]}
        />
      </div>

      <p class="cap">
        Bounded oracle efficiency on <strong>{TASK_LABELS[task]}</strong> — {MODEL_LABELS[model]}.
        100% = matches the ablation oracle. Bars sorted best→worst; whiskers are 95% CIs.
      </p>

      <div ref={ref} class="svg-wrap" tabIndex={0} role="group" aria-label="Scrollable chart — scroll horizontally to see all of it">
        <svg width={width} height={h} role="img" data-testid="efficiency-chart"
             aria-label={`Oracle efficiency by method for ${MODEL_LABELS[model]} on ${TASK_LABELS[task]}`}>
          {/* gridlines */}
          {[0, 25, 50, 75, 100].map((t) => (
            <g>
              <line x1={x(t)} x2={x(t)} y1={padT} y2={h} stroke="var(--hairline)" stroke-width="1" />
              <text x={x(t)} y={h - 2} fill="var(--ink-soft)" font-size="10" text-anchor="middle">{t}</text>
            </g>
          ))}
          {bars.map((b, i) => {
            const y = padT + i * rowH;
            const bw = x(b.eff) - x(0);
            return (
              <g data-testid={`eff-row-${b.key}`} data-eff={b.eff.toFixed(2)}>
                <text x={padL - 10} y={y + rowH / 2} fill="var(--ink-soft)" font-size="12"
                      text-anchor="end" dominant-baseline="middle">{b.label}</text>
                <rect x={x(0)} y={y + 7} width={Math.max(bw, 0)} height={rowH - 18}
                      rx="3" fill={b.color} opacity={b.key === 'oracle' ? 0.45 : 0.85}
                      data-testid={`eff-bar-${b.key}`} />
                {b.lo !== b.hi && (
                  <g stroke="var(--ink)" stroke-width="1.25" opacity="0.65">
                    <line x1={x(b.lo)} x2={x(b.hi)} y1={y + rowH / 2} y2={y + rowH / 2} />
                    <line x1={x(b.lo)} x2={x(b.lo)} y1={y + rowH / 2 - 4} y2={y + rowH / 2 + 4} />
                    <line x1={x(b.hi)} x2={x(b.hi)} y1={y + rowH / 2 - 4} y2={y + rowH / 2 + 4} />
                  </g>
                )}
                <text x={x(b.eff) + 7} y={y + rowH / 2} fill="var(--ink)" font-size="11.5"
                      dominant-baseline="middle">{fmtPct(b.eff)}</text>
              </g>
            );
          })}
        </svg>
      </div>

      {h1stat && (
        <p class="h1" data-testid="h1-readout">
          <strong>H1 (beats random):</strong> the POMDP agent improves on random selection by{' '}
          <span class="mono">{h1stat.improvement_pct.toFixed(0)}%</span>{' '}
          (permutation <span class="mono">p = {h1stat.p_permutation_onesided.toFixed(3)}</span>
          {h1stat.p_permutation_onesided < 0.05 ? ' — significant' : ' — not significant'}).
        </p>
      )}

      <style>{`
        .eff { margin: 1.5rem 0; }
        .controls { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 0.9rem; }
        .cap { font-size: 0.9rem; color: var(--ink-soft); margin: 0 0 0.8rem; }
        .cap strong { color: var(--ink-soft); }
        .svg-wrap { width: 100%; }
        .h1 { font-size: 0.92rem; color: var(--ink-soft); margin-top: 0.9rem;
              border-top: 1px solid var(--hairline); padding-top: 0.8rem; }
        .mono { font-family: var(--font-mono); color: var(--cyan); }
      `}</style>
    </div>
  );
}

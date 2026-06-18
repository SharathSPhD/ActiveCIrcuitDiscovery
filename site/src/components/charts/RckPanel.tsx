import { useState } from 'preact/hooks';
import { scaleLinear } from 'd3';
import Toggle from '../interactive/Toggle';
import { useChartWidth } from './useChartWidth';
import { rckBars, type Task } from '../../data';
import { MODEL_LABELS, type ModelKey } from '../../utils/theme';

/**
 * Scene 7a — the 1255% that isn't what it looks like.
 * Relative Cumulative KL: steering-enabled methods all blow past 100%, so the
 * amplification is mostly steering's mechanical power, not better discovery.
 */
export default function RckPanel() {
  const [model, setModel] = useState<ModelKey>('gemma');
  const [task, setTask] = useState<Task>('ioi');
  const { ref, width } = useChartWidth(680);
  const bars = rckBars(model, task);

  const rowH = 40;
  const padL = 170;
  const padR = 50;
  const padT = 8;
  const h = bars.length * rowH + padT + 16;
  const maxV = Math.max(100, ...bars.map((b) => b.hi ?? b.rck)) * 1.05;
  const x = scaleLinear().domain([0, maxV]).range([padL, Math.max(width - padR, padL + 120)]);

  return (
    <div class="rck" data-testid="rck-panel">
      <div class="controls">
        <Toggle ariaLabel="Model" testid="rck-model" value={model} onChange={(v) => setModel(v as ModelKey)}
          options={[{ value: 'gemma', label: MODEL_LABELS.gemma }, { value: 'llama', label: MODEL_LABELS.llama }]} />
        <Toggle ariaLabel="Task" testid="rck-task" value={task} onChange={(v) => setTask(v as Task)}
          options={[{ value: 'ioi', label: 'IOI' }, { value: 'multistep', label: 'Multi-step' }, { value: 'domain', label: 'Domains' }]} />
      </div>

      <p class="cap">
        Relative Cumulative KL vs the ablation oracle — {MODEL_LABELS[model]}. The dashed line is
        100% (the oracle). Notice: even <strong>random + steering</strong> clears it.
      </p>

      <div ref={ref} class="svg-wrap" tabIndex={0} role="group" aria-label="Scrollable chart — scroll horizontally to see all of it">
        <svg width={width} height={h} role="img" data-testid="rck-chart"
             aria-label={`Relative cumulative KL by method for ${MODEL_LABELS[model]}`}>
          {/* 100% reference */}
          <line x1={x(100)} x2={x(100)} y1={padT} y2={h - 14} stroke="var(--amber)" stroke-width="1.5" stroke-dasharray="4 3" />
          <text x={x(100)} y={h - 2} fill="var(--amber)" font-size="10" text-anchor="middle">100% (oracle)</text>
          {bars.map((b, i) => {
            const y = padT + i * rowH;
            return (
              <g data-testid={`rck-row-${b.key}`} data-rck={b.rck.toFixed(1)}>
                <text x={padL - 10} y={y + rowH / 2} fill="var(--ink-soft)" font-size="12" text-anchor="end" dominant-baseline="middle">{b.label}</text>
                <rect x={x(0)} y={y + 8} width={Math.max(x(b.rck) - x(0), 0)} height={rowH - 18} rx="3" fill={b.color} opacity="0.85" data-testid={`rck-bar-${b.key}`} />
                <text x={x(b.rck) + 6} y={y + rowH / 2} fill="var(--ink)" font-size="11.5" dominant-baseline="middle">{Math.round(b.rck)}%</text>
              </g>
            );
          })}
        </svg>
      </div>

      <style>{`
        .rck { margin: 1.4rem 0; }
        .controls { display:flex; gap:0.6rem; flex-wrap:wrap; margin-bottom:0.8rem; }
        .cap { font-size:0.9rem; color:var(--ink-soft); margin:0 0 0.7rem; }
        .cap strong { color: var(--amber); }
        .svg-wrap { width:100%; }
      `}</style>
    </div>
  );
}

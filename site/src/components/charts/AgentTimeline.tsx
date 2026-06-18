import { useState } from 'preact/hooks';
import { scaleLinear, line as d3line, curveMonotoneX } from 'd3';
import Toggle from '../interactive/Toggle';
import { useChartWidth } from './useChartWidth';
import { TIMELINES } from '../../data';
import { ACTION_COLORS, ACTION_LABELS, MODEL_LABELS, type ModelKey } from '../../utils/theme';
import { fmtKL } from '../../utils/math';

/**
 * Scene 5 — "Watch the agent think." Real 20-step trajectory: action chosen,
 * KL achieved, and belief-entropy / Expected-Free-Energy curves. A scrubber
 * lets you step through and read the exact values straight from the data.
 */
export default function AgentTimeline() {
  const [model, setModel] = useState<ModelKey>('gemma');
  const [pidx, setPidx] = useState(0);
  const [step, setStep] = useState(0);
  const { ref, width } = useChartWidth(680);

  const tl: any = TIMELINES[model];
  const prompt = tl.per_prompt[pidx];
  const kls: number[] = prompt.ai_kls;
  const actions: string[] = prompt.ai_actions;
  const entropy: number[] = prompt.agent_entropy_history;
  const efe: number[] = prompt.agent_efe_history;
  const n = kls.length;

  const W = Math.max(width, 320);
  const padL = 34;
  const padR = 14;
  const barTop = 14;
  const barH = 110;
  const gap = 18;
  const lineH = 92;
  const innerW = W - padL - padR;
  const bw = innerW / n;

  const klMax = Math.max(...kls, 1e-9);
  const yBar = scaleLinear().domain([0, klMax]).range([0, barH]);

  const eAll = [...entropy, ...efe];
  const eMin = Math.min(...eAll);
  const eMax = Math.max(...eAll);
  const xStep = (i: number) => padL + i * bw + bw / 2;
  const yLine = scaleLinear().domain([eMin, eMax]).range([barTop + barH + gap + lineH, barTop + barH + gap]);

  const mkLine = d3line<number>().x((_, i) => xStep(i)).y((d) => yLine(d)).curve(curveMonotoneX);
  const entropyPath = mkLine(entropy) ?? '';
  const efePath = mkLine(efe) ?? '';
  const totalH = barTop + barH + gap + lineH + 24;

  const curAction = actions[step];

  return (
    <div class="tl" data-testid="agent-timeline">
      <div class="controls">
        <Toggle ariaLabel="Model" testid="tl-model" value={model}
          onChange={(v) => { setModel(v as ModelKey); setStep(0); setPidx(0); }}
          options={[{ value: 'gemma', label: MODEL_LABELS.gemma }, { value: 'llama', label: MODEL_LABELS.llama }]} />
        <label class="prompt-pick">
          <span class="pick-label">Prompt</span>
          <select data-testid="tl-prompt" value={String(pidx)}
            onChange={(e) => { setPidx(Number((e.target as HTMLSelectElement).value)); setStep(0); }}>
            {tl.per_prompt.map((p: any, i: number) => (
              <option value={String(i)}>{`Prompt ${i + 1}: ${p.prompt.slice(0, 40)}${p.prompt.length > 40 ? '…' : ''}`}</option>
            ))}
          </select>
        </label>
      </div>

      <div class="legend">
        {Object.keys(ACTION_LABELS).map((a) => (
          <span class="leg"><span class="sw" style={`background:${ACTION_COLORS[a]}`} />{ACTION_LABELS[a]}</span>
        ))}
      </div>

      <div ref={ref} class="svg-wrap" tabIndex={0} role="group" aria-label="Scrollable chart — scroll horizontally to see all of it">
        <svg width={W} height={totalH} role="img" data-testid="timeline-svg"
             aria-label={`Agent 20-step trajectory for ${MODEL_LABELS[model]}, prompt ${pidx + 1}`}>
          <text x={padL} y={10} fill="var(--ink-soft)" font-size="10">KL divergence per intervention</text>
          {kls.map((k, i) => {
            const bh = Math.max(yBar(k), 1.5);
            const selected = i === step;
            return (
              <g data-action={actions[i]} data-kl={k}>
                <rect x={padL + i * bw + 1} y={barTop + barH - bh} width={bw - 2} height={bh}
                      fill={ACTION_COLORS[actions[i]]} opacity={selected ? 1 : 0.55} rx="1.5" />
                {selected && (
                  <rect x={padL + i * bw + 0.5} y={barTop - 2} width={bw - 1} height={barH + 4}
                        fill="none" stroke="var(--ink)" stroke-width="1" opacity="0.5" rx="2" />
                )}
              </g>
            );
          })}

          {/* entropy + EFE curves */}
          <text x={padL} y={barTop + barH + gap - 2} fill="var(--ink-soft)" font-size="10">
            belief entropy (cyan) &amp; expected free energy (violet)
          </text>
          <path d={entropyPath} fill="none" stroke="var(--cyan)" stroke-width="2" />
          <path d={efePath} fill="none" stroke="var(--violet)" stroke-width="1.5" opacity="0.8" stroke-dasharray="3 3" />
          <line x1={xStep(step)} x2={xStep(step)} y1={barTop + barH + gap} y2={barTop + barH + gap + lineH}
                stroke="var(--ink)" stroke-width="1" opacity="0.4" />
          <circle cx={xStep(step)} cy={yLine(entropy[step])} r="4" fill="var(--cyan)" />
          <circle cx={xStep(step)} cy={yLine(efe[step])} r="3" fill="var(--violet)" />

          {/* transparent hit layer (last in DOM so nothing intercepts clicks) */}
          {kls.map((k, i) => (
            <rect data-testid={`tl-bar-${i}`} data-action={actions[i]} data-kl={k}
                  x={padL + i * bw} y={0} width={bw} height={totalH}
                  fill="transparent" style="cursor:pointer"
                  onClick={() => setStep(i)} />
          ))}
        </svg>
      </div>

      <label class="scrub">
        <span class="scrub-label">Scrub the 20 interventions — or click a bar / use ← →</span>
        <input type="range" min="0" max={n - 1} value={step} data-testid="tl-scrubber"
               aria-label="Intervention step"
               onInput={(e) => setStep(Number((e.target as HTMLInputElement).value))} />
      </label>

      <div class="readout" data-testid="tl-readout">
        <div class="ro"><span class="k">Step</span><span class="v" data-testid="ro-step">{step + 1}<span class="dim"> / {n}</span></span></div>
        <div class="ro"><span class="k">Action</span><span class="v" style={`color:${ACTION_COLORS[curAction]}`} data-testid="ro-action">{ACTION_LABELS[curAction]}</span></div>
        <div class="ro"><span class="k">KL</span><span class="v mono" data-testid="ro-kl">{fmtKL(kls[step])}</span></div>
        <div class="ro"><span class="k">Belief entropy</span><span class="v mono" data-testid="ro-entropy">{entropy[step].toFixed(4)}</span></div>
        <div class="ro"><span class="k">EFE</span><span class="v mono" data-testid="ro-efe">{efe[step].toFixed(4)}</span></div>
      </div>

      <style>{`
        .tl { margin: 1.5rem 0; }
        .controls { display:flex; gap:0.6rem; flex-wrap:wrap; align-items:center; margin-bottom:0.7rem; }
        .prompt-pick select {
          background: var(--bg-inset); color: var(--ink-soft); border: 1px solid var(--hairline);
          border-radius: 8px; padding: 0.4rem 0.6rem; font: inherit; font-size: 0.85rem; max-width: 320px;
        }
        .legend { display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:0.4rem; }
        .leg { display:flex; align-items:center; gap:0.4rem; font-size:0.8rem; color:var(--ink-soft); }
        .sw { width:11px; height:11px; border-radius:3px; display:inline-block; }
        .svg-wrap { width:100%; overflow-x:auto; }
        .pick-label { font-size:0.78rem; color:var(--ink-soft); margin-right:0.4rem; }
        .scrub { display:block; }
        .scrub-label { display:block; font-size:0.78rem; color:var(--ink-soft); margin-bottom:0.3rem; }
        .scrub input { width:100%; accent-color: var(--cyan); }
        .readout { display:flex; gap:1.4rem; flex-wrap:wrap; margin-top:0.6rem;
                   border-top:1px solid var(--hairline); padding-top:0.8rem; }
        .ro { display:flex; flex-direction:column; gap:0.15rem; }
        .ro .k { font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--ink-soft); }
        .ro .v { font-size:1.05rem; color:var(--ink); font-weight:600; }
        .ro .v.mono { font-family:var(--font-mono); font-weight:500; }
        .ro .dim { color:var(--ink-soft); font-weight:400; }
      `}</style>
    </div>
  );
}

import { useState } from 'preact/hooks';
import Toggle from '../interactive/Toggle';
import { domains } from '../../data';
import { MODEL_LABELS, type ModelKey } from '../../utils/theme';
import { fmtPct } from '../../utils/math';

/**
 * Scene "Different thoughts in different layers."
 * The hero is a STACKED early/mid/late bar (where the top features live);
 * efficiency is a small side chip so the two never compete visually.
 */
const SEGS = [
  { key: 'early', label: 'Early', color: '#22d3ee' },
  { key: 'mid', label: 'Middle', color: '#a855f7' },
  { key: 'late', label: 'Late', color: '#f59e0b' },
] as const;

export default function DomainPanel() {
  const [model, setModel] = useState<ModelKey>('gemma');
  const data = domains(model);
  // common scale: largest total across domains so bars are comparable
  const maxTotal = Math.max(...data.map((d) => d.layers.early + d.layers.mid + d.layers.late), 1);

  return (
    <div class="dom" data-testid="domain-panel">
      <div class="controls">
        <Toggle ariaLabel="Model" testid="dom-model" value={model} onChange={(v) => setModel(v as ModelKey)}
          options={[{ value: 'gemma', label: MODEL_LABELS.gemma }, { value: 'llama', label: MODEL_LABELS.llama }]} />
        <div class="legend">
          {SEGS.map((s) => (
            <span class="leg"><span class="sw" style={`background:${s.color}`} />{s.label} layers</span>
          ))}
        </div>
      </div>

      <p class="cap">
        Where the <strong>top features</strong> live for each domain on {MODEL_LABELS[model]} — counted in the
        early / middle / late thirds of the network. The chip on the right is the agent's oracle efficiency there.
      </p>

      <div class="rows">
        {data.map((d) => {
          const segs = [d.layers.early, d.layers.mid, d.layers.late];
          const tot = segs.reduce((a, b) => a + b, 0) || 1;
          return (
            <div class="row" data-testid={`dom-row-${d.name}`}>
              <div class="name">{d.name}</div>
              <div class="track" role="img"
                   aria-label={`${d.name}: ${d.layers.early} early, ${d.layers.mid} middle, ${d.layers.late} late features`}>
                <div class="stack" style={`width:${(tot / maxTotal) * 100}%`}>
                  {SEGS.map((s, i) => {
                    const v = segs[i];
                    if (v <= 0) return null;
                    return (
                      <span class="seg" data-testid={`dom-${s.key}-${d.name}`} data-count={v}
                            style={`flex:${v};background:${s.color}`} title={`${s.label}: ${v}`}>
                        {v / tot > 0.12 ? v : ''}
                      </span>
                    );
                  })}
                </div>
              </div>
              <div class="eff" data-testid={`dom-eff-${d.name}`} data-eff={d.efficiency.toFixed(1)}>{fmtPct(d.efficiency, 0)}</div>
            </div>
          );
        })}
      </div>

      <p class="foot">Each bar = the top causal features for that domain, split by depth. Longer bars simply found more high-influence features.</p>

      <style>{`
        .dom { margin: 1.4rem 0; }
        .controls { display:flex; gap:1rem; align-items:center; flex-wrap:wrap; margin-bottom:0.7rem; }
        .legend { display:flex; gap:0.9rem; flex-wrap:wrap; }
        .leg { display:flex; align-items:center; gap:0.35rem; font-size:0.78rem; color:var(--ink-soft); }
        .sw { width:11px; height:11px; border-radius:3px; }
        .cap { font-size:0.9rem; color:var(--ink-soft); margin:0 0 1rem; }
        .rows { display:flex; flex-direction:column; gap:0.55rem; }
        .row { display:grid; grid-template-columns: 92px 1fr 52px; gap:0.8rem; align-items:center; }
        .name { text-transform:capitalize; color:var(--ink); font-size:0.92rem; }
        .track { background:var(--bg-inset); border-radius:6px; padding:3px; }
        .stack { display:flex; height:24px; border-radius:4px; overflow:hidden; min-width:14px; }
        .seg { display:flex; align-items:center; justify-content:center; color:#04121a; font-size:0.72rem; font-weight:700; min-width:10px; }
        .eff { text-align:right; font-family:var(--font-mono); font-size:0.82rem; color:var(--cyan); }
        .foot { margin-top:0.9rem; font-size:0.8rem; color:var(--ink-faint); }
        @media (max-width:560px){ .row{ grid-template-columns: 70px 1fr 44px; } }
      `}</style>
    </div>
  );
}

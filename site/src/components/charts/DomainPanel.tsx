import { useState } from 'preact/hooks';
import Toggle from '../interactive/Toggle';
import { domains } from '../../data';
import { MODEL_LABELS, type ModelKey } from '../../utils/theme';
import { fmtPct } from '../../utils/math';

/**
 * Scene 9 — different thoughts live in different layers. Per-domain efficiency
 * plus where the top features sit (early / mid / late thirds of the network).
 */
const LAYER_COLORS = { early: '#22d3ee', mid: '#a855f7', late: '#f59e0b' };

export default function DomainPanel() {
  const [model, setModel] = useState<ModelKey>('gemma');
  const data = domains(model);

  return (
    <div class="dom" data-testid="domain-panel">
      <div class="controls">
        <Toggle ariaLabel="Model" testid="dom-model" value={model} onChange={(v) => setModel(v as ModelKey)}
          options={[{ value: 'gemma', label: MODEL_LABELS.gemma }, { value: 'llama', label: MODEL_LABELS.llama }]} />
        <div class="legend">
          {Object.entries(LAYER_COLORS).map(([k, c]) => (
            <span class="leg"><span class="sw" style={`background:${c}`} />{k}</span>
          ))}
        </div>
      </div>

      <p class="cap">Knowledge domains — {MODEL_LABELS[model]}. Left: bounded oracle efficiency. Right: layer location of the top features.</p>

      <div class="rows">
        {data.map((d) => {
          const tot = d.layers.early + d.layers.mid + d.layers.late || 1;
          return (
            <div class="row" data-testid={`dom-row-${d.name}`}>
              <div class="name">{d.name}</div>
              <div class="effbar">
                <div class="fill" style={`width:${d.efficiency}%`} data-testid={`dom-eff-${d.name}`} data-eff={d.efficiency.toFixed(1)} />
                <span class="efflabel">{fmtPct(d.efficiency)}</span>
              </div>
              <div class="layers" title={`early ${d.layers.early} · mid ${d.layers.mid} · late ${d.layers.late}`}>
                {(['early', 'mid', 'late'] as const).map((seg) => (
                  d.layers[seg] > 0 && (
                    <span class="seg" style={`width:${(d.layers[seg] / tot) * 100}%;background:${LAYER_COLORS[seg]}`} />
                  )
                ))}
              </div>
            </div>
          );
        })}
      </div>

      <style>{`
        .dom { margin: 1.4rem 0; }
        .controls { display:flex; gap:1rem; align-items:center; flex-wrap:wrap; margin-bottom:0.7rem; }
        .legend { display:flex; gap:0.9rem; }
        .leg { display:flex; align-items:center; gap:0.35rem; font-size:0.78rem; color:var(--ink-faint); text-transform:capitalize; }
        .sw { width:10px; height:10px; border-radius:3px; }
        .cap { font-size:0.9rem; color:var(--ink-faint); margin:0 0 0.8rem; }
        .rows { display:flex; flex-direction:column; gap:0.6rem; }
        .row { display:grid; grid-template-columns: 90px 1fr 120px; gap:0.8rem; align-items:center; }
        .name { text-transform:capitalize; color:var(--ink-soft); font-size:0.9rem; }
        .effbar { position:relative; background:var(--bg-inset); border-radius:6px; height:22px; overflow:hidden; }
        .effbar .fill { height:100%; background:linear-gradient(90deg, var(--cyan), var(--violet)); border-radius:6px; }
        .efflabel { position:absolute; right:8px; top:50%; transform:translateY(-50%); font-size:0.74rem; color:var(--ink); font-family:var(--font-mono); }
        .layers { display:flex; height:14px; border-radius:4px; overflow:hidden; background:var(--bg-inset); }
        .seg { height:100%; }
        @media (max-width:560px){ .row{ grid-template-columns: 70px 1fr; } .layers{ grid-column: 1 / -1; } }
      `}</style>
    </div>
  );
}

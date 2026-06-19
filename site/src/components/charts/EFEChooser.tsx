import { useState } from 'preact/hooks';
import { ACTION_COLORS } from '../../utils/theme';

/**
 * Interactive Expected-Free-Energy explainer. A "belief uncertainty" slider
 * shows how the agent's preferred intervention shifts from exploration
 * (ablation, max information) to exploitation (steering) as it grows confident
 * — the emergent explore→exploit signature the paper actually observes.
 */
const ACTIONS = [
  { key: 'ablation', label: 'Ablation', epistemic: 1.0, pragmatic: 0.35, blurb: 'switch the feature off — most informative' },
  { key: 'activation_patching', label: 'Patching', epistemic: 0.55, pragmatic: 0.6, blurb: 'swap in a reference — a middle probe' },
  { key: 'feature_steering', label: 'Steering', epistemic: 0.2, pragmatic: 1.0, blurb: 'amplify it — confirm what you believe' },
];

export default function EFEChooser() {
  const [u, setU] = useState(0.9); // belief uncertainty 0..1

  // Expected Free Energy (lower is better) ≈ -(epistemic·uncertainty + pragmatic·(1-uncertainty)).
  const scored = ACTIONS.map((a) => {
    const value = a.epistemic * u + a.pragmatic * (1 - u);
    return { ...a, value };
  });
  const best = scored.reduce((m, a) => (a.value > m.value ? a : m), scored[0]);
  const maxV = Math.max(...scored.map((a) => a.value));

  return (
    <div class="efe" data-testid="efe-chooser">
      <label class="u-row">
        <span class="u-label">Belief uncertainty</span>
        <input type="range" min="0" max="1" step="0.01" value={u} data-testid="efe-uncertainty"
               aria-label="Belief uncertainty"
               onInput={(e) => setU(Number((e.target as HTMLInputElement).value))} />
        <span class="u-val mono">{u > 0.66 ? 'high' : u > 0.33 ? 'medium' : 'low'}</span>
      </label>

      <div class="cards">
        {scored.map((a) => {
          const win = a.key === best.key;
          return (
            <div class={win ? 'card win' : 'card'} data-testid={`efe-card-${a.key}`} data-win={win}>
              <div class="c-top">
                <span class="dot" style={`background:${ACTION_COLORS[a.key]}`} />
                <span class="c-name">{a.label}</span>
                {win && <span class="pick">agent picks ✓</span>}
              </div>
              <p class="c-blurb">{a.blurb}</p>
              <div class="bars">
                <Bar label="info gain" v={a.epistemic} color="var(--cyan)" />
                <Bar label="goal value" v={a.pragmatic} color="var(--violet)" />
              </div>
              <div class="score">
                <span>combined value</span>
                <div class="track"><div class="fill" style={`width:${(a.value / maxV) * 100}%;background:${ACTION_COLORS[a.key]}`} /></div>
              </div>
            </div>
          );
        })}
      </div>

      <p class="efe-cap" data-testid="efe-caption">
        {u > 0.66
          ? 'Uncertain → information wins. The agent explores by ablating — exactly what it does at step 0 on every prompt.'
          : u < 0.34
          ? 'Confident → goal value wins. The agent exploits by steering, just as its real trajectory does once entropy drops.'
          : 'In between, patching becomes competitive — a confirmatory middle probe.'}
      </p>

      <style>{`
        .efe { margin: 1.4rem 0; }
        .u-row { display:flex; align-items:center; gap:0.8rem; margin-bottom:1rem; }
        .u-label { font-size:0.82rem; color:var(--ink-soft); white-space:nowrap; }
        .u-row input { flex:1; accent-color:var(--cyan); }
        .u-val { font-size:0.82rem; color:var(--cyan); width:4rem; }
        .cards { display:grid; grid-template-columns:repeat(3,1fr); gap:0.8rem; }
        .card { border:1px solid var(--hairline); border-radius:var(--radius); background:var(--bg-panel);
          padding:0.9rem; transition:border-color 0.2s ease, transform 0.2s ease; }
        .card.win { border-color:var(--cyan); box-shadow:var(--glow-cyan); transform:translateY(-2px); }
        .c-top { display:flex; align-items:center; gap:0.45rem; }
        .dot { width:10px; height:10px; border-radius:3px; }
        .c-name { font-weight:650; color:var(--ink); }
        .pick { margin-left:auto; font-size:0.68rem; color:var(--cyan); font-family:var(--font-mono); }
        .c-blurb { font-size:0.78rem; color:var(--ink-faint); margin:0.4rem 0 0.7rem; min-height:2.4em; }
        .bars { display:flex; flex-direction:column; gap:0.35rem; }
        .mono { font-family:var(--font-mono); }
        .bars :global(.b) { display:flex; align-items:center; gap:0.4rem; font-size:0.7rem; color:var(--ink-soft); }
        .bars :global(.b .bt) { flex:1; height:6px; background:var(--bg-inset); border-radius:3px; overflow:hidden; }
        .bars :global(.b .bf) { height:100%; border-radius:3px; transition:width 0.25s ease; }
        .bars :global(.b .bl) { width:4.5rem; }
        .score { margin-top:0.7rem; font-size:0.7rem; color:var(--ink-soft); }
        .score .track { height:7px; background:var(--bg-inset); border-radius:4px; overflow:hidden; margin-top:3px; }
        .score .fill { height:100%; transition:width 0.25s ease; }
        .efe-cap { margin-top:1rem; font-size:0.92rem; color:var(--ink); border-top:1px solid var(--hairline); padding-top:0.8rem; }
        @media (max-width:680px){ .cards{ grid-template-columns:1fr; } }
        @media (prefers-reduced-motion: reduce){ .card,.bf,.fill{ transition:none !important; } }
      `}</style>
    </div>
  );
}

function Bar({ label, v, color }: { label: string; v: number; color: string }) {
  return (
    <div class="b">
      <span class="bl">{label}</span>
      <span class="bt"><span class="bf" style={`width:${v * 100}%;background:${color}`} /></span>
    </div>
  );
}

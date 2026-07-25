/* Part II diagrams: the perception-action loop mapped onto circuit discovery. */

const G = 'var(--grotesk)';

export function LoopSVG() {
  return (
    <svg viewBox="0 0 900 430" className="chart-svg" style={{ width: '100%', height: 'auto', color: 'var(--cream)' }} role="img"
      aria-label="Active inference loop mapped onto circuit discovery">
      <defs>
        <marker id="lp-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#4fd8ce" />
        </marker>
        <marker id="lp-v" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#b49bf0" />
        </marker>
      </defs>

      {/* Agent box */}
      <rect x={40} y={40} width={380} height={350} rx={16} fill="rgba(79,216,206,.05)" stroke="#4fd8ce" strokeWidth={1.4} />
      <text x={230} y={70} fontSize={14} fontFamily={G} fontWeight={700} fill="#4fd8ce" textAnchor="middle">POMDP AGENT (pymdp)</text>

      <rect x={70} y={90} width={320} height={64} rx={10} fill="#10151f" stroke="#1e2836" />
      <text x={230} y={116} fontSize={12.5} fontFamily={G} fontWeight={600} fill="#ede8dc" textAnchor="middle">beliefs q(s₀, s₁, s₂) per candidate</text>
      <text x={230} y={136} fontSize={11} fontFamily={G} fill="#b9b4a6" textAnchor="middle">importance (4) × layer role (3) × causal influence (3)</text>

      <rect x={70} y={172} width={320} height={64} rx={10} fill="#10151f" stroke="#1e2836" />
      <text x={230} y={198} fontSize={12.5} fontFamily={G} fontWeight={600} fill="#ede8dc" textAnchor="middle">EFE over joint (feature i, action u)</text>
      <text x={230} y={218} fontSize={11} fontFamily={G} fill="#b9b4a6" textAnchor="middle">epistemic (state + novelty) + pragmatic (C)</text>

      <rect x={70} y={254} width={320} height={58} rx={10} fill="#10151f" stroke="#1e2836" />
      <text x={230} y={278} fontSize={12.5} fontFamily={G} fontWeight={600} fill="#ede8dc" textAnchor="middle">softmax(−γ·G), γ = 16 → (i*, u*)</text>
      <text x={230} y={297} fontSize={11} fontFamily={G} fill="#b9b4a6" textAnchor="middle">stochastic action selection</text>

      <rect x={70} y={328} width={320} height={46} rx={10} fill="rgba(180,155,240,.08)" stroke="#b49bf0" />
      <text x={230} y={347} fontSize={12} fontFamily={G} fontWeight={600} fill="#b49bf0" textAnchor="middle">learn: pA ← pA + η·(o ⊗ q(s))</text>
      <text x={230} y={364} fontSize={10.5} fontFamily={G} fill="#b9b4a6" textAnchor="middle">Dirichlet update of the likelihood (novelty-bearing)</text>

      {/* Environment box */}
      <rect x={520} y={40} width={340} height={350} rx={16} fill="rgba(240,162,75,.05)" stroke="#f0a24b" strokeWidth={1.4} />
      <text x={690} y={70} fontSize={14} fontFamily={G} fontWeight={700} fill="#f0a24b" textAnchor="middle">ENVIRONMENT = THE TRANSFORMER</text>

      <rect x={550} y={95} width={280} height={70} rx={10} fill="#10151f" stroke="#1e2836" />
      <text x={690} y={121} fontSize={12.5} fontFamily={G} fontWeight={600} fill="#ede8dc" textAnchor="middle">Gemma-2-2B / Llama-3.2-1B</text>
      <text x={690} y={141} fontSize={11} fontFamily={G} fill="#b9b4a6" textAnchor="middle">+ transcoders + attribution graph (hidden circuit)</text>

      <rect x={550} y={190} width={280} height={62} rx={10} fill="#10151f" stroke="#1e2836" />
      <text x={690} y={214} fontSize={12.5} fontFamily={G} fontWeight={600} fill="#ede8dc" textAnchor="middle">feature_intervention(l, p, f, v)</text>
      <text x={690} y={234} fontSize={11} fontFamily={G} fill="#b9b4a6" textAnchor="middle">ablate · patch · steer — real causal probe</text>

      <rect x={550} y={278} width={280} height={78} rx={10} fill="#10151f" stroke="#1e2836" />
      <text x={690} y={302} fontSize={12.5} fontFamily={G} fontWeight={600} fill="#ede8dc" textAnchor="middle">observation o = (KL bin, act bin, degree bin)</text>
      <text x={690} y={322} fontSize={11} fontFamily={G} fill="#b9b4a6" textAnchor="middle">KL(clean ‖ intervened) discretised at</text>
      <text x={690} y={340} fontSize={11} fontFamily="var(--mono)" fill="#b9b4a6" textAnchor="middle">10⁻⁴ / 10⁻³ / 10⁻²</text>

      {/* arrows */}
      <path d="M420 283 C 470 283, 480 221, 548 221" fill="none" stroke="#4fd8ce" strokeWidth={2.4} markerEnd="url(#lp-a)" />
      <text x={470} y={238} fontSize={11.5} fontFamily={G} fill="#4fd8ce">action u*</text>
      <path d="M550 317 C 470 317, 480 122, 422 122" fill="none" stroke="#b49bf0" strokeWidth={2.4} markerEnd="url(#lp-v)" />
      <text x={452} y={110} fontSize={11.5} fontFamily={G} fill="#b49bf0">observation o</text>
    </svg>
  );
}

export function DictionarySVG() {
  /* The mapping table: active inference concept ↔ circuit discovery realisation */
  const rows: [string, string][] = [
    ['hidden state s', 'a feature’s (importance, layer role, causal influence)'],
    ['observation o', 'discretised (KL, activation, connectivity) after a probe'],
    ['action u', 'which intervention type to run: ablate / patch / steer'],
    ['policy π (len 1)', 'the joint choice (candidate feature, intervention)'],
    ['likelihood A', 'how importance maps to observable KL — learned online'],
    ['transition B(u)', 'how an intervention revises importance beliefs'],
    ['preferences C', 'wanting to see large causal effects (high KL bins)'],
    ['prior D', 'sparsity: most features are unimportant'],
    ['free energy F', 'belief updating after each real intervention'],
    ['expected free energy G', 'the value of the next causal experiment'],
  ];
  return (
    <div className="tbl-wrap">
      <table className="tbl">
        <thead>
          <tr><th>Active inference object</th><th>Circuit-discovery realisation in ACD</th></tr>
        </thead>
        <tbody>
          {rows.map(([a, b]) => (
            <tr key={a}>
              <td className="mono" style={{ whiteSpace: 'nowrap' }}>{a}</td>
              <td>{b}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

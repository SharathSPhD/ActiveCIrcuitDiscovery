'use client';

import { useMemo, useState } from 'react';
import gm from '../data/genmodel.json';

/* Interactive A/B/C/D inspector for the exact generative model shipped with the paper
   (results/generative_model.json → data/genmodel.json). */

const IMP = ['negligible', 'low', 'moderate', 'high'];
const ROLE = ['early', 'middle', 'late'];
const CAUSAL = ['weak', 'moderate', 'strong'];
const KL = ['<1e-4', '<1e-3', '<1e-2', '≥1e-2'];
const ACT = ['<0.5', '<5', '<50', '≥50'];
const CONN = ['sparse', 'moderate', 'dense'];
const ACTIONS = ['ablation', 'activation patching', 'feature steering'];
const MODALITIES = [
  { name: 'KL magnitude', labels: KL },
  { name: 'activation magnitude', labels: ACT },
  { name: 'graph connectivity', labels: CONN },
];

function heat(v: number, max: number, dark: boolean) {
  const t = max > 0 ? v / max : 0;
  return dark
    ? `rgba(79, 216, 206, ${0.06 + 0.72 * t})`
    : `rgba(14, 124, 134, ${0.05 + 0.7 * t})`;
}

function Matrix({
  data,
  rows,
  cols,
  rowTitle,
  colTitle,
  dark = true,
}: {
  data: number[][];
  rows: string[];
  cols: string[];
  rowTitle: string;
  colTitle: string;
  dark?: boolean;
}) {
  const max = Math.max(...data.flat());
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', fontFamily: 'var(--mono)', fontSize: '0.72rem' }}>
        <thead>
          <tr>
            <th style={{ padding: '4px 8px', textAlign: 'left', fontFamily: 'var(--grotesk)', fontWeight: 500, opacity: 0.7 }}>
              {rowTitle} ↓ · {colTitle} →
            </th>
            {cols.map((c) => (
              <th key={c} style={{ padding: '4px 8px', fontFamily: 'var(--grotesk)', fontWeight: 500, opacity: 0.8, whiteSpace: 'nowrap' }}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i}>
              <td style={{ padding: '4px 8px', fontFamily: 'var(--grotesk)', opacity: 0.8, whiteSpace: 'nowrap' }}>{rows[i]}</td>
              {row.map((v, j) => (
                <td
                  key={j}
                  style={{
                    padding: '5px 10px',
                    textAlign: 'center',
                    background: heat(v, max, dark),
                    border: dark ? '1px solid rgba(30,40,54,.9)' : '1px solid rgba(27,26,22,.12)',
                    minWidth: 52,
                  }}
                >
                  {v.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function GenModelInspector() {
  const [tab, setTab] = useState<'A' | 'B' | 'C' | 'D'>('B');
  const [modality, setModality] = useState(0);
  const [role, setRole] = useState(2);
  const [causal, setCausal] = useState(2);
  const [action, setAction] = useState(0);

  const A = gm.A as unknown as number[][][][][];
  const B0 = gm.B[0] as unknown as number[][][];
  const C = gm.C as unknown as number[][];
  const D = gm.D as unknown as number[][];

  // A[modality][obs][imp][role][causal] — slice at (role, causal): matrix obs × importance
  const aSlice = useMemo(() => {
    const mod = A[modality];
    const nObs = mod.length;
    const out: number[][] = [];
    for (let o = 0; o < nObs; o++) {
      const row: number[] = [];
      for (let s = 0; s < 4; s++) row.push((mod[o][s] as unknown as number[][])[role][causal]);
      out.push(row);
    }
    return out;
  }, [A, modality, role, causal]);

  // B0[s'][s][action]
  const bSlice = useMemo(() => B0.map((row) => row.map((cell) => cell[action])), [B0, action]);

  const btn = (active: boolean): React.CSSProperties => ({
    fontFamily: 'var(--grotesk)',
    fontSize: '0.76rem',
    fontWeight: 600,
    padding: '5px 12px',
    borderRadius: 999,
    cursor: 'pointer',
    border: `1px solid ${active ? 'var(--teal-bright)' : 'var(--navy-hairline)'}`,
    background: active ? 'rgba(79,216,206,.14)' : 'transparent',
    color: active ? 'var(--teal-bright)' : 'var(--cream-soft)',
  });

  return (
    <div style={{ border: '1px solid var(--navy-hairline)', borderRadius: 14, padding: '1.1rem 1.2rem', background: 'var(--navy-panel)' }}>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 14 }}>
        {(['A', 'B', 'C', 'D'] as const).map((t) => (
          <button key={t} style={btn(tab === t)} onClick={() => setTab(t)}>
            {t} — {{ A: 'likelihood', B: 'transitions', C: 'preferences', D: 'priors' }[t]}
          </button>
        ))}
      </div>

      {tab === 'A' && (
        <>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
            {MODALITIES.map((m, i) => (
              <button key={m.name} style={btn(modality === i)} onClick={() => setModality(i)}>
                o{i}: {m.name}
              </button>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12, alignItems: 'center' }}>
            <span style={{ fontFamily: 'var(--grotesk)', fontSize: '.75rem', opacity: 0.7 }}>slice at role =</span>
            {ROLE.map((r, i) => (
              <button key={r} style={btn(role === i)} onClick={() => setRole(i)}>{r}</button>
            ))}
            <span style={{ fontFamily: 'var(--grotesk)', fontSize: '.75rem', opacity: 0.7 }}>causal =</span>
            {CAUSAL.map((c, i) => (
              <button key={c} style={btn(causal === i)} onClick={() => setCausal(i)}>{c}</button>
            ))}
          </div>
          <Matrix data={aSlice} rows={MODALITIES[modality].labels} cols={IMP} rowTitle="observation" colTitle="importance s₀" />
          <p style={{ fontFamily: 'var(--grotesk)', fontSize: '.78rem', opacity: 0.75, marginBottom: 0 }}>
            P(o{modality} | s₀, s₁={ROLE[role]}, s₂={CAUSAL[causal]}) as shipped after learning started from the
            calibrated prior. This is the matrix the Dirichlet updates sculpt online (pA = 10·A, η = 1).
          </p>
        </>
      )}

      {tab === 'B' && (
        <>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
            {ACTIONS.map((a, i) => (
              <button key={a} style={btn(action === i)} onClick={() => setAction(i)}>u = {a}</button>
            ))}
          </div>
          <Matrix data={bSlice} rows={IMP.map((s) => `${s}′`)} cols={IMP} rowTitle="s₀′" colTitle="s₀" />
          <p style={{ fontFamily: 'var(--grotesk)', fontSize: '.78rem', opacity: 0.75, marginBottom: 0 }}>
            Factor-0 transitions under {ACTIONS[action]}: diagonal {action === 0 ? '0.50 — broadest, highest entropy (exploratory)' : action === 1 ? '0.70 — intermediate (confirmatory)' : '0.90 — near-identity, lowest entropy (belief-preserving)'}.
            The entropy ordering H(B_abl) &gt; H(B_patch) &gt; H(B_steer) is the design commitment; the exact
            fractions are round numbers. Factors 1–2 (role, causal) are identity across actions.
          </p>
        </>
      )}

      {tab === 'C' && (
        <>
          {C.map((c, m) => (
            <div key={m} style={{ marginBottom: 10 }}>
              <div style={{ fontFamily: 'var(--grotesk)', fontSize: '.78rem', opacity: 0.8, marginBottom: 4 }}>
                C{m} — {MODALITIES[m].name}
              </div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {c.map((v, i) => (
                  <div key={i} style={{
                    fontFamily: 'var(--mono)', fontSize: '.72rem', padding: '6px 10px', borderRadius: 8,
                    background: heat(Math.max(v, 0), Math.max(...c.map((x) => Math.max(x, 0)), 0.01), true),
                    border: '1px solid var(--navy-hairline)',
                  }}>
                    {MODALITIES[m].labels[i]}: {v.toFixed(2)}
                  </div>
                ))}
              </div>
            </div>
          ))}
          <p style={{ fontFamily: 'var(--grotesk)', fontSize: '.78rem', opacity: 0.75, marginBottom: 0 }}>
            Log-preferences over observations, per modality: monotone in KL and activation — the agent
            “prefers to observe” large causal effects. This is where the discovery goal enters, and where
            reward-relabelling questions aim (Part V, Q2).
          </p>
        </>
      )}

      {tab === 'D' && (
        <>
          {D.map((d, f) => (
            <div key={f} style={{ marginBottom: 10 }}>
              <div style={{ fontFamily: 'var(--grotesk)', fontSize: '.78rem', opacity: 0.8, marginBottom: 4 }}>
                D{f} — {['importance', 'layer role', 'causal influence'][f]}
              </div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {d.map((v, i) => (
                  <div key={i} style={{
                    fontFamily: 'var(--mono)', fontSize: '.72rem', padding: '6px 10px', borderRadius: 8,
                    background: heat(v, Math.max(...d), true),
                    border: '1px solid var(--navy-hairline)',
                  }}>
                    {[IMP, ROLE, CAUSAL][f][i]}: {v.toFixed(2)}
                  </div>
                ))}
              </div>
            </div>
          ))}
          <p style={{ fontFamily: 'var(--grotesk)', fontSize: '.78rem', opacity: 0.75, marginBottom: 0 }}>
            Initial state priors: importance biased low (most features are not causally critical — a
            sparsity prior), layer role set from the feature&rsquo;s actual depth, causal influence uniform.
          </p>
        </>
      )}
    </div>
  );
}

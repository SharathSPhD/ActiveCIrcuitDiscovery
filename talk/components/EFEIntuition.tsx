'use client';

/* Interactive intuition pump for the per-step EFE.
   Two sliders describe one candidate feature; the widget shows how the balance of
   epistemic and pragmatic value picks an intervention. Toy numbers, honest shape. */

import { useState } from 'react';

const ACTIONS = [
  // name, info-gain capacity (from B entropy), effect amplification (observed KL scale), color
  { name: 'ablate', info: 1.0, amp: 0.55, color: '#f0a24b', note: 'wide transitions — learns the most' },
  { name: 'patch', info: 0.6, amp: 0.45, color: '#9fb6e8', note: 'moderate on both counts' },
  { name: 'steer', info: 0.25, amp: 1.0, color: '#b49bf0', note: 'narrow transitions — but big effects' },
];

export default function EFEIntuition() {
  const [unc, setUnc] = useState(0.85); // belief uncertainty about the feature
  const [eff, setEff] = useState(0.35); // expected effect size if probed

  const rows = ACTIONS.map((a) => {
    const epistemic = unc * a.info;
    const pragmatic = eff * a.amp;
    return { ...a, epistemic, pragmatic, total: epistemic + pragmatic };
  });
  const best = rows.reduce((m, r) => (r.total > m.total ? r : m), rows[0]);

  const sl = (label: string, v: number, set: (x: number) => void, lo: string, hi: string) => (
    <label style={{ display: 'block', fontFamily: 'var(--grotesk)', fontSize: '.85rem', marginBottom: '1rem' }}>
      <span style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <strong>{label}</strong>
        <span style={{ fontFamily: 'var(--mono)', opacity: 0.75 }}>{v.toFixed(2)}</span>
      </span>
      <input
        type="range" min={0} max={1} step={0.01} value={v}
        onChange={(e) => set(Number(e.target.value))}
        style={{ width: '100%' }}
      />
      <span style={{ display: 'flex', justifyContent: 'space-between', fontSize: '.68rem', opacity: 0.6 }}>
        <span>{lo}</span><span>{hi}</span>
      </span>
    </label>
  );

  return (
    <div style={{
      display: 'grid', gridTemplateColumns: 'minmax(240px, 1fr) 2fr', gap: '1.6rem',
      border: '1px solid var(--navy-hairline)', borderRadius: 14, background: 'var(--navy-panel)',
      padding: '1.2rem 1.4rem', alignItems: 'center',
    }}>
      <div>
        <div style={{ fontFamily: 'var(--grotesk)', fontSize: '.7rem', letterSpacing: '.14em', textTransform: 'uppercase', color: 'var(--teal-bright)', marginBottom: 10 }}>
          describe one candidate feature
        </div>
        {sl('How uncertain are the beliefs about it?', unc, setUnc, 'well understood', 'no idea yet')}
        {sl('How large an effect is expected?', eff, setEff, 'probably minor', 'probably load-bearing')}
        <p style={{ fontFamily: 'var(--grotesk)', fontSize: '.72rem', opacity: 0.6, margin: 0 }}>
          Toy numbers for intuition — the shape, not the paper&rsquo;s exact arithmetic. Try: drag
          uncertainty down as if evidence has come in, and watch the chosen action flip.
        </p>
      </div>
      <div>
        {rows.map((r) => (
          <div key={r.name} style={{ marginBottom: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--grotesk)', fontSize: '.8rem', marginBottom: 3 }}>
              <span style={{ color: r.color, fontWeight: 700 }}>
                {r.name}{best.name === r.name ? '  ← chosen' : ''}
              </span>
              <span style={{ opacity: 0.6, fontSize: '.7rem' }}>{r.note}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{ flex: 1, display: 'flex', height: 20, borderRadius: 6, overflow: 'hidden', background: 'rgba(237,232,220,.07)', outline: best.name === r.name ? `2px solid ${r.color}` : 'none' }}>
                <div style={{ width: `${r.epistemic * 48}%`, background: '#4fd8ce', transition: 'width .2s ease' }}
                  title={`epistemic ${r.epistemic.toFixed(2)}`} />
                <div style={{ width: `${r.pragmatic * 48}%`, background: '#f0a24b', transition: 'width .2s ease' }}
                  title={`pragmatic ${r.pragmatic.toFixed(2)}`} />
              </div>
              <span style={{ fontFamily: 'var(--mono)', fontSize: '.68rem', opacity: 0.8, width: 86, whiteSpace: 'nowrap' }}>
                {r.epistemic.toFixed(2)} + {r.pragmatic.toFixed(2)}
              </span>
            </div>
          </div>
        ))}
        <div style={{ display: 'flex', gap: '1.2rem', fontFamily: 'var(--grotesk)', fontSize: '.72rem', opacity: 0.85, flexWrap: 'wrap' }}>
          <span><span style={{ display: 'inline-block', width: 12, height: 8, background: '#4fd8ce', borderRadius: 2, marginRight: 5 }} />epistemic value — what the probe would teach</span>
          <span><span style={{ display: 'inline-block', width: 12, height: 8, background: '#f0a24b', borderRadius: 2, marginRight: 5 }} />pragmatic value — the effects it would find</span>
        </div>
        <p style={{ fontFamily: 'var(--grotesk)', fontSize: '.68rem', opacity: 0.55, margin: '0.6rem 0 0' }}>
          Bars show value = −G, so the tallest bar is the lowest G — the argmin from the previous
          slide. The epistemic axis follows each lever&rsquo;s B-matrix entropy, as in the shipped
          model; the effect axis is illustrative shorthand for the observed outcome statistics
          (in the shipped model the levers are differentiated epistemically, through B).
        </p>
      </div>
    </div>
  );
}

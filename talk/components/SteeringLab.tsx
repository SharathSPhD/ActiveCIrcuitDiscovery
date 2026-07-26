'use client';

/* Live steering laboratory — the Golden Gate experiment, runnable.
   LIVE: POST /api/dgx/steer {prompt, feature_rank, multipliers} → dose sweep on a real feature.
   OFFLINE: shows the paper's recorded dose-response (SteeringChart) instead. */

import { useEffect, useState } from 'react';
import { SteeringChart } from './Charts';

const PROMPTS = [
  'The Golden Gate Bridge is located in the city of',
  'The Eiffel Tower stands in the heart of',
  'When John and Mary went to the store, John gave the bag to',
  'The capital of the state containing Dallas is',
];
const MULTS = [0, 1.5, 2, 3, 5, 10];

type SweepPoint = { mult: number; kl: number; top_tokens: [string, number][]; top1_changed: boolean };

export default function SteeringLab() {
  const [live, setLive] = useState<boolean | null>(null);
  const [prompt, setPrompt] = useState(PROMPTS[0]);
  const [rank, setRank] = useState(0);
  const [running, setRunning] = useState(false);
  const [feature, setFeature] = useState<any>(null);
  const [sweep, setSweep] = useState<SweepPoint[]>([]);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch('/api/dgx/health')
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((h) => setLive(h.status === 'ok'))
      .catch(() => setLive(false));
  }, []);

  const run = async () => {
    setRunning(true); setError(''); setSweep([]); setFeature(null);
    try {
      const res = await fetch('/api/dgx/steer', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ prompt, feature_rank: rank, multipliers: MULTS }),
      });
      if (!res.ok) throw new Error(`backend ${res.status}`);
      const data = await res.json();
      setFeature(data.feature);
      setSweep(data.sweep);
    } catch (e: any) {
      setError(`live steering unavailable (${e?.message ?? e})`);
      setLive(false);
    }
    setRunning(false);
  };

  const maxKl = Math.max(...sweep.map((s) => s.kl), 1e-9);
  const panel: React.CSSProperties = {
    border: '1px solid var(--navy-hairline)', borderRadius: 14,
    background: 'var(--navy-panel)', padding: '1rem 1.1rem',
  };

  if (live === false && sweep.length === 0) {
    return (
      <div style={panel}>
        <div style={{ fontFamily: 'var(--grotesk)', fontSize: '.7rem', letterSpacing: '.12em', textTransform: 'uppercase', opacity: 0.7, marginBottom: 6 }}>
          steering laboratory · offline — showing the paper&rsquo;s recorded dose-response
        </div>
        <p style={{ fontFamily: 'var(--grotesk)', fontSize: '.8rem', opacity: 0.75, marginTop: 0 }}>
          When the GPU is reachable this panel runs the Golden-Gate experiment live: pick a
          prompt and a circuit-ranked feature, and sweep its activation from 0× to 10×. Offline,
          the recorded result stands in — circuit-selected features (teal) vs a random-feature
          control (grey), top-1 prediction changes out of 50 at each dose.
        </p>
        <SteeringChart />
      </div>
    );
  }

  return (
    <div style={{ display: 'grid', gap: '1rem' }}>
      <div style={{ ...panel, display: 'flex', flexWrap: 'wrap', gap: '0.8rem', alignItems: 'center' }}>
        <span style={{ fontFamily: 'var(--grotesk)', fontSize: '.7rem', letterSpacing: '.12em', textTransform: 'uppercase', opacity: 0.7, flexBasis: '100%' }}>
          steering laboratory — sweep one real feature from 0× to 10× of its activation
        </span>
        <select
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          disabled={running}
          style={{ fontFamily: 'var(--grotesk)', fontSize: '.8rem', padding: '7px 10px', borderRadius: 8, background: 'var(--navy-2)', color: 'var(--cream)', border: '1px solid var(--navy-hairline)', maxWidth: '100%', flex: '1 1 320px' }}
        >
          {PROMPTS.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
        <label style={{ fontFamily: 'var(--grotesk)', fontSize: '.75rem', opacity: 0.85, display: 'flex', alignItems: 'center', gap: 8 }}>
          feature rank
          <input type="range" min={0} max={9} step={1} value={rank} disabled={running}
            onChange={(e) => setRank(Number(e.target.value))} />
          <span style={{ fontFamily: 'var(--mono)' }}>#{rank + 1}</span>
        </label>
        <button
          onClick={run}
          disabled={running || live === null}
          style={{ fontFamily: 'var(--grotesk)', fontWeight: 700, fontSize: '.82rem', padding: '9px 20px', borderRadius: 999, cursor: 'pointer', border: 'none', background: 'var(--violet)', color: '#fff', opacity: running ? 0.6 : 1 }}
        >
          {running ? 'Sweeping…' : 'Run dose sweep'}
        </button>
        {error && <span style={{ fontFamily: 'var(--grotesk)', fontSize: '.72rem', color: 'var(--amber)', flexBasis: '100%' }}>{error} — recorded dose-response shown below.</span>}
      </div>

      {feature && (
        <div style={panel}>
          <div style={{ fontFamily: 'var(--grotesk)', fontSize: '.7rem', letterSpacing: '.12em', textTransform: 'uppercase', opacity: 0.7, marginBottom: 8 }}>
            feature under the dial · <span style={{ color: 'var(--teal-bright)' }}>{feature.fid}</span> · layer {feature.layer} · graph importance {feature.imp?.toFixed?.(2)}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0.7rem' }}>
            {sweep.map((s) => (
              <div key={s.mult} style={{ border: '1px solid var(--navy-hairline)', borderRadius: 10, padding: '0.6rem 0.7rem', background: s.top1_changed ? 'rgba(180,155,240,.1)' : 'var(--navy-2)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--grotesk)', fontSize: '.78rem', marginBottom: 5 }}>
                  <strong style={{ color: 'var(--violet-soft)' }}>{s.mult}×</strong>
                  {s.top1_changed && <span style={{ color: 'var(--violet-soft)', fontSize: '.65rem' }}>top-1 flipped</span>}
                </div>
                <div style={{ height: 8, background: 'rgba(127,127,127,.14)', borderRadius: 4, overflow: 'hidden', marginBottom: 6 }}>
                  <div style={{ width: `${(s.kl / maxKl) * 100}%`, height: '100%', background: 'var(--violet-soft)' }} />
                </div>
                <div style={{ fontFamily: 'var(--mono)', fontSize: '.62rem', opacity: 0.7, marginBottom: 4 }}>KL {s.kl >= 0.01 ? s.kl.toFixed(3) : s.kl.toExponential(1)}</div>
                {(s.top_tokens ?? []).slice(0, 3).map(([t, p]) => (
                  <div key={t} style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: '.64rem' }}>
                    <span style={{ opacity: 0.9, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '70%' }}>{JSON.stringify(t).slice(1, -1)}</span>
                    <span style={{ opacity: 0.6 }}>{(p * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
          <p style={{ fontFamily: 'var(--grotesk)', fontSize: '.72rem', opacity: 0.6, margin: '0.7rem 0 0' }}>
            Watch for: KL rising monotonically with dose, and the token list drifting toward the
            feature&rsquo;s concept — the quantified version of Golden Gate Claude, on a 2B model.
          </p>
        </div>
      )}
    </div>
  );
}

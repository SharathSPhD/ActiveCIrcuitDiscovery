'use client';

/* The live/replay demo: watch the POMDP agent spend its intervention budget.
   LIVE  — streams SSE from the DGX Spark via /api/dgx/episode.
   REPLAY — steps through the real recorded runs shipped in data/replays/. */

import { useEffect, useMemo, useRef, useState } from 'react';
import ioiGemma from '../data/replays/ioi_gemma.json';
import ioiLlama from '../data/replays/ioi_llama.json';
import msGemma from '../data/replays/multistep_gemma.json';

type Beliefs = { importance: number[]; layer_role: number[]; causal: number[] };
type StepEv = {
  step: number; fid: string; layer: number; action: string; efe: number;
  kl: number; cum_kl: number; obs_bins?: number[]; beliefs?: Beliefs;
  entropy?: number | null; converged?: boolean; step_seconds?: number;
};
type Candidate = { fid: string; layer: number; imp: number; eap?: number };

const ACTION_COLOR: Record<string, string> = {
  ablation: '#f0a24b',
  activation_patching: '#9fb6e8',
  feature_steering: '#b49bf0',
};
const ACTION_SHORT: Record<string, string> = {
  ablation: 'ablate',
  activation_patching: 'patch',
  feature_steering: 'steer',
};

const REPLAYS: { id: string; label: string; rep: any; pi: number }[] = [];
(ioiGemma as any).prompts.forEach((p: any, i: number) =>
  REPLAYS.push({ id: `g-ioi-${i}`, label: `Gemma · IOI · “${p.prompt.slice(0, 44)}…”`, rep: ioiGemma, pi: i })
);
REPLAYS.push({ id: 'l-ioi-0', label: `Llama · IOI · “${(ioiLlama as any).prompts[0].prompt.slice(0, 44)}…”`, rep: ioiLlama, pi: 0 });
REPLAYS.push({ id: 'g-ms-0', label: `Gemma · multi-step · “${(msGemma as any).prompts[0].prompt.slice(0, 40)}…”`, rep: msGemma, pi: 0 });

function fmtKL(v: number) {
  if (v === 0) return '0';
  if (v >= 0.01) return v.toFixed(3);
  return v.toExponential(1);
}

function BeliefBars({ title, labels, dist, color }: { title: string; labels: string[]; dist: number[]; color: string }) {
  return (
    <div style={{ flex: 1, minWidth: 150 }}>
      <div style={{ fontFamily: 'var(--grotesk)', fontSize: '.68rem', letterSpacing: '.1em', textTransform: 'uppercase', opacity: 0.7, marginBottom: 6 }}>{title}</div>
      {labels.map((l, i) => (
        <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
          <span style={{ fontFamily: 'var(--mono)', fontSize: '.62rem', width: 62, opacity: 0.7, textAlign: 'right' }}>{l}</span>
          <div style={{ flex: 1, height: 10, background: 'rgba(127,127,127,.14)', borderRadius: 5, overflow: 'hidden' }}>
            <div style={{ width: `${(dist?.[i] ?? 0) * 100}%`, height: '100%', background: color, transition: 'width .4s ease' }} />
          </div>
          <span style={{ fontFamily: 'var(--mono)', fontSize: '.62rem', width: 32, opacity: 0.75 }}>{((dist?.[i] ?? 0) * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );
}

export default function DemoApp() {
  const [live, setLive] = useState<boolean | null>(null); // null = probing
  const [health, setHealth] = useState<any>(null);
  const [source, setSource] = useState(REPLAYS[0].id);
  const [running, setRunning] = useState(false);
  const [steps, setSteps] = useState<StepEv[]>([]);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [oracleCum, setOracleCum] = useState<number[]>([]);
  const [eapCum, setEapCum] = useState<number[]>([]);
  const [randomish, setRandomish] = useState<number[]>([]);
  const [done, setDone] = useState<any>(null);
  const [phase, setPhase] = useState('');
  const [speed, setSpeed] = useState(900);
  const timer = useRef<any>(null);
  const abort = useRef<AbortController | null>(null);

  // Probe the DGX once on mount.
  useEffect(() => {
    fetch('/api/dgx/health')
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((h) => { setHealth(h); setLive(h.status === 'ok'); })
      .catch(() => setLive(false));
  }, []);

  const stop = () => {
    if (timer.current) clearInterval(timer.current);
    abort.current?.abort();
    setRunning(false);
  };
  useEffect(() => () => stop(), []);

  const reset = () => {
    stop();
    setSteps([]); setDone(null); setCandidates([]);
    setOracleCum([]); setEapCum([]); setRandomish([]); setPhase('');
  };

  /* ---------------- replay driver ---------------- */
  const runReplay = () => {
    reset();
    const sel = REPLAYS.find((r) => r.id === source)!;
    const p = sel.rep.prompts[sel.pi];
    setPhase(`replaying a real recorded run · ${p.n_candidates} candidates from ${p.n_features?.toLocaleString?.() ?? '—'} active features`);
    const cum = (xs: number[]) => xs.reduce<number[]>((a, v) => (a.push((a[a.length - 1] ?? 0) + v), a), []);
    const oc = cum(p.baselines.oracle); const ec = cum(p.baselines.eap); const gc = cum(p.baselines.greedy);
    const evs: StepEv[] = p.ai.kls.map((kl: number, i: number) => ({
      step: i + 1, fid: `step ${i + 1}`, layer: -1,
      action: p.ai.actions[i], efe: p.ai.efe[i], kl,
      cum_kl: cum(p.ai.kls)[i], entropy: p.ai.entropy[i], converged: p.ai.converged && i === p.ai.kls.length - 1,
    }));
    setRunning(true);
    let i = 0;
    timer.current = setInterval(() => {
      if (i >= evs.length) {
        clearInterval(timer.current);
        setRunning(false);
        setDone({ steps: evs.length, cum_kl: evs[evs.length - 1].cum_kl, converged: p.ai.converged });
        return;
      }
      const k = i;
      setSteps((s) => [...s, evs[k]]);
      setOracleCum(oc.slice(0, k + 1)); setEapCum(ec.slice(0, k + 1)); setRandomish(gc.slice(0, k + 1));
      i++;
    }, speed);
  };

  /* ---------------- live driver (SSE) ---------------- */
  const runLive = async () => {
    reset();
    const sel = REPLAYS.find((r) => r.id === source)!;
    const prompt = sel.rep.prompts[sel.pi].prompt;
    setPhase('requesting episode from the DGX Spark…');
    setRunning(true);
    abort.current = new AbortController();
    try {
      const res = await fetch('/api/dgx/episode', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ prompt, budget: 20, mode: 'multi' }),
        signal: abort.current.signal,
      });
      if (!res.ok || !res.body) throw new Error(`backend ${res.status}`);
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      for (;;) {
        const { value, done: rdone } = await reader.read();
        if (rdone) break;
        buf += dec.decode(value, { stream: true });
        const blocks = buf.split('\n\n');
        buf = blocks.pop() ?? '';
        for (const b of blocks) {
          const em = b.match(/^event: (\w+)/m);
          const dm = b.match(/^data: (.*)$/m);
          if (!em || !dm) continue;
          const payload = JSON.parse(dm[1]);
          if (em[1] === 'graph') {
            setCandidates(payload.candidates);
            setPhase(`graph ready · ${payload.n_candidates} candidates of ${payload.n_features.toLocaleString()} active features · streaming real interventions`);
          } else if (em[1] === 'step') {
            setSteps((s) => [...s, payload]);
          } else if (em[1] === 'done') {
            setDone(payload);
          }
        }
      }
    } catch (e: any) {
      if (e?.name !== 'AbortError') {
        setPhase(`live run unavailable (${e?.message ?? e}) — falling back to replay`);
        setLive(false);
        runReplay();
        return;
      }
    }
    setRunning(false);
  };

  const run = () => (live ? runLive() : runReplay());

  /* ---------------- derived ---------------- */
  const last = steps[steps.length - 1];
  const maxCum = Math.max(
    last?.cum_kl ?? 0,
    oracleCum[oracleCum.length - 1] ?? 0,
    eapCum[eapCum.length - 1] ?? 0,
    1e-9
  );
  const raceRows: [string, number, string][] = useMemo(() => {
    const rows: [string, number, string][] = [['POMDP agent', last?.cum_kl ?? 0, '#4fd8ce']];
    if (oracleCum.length) rows.push(['Ablation oracle', oracleCum[oracleCum.length - 1], '#8c94a4']);
    if (eapCum.length) rows.push(['EAP ranking', eapCum[eapCum.length - 1], '#f0a24b']);
    if (randomish.length) rows.push(['Greedy', randomish[randomish.length - 1], '#e0567a']);
    return rows.sort((a, b) => b[1] - a[1]);
  }, [last, oracleCum, eapCum, randomish]);

  const actionCounts = useMemo(() => {
    const c: Record<string, number> = {};
    steps.forEach((s) => (c[s.action] = (c[s.action] ?? 0) + 1));
    return c;
  }, [steps]);

  const layers = useMemo(() => {
    if (!candidates.length) return null;
    const maxL = Math.max(...candidates.map((c) => c.layer));
    return { maxL, probed: new Set(steps.map((s) => s.fid)) };
  }, [candidates, steps]);

  const panel: React.CSSProperties = {
    border: '1px solid var(--navy-hairline)', borderRadius: 14,
    background: 'var(--navy-panel)', padding: '1rem 1.1rem',
  };

  return (
    <div style={{ display: 'grid', gap: '1rem' }}>
      {/* control bar */}
      <div style={{ ...panel, display: 'flex', flexWrap: 'wrap', gap: '0.8rem', alignItems: 'center' }}>
        <span className={`badge-live ${live ? 'on' : 'off'}`}>
          <span className="pulse" />
          {live === null ? 'PROBING DGX…' : live ? `LIVE · ${health?.gpu ?? 'GPU'}` : 'OFFLINE · REPLAY MODE'}
        </span>
        <select
          value={source}
          onChange={(e) => { setSource(e.target.value); reset(); }}
          disabled={running}
          style={{ fontFamily: 'var(--grotesk)', fontSize: '.8rem', padding: '7px 10px', borderRadius: 8, background: 'var(--navy-2)', color: 'var(--cream)', border: '1px solid var(--navy-hairline)', maxWidth: '100%' }}
        >
          {REPLAYS.map((r) => <option key={r.id} value={r.id}>{r.label}</option>)}
        </select>
        {!live && (
          <label style={{ fontFamily: 'var(--grotesk)', fontSize: '.72rem', opacity: 0.75, display: 'flex', alignItems: 'center', gap: 6 }}>
            speed
            <input type="range" min={200} max={1600} step={100} value={1800 - speed} onChange={(e) => setSpeed(1800 - Number(e.target.value))} />
          </label>
        )}
        <button
          onClick={running ? stop : run}
          style={{ fontFamily: 'var(--grotesk)', fontWeight: 700, fontSize: '.82rem', padding: '9px 20px', borderRadius: 999, cursor: 'pointer', border: 'none', background: running ? 'var(--rose)' : 'var(--teal)', color: '#fff' }}
        >
          {running ? 'Stop' : live ? 'Run live on the DGX' : 'Replay recorded run'}
        </button>
        {phase && <span style={{ fontFamily: 'var(--grotesk)', fontSize: '.72rem', opacity: 0.7, flexBasis: '100%' }}>{phase}</span>}
      </div>

      {/* main grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
        {/* step feed */}
        <div style={{ ...panel, maxHeight: 420, overflowY: 'auto' }}>
          <div style={{ fontFamily: 'var(--grotesk)', fontSize: '.7rem', letterSpacing: '.12em', textTransform: 'uppercase', opacity: 0.7, marginBottom: 8 }}>
            intervention log
          </div>
          {steps.length === 0 && <div style={{ fontSize: '.85rem', opacity: 0.55 }}>Press run. Each row is one causal experiment chosen by EFE.</div>}
          {[...steps].reverse().map((s) => (
            <div key={s.step} style={{ display: 'flex', gap: 8, alignItems: 'baseline', padding: '5px 0', borderBottom: '1px solid rgba(30,40,54,.6)', fontSize: '.78rem', fontFamily: 'var(--mono)' }}>
              <span style={{ opacity: 0.5, width: 22 }}>{s.step}</span>
              <span style={{ color: ACTION_COLOR[s.action], width: 52, fontFamily: 'var(--grotesk)', fontWeight: 600 }}>{ACTION_SHORT[s.action] ?? s.action}</span>
              <span style={{ opacity: 0.85, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.fid}</span>
              <span style={{ color: '#4fd8ce' }}>KL {fmtKL(s.kl)}</span>
              {s.converged && <span style={{ color: 'var(--ok)', fontFamily: 'var(--grotesk)' }}>✓ converged</span>}
            </div>
          ))}
        </div>

        {/* beliefs or entropy */}
        <div style={panel}>
          <div style={{ fontFamily: 'var(--grotesk)', fontSize: '.7rem', letterSpacing: '.12em', textTransform: 'uppercase', opacity: 0.7, marginBottom: 8 }}>
            {last?.beliefs ? 'posterior beliefs (current feature)' : 'belief entropy (total, per step)'}
          </div>
          {last?.beliefs ? (
            <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
              <BeliefBars title="importance" labels={['negl.', 'low', 'mod.', 'high']} dist={last.beliefs.importance} color="#4fd8ce" />
              <BeliefBars title="layer role" labels={['early', 'middle', 'late']} dist={last.beliefs.layer_role} color="#9fb6e8" />
              <BeliefBars title="causal" labels={['weak', 'mod.', 'strong']} dist={last.beliefs.causal} color="#b49bf0" />
            </div>
          ) : (
            <svg viewBox="0 0 300 120" style={{ width: '100%', height: 'auto' }}>
              {steps.length > 1 && (() => {
                const es = steps.map((s) => s.entropy ?? 0);
                const mn = Math.min(...es), mx = Math.max(...es);
                const x = (i: number) => 8 + (i / Math.max(19, es.length - 1)) * 284;
                const y = (v: number) => 108 - ((v - mn) / (mx - mn || 1)) * 96;
                return (
                  <>
                    <path d={es.map((v, i) => `${i ? 'L' : 'M'}${x(i)},${y(v)}`).join(' ')} fill="none" stroke="#4fd8ce" strokeWidth={2} />
                    {es.map((v, i) => <circle key={i} cx={x(i)} cy={y(v)} r={2.2} fill="#4fd8ce" />)}
                    <text x={8} y={116} fontSize={8.5} fill="currentColor" opacity={0.6} fontFamily="var(--grotesk)">
                      total belief entropy — falls as the agent accumulates evidence
                    </text>
                  </>
                );
              })()}
              {steps.length <= 1 && <text x={10} y={60} fontSize={10} fill="currentColor" opacity={0.5}>waiting for steps…</text>}
            </svg>
          )}
          <div style={{ display: 'flex', gap: 10, marginTop: 12, flexWrap: 'wrap' }}>
            {Object.entries(ACTION_SHORT).map(([k, v]) => (
              <span key={k} style={{ fontFamily: 'var(--grotesk)', fontSize: '.72rem', color: ACTION_COLOR[k] }}>
                {v}: {actionCounts[k] ?? 0}
              </span>
            ))}
            {last && <span style={{ fontFamily: 'var(--mono)', fontSize: '.72rem', opacity: 0.7 }}>EFE {last.efe?.toFixed?.(2)}</span>}
          </div>
        </div>

        {/* KL race */}
        <div style={panel}>
          <div style={{ fontFamily: 'var(--grotesk)', fontSize: '.7rem', letterSpacing: '.12em', textTransform: 'uppercase', opacity: 0.7, marginBottom: 8 }}>
            cumulative KL race {live ? '(live agent)' : '(vs recorded baselines)'}
          </div>
          {raceRows.map(([name, v, c]) => (
            <div key={name} style={{ marginBottom: 9 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--grotesk)', fontSize: '.72rem', marginBottom: 3 }}>
                <span style={{ color: c }}>{name}</span>
                <span style={{ fontFamily: 'var(--mono)', opacity: 0.8 }}>{fmtKL(v)}</span>
              </div>
              <div style={{ height: 11, background: 'rgba(127,127,127,.13)', borderRadius: 6, overflow: 'hidden' }}>
                <div style={{ width: `${(v / maxCum) * 100}%`, height: '100%', background: c, transition: 'width .4s ease' }} />
              </div>
            </div>
          ))}
          {done && (
            <div style={{ marginTop: 10, fontFamily: 'var(--grotesk)', fontSize: '.78rem', color: 'var(--teal-bright)' }}>
              done · {done.steps} interventions · cumulative KL {fmtKL(done.cum_kl)}{done.converged ? ' · beliefs converged' : ''}
            </div>
          )}
          {!live && steps.length > 0 && (
            <p style={{ fontSize: '.72rem', opacity: 0.6, fontFamily: 'var(--grotesk)', marginBottom: 0 }}>
              Note: the agent trace is the multi-action run — steering steps produce KLs far above the
              ablation-only baselines. That gap is the RCK story from Part III, visible live.
            </p>
          )}
        </div>
      </div>

      {/* candidate layer map (live only, when graph known) */}
      {layers && candidates.length > 0 && (
        <div style={panel}>
          <div style={{ fontFamily: 'var(--grotesk)', fontSize: '.7rem', letterSpacing: '.12em', textTransform: 'uppercase', opacity: 0.7, marginBottom: 8 }}>
            candidate features by layer — probed ones light up
          </div>
          <svg viewBox={`0 0 760 ${Math.ceil((layers.maxL + 1) / 2) * 0 + 120}`} style={{ width: '100%', height: 'auto' }}>
            {candidates.map((c, i) => {
              const x = 30 + (c.layer / Math.max(1, layers.maxL)) * 700;
              const y = 26 + (i % 5) * 18;
              const hit = layers.probed.has(c.fid);
              return (
                <circle key={c.fid} cx={x} cy={y} r={4 + c.imp * 6}
                  fill={hit ? '#4fd8ce' : '#3d4d63'} opacity={hit ? 1 : 0.65}>
                  <title>{c.fid} · imp {c.imp.toFixed(2)}</title>
                </circle>
              );
            })}
            <text x={30} y={116} fontSize={10} fill="currentColor" opacity={0.6} fontFamily="var(--grotesk)">layer 0</text>
            <text x={700} y={116} fontSize={10} fill="currentColor" opacity={0.6} fontFamily="var(--grotesk)">layer {layers.maxL}</text>
          </svg>
        </div>
      )}
    </div>
  );
}

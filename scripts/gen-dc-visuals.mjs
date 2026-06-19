#!/usr/bin/env node
/**
 * Generate static SVG visuals (themed for the editorial DC build) from the REAL
 * data, to inject into web/index.html:
 *   - architecture / loop / step flowcharts (clean SVG, my FlowDiagram layout)
 *   - a feature-activation map: prompt tokens x layers, real attribution graph
 * Writes snippets to /tmp/dcgen/*.html
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUT = '/tmp/dcgen';
mkdirSync(OUT, { recursive: true });

/* ---- load paperDiagrams.ts by stripping TS types into a temp ESM module ---- */
let pd = readFileSync(resolve(ROOT, 'site/src/components/charts/paperDiagrams.ts'), 'utf8');
pd = pd.replace(/import type[^;]*;/g, '')
       .replace(/export const (\w+)\s*:[^=]*=/g, 'export const $1 =');
const tmp = `${OUT}/_pd.mjs`;
writeFileSync(tmp, pd);
const { architecture, aiLoop, stepFlow } = await import(pathToFileURL(tmp).href);
const diagrams = { architecture, aiLoop, stepFlow };

const GROUP_COLOR = { backend: '#4FD8CE', agent: '#B49BF0', genmodel: '#F0A24B', io: '#5EE6B8', neutral: '#8b98ac' };
const NODE_FILL = '#111824', INK = '#E6EDF3', INK_SOFT = '#AEB7C7', FAINT = '#8C93A3';
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function anchor(n, side) {
  switch (side) {
    case 'top': return [n.x + n.w / 2, n.y];
    case 'bottom': return [n.x + n.w / 2, n.y + n.h];
    case 'left': return [n.x, n.y + n.h / 2];
    case 'right': return [n.x + n.w, n.y + n.h / 2];
  }
}
function inferSides(a, b) {
  const dx = b.x + b.w / 2 - (a.x + a.w / 2);
  const dy = b.y + b.h / 2 - (a.y + a.h / 2);
  if (Math.abs(dy) >= Math.abs(dx)) return dy >= 0 ? ['bottom', 'top'] : ['top', 'bottom'];
  return dx >= 0 ? ['right', 'left'] : ['left', 'right'];
}
function edgePath(a, b, e) {
  const [fs, ts] = e.fromSide && e.toSide ? [e.fromSide, e.toSide] : inferSides(a, b);
  const [x1, y1] = anchor(a, fs);
  const [x2, y2] = anchor(b, ts);
  if (e.curve) { const mx = (x1 + x2) / 2 + e.curve; return `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`; }
  const dx = x2 - x1, dy = y2 - y1, c = 0.45;
  if (fs === 'bottom' || fs === 'top') return `M ${x1} ${y1} C ${x1} ${y1 + dy * c}, ${x2} ${y2 - dy * c}, ${x2} ${y2}`;
  return `M ${x1} ${y1} C ${x1 + dx * c} ${y1}, ${x2 - dx * c} ${y2}, ${x2} ${y2}`;
}

function flowSVG(d, id) {
  const byId = Object.fromEntries(d.nodes.map((n) => [n.id, n]));
  let s = `<svg viewBox="0 0 ${d.viewW} ${d.viewH}" preserveAspectRatio="xMidYMid meet" style="display:block;width:100%;height:auto;font-family:'Space Grotesk',sans-serif;">`;
  s += `<defs><marker id="${id}-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${INK_SOFT}"/></marker>`;
  s += `<filter id="${id}-glow" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="2.2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;
  for (const g of d.groups || []) {
    s += `<rect x="${g.bbox.x}" y="${g.bbox.y}" width="${g.bbox.w}" height="${g.bbox.h}" rx="14" fill="${g.color}0d" stroke="${g.color}55" stroke-dasharray="5 4"/>`;
    s += `<text x="${g.bbox.x + 12}" y="${g.bbox.y + 20}" fill="${g.color}" font-size="13" font-weight="600">${esc(g.label)}</text>`;
  }
  d.edges.forEach((e) => {
    const a = byId[e.from], b = byId[e.to]; if (!a || !b) return;
    const dd = edgePath(a, b, e);
    const color = e.feedback ? GROUP_COLOR.agent : INK_SOFT;
    const dash = e.feedback ? `stroke-dasharray="5 4"` : `stroke-dasharray="4 6" class="acd-edge"`;
    s += `<path d="${dd}" fill="none" stroke="${color}" stroke-width="1.7" stroke-opacity="${e.feedback ? 0.7 : 0.9}" ${dash} marker-end="url(#${id}-arr)"/>`;
    if (e.label) {
      const mx = ((a.x + a.w / 2) + (b.x + b.w / 2)) / 2, my = ((a.y + a.h / 2) + (b.y + b.h / 2)) / 2;
      s += `<text x="${mx}" y="${my - 4}" fill="${FAINT}" font-size="10.5" text-anchor="middle">${esc(e.label)}</text>`;
    }
  });
  for (const n of d.nodes) {
    const color = GROUP_COLOR[n.group] || GROUP_COLOR.neutral;
    const cx = n.x + n.w / 2;
    const lines = String(n.label).split('\n');
    if (n.shape === 'diamond') {
      s += `<polygon points="${cx},${n.y} ${n.x + n.w},${n.y + n.h / 2} ${cx},${n.y + n.h} ${n.x},${n.y + n.h / 2}" fill="${NODE_FILL}" stroke="${color}" stroke-width="${n.emphasis ? 2 : 1.4}"/>`;
    } else {
      const rx = n.shape === 'round' ? n.h / 2 : 9;
      s += `<rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="${rx}" fill="${NODE_FILL}" stroke="${color}" stroke-width="${n.emphasis ? 2.2 : 1.4}"${n.emphasis ? ` filter="url(#${id}-glow)"` : ''}/>`;
    }
    s += `<text x="${cx}" y="${n.y + n.h / 2}" fill="${INK}" font-size="11.5" text-anchor="middle" dominant-baseline="middle">`;
    lines.forEach((ln, li) => {
      const dy = li === 0 ? `${-(lines.length - 1) * 0.6}em` : '1.2em';
      s += `<tspan x="${cx}" dy="${dy}">${esc(ln)}</tspan>`;
    });
    s += `</text>`;
  }
  s += `</svg>`;
  return s;
}

function panelFigure(svg, caption, maxw = 1060) {
  return `<figure class="r" style="max-width:${maxw}px;margin:1.8rem auto 0;padding:0 24px;">
<div style="background:linear-gradient(180deg,#0A0D14,#0C111B);border:1px solid #1B2230;border-radius:6px;padding:18px;box-shadow:0 24px 60px -28px rgba(10,13,20,.6);">${svg}</div>
<figcaption style="font-family:'Newsreader',serif;font-style:italic;font-size:.92rem;color:#7A746A;margin-top:1rem;text-align:center;">${caption}</figcaption>
</figure>`;
}

writeFileSync(`${OUT}/arch.html`, panelFigure(flowSVG(diagrams.architecture, 'arch'),
  'The Active Circuit Discovery architecture, recreated from the paper. The backend builds the candidate features; the agent loop probes them. The violet dashed edge is the feedback loop that lets each experiment reshape the next choice.'));
writeFileSync(`${OUT}/loop.html`, panelFigure(flowSVG(diagrams.aiLoop, 'loop'),
  'The agent’s perception–action loop: infer beliefs → evaluate Expected Free Energy → act → learn → repeat.', 760));
writeFileSync(`${OUT}/step.html`, panelFigure(flowSVG(diagrams.stepFlow, 'step'),
  'One step of the agent, from candidate features to the convergence check.', 460));

/* ---- SAE feature-activation map: tokens x layers, real attribution graph ---- */
const g = JSON.parse(readFileSync(resolve(ROOT, 'site/src/data/graphs/ioi_gemma.json'), 'utf8'));
const toks = g.prompt_tokens;
const W = 1000, H = 560, padL = 64, padR = 24, padT = 26, padB = 70;
const maxLayer = 25;
const innerW = W - padL - padR, innerH = H - padT - padB;
const xOf = (ctx) => padL + (ctx / (toks.length - 1)) * innerW;
const yOf = (layer) => padT + (1 - (layer ?? maxLayer) / maxLayer) * innerH;
const maxInfl = Math.max(...g.nodes.map((n) => n.influence || 0)) || 1;
const lerp = (a, b, t) => Math.round(a + (b - a) * t);
const depthColor = (layer) => {
  if (layer == null) return '#F0A24B';
  const t = Math.min(1, layer / maxLayer);
  return `rgb(${lerp(79, 180, t)},${lerp(216, 155, t)},${lerp(206, 240, t)})`;
};
const idMap = Object.fromEntries(g.nodes.map((n) => [n.id, n]));

let sae = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" style="display:block;width:100%;height:auto;font-family:'Space Grotesk',sans-serif;">`;
sae += `<defs><filter id="sae-glow" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="2.4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;
for (const L of [0, 6, 13, 19, 25]) {
  const y = yOf(L);
  sae += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#1B2230" stroke-width="1"/>`;
  sae += `<text x="${padL - 10}" y="${y}" fill="${FAINT}" font-size="11" text-anchor="end" dominant-baseline="middle">L${L}</text>`;
}
sae += `<text x="16" y="${padT - 8}" fill="${FAINT}" font-size="11" font-weight="600">layer</text>`;
let drawn = 0;
for (const l of g.links) {
  if (drawn > 240) break;
  const a = idMap[l.s], b = idMap[l.t]; if (!a || !b) continue;
  const x1 = xOf(a.ctx), y1 = yOf(a.layer), x2 = xOf(b.ctx), y2 = yOf(b.layer);
  const mx = (x1 + x2) / 2 + (drawn % 2 ? 16 : -16);
  sae += `<path d="M${x1.toFixed(1)},${y1.toFixed(1)} Q${mx.toFixed(1)},${((y1 + y2) / 2).toFixed(1)} ${x2.toFixed(1)},${y2.toFixed(1)}" fill="none" stroke="#3a4a63" stroke-width="0.8" stroke-opacity="0.35"/>`;
  drawn++;
}
toks.forEach((t, ctx) => {
  if (ctx === 0) return;
  const x = xOf(ctx);
  sae += `<line x1="${x}" y1="${padT}" x2="${x}" y2="${H - padB}" stroke="#141b27" stroke-width="1"/>`;
  const label = t.replace(/^ /, '') || '·';
  sae += `<text x="${x}" y="${H - padB + 16}" fill="${INK_SOFT}" font-size="11" text-anchor="end" transform="rotate(-35 ${x} ${H - padB + 16})">${esc(label)}</text>`;
});
for (const n of g.nodes) {
  const x = xOf(n.ctx), y = yOf(n.layer);
  const r = n.logit ? 9 : 3 + (n.influence / maxInfl) * 8;
  const fill = depthColor(n.logit ? null : n.layer);
  const glow = (n.logit || n.influence > 0.5) ? ` filter="url(#sae-glow)"` : '';
  const tip = (n.logit ? 'logit output' : 'layer ' + n.layer) + ' · token "' + (toks[n.ctx] || '') + '" · activation ' + n.activation.toFixed(1) + ' · influence ' + (n.influence * 100).toFixed(0) + '%';
  sae += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="${fill}" fill-opacity="${n.logit ? 0.95 : 0.8}"${glow}><title>${esc(tip)}</title></circle>`;
}
sae += `</svg>`;

const saeFig = `<div data-graph="true" style="position:relative;background:linear-gradient(180deg,#0A0D14,#0C111B);border-radius:6px;overflow:hidden;box-shadow:0 24px 60px -28px rgba(10,13,20,.6);">
${sae}
<div style="display:flex;flex-wrap:wrap;gap:18px;align-items:center;padding:13px 18px;border-top:1px solid #1B2230;font-family:'Space Grotesk',sans-serif;font-size:.76rem;color:#8C93A3;">
<span style="display:flex;align-items:center;gap:7px;"><span style="width:10px;height:10px;border-radius:50%;background:#4FD8CE;"></span>early layers</span>
<span style="display:flex;align-items:center;gap:7px;"><span style="width:10px;height:10px;border-radius:50%;background:#B49BF0;"></span>late layers</span>
<span style="display:flex;align-items:center;gap:7px;"><span style="width:10px;height:10px;border-radius:50%;background:#F0A24B;"></span>logit output</span>
<span style="margin-left:auto;font-style:italic;color:#6F7686;">dot size = causal influence · ${g.nodes.length} features from ${g.n_links_original.toLocaleString()} edges, on the real IOI prompt</span>
</div></div>`;
writeFileSync(`${OUT}/sae.html`, saeFig);

/* ---- prompt catalog: real experiment prompts + per-domain layer fingerprint ---- */
const stats = JSON.parse(readFileSync(resolve(ROOT, 'results/paper_stats.json'), 'utf8'));
// Verified verbatim from src/experiments/run_real_experiments.py
const DOMAIN_PROMPTS = {
  geography: ['The capital of France is', 'The Golden Gate Bridge connects San Francisco to'],
  math: ['The square root of 64 is', 'If 2 + 3 = 5 then 3 + 4 ='],
  science: ['Water is made of hydrogen and', 'The speed of light is approximately'],
  logic: ['All mammals are warm-blooded. A whale is a mammal. Therefore a whale is', 'All birds have wings. A penguin is a bird. Therefore a penguin has'],
  history: ['The year World War II ended was', 'The first person to walk on the moon was'],
};
const TASK_PROMPTS = {
  'Indirect Object Identification': [
    'When John and Mary went to the store, John gave the bag to',
    'After Alice and Bob finished lunch, Alice handed the receipt to',
    'While Sarah and Tom were at the park, Sarah threw the ball to',
    'When Emma and David arrived at the office, Emma passed the keys to',
    'As Lisa and Mike left the restaurant, Lisa returned the coat to',
  ],
  'Multi-step reasoning': [
    'If Alice is taller than Bob, and Bob is taller than Carol, then the tallest person is',
    'The capital of France is Paris. Paris is in Europe. The continent containing Paris is',
    'All dogs are animals. Fido is a dog. Therefore Fido is',
  ],
  'Concept steering': [
    'The Golden Gate Bridge is', 'The Eiffel Tower is located in', 'Mount Everest is the tallest',
    'The Great Wall of China was built', 'The Statue of Liberty stands in',
  ],
};
const DOM_LABEL = { geography: 'Geography', math: 'Mathematics', science: 'Science', logic: 'Logic', history: 'History' };

function fingerprintBar(ld) {
  const tot = (ld.early + ld.mid + ld.late) || 1;
  const seg = (v, c) => v > 0 ? `<span style="flex:${v};background:${c};"></span>` : '';
  return `<span style="display:flex;height:8px;border-radius:4px;overflow:hidden;background:#E1DAC9;width:120px;">${seg(ld.early, '#0E7C86')}${seg(ld.mid, '#6D28D9')}${seg(ld.late, '#B5610E')}</span>`;
}
const promptChip = (p) => `<div style="font-family:'JetBrains Mono',monospace;font-size:.82rem;line-height:1.5;color:#2A2720;background:#EAE4D6;border-left:2px solid #0E7C86;padding:.45rem .65rem;border-radius:0 3px 3px 0;margin:.35rem 0;">"${esc(p)}"</div>`;

let cat = `<div class="r" style="max-width:1060px;margin:3rem auto 0;padding:0 24px;border-top:1px solid #D2CAB8;padding-top:2rem;">
<div style="font-family:'Space Grotesk',sans-serif;font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:#B5610E;font-weight:600;margin-bottom:.6rem;">The actual prompts</div>
<h3 style="font-family:'Newsreader',serif;font-weight:500;font-size:1.7rem;line-height:1.15;margin:0 0 .8rem;">Every run, the exact prompts — and where each domain lights up</h3>
<p style="font-family:'Newsreader',serif;font-size:1.14rem;line-height:1.65;color:#2A2720;margin:0 0 1.6rem;max-width:680px;">These are the verbatim prompts from the experiment code. For each knowledge domain the bar shows its <em>layer fingerprint</em> on Gemma — how the top causal features split across <span style="color:#0E7C86;font-weight:600;">early</span> / <span style="color:#6D28D9;font-weight:600;">middle</span> / <span style="color:#B5610E;font-weight:600;">late</span> layers. Factual lookups and compositional reasoning leave visibly different prints.</p>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;">`;
for (const key of ['geography', 'math', 'science', 'logic', 'history']) {
  const ld = stats.gemma.domain.domains[key].layer_distribution;
  const tot = ld.early + ld.mid + ld.late;
  cat += `<div style="border:1px solid #D2CAB8;border-radius:8px;padding:1rem 1.1rem;background:#F7F3EA;">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:.5rem;">
    <span style="font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:1rem;color:#1B1A16;">${DOM_LABEL[key]}</span>
    ${fingerprintBar(ld)}
  </div>
  <div style="font-family:'Space Grotesk',sans-serif;font-size:.72rem;color:#8A8372;margin-bottom:.3rem;">${ld.early} early · ${ld.mid} mid · ${ld.late} late features</div>
  ${DOMAIN_PROMPTS[key].map(promptChip).join('')}
  </div>`;
}
cat += `</div>`;
// task families
cat += `<h4 style="font-family:'Space Grotesk',sans-serif;font-size:.95rem;font-weight:600;color:#1B1A16;margin:2.2rem 0 .4rem;">The three benchmark task families</h4>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;">`;
for (const [task, prompts] of Object.entries(TASK_PROMPTS)) {
  cat += `<div style="border:1px solid #D2CAB8;border-radius:8px;padding:1rem 1.1rem;background:#F7F3EA;">
  <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:.95rem;color:#1B1A16;margin-bottom:.2rem;">${task}</div>
  <div style="font-family:'Space Grotesk',sans-serif;font-size:.72rem;color:#8A8372;margin-bottom:.3rem;">${prompts.length} prompts</div>
  ${prompts.map(promptChip).join('')}
  </div>`;
}
cat += `</div></div>`;
writeFileSync(`${OUT}/catalog.html`, cat);

console.log('generated arch/loop/step/sae/catalog in', OUT);

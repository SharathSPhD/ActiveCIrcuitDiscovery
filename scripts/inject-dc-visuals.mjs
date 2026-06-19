#!/usr/bin/env node
/** Inject the generated static visuals into web/index.html, replacing the
 *  distorted DC flowcharts and the force-graph with my clean versions. */
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const file = resolve(ROOT, 'web/index.html');
let html = readFileSync(file, 'utf8');
const read = (p) => readFileSync(`/tmp/dcgen/${p}`, 'utf8').trim();

/** From `startIdx` (pointing at a `<tag`), return index just past its matching `</tag>`. */
function elementEnd(s, startIdx, tag) {
  const open = `<${tag}`, close = `</${tag}>`;
  let i = startIdx, depth = 0;
  while (i < s.length) {
    const no = s.indexOf(open, i), nc = s.indexOf(close, i);
    if (nc === -1) throw new Error(`no close for ${tag}`);
    if (no !== -1 && no < nc) { depth++; i = no + open.length; }
    else { depth--; i = nc + close.length; if (depth === 0) return i; }
  }
  throw new Error(`unbalanced ${tag}`);
}
function replaceElement(anchor, tag, replacement, label) {
  const a = html.indexOf(anchor);
  if (a === -1) throw new Error(`anchor not found: ${label}`);
  const start = html.indexOf(`<${tag}`, a);
  const end = elementEnd(html, start, tag);
  html = html.slice(0, start) + replacement + html.slice(end);
  console.log(`replaced ${label} (${end - start} -> ${replacement.length} chars)`);
}

// 1. Architecture flowchart (figure right after the ARCHITECTURE FLOW comment)
replaceElement('<!-- ARCHITECTURE FLOW -->', 'figure', read('arch.html'), 'architecture');
// 2. Loop flowchart
replaceElement('<figure style="margin:0 auto;max-width:600px;">', 'figure', read('loop.html'), 'loop');
// 3. Step flowchart (inside the deep-dive)
replaceElement('<figure style="margin:0;max-width:420px;', 'figure', read('step.html'), 'step');
// 4. SAE feature-activation map (the data-graph div in #microscope)
replaceElement('<div data-graph="true"', 'div', read('sae.html'), 'sae-map');

// 5b. inject the prompt catalog into #domains, before its closing </section>
{
  const di = html.indexOf('Toggle the models to compare.');
  if (di === -1) throw new Error('domains anchor not found');
  const secEnd = html.indexOf('</section>', di);
  html = html.slice(0, secEnd) + '\n' + read('catalog.html') + '\n  ' + html.slice(secEnd);
  console.log('injected prompt catalog into #domains');
}

// 5. ensure the flowing-edge animation class exists (DC build already has @keyframes acd-flow)
if (!html.includes('.acd-edge{')) {
  html = html.replace('@keyframes acd-flow{', '.acd-edge{ animation:acd-flow 2.2s linear infinite; }\n  @keyframes acd-flow{');
  console.log('added .acd-edge animation rule');
}

writeFileSync(file, html);
console.log('done; new size', html.length);

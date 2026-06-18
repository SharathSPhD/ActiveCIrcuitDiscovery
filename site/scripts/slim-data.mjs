#!/usr/bin/env node
/**
 * Build-time data pipeline.
 *
 * Reads the REAL experiment outputs in <repo>/results and emits small,
 * web-ready JSON into site/src/data/. Nothing here fabricates numbers — it
 * only selects, reshapes, and (for the huge attribution graphs) subsamples.
 *
 * Outputs:
 *   src/data/stats.json            — canonical paper_stats.json (source of truth)
 *   src/data/genmodel.json         — learned POMDP A/B/C/D + config
 *   src/data/timeline_<model>.json — per-prompt agent trajectories (IOI)
 *   src/data/graph_ioi_gemma.json  — slim attribution subgraph (~hundreds of nodes)
 *   src/data/manifest.json         — provenance + sizes
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SITE = resolve(__dirname, '..');
const REPO = resolve(SITE, '..');
const RESULTS = join(REPO, 'results');
const OUT = join(SITE, 'src', 'data');

mkdirSync(OUT, { recursive: true });
mkdirSync(join(OUT, 'graphs'), { recursive: true });

const readJSON = (p) => JSON.parse(readFileSync(p, 'utf8'));
const writeJSON = (name, obj) => {
  const p = join(OUT, name);
  mkdirSync(dirname(p), { recursive: true });
  writeFileSync(p, JSON.stringify(obj));
  const kb = (statSync(p).size / 1024).toFixed(1);
  console.log(`  wrote ${name.padEnd(30)} ${kb.padStart(8)} KB`);
  return Number(kb);
};

const manifest = { generatedFrom: 'results/', files: {}, note: 'All values are real experiment outputs.' };

console.log('· canonical stats + generative model');
const stats = readJSON(join(RESULTS, 'paper_stats.json'));
manifest.files['stats.json'] = writeJSON('stats.json', stats);
if (existsSync(join(RESULTS, 'generative_model.json'))) {
  manifest.files['genmodel.json'] = writeJSON('genmodel.json', readJSON(join(RESULTS, 'generative_model.json')));
}

/* ---- Per-prompt agent trajectories (IOI) ---------------------------------- */
function slimTimeline(model) {
  const src = join(RESULTS, `ioi_results_${model}.json`);
  if (!existsSync(src)) return;
  const d = readJSON(src);
  const out = {
    task: d.task,
    model,
    budget: d.budget,
    n_prompts: d.n_prompts,
    per_prompt: d.per_prompt.map((p) => ({
      prompt: p.prompt,
      n_candidates: p.n_candidates,
      ai_kls: p.ai_kls,
      ai_actions: p.ai_actions,
      ai_abl_kls: p.ai_abl_kls,
      ai_cumkl: p.ai_cumkl,
      ai_abl_cumkl: p.ai_abl_cumkl,
      agent_entropy_history: p.agent_entropy_history,
      agent_efe_history: p.agent_efe_history,
      agent_converged: p.agent_converged ?? null,
      layer_distribution: p.layer_distribution,
      // baseline cumulative curves for the same prompt (per-step)
      oracle_kls: p.oracle_kls,
      eap_kls: p.eap_kls,
      random_kls: p.random_kls ?? null,
    })),
  };
  manifest.files[`timeline_${model}.json`] = writeJSON(`timeline_${model}.json`, out);
}
console.log('· agent trajectories');
slimTimeline('gemma');
slimTimeline('llama');

/* ---- Attribution graph slimming ------------------------------------------ */
/**
 * Reduce a circuit-tracer graph (thousands of nodes, ~680k links) to a
 * legible subgraph: the top-K transcoder features by influence, plus the
 * target-logit nodes, with only the strongest links among survivors.
 */
function slimGraph(slug, { topK = 120, maxLinks = 500 } = {}) {
  const src = join(RESULTS, 'graphs', `${slug}.json`);
  if (!existsSync(src)) {
    console.log(`  (skip ${slug}: not present)`);
    return;
  }
  const g = readJSON(src);
  const nodes = g.nodes ?? [];
  const links = g.links ?? [];

  // Full layer histogram (over all feature nodes) for context.
  const layerHist = {};
  for (const n of nodes) {
    if (n.is_target_logit) continue;
    const L = Number(n.layer);
    if (Number.isFinite(L)) layerHist[L] = (layerHist[L] ?? 0) + 1;
  }

  const featureNodes = nodes.filter((n) => !n.is_target_logit && Number.isFinite(Number(n.layer)));
  featureNodes.sort((a, b) => (b.influence ?? 0) - (a.influence ?? 0));
  const kept = featureNodes.slice(0, topK);
  const logits = nodes.filter((n) => n.is_target_logit);
  const keptIds = new Set([...kept, ...logits].map((n) => n.node_id));

  const slimLinks = links
    .filter((l) => keptIds.has(l.source) && keptIds.has(l.target))
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
    .slice(0, maxLinks)
    .map((l) => ({ s: l.source, t: l.target, w: Number(l.weight.toFixed(3)) }));

  // Only keep nodes that participate in at least one surviving link (or logits).
  const connected = new Set();
  for (const l of slimLinks) {
    connected.add(l.s);
    connected.add(l.t);
  }
  const outNodes = [...kept, ...logits]
    .filter((n) => connected.has(n.node_id) || n.is_target_logit)
    .map((n) => ({
      id: n.node_id,
      layer: n.is_target_logit ? null : Number(n.layer),
      feature: n.feature,
      ctx: n.ctx_idx,
      influence: Number((n.influence ?? 0).toFixed(4)),
      activation: Number((n.activation ?? 0).toFixed(4)),
      logit: !!n.is_target_logit,
    }));

  const out = {
    slug,
    prompt: g.metadata?.prompt ?? '',
    prompt_tokens: g.metadata?.prompt_tokens ?? [],
    n_nodes_original: nodes.length,
    n_links_original: links.length,
    layer_hist: layerHist,
    nodes: outNodes,
    links: slimLinks,
  };
  manifest.files[`graphs/${slug}.json`] = writeJSON(`graphs/${slug}.json`, out);
}
console.log('· attribution subgraphs');
slimGraph('ioi_gemma', { topK: 110, maxLinks: 420 });
slimGraph('steering_gemma', { topK: 90, maxLinks: 320 });

writeJSON('manifest.json', manifest);
console.log('✓ data pipeline complete');

// Build compact data JSONs for the talk site from the repo's results files.
// Source of truth: results/*.json + site/src/data/*.json (already computed
// by scripts/compute_paper_stats.py in the main repo — numbers match the paper).
// Run from talk/: node scripts/build-data.mjs
// Idempotent; skips gracefully if sources are absent (data/ is checked in).

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const talk = join(here, '..');
const repo = process.env.ACD_REPO ?? join(talk, '..');
const out = join(talk, 'data');
mkdirSync(out, { recursive: true });
mkdirSync(join(out, 'replays'), { recursive: true });

const src = (p) => join(repo, p);
const load = (p) => JSON.parse(readFileSync(src(p), 'utf8'));
const save = (name, obj) =>
  writeFileSync(join(out, name), JSON.stringify(obj, null, 1));

if (!existsSync(src('results/paper_stats.json'))) {
  console.log('[build-data] repo results not found at', repo, '— keeping checked-in data/');
  process.exit(0);
}

// ---------- 1. stats.json passthrough (already paper-exact) ----------
const stats = load('site/src/data/stats.json');
save('stats.json', stats);

// ---------- 2. generative model ----------
save('genmodel.json', load('site/src/data/genmodel.json'));

// ---------- 3. attribution graphs for viz ----------
save('graph_ioi_gemma.json', load('site/src/data/graphs/ioi_gemma.json'));
save('graph_steering_gemma.json', load('site/src/data/graphs/steering_gemma.json'));

// ---------- 4. replay traces for the demo ----------
// Per-prompt, per-step traces straight out of the real experiment runs.
function buildReplay(model, file) {
  const r = load(file);
  const prompts = r.per_prompt.map((p, i) => ({
    id: `${model}-ioi-${i}`,
    prompt: p.prompt,
    n_candidates: p.n_candidates,
    n_features: p.n_features,
    // POMDP multi-action trace
    ai: {
      kls: p.ai_kls,
      actions: p.ai_actions,
      abl_kls: p.ai_abl_kls,
      entropy: p.agent_entropy_history,
      efe: p.agent_efe_history,
      a_drift: p.agent_a_convergence,
      converged: p.agent_converged,
      cumkl: p.ai_cumkl,
      abl_cumkl: p.ai_abl_cumkl,
    },
    baselines: {
      oracle: p.oracle_kls,
      eap: p.eap_kls,
      bandit: p.bandit_kls,
      ucb: p.ucb_kls,
      greedy: p.greedy_kls,
    },
    layer_distribution: p.layer_distribution,
    top5: p.ground_truth_top5,
  }));
  return { model, task: r.task, budget: r.budget, prompts };
}
save('replays/ioi_gemma.json', buildReplay('gemma', 'results/ioi_results_gemma.json'));
save('replays/ioi_llama.json', buildReplay('llama', 'results/ioi_results_llama.json'));
save('replays/multistep_gemma.json', buildReplay('gemma', 'results/multistep_results_gemma.json'));
save('replays/multistep_llama.json', buildReplay('llama', 'results/multistep_results_llama.json'));

// ---------- 5. steering summary (dose-response for demo + results) ----------
function steeringSummary(model, file) {
  const s = load(file);
  // keep only compact aggregates; the full file is 400 KB
  const g = stats[model].steering;
  return {
    model,
    multipliers: g.multipliers,
    top_changes: g.top_changes_per_mult,
    control_changes: g.control_changes_per_mult,
    top_max_kl: g.top_max_kl,
    control_max_kl: g.control_max_kl,
    concept_amplification: g.concept_amplification,
    prompts: (s.per_prompt ?? s.prompts ?? []).slice(0, 5).map((p) => p.prompt ?? p),
  };
}
try {
  save('steering_gemma.json', steeringSummary('gemma', 'results/steering_results_gemma.json'));
  save('steering_llama.json', steeringSummary('llama', 'results/steering_results_llama.json'));
} catch (e) {
  console.log('[build-data] steering summary skipped:', e.message);
}

console.log('[build-data] wrote data/ from', repo);

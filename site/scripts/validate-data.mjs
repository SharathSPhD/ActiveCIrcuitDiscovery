#!/usr/bin/env node
/**
 * Validate the slimmed data layer. Binary pass/fail gate for milestone M1.
 * Checks structural shape, size budget (<5MB), and a few spot values that
 * must match the paper (so the site can never silently drift from results).
 */
import { readFileSync, statSync, existsSync, readdirSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'src', 'data');
const read = (p) => JSON.parse(readFileSync(join(OUT, p), 'utf8'));

let failures = 0;
const check = (cond, msg) => {
  if (cond) {
    console.log(`  ✓ ${msg}`);
  } else {
    console.error(`  ✗ ${msg}`);
    failures++;
  }
};
const near = (a, b, tol = 0.05) => Math.abs(a - b) <= tol;

// 1. Required files exist.
for (const f of ['stats.json', 'genmodel.json', 'timeline_gemma.json', 'timeline_llama.json', 'manifest.json']) {
  check(existsSync(join(OUT, f)), `${f} exists`);
}

// 2. Size budget: every emitted JSON < 5MB.
const walk = (dir) => {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) walk(p);
    else if (e.name.endsWith('.json')) {
      const mb = statSync(p).size / 1024 / 1024;
      check(mb < 5, `${e.name} is ${mb.toFixed(2)}MB (<5MB)`);
    }
  }
};
walk(OUT);

// 3. Canonical spot values must match the paper.
const stats = read('stats.json');
check(near(stats.gemma.ioi.methods.ai_abl.oracle_eff.point, 81.98, 0.1), 'Gemma IOI POMDP-abl oracle efficiency ≈ 82.0%');
check(near(stats.llama.ioi.methods.ai_abl.oracle_eff.point, 52.14, 0.1), 'Llama IOI POMDP-abl oracle efficiency ≈ 52.1%');
check(near(stats.gemma.multistep.methods.ai_abl.oracle_eff.point, 73.04, 0.1), 'Gemma multi-step efficiency ≈ 73.0%');
check(near(stats.llama.multistep.methods.ai_abl.oracle_eff.point, 9.26, 0.1), 'Llama multi-step efficiency ≈ 9.3% (the honest failure)');
check(near(stats.llama.multistep_finebins.methods.ai_abl.oracle_eff.point, 37.78, 0.1), 'Llama multi-step 6-bin fix ≈ 37.8%');
check(stats.gemma.ioi.methods.ai.oracle_eff === undefined || stats.head_to_head.gemma.ioi.ai_rck > 1000, 'Gemma IOI RCK > 1000% (steering amplification)');

// 4. Timeline shape: IOI has the expected prompts and 20-step budgets.
const tl = read('timeline_gemma.json');
check(tl.per_prompt.length === 5, 'Gemma IOI timeline has 5 prompts');
check(tl.per_prompt[0].ai_kls.length === 20, 'per-prompt has a 20-step KL trajectory');
check(tl.per_prompt[0].ai_actions.length === 20, 'per-prompt has a 20-step action trajectory');
check(tl.per_prompt[0].agent_entropy_history.length === 20, 'per-prompt has a 20-step entropy trajectory');
check(tl.per_prompt[0].ai_actions[0] === 'ablation', 'agent explores first (step 0 = ablation)');

// 5. Generative model shape.
const gm = read('genmodel.json');
check(Array.isArray(gm.A) && gm.A.length === 3, 'generative model has 3 observation modalities (A)');
check(Array.isArray(gm.B) && gm.B.length === 3, 'generative model has 3 state factors (B)');
check(gm.config?.n_importance === 4, 'importance factor has 4 levels');

if (failures) {
  console.error(`\n✗ validate-data: ${failures} check(s) failed`);
  process.exit(1);
}
console.log('\n✓ validate-data: all checks passed');

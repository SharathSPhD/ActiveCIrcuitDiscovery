/**
 * Typed accessors over the slimmed, REAL experiment data.
 * Every number here traces to results/paper_stats.json (the paper's own stats).
 */
import stats from './stats.json';
import genmodel from './genmodel.json';
import timelineGemma from './timeline_gemma.json';
import timelineLlama from './timeline_llama.json';
import graphIoiGemma from './graphs/ioi_gemma.json';
import { METHODS, type MethodKey, type ModelKey } from '../utils/theme';

export type Task = 'ioi' | 'multistep' | 'domain';

interface CI {
  point: number;
  lo: number;
  hi: number;
}

const S: any = stats;

export const TASK_LABELS: Record<Task, string> = {
  ioi: 'Indirect Object Identification',
  multistep: 'Multi-step reasoning',
  domain: 'Knowledge domains (pooled)',
};

/** Methods that report a bounded oracle efficiency (ablation-based, ≤100%). */
const EFF_METHODS: MethodKey[] = ['ai_abl', 'eap', 'bandit', 'ucb', 'greedy', 'random'];

export interface EffBar {
  key: MethodKey | 'oracle';
  label: string;
  color: string;
  eff: number;
  lo: number;
  hi: number;
}

/** Bounded oracle-efficiency bars for a model+task, sorted high→low, oracle pinned at 100. */
export function efficiencyBars(model: ModelKey, task: Task): EffBar[] {
  const node = S[model][task];
  const bars: EffBar[] = [];
  if (task === 'domain') {
    const agg = node.aggregate;
    const map: [MethodKey, string][] = [
      ['ai_abl', 'ai_abl_oracle_efficiency'],
      ['eap', 'eap_oracle_efficiency'],
      ['bandit', 'bandit_oracle_efficiency'],
      ['ucb', 'ucb_oracle_efficiency'],
    ];
    for (const [key, field] of map) {
      if (agg[field] == null) continue;
      bars.push({ key, label: METHODS[key].label, color: METHODS[key].color, eff: agg[field], lo: agg[field], hi: agg[field] });
    }
  } else {
    for (const key of EFF_METHODS) {
      const m = node.methods?.[key];
      if (!m?.oracle_eff) continue;
      const e: CI = m.oracle_eff;
      bars.push({ key, label: METHODS[key].label, color: METHODS[key].color, eff: e.point, lo: e.lo, hi: e.hi });
    }
  }
  bars.sort((a, b) => b.eff - a.eff);
  bars.push({ key: 'oracle', label: METHODS.oracle.label, color: METHODS.oracle.color, eff: 100, lo: 100, hi: 100 });
  return bars;
}

export interface RckBar {
  key: string;
  label: string;
  color: string;
  rck: number;
  lo?: number;
  hi?: number;
}

/** Relative-Cumulative-KL (amplification) bars — values can exceed 100%. */
export function rckBars(model: ModelKey, task: Task): RckBar[] {
  const node = S[model][task];
  const out: RckBar[] = [];
  const push = (key: string, label: string, color: string, v: any) => {
    if (v == null) return;
    if (typeof v === 'number') out.push({ key, label, color, rck: v });
    else out.push({ key, label, color, rck: v.point, lo: v.lo, hi: v.hi });
  };
  if (task === 'domain') {
    push('ai', 'POMDP (multi-action)', METHODS.ai.color, node.aggregate?.ai_rck);
  } else {
    const m = node.methods ?? {};
    push('ai', 'POMDP (multi-action)', METHODS.ai.color, m.ai?.rck);
    push('bandit_steer', 'Bandit + steering', METHODS.bandit.color, m.bandit_steer?.rck);
    push('greedy_steer', 'Greedy + steering', METHODS.greedy.color, m.greedy_steer?.rck);
    push('random_steer', 'Random + steering', METHODS.random.color, m.random_steer?.rck);
    push('random_action', 'Random feature + action', '#475569', m.random_action?.rck);
  }
  out.sort((a, b) => b.rck - a.rck);
  return out;
}

/** Action-type counts for the agent over all steps (explore→exploit signature). */
export function actionCounts(model: ModelKey, task: 'ioi' | 'multistep') {
  return S[model][task].actions as {
    counts: Record<string, number>;
    total: number;
    first_step: Record<string, number>;
    post_first: Record<string, number>;
  };
}

/** H1 (efficiency vs random) stats for the ablation-only agent. */
export function h1(model: ModelKey, task: 'ioi' | 'multistep') {
  return S[model][task].H1.ai_abl as {
    mean_diff: number;
    t_stat: number;
    p_ttest_onesided: number;
    p_permutation_onesided: number;
    improvement_pct: number;
  };
}

/** Steering dose-response: per-multiplier change counts for selected vs control. */
export function steeringDose(model: ModelKey) {
  const s = S[model].steering;
  const mults: number[] = s.multipliers;
  return mults.map((m) => {
    const key = m.toFixed(1);
    return {
      multiplier: m,
      selected: s.top_changes_per_mult[key],
      control: s.control_changes_per_mult[key],
    } as { multiplier: number; selected: { changed: number; total: number }; control: { changed: number; total: number } };
  });
}

/** Steering hypothesis verdicts (H2a manipulability, H2b selectivity). */
export function steeringHypotheses(model: ModelKey) {
  const s = S[model].steering;
  return {
    h2a: s.H2_binomial as { multiplier: number; changed: number; total: number; rate: number; p_value: number },
    h2b: s.H2_selected_vs_control as { multiplier: number; selected_rate: number; control_rate: number; odds_ratio: number; p_value: number },
    h2bPooled: s.H2_pooled_selected_vs_control as { selected: string; control: string; selected_rate: number; control_rate: number; odds_ratio: number; p_value: number },
    topMaxKl: s.top_max_kl.mean as number,
    controlMaxKl: s.control_max_kl.mean as number,
  };
}

/** Per-domain efficiency + layer distribution. */
export function domains(model: ModelKey) {
  const d = S[model].domain.domains;
  return Object.entries(d).map(([name, v]: [string, any]) => ({
    name,
    efficiency: v.ai_abl_oracle_efficiency as number,
    layers: v.layer_distribution as { early: number; mid: number; late: number },
  }));
}

export const TIMELINES = { gemma: timelineGemma, llama: timelineLlama } as const;
export const GRAPHS = { ioi_gemma: graphIoiGemma } as const;
export const GENMODEL: any = genmodel;
export const STATS: any = stats;

/** Headline numbers reused in prose. */
export const HEADLINE = {
  gemmaIoiEff: S.gemma.ioi.methods.ai_abl.oracle_eff.point as number,
  llamaIoiEff: S.llama.ioi.methods.ai_abl.oracle_eff.point as number,
  gemmaMultistepEff: S.gemma.multistep.methods.ai_abl.oracle_eff.point as number,
  llamaMultistepEff: S.llama.multistep.methods.ai_abl.oracle_eff.point as number,
  llamaMultistepFinebins: S.llama.multistep_finebins.methods.ai_abl.oracle_eff.point as number,
  gemmaIoiRck: S.head_to_head.gemma.ioi.ai_rck as number,
};

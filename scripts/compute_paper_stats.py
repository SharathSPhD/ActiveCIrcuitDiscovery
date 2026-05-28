#!/usr/bin/env python3
"""Compute all paper statistics from raw experiment JSON.

This is the single source of truth for every numeric value reported in the
paper tables and text.  It reads ``results/*.json`` and derives:

  * Oracle mean / cumulative KL (the values the reviewers asked to be shown).
  * Mean KL +/- std, plus median and inter-quartile range (Q1, Q3) per method.
  * Bounded oracle efficiency for ablation-comparable methods
    (POMDP-abl, bandit, EAP, greedy, random) -- all <= the ablation oracle.
  * Relative Cumulative KL (RCK) for multi-action methods
    (POMDP, greedy+steer, bandit+steer, random+steer, random-action), each as a
    ratio to the ablation-oracle cumulative KL.  This is the renamed ">100%"
    metric and is what decomposes EFE-selection gains from steering gains.
  * Bootstrap 95% CIs (resampling prompts) for mean KL and efficiency/RCK.
  * H1: one-sided paired t-test AND exact paired-permutation test of
    (method mean KL) vs (random mean KL), per model/task.
  * H2: binomial test on steering-induced prediction changes, with a matched
    random-feature control.
  * Action-type tallies per step (e.g. "steering selected X/100 steps").

Outputs ``results/paper_stats.json`` (machine-readable) and prints a summary.

Usage:
    python -m scripts.compute_paper_stats [--results-dir results]
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

BOOTSTRAP_N = 10000
RNG = np.random.default_rng(12345)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _load(path: Path) -> Optional[dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _q(vals: List[float]) -> Dict[str, float]:
    """Mean/std/median/quartile summary of a list."""
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "q1": 0.0, "q3": 0.0, "n": 0}
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "q1": float(np.percentile(a, 25)),
        "q3": float(np.percentile(a, 75)),
        "cv": float(np.std(a, ddof=1) / np.mean(a)) if a.size > 1 and np.mean(a) != 0 else 0.0,
        "n": int(a.size),
    }


def _bootstrap_ratio_ci(num_per_prompt: List[float], den_per_prompt: List[float]) -> Dict[str, float]:
    """Bootstrap 95% CI for the ratio of summed numerator to summed denominator
    (i.e. aggregate cumulative-KL efficiency) by resampling prompts."""
    num = np.asarray(num_per_prompt, dtype=float)
    den = np.asarray(den_per_prompt, dtype=float)
    n = num.size
    if n == 0:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0}
    point = float(num.mean() / den.mean() * 100) if den.mean() > 0 else 0.0
    samples = []
    for _ in range(BOOTSTRAP_N):
        idx = RNG.integers(0, n, size=n)
        d = den[idx].mean()
        if d > 0:
            samples.append(num[idx].mean() / d * 100)
    if not samples:
        return {"point": point, "lo": point, "hi": point}
    return {
        "point": point,
        "lo": float(np.percentile(samples, 2.5)),
        "hi": float(np.percentile(samples, 97.5)),
    }


def _bootstrap_mean_ci(vals: List[float]) -> Dict[str, float]:
    a = np.asarray(vals, dtype=float)
    n = a.size
    if n == 0:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0}
    point = float(a.mean())
    samples = [a[RNG.integers(0, n, size=n)].mean() for _ in range(BOOTSTRAP_N)]
    return {
        "point": point,
        "lo": float(np.percentile(samples, 2.5)),
        "hi": float(np.percentile(samples, 97.5)),
    }


def _paired_permutation_p(diff: np.ndarray) -> float:
    """Exact one-sided paired permutation (sign-flip) test.

    H0: mean paired difference <= 0.  Returns the fraction of sign-flip
    permutations whose mean >= observed.  Exact for small n (2^n perms).
    """
    n = diff.size
    if n == 0:
        return 1.0
    obs = diff.mean()
    if n <= 20:
        ge = 0
        total = 0
        for signs in itertools.product([1, -1], repeat=n):
            total += 1
            if (diff * np.array(signs)).mean() >= obs - 1e-12:
                ge += 1
        return ge / total
    # fall back to Monte-Carlo for larger n
    ge = 0
    for _ in range(BOOTSTRAP_N):
        signs = RNG.choice([1, -1], size=n)
        if (diff * signs).mean() >= obs - 1e-12:
            ge += 1
    return ge / BOOTSTRAP_N


# ----------------------------------------------------------------------
# Per-task analysis (IOI / MultiStep share the same per_prompt schema)
# ----------------------------------------------------------------------

ABLATION_METHODS = {
    "ai_abl": ("ai_abl_mean", "ai_abl_cumkl"),
    "bandit": ("bandit_mean", "bandit_cumkl"),
    "eap": ("eap_mean", "eap_cumkl"),
    "greedy": ("greedy_mean", "greedy_cumkl"),
    "random": ("random_mean", "random_cumkl"),
}

MULTIACTION_METHODS = {
    "ai": ("ai_mean", "ai_cumkl"),
    "greedy_steer": (None, "greedy_steer_cumkl"),
    "bandit_steer": (None, "bandit_steer_cumkl"),
    "random_steer": (None, "random_steer_cumkl"),
    "random_action": ("random_action_mean", "random_action_cumkl"),
}


def analyse_task(data: dict) -> dict:
    pp = data["per_prompt"]
    out: Dict[str, Any] = {"n_prompts": len(pp), "budget": data.get("budget")}

    # --- oracle KL (the values reviewers asked to see) ---
    oracle_cum = [p["oracle_cumkl"] for p in pp]
    oracle_mean_per_prompt = [float(np.mean(p["oracle_kls"])) for p in pp if p.get("oracle_kls")]
    out["oracle_mean_kl"] = _q(oracle_mean_per_prompt)
    out["oracle_cumkl"] = _q(oracle_cum)
    if all("oracle_steer_cumkl" in p for p in pp):
        out["oracle_steer_cumkl"] = _q([p["oracle_steer_cumkl"] for p in pp])

    # --- per-method mean KL summaries (mean/std/median/IQR) ---
    methods: Dict[str, Any] = {}
    for name, (mean_key, cum_key) in {**ABLATION_METHODS, **MULTIACTION_METHODS}.items():
        entry: Dict[str, Any] = {}
        if mean_key and all(mean_key in p for p in pp):
            per_prompt_mean = [p[mean_key] for p in pp]
            entry["mean_kl"] = _q(per_prompt_mean)
            entry["mean_kl_ci"] = _bootstrap_mean_ci(per_prompt_mean)
        if cum_key and all(cum_key in p for p in pp):
            num = [p[cum_key] for p in pp]
            ci = _bootstrap_ratio_ci(num, oracle_cum)
            if name in ABLATION_METHODS:
                entry["oracle_eff"] = ci  # bounded efficiency
            else:
                entry["rck"] = ci          # relative cumulative KL
        methods[name] = entry
    out["methods"] = methods

    # --- H1: method vs random, one-sided paired test (per-prompt mean KL) ---
    rand = np.array([p["random_mean"] for p in pp])
    h1: Dict[str, Any] = {}
    for name in ("ai", "ai_abl"):
        mk = ABLATION_METHODS.get(name, MULTIACTION_METHODS.get(name))[0]
        if mk and all(mk in p for p in pp):
            vals = np.array([p[mk] for p in pp])
            diff = vals - rand
            t = stats.ttest_rel(vals, rand, alternative="greater")
            h1[name] = {
                "mean_diff": float(diff.mean()),
                "t_stat": float(t.statistic),
                "p_ttest_onesided": float(t.pvalue),
                "p_permutation_onesided": _paired_permutation_p(diff),
                "improvement_pct": float((vals.mean() - rand.mean()) / max(rand.mean(), 1e-12) * 100),
            }
    out["H1"] = h1

    # --- action distribution tallies ---
    action_counts: Dict[str, int] = {}
    total_actions = 0
    first_step_counts: Dict[str, int] = {}
    post_first: Dict[str, int] = {}
    post_first_total = 0
    for p in pp:
        acts = p.get("ai_actions", [])
        for s, a in enumerate(acts):
            action_counts[a] = action_counts.get(a, 0) + 1
            total_actions += 1
            if s == 0:
                first_step_counts[a] = first_step_counts.get(a, 0) + 1
            else:
                post_first[a] = post_first.get(a, 0) + 1
                post_first_total += 1
    out["actions"] = {
        "counts": action_counts,
        "total": total_actions,
        "first_step": first_step_counts,
        "post_first": post_first,
        "post_first_total": post_first_total,
    }
    return out


def analyse_domain(data: dict) -> dict:
    by_domain = data.get("by_domain", {})
    out = {"domains": {}}
    ai_all, rand_all = [], []
    for dom, d in by_domain.items():
        pp = d.get("per_prompt", [])
        ai = [p["ai_mean"] for p in pp]
        band = [p["bandit_mean"] for p in pp]
        rnd = [p["random_mean"] for p in pp]
        ai_all += ai
        rand_all += rnd
        out["domains"][dom] = {
            "ai_mean_kl": float(np.mean(ai)) if ai else 0.0,
            "bandit_mean_kl": float(np.mean(band)) if band else 0.0,
            "random_mean_kl": float(np.mean(rnd)) if rnd else 0.0,
            "ai_vs_random_pct": d.get("ai_vs_random_pct"),
            "ai_vs_greedy_pct": d.get("ai_vs_greedy_pct"),
            "layer_distribution": d.get("layer_distribution"),
        }
    if ai_all:
        diff = np.array(ai_all) - np.array(rand_all)
        t = stats.ttest_rel(ai_all, rand_all, alternative="greater")
        out["H1_pooled"] = {
            "mean_diff": float(diff.mean()),
            "p_ttest_onesided": float(t.pvalue),
            "p_permutation_onesided": _paired_permutation_p(diff),
            "improvement_pct": float((np.mean(ai_all) - np.mean(rand_all)) / max(np.mean(rand_all), 1e-12) * 100),
        }
    return out


def analyse_steering(data: dict) -> dict:
    pp = data["per_prompt"]
    mults = data.get("multipliers", [])
    out: Dict[str, Any] = {"multipliers": mults, "n_prompts": len(pp)}

    def _changes(prompts, key="features"):
        per_mult = {m: {"changed": 0, "total": 0} for m in mults}
        max_kls = []
        for p in prompts:
            for feat in p.get(key, []):
                fkls = []
                for m in mults:
                    cell = feat.get(f"mult_{m}")
                    if isinstance(cell, dict):
                        per_mult[m]["total"] += 1
                        if cell.get("prediction_changed"):
                            per_mult[m]["changed"] += 1
                        fkls.append(cell.get("kl", 0.0))
                if fkls:
                    max_kls.append(max(fkls))
        return per_mult, max_kls

    top_changes, top_maxkl = _changes(pp, "features")
    ctrl_changes, ctrl_maxkl = _changes(pp, "control_features")
    out["top_changes_per_mult"] = top_changes
    out["control_changes_per_mult"] = ctrl_changes
    out["top_max_kl"] = _q(top_maxkl)
    out["control_max_kl"] = _q(ctrl_maxkl)

    # H2 binomial test at the largest multiplier, vs 1% chance.
    if mults:
        m_max = max(mults)
        c = top_changes.get(m_max, {"changed": 0, "total": 0})
        if c["total"] > 0:
            bt = stats.binomtest(c["changed"], c["total"], 0.01, alternative="greater")
            out["H2_binomial"] = {
                "multiplier": m_max,
                "changed": c["changed"], "total": c["total"],
                "p_value": float(bt.pvalue),
            }
        cc = ctrl_changes.get(m_max, {"changed": 0, "total": 0})
        out["H2_control"] = {
            "multiplier": m_max,
            "changed": cc["changed"], "total": cc["total"],
        }
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute paper statistics from results JSON.")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/paper_stats.json")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    report: Dict[str, Any] = {}

    for model in ("gemma", "llama"):
        m: Dict[str, Any] = {}
        for task, fn in (("ioi", analyse_task), ("multistep", analyse_task)):
            data = _load(rd / f"{task}_results_{model}.json")
            if data:
                m[task] = fn(data)
        dom = _load(rd / f"domain_results_{model}.json")
        if dom:
            m["domain"] = analyse_domain(dom)
        st = _load(rd / f"steering_results_{model}.json")
        if st:
            m["steering"] = analyse_steering(st)
        # finer-bin Llama multistep remedy
        fb = _load(rd / f"multistep_results_{model}_finebins.json")
        if fb:
            m["multistep_finebins"] = analyse_task(fb)
        report[model] = m

    # sensitivity sweeps (IOI; whatever suffixes exist)
    sens: Dict[str, Any] = {}
    for p in sorted(rd.glob("ioi_results_*_sens*.json")):
        data = _load(p)
        if data:
            sens[p.stem] = {
                "agent_kwargs": data.get("agent_kwargs"),
                "ai_abl_oracle_efficiency": data.get("aggregate", {}).get("ai_abl_oracle_efficiency"),
                "ai_rck": data.get("aggregate", {}).get("ai_rck"),
                "bandit_oracle_efficiency": data.get("aggregate", {}).get("bandit_oracle_efficiency"),
            }
    if sens:
        report["sensitivity"] = sens

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # --- human-readable summary ---
    print("=" * 72)
    print("PAPER STATISTICS SUMMARY")
    print("=" * 72)
    for model in ("gemma", "llama"):
        if model not in report:
            continue
        print(f"\n### {model.upper()}")
        for task in ("ioi", "multistep"):
            t = report[model].get(task)
            if not t:
                continue
            print(f"\n  [{task}] n={t['n_prompts']}  oracle mean KL="
                  f"{t['oracle_mean_kl']['mean']:.6f}  oracle cumKL={t['oracle_cumkl']['mean']:.6f}")
            for name, e in t["methods"].items():
                eff = e.get("oracle_eff") or e.get("rck")
                mk = e.get("mean_kl", {})
                tag = "eff" if "oracle_eff" in e else "RCK"
                if eff:
                    print(f"    {name:14s} meanKL={mk.get('mean',0):.6f} "
                          f"(med={mk.get('median',0):.6f}) {tag}={eff['point']:.1f}% "
                          f"[{eff['lo']:.1f}, {eff['hi']:.1f}]")
            for name, h in t.get("H1", {}).items():
                print(f"    H1 {name} vs random: +{h['improvement_pct']:.1f}%  "
                      f"p_t={h['p_ttest_onesided']:.4f}  p_perm={h['p_permutation_onesided']:.4f}")
            act = t.get("actions", {})
            if act.get("total"):
                pf = act.get("post_first", {})
                print(f"    actions total={act['total']}  counts={act['counts']}  "
                      f"post-first steering={pf.get('feature_steering',0)}/{act['post_first_total']}")
        st = report[model].get("steering")
        if st and "H2_binomial" in st:
            h2 = st["H2_binomial"]; hc = st.get("H2_control", {})
            print(f"  [steering] H2 m={h2['multiplier']}: top {h2['changed']}/{h2['total']} "
                  f"(p={h2['p_value']:.2e}); control {hc.get('changed',0)}/{hc.get('total',0)}")
        fb = report[model].get("multistep_finebins")
        if fb:
            aieff = fb["methods"].get("ai_abl", {}).get("oracle_eff", {})
            print(f"  [multistep finebins] POMDP-abl eff={aieff.get('point',0):.1f}%")
    if "sensitivity" in report:
        print("\n### SENSITIVITY (IOI)")
        for k, v in report["sensitivity"].items():
            print(f"  {k}: kwargs={v['agent_kwargs']} POMDP-abl eff={v['ai_abl_oracle_efficiency']}")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

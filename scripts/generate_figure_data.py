#!/usr/bin/env python3
"""Generate LaTeX pgfplots data from JSON experiment results.

Reads results/*.json and writes paper/figures/*.tex with data
that exactly matches the experiment outputs.  Generates side-by-side
(groupplot) figures for Gemma-2-2B and Llama-3.2-1B where both
results are available.

Usage:
    python -m scripts.generate_figure_data [--results-dir results] [--figures-dir paper/figures]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _try_load(results_dir: Path, model_name: str, task: str) -> Optional[dict]:
    path = results_dir / f"{task}_results_{model_name}.json"
    if not path.exists():
        path = results_dir / f"{task}_results.json"
    if path.exists():
        return load_json(str(path))
    return None


# ======================================================================
# IOI aggregation
# ======================================================================

def compute_ioi_aggregate(data: dict) -> dict:
    prompts = data["per_prompt"]
    n = len(prompts)

    def avg(key):
        return sum(p.get(key, 0) for p in prompts) / n

    ai_cum     = avg("ai_cumkl")
    bandit_cum = avg("bandit_cumkl") if "bandit_cumkl" in prompts[0] else avg("ai_cumkl")
    greedy_cum = avg("greedy_cumkl")
    random_cum = avg("random_cumkl")
    oracle_cum = avg("oracle_cumkl")

    def eff(cum):
        return cum / oracle_cum * 100 if oracle_cum > 0 else 0

    return {
        "ai_mean": avg("ai_mean"),
        "bandit_mean": avg("bandit_mean") if "bandit_mean" in prompts[0] else avg("ai_mean"),
        "greedy_mean": avg("greedy_mean"),
        "random_mean": avg("random_mean"),
        "ai_oracle_eff": eff(ai_cum),
        "bandit_oracle_eff": eff(bandit_cum),
        "greedy_oracle_eff": eff(greedy_cum),
        "random_oracle_eff": eff(random_cum),
    }


# ======================================================================
# Cumulative KL line graph (side-by-side Gemma + Llama)
# ======================================================================

def _compute_cumkl_arrays(data: dict) -> Tuple[np.ndarray, ...]:
    prompts = data["per_prompt"]
    budget = data.get("budget", 20)
    n = len(prompts)

    ai_cum     = np.zeros(budget)
    bandit_cum = np.zeros(budget)
    greedy_cum = np.zeros(budget)
    oracle_cum = np.zeros(budget)

    for p in prompts:
        ai_kl     = p.get("ai_kls", [])
        bandit_kl = p.get("bandit_kls", ai_kl)
        greedy_kl = p.get("greedy_kls", [])
        oracle_kl = p.get("oracle_kls", [])

        for arr, cum in [
            (ai_kl, ai_cum), (bandit_kl, bandit_cum),
            (greedy_kl, greedy_cum), (oracle_kl, oracle_cum),
        ]:
            for i in range(min(budget, len(arr))):
                cum[i] += sum(arr[: i + 1]) / n

    return oracle_cum, ai_cum, bandit_cum, greedy_cum, budget


def _write_axis_block(f, oracle, ai, bandit, greedy, budget, title, show_legend=True):
    """Write a single axis block for cumulative KL."""
    f.write(
        f"\\nextgroupplot[title={{{title}}},\n"
        f"    xlabel={{Intervention step}},\n"
        f"    ylabel={{Cumulative KL}},\n"
        f"    xmin=1, xmax={budget},\n"
        f"]\n\n"
    )
    def coords(vals):
        return " ".join(f"({i+1},{v:.6f})" for i, v in enumerate(vals))

    f.write(f"\\addplot[black, dashed, thick] coordinates {{ {coords(oracle)} }};\n")
    f.write(f"\\addplot[red, thick] coordinates {{ {coords(ai)} }};\n")
    f.write(f"\\addplot[orange, thick] coordinates {{ {coords(bandit)} }};\n")
    f.write(f"\\addplot[blue, thick] coordinates {{ {coords(greedy)} }};\n\n")
    if show_legend:
        f.write("\\legend{Oracle, POMDP Agent, Bandit, Greedy}\n\n")


def write_cumulative_kl(gemma_data: dict, llama_data: Optional[dict], out_path: str):
    g_oracle, g_ai, g_bandit, g_greedy, g_budget = _compute_cumkl_arrays(gemma_data)

    with open(out_path, "w") as f:
        f.write("% Auto-generated from experiment results.\n")

        if llama_data:
            l_oracle, l_ai, l_bandit, l_greedy, l_budget = _compute_cumkl_arrays(llama_data)
            f.write(
                "\\begin{tikzpicture}\n"
                "\\begin{groupplot}[\n"
                "    group style={\n"
                "        group size=2 by 1,\n"
                "        horizontal sep=1.2cm,\n"
                "        ylabels at=edge left,\n"
                "    },\n"
                "    width=0.52\\columnwidth,\n"
                "    height=4.5cm,\n"
                "    x label style={font=\\scriptsize},\n"
                "    y label style={font=\\scriptsize},\n"
                "    tick label style={font=\\scriptsize},\n"
                "    title style={font=\\scriptsize\\bfseries},\n"
                "    legend style={\n"
                "        font=\\tiny,\n"
                "        at={(0.02,0.98)},\n"
                "        anchor=north west\n"
                "    },\n"
                "]\n\n"
            )
            _write_axis_block(f, g_oracle, g_ai, g_bandit, g_greedy, g_budget, "Gemma-2-2B", show_legend=True)
            _write_axis_block(f, l_oracle, l_ai, l_bandit, l_greedy, l_budget, "Llama-3.2-1B", show_legend=False)
            f.write("\\end{groupplot}\n\\end{tikzpicture}\n")
        else:
            f.write(
                "\\begin{tikzpicture}\n"
                "\\begin{axis}[\n"
                "    width=\\columnwidth,\n"
                "    height=5cm,\n"
                "    xlabel={Intervention step},\n"
                "    ylabel={Cumulative KL},\n"
                "    x label style={font=\\scriptsize},\n"
                "    y label style={font=\\scriptsize},\n"
                "    tick label style={font=\\scriptsize},\n"
                "    legend style={\n"
                "        font=\\tiny,\n"
                "        at={(0.02,0.98)},\n"
                "        anchor=north west\n"
                "    },\n"
                f"    xmin=1, xmax={g_budget},\n"
                "]\n\n"
            )
            def coords(vals):
                return " ".join(f"({i+1},{v:.6f})" for i, v in enumerate(vals))
            f.write(f"\\addplot[black, dashed, thick] coordinates {{ {coords(g_oracle)} }};\n")
            f.write(f"\\addplot[red, thick] coordinates {{ {coords(g_ai)} }};\n")
            f.write(f"\\addplot[orange, thick] coordinates {{ {coords(g_bandit)} }};\n")
            f.write(f"\\addplot[blue, thick] coordinates {{ {coords(g_greedy)} }};\n\n")
            f.write("\\legend{Oracle, POMDP Agent, Bandit, Greedy}\n\n")
            f.write("\\end{axis}\n\\end{tikzpicture}\n")

    print(f"  Wrote {out_path}")


# ======================================================================
# IOI comparison bar chart (Gemma + Llama grouped)
# ======================================================================

def _panel_ymax(aggs: list) -> float:
    """Compute a sensible ymax from a list of aggregate dicts."""
    vals = []
    for a in aggs:
        if a is None:
            continue
        for k in ('random_oracle_eff', 'greedy_oracle_eff',
                  'bandit_oracle_eff', 'ai_oracle_eff'):
            vals.append(a.get(k, 0))
    peak = max(vals) if vals else 100
    headroom = peak * 1.15
    return max(headroom, 110)


def write_ioi_comparison(
    g_ioi_agg: dict, g_ms_agg: Optional[dict],
    l_ioi_agg: Optional[dict], l_ms_agg: Optional[dict],
    out_path: str
):
    with open(out_path, "w") as f:
        f.write("% Auto-generated from experiment results.\n")

        if l_ioi_agg:
            g_ymax = _panel_ymax([g_ioi_agg, g_ms_agg])
            l_ymax = _panel_ymax([l_ioi_agg, l_ms_agg])
            f.write(
                "\\begin{tikzpicture}\n"
                "\\begin{groupplot}[\n"
                "    group style={\n"
                "        group size=2 by 1,\n"
                "        horizontal sep=1.2cm,\n"
                "        ylabels at=edge left,\n"
                "    },\n"
                "    width=0.52\\columnwidth,\n"
                "    height=4.5cm,\n"
                "    ylabel={Oracle Efficiency (\\%)},\n"
                "    symbolic x coords={Random, Greedy, Bandit, POMDP},\n"
                "    xtick=data,\n"
                "    x tick label style={font=\\tiny},\n"
                "    y tick label style={font=\\scriptsize},\n"
                "    ylabel style={font=\\scriptsize},\n"
                "    title style={font=\\scriptsize\\bfseries},\n"
                "    ymin=0,\n"
                "    enlarge x limits=0.2,\n"
                "    nodes near coords,\n"
                "    nodes near coords style={font=\\tiny},\n"
                "    every node near coord/.append style={/pgf/number format/fixed,\n"
                "        /pgf/number format/precision=1},\n"
                "    legend style={\n"
                "        font=\\tiny,\n"
                "        at={(0.5,-0.28)},\n"
                "        anchor=north,\n"
                "        legend columns=2\n"
                "    },\n"
                "]\n\n"
            )
            # Gemma panel
            f.write(f"\\nextgroupplot[ybar, bar width=5pt, title={{Gemma-2-2B}}, ymax={g_ymax:.0f}]\n")
            f.write(
                f"\\addplot[fill=blue!40, draw=blue!60] coordinates {{\n"
                f"    (Random, {g_ioi_agg['random_oracle_eff']:.1f}) "
                f"(Greedy, {g_ioi_agg['greedy_oracle_eff']:.1f}) "
                f"(Bandit, {g_ioi_agg['bandit_oracle_eff']:.1f}) "
                f"(POMDP, {g_ioi_agg['ai_oracle_eff']:.1f})\n"
                f"}};\n\n"
            )
            if g_ms_agg:
                f.write(
                    f"\\addplot[fill=red!40, draw=red!60] coordinates {{\n"
                    f"    (Random, {g_ms_agg['random_oracle_eff']:.1f}) "
                    f"(Greedy, {g_ms_agg['greedy_oracle_eff']:.1f}) "
                    f"(Bandit, {g_ms_agg['bandit_oracle_eff']:.1f}) "
                    f"(POMDP, {g_ms_agg['ai_oracle_eff']:.1f})\n"
                    f"}};\n\n"
                )
            f.write("\\legend{IOI, Multi-step}\n\n")

            # Llama panel
            f.write(f"\\nextgroupplot[ybar, bar width=5pt, title={{Llama-3.2-1B}}, ymax={l_ymax:.0f}]\n")
            f.write(
                f"\\addplot[fill=blue!40, draw=blue!60] coordinates {{\n"
                f"    (Random, {l_ioi_agg['random_oracle_eff']:.1f}) "
                f"(Greedy, {l_ioi_agg['greedy_oracle_eff']:.1f}) "
                f"(Bandit, {l_ioi_agg['bandit_oracle_eff']:.1f}) "
                f"(POMDP, {l_ioi_agg['ai_oracle_eff']:.1f})\n"
                f"}};\n\n"
            )
            if l_ms_agg:
                f.write(
                    f"\\addplot[fill=red!40, draw=red!60] coordinates {{\n"
                    f"    (Random, {l_ms_agg['random_oracle_eff']:.1f}) "
                    f"(Greedy, {l_ms_agg['greedy_oracle_eff']:.1f}) "
                    f"(Bandit, {l_ms_agg['bandit_oracle_eff']:.1f}) "
                    f"(POMDP, {l_ms_agg['ai_oracle_eff']:.1f})\n"
                    f"}};\n\n"
                )
            f.write("\\end{groupplot}\n\\end{tikzpicture}\n")
        else:
            # Gemma-only fallback
            g_ymax = _panel_ymax([g_ioi_agg, g_ms_agg])
            f.write(
                "\\begin{tikzpicture}\n"
                "\\begin{axis}[\n"
                "    ybar,\n"
                "    width=\\columnwidth,\n"
                "    height=5cm,\n"
                "    bar width=6pt,\n"
                "    ylabel={Oracle Efficiency (\\%)},\n"
                "    symbolic x coords={Random, Greedy, Bandit, POMDP},\n"
                "    xtick=data,\n"
                "    x tick label style={font=\\scriptsize},\n"
                "    y tick label style={font=\\scriptsize},\n"
                "    ylabel style={font=\\scriptsize},\n"
                "    legend style={\n"
                "        font=\\scriptsize,\n"
                "        at={(0.5,-0.22)},\n"
                "        anchor=north,\n"
                "        legend columns=2\n"
                "    },\n"
                f"    ymin=0, ymax={g_ymax:.0f},\n"
                "    enlarge x limits=0.2,\n"
                "    nodes near coords,\n"
                "    nodes near coords style={font=\\tiny},\n"
                "    every node near coord/.append style={/pgf/number format/fixed,\n"
                "        /pgf/number format/precision=1},\n"
                "]\n\n"
            )
            f.write(
                f"\\addplot[fill=blue!40, draw=blue!60] coordinates {{\n"
                f"    (Random, {g_ioi_agg['random_oracle_eff']:.1f}) "
                f"(Greedy, {g_ioi_agg['greedy_oracle_eff']:.1f}) "
                f"(Bandit, {g_ioi_agg['bandit_oracle_eff']:.1f}) "
                f"(POMDP, {g_ioi_agg['ai_oracle_eff']:.1f})\n"
                f"}};\n\n"
            )
            if g_ms_agg:
                f.write(
                    f"\\addplot[fill=red!40, draw=red!60] coordinates {{\n"
                    f"    (Random, {g_ms_agg['random_oracle_eff']:.1f}) "
                    f"(Greedy, {g_ms_agg['greedy_oracle_eff']:.1f}) "
                    f"(Bandit, {g_ms_agg['bandit_oracle_eff']:.1f}) "
                    f"(POMDP, {g_ms_agg['ai_oracle_eff']:.1f})\n"
                    f"}};\n\n"
                )
                f.write("\\legend{IOI, Multi-step}\n\n")
            else:
                f.write("\\legend{IOI}\n\n")
            f.write("\\end{axis}\n\\end{tikzpicture}\n")
    print(f"  Wrote {out_path}")


# ======================================================================
# Steering KL line graph (side-by-side Gemma + Llama)
# ======================================================================

def _steering_coords(data: dict, multipliers: list) -> list:
    """Return list of (concept_label, coords_string) for up to 3 concepts."""
    prompts = data["per_prompt"]
    result = []
    for idx, p in enumerate(prompts[:3]):
        mean_kls = p.get("mean_kl_per_multiplier", None)
        if mean_kls is None or len(mean_kls) != len(multipliers):
            features = p.get("features", [])
            mean_kls = []
            for m in multipliers:
                kls = []
                for feat in features:
                    key = f"mult_{m}"
                    if key in feat and isinstance(feat[key], dict):
                        kls.append(feat[key].get("kl", 0))
                mean_kls.append(float(np.mean(kls)) if kls else 0.0)
        coords_str = " ".join(f"({m},{kl:.6f})" for m, kl in zip(multipliers, mean_kls))
        concept = p.get("concept", p.get("prompt", f"Concept {idx+1}"))
        label = concept[:20].replace("_", " ").strip()
        result.append((label, coords_str))
    return result


def write_steering_heatmap(gemma_data: dict, llama_data: Optional[dict], out_path: str):
    multipliers = gemma_data.get("multipliers", [0, 2, 5, 10])
    colors = ["blue", "red", "green!60!black"]

    with open(out_path, "w") as f:
        f.write("% Auto-generated from experiment results.\n")

        if llama_data:
            g_coords = _steering_coords(gemma_data, multipliers)
            l_coords = _steering_coords(llama_data, multipliers)
            f.write(
                "\\begin{tikzpicture}\n"
                "\\begin{groupplot}[\n"
                "    group style={\n"
                "        group size=2 by 1,\n"
                "        horizontal sep=1.2cm,\n"
                "        ylabels at=edge left,\n"
                "    },\n"
                "    width=0.52\\columnwidth,\n"
                "    height=4.5cm,\n"
                "    xlabel={Steering multiplier $m$},\n"
                "    ylabel={Mean KL divergence},\n"
                "    x label style={font=\\scriptsize},\n"
                "    y label style={font=\\scriptsize},\n"
                "    tick label style={font=\\scriptsize},\n"
                "    title style={font=\\scriptsize\\bfseries},\n"
                "    legend style={\n"
                "        font=\\tiny,\n"
                "        at={(0.02,0.98)},\n"
                "        anchor=north west\n"
                "    },\n"
                "    xtick={0,2,5,10},\n"
                "]\n\n"
            )
            # Gemma panel
            f.write("\\nextgroupplot[title={Gemma-2-2B}]\n")
            for idx, (label, cs) in enumerate(g_coords):
                f.write(f"\\addplot[{colors[idx]}, thick, mark=*] coordinates {{ {cs} }};\n")
            legends = ", ".join(lbl for lbl, _ in g_coords)
            f.write(f"\\legend{{{legends}}}\n\n")

            # Llama panel
            f.write("\\nextgroupplot[title={Llama-3.2-1B}]\n")
            for idx, (label, cs) in enumerate(l_coords):
                f.write(f"\\addplot[{colors[idx]}, thick, mark=*] coordinates {{ {cs} }};\n")
            f.write("\n")

            f.write("\\end{groupplot}\n\\end{tikzpicture}\n")
        else:
            f.write(
                "\\begin{tikzpicture}\n"
                "\\begin{axis}[\n"
                "    width=\\columnwidth,\n"
                "    height=5cm,\n"
                "    xlabel={Steering multiplier $m$},\n"
                "    ylabel={Mean KL divergence},\n"
                "    x label style={font=\\scriptsize},\n"
                "    y label style={font=\\scriptsize},\n"
                "    tick label style={font=\\scriptsize},\n"
                "    legend style={\n"
                "        font=\\tiny,\n"
                "        at={(0.02,0.98)},\n"
                "        anchor=north west\n"
                "    },\n"
                "    xtick={0,2,5,10},\n"
                "]\n\n"
            )
            g_coords = _steering_coords(gemma_data, multipliers)
            for idx, (label, cs) in enumerate(g_coords):
                f.write(f"\\addplot[{colors[idx]}, thick, mark=*] coordinates {{ {cs} }};\n")
            legends = ", ".join(lbl for lbl, _ in g_coords)
            f.write(f"\\legend{{{legends}}}\n\n")
            f.write("\\end{axis}\n\\end{tikzpicture}\n")
    print(f"  Wrote {out_path}")


# ======================================================================
# Layer distribution bar chart (side-by-side Gemma + Llama)
# ======================================================================

def _aggregate_layer_dist(data: dict) -> dict:
    """Aggregate layer_distribution across prompts."""
    prompts = data["per_prompt"]
    total = {"early": 0, "mid": 0, "late": 0}
    for p in prompts:
        ld = p.get("layer_distribution", {})
        for k in total:
            total[k] += ld.get(k, 0)
    return total


def write_layer_distribution(
    gemma_ioi_ld: Optional[dict], gemma_ms_ld: dict,
    llama_ioi_ld: Optional[dict], llama_ms_ld: Optional[dict],
    out_path: str
):
    with open(out_path, "w") as f:
        f.write("% Auto-generated from experiment results.\n")

        if llama_ms_ld:
            f.write(
                "\\begin{tikzpicture}\n"
                "\\begin{groupplot}[\n"
                "    group style={\n"
                "        group size=2 by 1,\n"
                "        horizontal sep=1.2cm,\n"
                "        ylabels at=edge left,\n"
                "    },\n"
                "    width=0.52\\columnwidth,\n"
                "    height=4.5cm,\n"
                "    ylabel={Count (top-10 features)},\n"
                "    symbolic x coords={Early, Mid, Late},\n"
                "    xtick=data,\n"
                "    x tick label style={font=\\scriptsize},\n"
                "    y tick label style={font=\\scriptsize},\n"
                "    ylabel style={font=\\scriptsize},\n"
                "    title style={font=\\scriptsize\\bfseries},\n"
                "    ymin=0,\n"
                "    enlarge x limits=0.3,\n"
                "    nodes near coords,\n"
                "    nodes near coords style={font=\\tiny},\n"
                "    legend style={\n"
                "        font=\\tiny,\n"
                "        at={(0.5,-0.28)},\n"
                "        anchor=north,\n"
                "        legend columns=2\n"
                "    },\n"
                "]\n\n"
            )
            # Gemma panel
            f.write("\\nextgroupplot[ybar, bar width=8pt, title={Gemma-2-2B}]\n")
            f.write(
                f"\\addplot[fill=blue!40, draw=blue!60] coordinates {{\n"
                f"    (Early, {gemma_ms_ld['early']}) (Mid, {gemma_ms_ld['mid']}) (Late, {gemma_ms_ld['late']})\n"
                f"}};\n\n"
            )
            if gemma_ioi_ld:
                f.write(
                    f"\\addplot[fill=red!40, draw=red!60] coordinates {{\n"
                    f"    (Early, {gemma_ioi_ld['early']}) (Mid, {gemma_ioi_ld['mid']}) (Late, {gemma_ioi_ld['late']})\n"
                    f"}};\n\n"
                )
            f.write("\\legend{Multi-step, IOI}\n\n")

            # Llama panel
            f.write("\\nextgroupplot[ybar, bar width=8pt, title={Llama-3.2-1B}]\n")
            f.write(
                f"\\addplot[fill=blue!40, draw=blue!60] coordinates {{\n"
                f"    (Early, {llama_ms_ld['early']}) (Mid, {llama_ms_ld['mid']}) (Late, {llama_ms_ld['late']})\n"
                f"}};\n\n"
            )
            if llama_ioi_ld:
                f.write(
                    f"\\addplot[fill=red!40, draw=red!60] coordinates {{\n"
                    f"    (Early, {llama_ioi_ld['early']}) (Mid, {llama_ioi_ld['mid']}) (Late, {llama_ioi_ld['late']})\n"
                    f"}};\n\n"
                )
            f.write("\\end{groupplot}\n\\end{tikzpicture}\n")
        else:
            # Gemma-only fallback
            f.write(
                "\\begin{tikzpicture}\n"
                "\\begin{axis}[\n"
                "    ybar,\n"
                "    width=\\columnwidth,\n"
                "    height=5cm,\n"
                "    bar width=10pt,\n"
                "    ylabel={Count (top-10 features)},\n"
                "    symbolic x coords={Early, Mid, Late},\n"
                "    xtick=data,\n"
                "    x tick label style={font=\\scriptsize},\n"
                "    y tick label style={font=\\scriptsize},\n"
                "    ylabel style={font=\\scriptsize},\n"
                "    legend style={\n"
                "        font=\\scriptsize,\n"
                "        at={(0.5,-0.22)},\n"
                "        anchor=north,\n"
                "        legend columns=2\n"
                "    },\n"
                "    ymin=0,\n"
                "    enlarge x limits=0.3,\n"
                "    nodes near coords,\n"
                "    nodes near coords style={font=\\tiny},\n"
                "]\n\n"
            )
            f.write(
                f"\\addplot[fill=blue!40, draw=blue!60] coordinates {{\n"
                f"    (Early, {gemma_ms_ld['early']}) (Mid, {gemma_ms_ld['mid']}) (Late, {gemma_ms_ld['late']})\n"
                f"}};\n\n"
            )
            if gemma_ioi_ld:
                f.write(
                    f"\\addplot[fill=red!40, draw=red!60] coordinates {{\n"
                    f"    (Early, {gemma_ioi_ld['early']}) (Mid, {gemma_ioi_ld['mid']}) (Late, {gemma_ioi_ld['late']})\n"
                    f"}};\n\n"
                )
            f.write("\\legend{Multi-step, IOI}\n\n")
            f.write("\\end{axis}\n\\end{tikzpicture}\n")
    print(f"  Wrote {out_path}")


# ======================================================================
# Domain layer distribution (side-by-side Gemma + Llama)
# ======================================================================

def _extract_domain_layers(data: dict) -> dict:
    """Return {domain: {early, mid, late}} from domain results."""
    domains_data = {}
    by_domain = data.get("by_domain", data.get("per_domain", {}))
    for domain_name, domain_info in by_domain.items():
        ld = domain_info.get("layer_distribution", {})
        domains_data[domain_name] = {
            "early": ld.get("early", 0),
            "mid": ld.get("mid", 0),
            "late": ld.get("late", 0),
        }
    return domains_data


def write_domain_layers(gemma_domain: dict, llama_domain: Optional[dict], out_path: str):
    g_dl = _extract_domain_layers(gemma_domain)
    domain_order = ["geography", "mathematics", "science", "logic", "history"]
    short_labels = {"geography": "Geo.", "mathematics": "Math", "science": "Sci.",
                    "logic": "Logic", "history": "Hist."}

    with open(out_path, "w") as f:
        f.write("% Auto-generated from experiment results.\n")

        if llama_domain:
            l_dl = _extract_domain_layers(llama_domain)
            f.write(
                "\\begin{tikzpicture}\n"
                "\\begin{groupplot}[\n"
                "    group style={\n"
                "        group size=2 by 1,\n"
                "        horizontal sep=1.2cm,\n"
                "        ylabels at=edge left,\n"
                "    },\n"
                "    width=0.52\\columnwidth,\n"
                "    height=5cm,\n"
                "    ylabel={Count (top-10 features)},\n"
                f"    symbolic x coords={{{', '.join(short_labels[d] for d in domain_order)}}},\n"
                "    xtick=data,\n"
                "    x tick label style={font=\\tiny},\n"
                "    y tick label style={font=\\scriptsize},\n"
                "    ylabel style={font=\\scriptsize},\n"
                "    title style={font=\\scriptsize\\bfseries},\n"
                "    ymin=0,\n"
                "    enlarge x limits=0.15,\n"
                "    nodes near coords,\n"
                "    nodes near coords style={font=\\tiny, rotate=90, anchor=west},\n"
                "    legend style={\n"
                "        font=\\tiny,\n"
                "        at={(0.5,-0.30)},\n"
                "        anchor=north,\n"
                "        legend columns=3\n"
                "    },\n"
                "]\n\n"
            )
            # Gemma panel
            f.write("\\nextgroupplot[ybar, bar width=4pt, title={Gemma-2-2B}]\n")
            for region, color in [("early", "blue!50, draw=blue!70"),
                                   ("mid", "green!40, draw=green!60"),
                                   ("late", "red!40, draw=red!60")]:
                coords = " ".join(f"({short_labels[d]}, {g_dl.get(d, {}).get(region, 0)})" for d in domain_order)
                f.write(f"\\addplot[fill={color}] coordinates {{ {coords} }};\n")
            f.write("\\legend{Early, Mid, Late}\n\n")

            # Llama panel
            f.write("\\nextgroupplot[ybar, bar width=4pt, title={Llama-3.2-1B}]\n")
            for region, color in [("early", "blue!50, draw=blue!70"),
                                   ("mid", "green!40, draw=green!60"),
                                   ("late", "red!40, draw=red!60")]:
                coords = " ".join(f"({short_labels[d]}, {l_dl.get(d, {}).get(region, 0)})" for d in domain_order)
                f.write(f"\\addplot[fill={color}] coordinates {{ {coords} }};\n")
            f.write("\n")

            f.write("\\end{groupplot}\n\\end{tikzpicture}\n")
        else:
            # Gemma-only fallback
            f.write(
                "\\begin{tikzpicture}\n"
                "\\begin{axis}[\n"
                "    ybar,\n"
                "    width=\\columnwidth,\n"
                "    height=5.5cm,\n"
                "    bar width=5pt,\n"
                "    ylabel={Count (top-10 features)},\n"
                f"    symbolic x coords={{{', '.join(short_labels[d] for d in domain_order)}}},\n"
                "    xtick=data,\n"
                "    x tick label style={font=\\scriptsize},\n"
                "    y tick label style={font=\\scriptsize},\n"
                "    ylabel style={font=\\scriptsize},\n"
                "    legend style={\n"
                "        font=\\tiny,\n"
                "        at={(0.5,-0.25)},\n"
                "        anchor=north,\n"
                "        legend columns=3\n"
                "    },\n"
                "    ymin=0,\n"
                "    enlarge x limits=0.15,\n"
                "    nodes near coords,\n"
                "    nodes near coords style={font=\\tiny, rotate=90, anchor=west},\n"
                "]\n\n"
            )
            for region, color in [("early", "blue!50, draw=blue!70"),
                                   ("mid", "green!40, draw=green!60"),
                                   ("late", "red!40, draw=red!60")]:
                coords = " ".join(f"({short_labels[d]}, {g_dl.get(d, {}).get(region, 0)})" for d in domain_order)
                f.write(f"\\addplot[fill={color}] coordinates {{ {coords} }};\n")
            f.write("\\legend{Early, Mid, Late}\n\n")
            f.write("\\end{axis}\n\\end{tikzpicture}\n")
    print(f"  Wrote {out_path}")


# ======================================================================
# Attribution graph (add Llama panel)
# ======================================================================

def write_attribution_graph(out_path: str):
    """Write a side-by-side attribution graph: Gemma (L0-L25) and Llama (L0-L15)."""
    with open(out_path, "w") as f:
        f.write(
            "% Simplified attribution graph visualisation\n"
            "% Side-by-side: Gemma-2-2B (26 layers) and Llama-3.2-1B (16 layers)\n"
            "\\begin{tikzpicture}[\n"
            "  node distance=0.6cm and 1.2cm,\n"
            "  feat/.style={circle, draw, minimum size=0.4cm, inner sep=0pt, font=\\tiny},\n"
            "  high/.style={feat, fill=red!40, thick},\n"
            "  med/.style={feat, fill=orange!30},\n"
            "  low/.style={feat, fill=gray!15},\n"
            "  edge/.style={->, >=stealth, thin, gray!60},\n"
            "  strong/.style={->, >=stealth, thick, red!70},\n"
            "  label/.style={font=\\scriptsize, text=black!70},\n"
            "  ptitle/.style={font=\\scriptsize\\bfseries, anchor=south},\n"
            "]\n\n"
            "% --- Gemma-2-2B (left) ---\n"
            "\\node[ptitle] at (1.2, 5.4) {Gemma-2-2B};\n"
            "\\node[label, rotate=90] at (-0.7, 0) {L0};\n"
            "\\node[label, rotate=90] at (-0.7, 1.2) {L8};\n"
            "\\node[label, rotate=90] at (-0.7, 2.4) {L17};\n"
            "\\node[label, rotate=90] at (-0.7, 3.6) {L24};\n"
            "\\node[label, rotate=90] at (-0.7, 4.8) {L25};\n\n"
            "\\node[low] (g00) at (0, 0) {};\n"
            "\\node[low] (g01) at (0.7, 0) {};\n"
            "\\node[med] (g02) at (1.4, 0) {};\n"
            "\\node[low] (g03) at (2.1, 0) {};\n\n"
            "\\node[med] (g10) at (0.2, 1.2) {};\n"
            "\\node[low] (g11) at (0.9, 1.2) {};\n"
            "\\node[med] (g12) at (1.6, 1.2) {};\n\n"
            "\\node[low] (g20) at (0.4, 2.4) {};\n"
            "\\node[med] (g21) at (1.1, 2.4) {};\n"
            "\\node[low] (g22) at (1.8, 2.4) {};\n\n"
            "\\node[med] (g30) at (0.3, 3.6) {};\n"
            "\\node[high] (g31) at (1.0, 3.6) {};\n"
            "\\node[med] (g32) at (1.7, 3.6) {};\n\n"
            "\\node[high] (g40) at (0.6, 4.8) {};\n"
            "\\node[high] (g41) at (1.4, 4.8) {};\n\n"
            "\\draw[edge] (g00) -- (g10);\n"
            "\\draw[edge] (g01) -- (g11);\n"
            "\\draw[edge] (g02) -- (g10);\n"
            "\\draw[edge] (g02) -- (g12);\n"
            "\\draw[edge] (g03) -- (g12);\n"
            "\\draw[edge] (g10) -- (g20);\n"
            "\\draw[edge] (g11) -- (g21);\n"
            "\\draw[edge] (g12) -- (g21);\n"
            "\\draw[edge] (g12) -- (g22);\n"
            "\\draw[edge] (g20) -- (g30);\n"
            "\\draw[edge] (g22) -- (g32);\n"
            "\\draw[strong] (g10) -- (g21);\n"
            "\\draw[strong] (g21) -- (g31);\n"
            "\\draw[strong] (g31) -- (g40);\n"
            "\\draw[strong] (g31) -- (g41);\n"
            "\\draw[strong] (g32) -- (g41);\n\n"
            "% --- Llama-3.2-1B (right) ---\n"
            "\\begin{scope}[xshift=4.2cm]\n"
            "\\node[ptitle] at (1.2, 5.4) {Llama-3.2-1B};\n"
            "\\node[label, rotate=90] at (-0.7, 0) {L0};\n"
            "\\node[label, rotate=90] at (-0.7, 1.6) {L4};\n"
            "\\node[label, rotate=90] at (-0.7, 3.2) {L11};\n"
            "\\node[label, rotate=90] at (-0.7, 4.8) {L15};\n\n"
            "\\node[med] (l00) at (0, 0) {};\n"
            "\\node[low] (l01) at (0.7, 0) {};\n"
            "\\node[med] (l02) at (1.4, 0) {};\n"
            "\\node[low] (l03) at (2.1, 0) {};\n\n"
            "\\node[high] (l10) at (0.2, 1.6) {};\n"
            "\\node[med] (l11) at (0.9, 1.6) {};\n"
            "\\node[med] (l12) at (1.6, 1.6) {};\n\n"
            "\\node[low] (l20) at (0.4, 3.2) {};\n"
            "\\node[med] (l21) at (1.1, 3.2) {};\n\n"
            "\\node[high] (l30) at (0.6, 4.8) {};\n"
            "\\node[high] (l31) at (1.4, 4.8) {};\n\n"
            "\\draw[edge] (l00) -- (l10);\n"
            "\\draw[edge] (l01) -- (l11);\n"
            "\\draw[edge] (l02) -- (l10);\n"
            "\\draw[edge] (l02) -- (l12);\n"
            "\\draw[edge] (l03) -- (l12);\n"
            "\\draw[edge] (l10) -- (l20);\n"
            "\\draw[edge] (l11) -- (l21);\n"
            "\\draw[strong] (l00) -- (l10);\n"
            "\\draw[strong] (l10) -- (l21);\n"
            "\\draw[strong] (l21) -- (l30);\n"
            "\\draw[strong] (l12) -- (l31);\n"
            "\\end{scope}\n\n"
            "% Legend\n"
            "\\node[low, label=right:{\\tiny Low}] at (9.0, 0.3) {};\n"
            "\\node[med, label=right:{\\tiny Moderate}] at (9.0, 0.9) {};\n"
            "\\node[high, label=right:{\\tiny High}] at (9.0, 1.5) {};\n\n"
            "\\end{tikzpicture}\n"
        )
    print(f"  Wrote {out_path}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX figure data from results")
    parser.add_argument("--results-dir", default="results", help="Directory with JSON results")
    parser.add_argument("--figures-dir", default="paper/figures", help="Output directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figure data from experiment results...")

    # Load Gemma data
    g_ioi = _try_load(results_dir, "gemma", "ioi")
    g_ms  = _try_load(results_dir, "gemma", "multistep")
    g_steer = _try_load(results_dir, "gemma", "steering")
    g_domain = _try_load(results_dir, "gemma", "domain")

    # Load Llama data
    l_ioi = _try_load(results_dir, "llama", "ioi")
    l_ms  = _try_load(results_dir, "llama", "multistep")
    l_steer = _try_load(results_dir, "llama", "steering")
    l_domain = _try_load(results_dir, "llama", "domain")

    g_ioi_agg = None
    g_ms_agg = None
    l_ioi_agg = None
    l_ms_agg = None

    # --- Cumulative KL (Fig 4) ---
    if g_ioi:
        g_ioi_agg = compute_ioi_aggregate(g_ioi)
        print(f"\n  IOI Gemma: POMDP oracle_eff={g_ioi_agg['ai_oracle_eff']:.1f}%")
        write_cumulative_kl(g_ioi, l_ioi, str(figures_dir / "cumulative_kl.tex"))
    else:
        print("  WARNING: No Gemma IOI results found")

    if l_ioi:
        l_ioi_agg = compute_ioi_aggregate(l_ioi)
        print(f"  IOI Llama: POMDP oracle_eff={l_ioi_agg['ai_oracle_eff']:.1f}%")

    # --- Multi-step aggregation ---
    if g_ms:
        g_ms_agg = compute_ioi_aggregate(g_ms)
        print(f"  Multi-step Gemma: POMDP oracle_eff={g_ms_agg['ai_oracle_eff']:.1f}%")
    if l_ms:
        l_ms_agg = compute_ioi_aggregate(l_ms)
        print(f"  Multi-step Llama: POMDP oracle_eff={l_ms_agg['ai_oracle_eff']:.1f}%")

    # --- IOI comparison bar chart (Fig 6) ---
    if g_ioi_agg:
        write_ioi_comparison(g_ioi_agg, g_ms_agg, l_ioi_agg, l_ms_agg,
                             str(figures_dir / "ioi_comparison.tex"))

    # --- Steering heatmap (Fig 7) ---
    if g_steer:
        write_steering_heatmap(g_steer, l_steer, str(figures_dir / "steering_heatmap.tex"))
    else:
        print("  WARNING: No Gemma steering results found")

    # --- Layer distribution (Fig 8) ---
    if g_ms:
        g_ms_ld = _aggregate_layer_dist(g_ms)
        g_ioi_ld = _aggregate_layer_dist(g_ioi) if g_ioi and "layer_distribution" in g_ioi.get("per_prompt", [{}])[0] else None
        l_ms_ld = _aggregate_layer_dist(l_ms) if l_ms else None
        l_ioi_ld = _aggregate_layer_dist(l_ioi) if l_ioi and "layer_distribution" in l_ioi.get("per_prompt", [{}])[0] else None
        write_layer_distribution(g_ioi_ld, g_ms_ld, l_ioi_ld, l_ms_ld,
                                 str(figures_dir / "layer_distribution.tex"))

    # --- Domain layers (Fig 9) ---
    if g_domain:
        write_domain_layers(g_domain, l_domain, str(figures_dir / "domain_layers.tex"))

    # --- Attribution graph (Fig 5) ---
    write_attribution_graph(str(figures_dir / "attribution_graph.tex"))

    # --- Action distribution (new figure for Section VI.B) ---
    write_action_distribution(g_ioi, g_ms, l_ioi, l_ms,
                              str(figures_dir / "action_distribution.tex"))

    print("\nDone. Re-compile the paper to see updated figures.")


def _collect_action_proportions(data: Optional[dict], budget: int = 20) -> Optional[List[Dict[str, float]]]:
    """Aggregate action proportions per step across prompts."""
    if data is None:
        return None
    prompts = data.get("per_prompt", [])
    step_counts: List[Dict[str, int]] = [{} for _ in range(budget)]
    step_totals = [0] * budget
    for pr in prompts:
        actions = pr.get("ai_actions", [])
        for s, a in enumerate(actions[:budget]):
            step_counts[s][a] = step_counts[s].get(a, 0) + 1
            step_totals[s] += 1
    result = []
    for s in range(budget):
        if step_totals[s] == 0:
            break
        abl = step_counts[s].get("ablation", 0) / step_totals[s]
        pat = step_counts[s].get("activation_patching", 0) / step_totals[s]
        ste = step_counts[s].get("feature_steering", 0) / step_totals[s]
        result.append({"ablation": abl, "patching": pat, "steering": ste})
    return result if result else None


def write_action_distribution(
    g_ioi: Optional[dict], g_ms: Optional[dict],
    l_ioi: Optional[dict], l_ms: Optional[dict],
    out_path: str,
) -> None:
    """Write a stacked bar chart of action distribution over steps."""
    panels = []
    for label, data in [("Gemma IOI", g_ioi), ("Gemma Multi-step", g_ms),
                        ("Llama IOI", l_ioi), ("Llama Multi-step", l_ms)]:
        props = _collect_action_proportions(data)
        if props:
            panels.append((label, props))

    if not panels:
        print("  WARNING: No ai_actions data found; skipping action_distribution figure.")
        return

    n_panels = len(panels)
    n_cols = min(n_panels, 2)
    n_rows = (n_panels + 1) // 2
    lines = [
        "% Auto-generated by generate_figure_data.py",
        "\\begin{tikzpicture}",
        "\\begin{groupplot}[",
        f"  group style={{group size={n_cols} by {n_rows}, horizontal sep=1.0cm, vertical sep=1.8cm}},",
        "  width=0.48\\columnwidth, height=3.8cm,",
        "  xlabel={Step}, ylabel={Proportion},",
        "  x label style={font=\\scriptsize},",
        "  y label style={font=\\scriptsize},",
        "  tick label style={font=\\scriptsize},",
        "  title style={font=\\scriptsize\\bfseries},",
        "  ymin=0, ymax=1,",
        "  legend style={at={(0.5,-0.35)}, anchor=north, legend columns=3, font=\\tiny},",
        "]",
    ]

    for i, (label, props) in enumerate(panels):
        n_steps = len(props)
        lines.append(f"\\nextgroupplot[title={{{label}}}, ybar stacked, bar width=3pt]")
        for action_key, colour in [("ablation", "blue!70"), ("patching", "orange!80"),
                                    ("steering", "green!60!black")]:
            coords = " ".join(f"({s+1},{props[s][action_key]:.3f})" for s in range(n_steps))
            lines.append(f"\\addplot[fill={colour}, draw=none] coordinates {{{coords}}};")
        if i == 0:
            lines.append("\\legend{Ablation, Patching, Steering}")

    lines += ["\\end{groupplot}", "\\end{tikzpicture}"]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()

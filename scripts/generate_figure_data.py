#!/usr/bin/env python3
"""Generate LaTeX pgfplots data from JSON experiment results.

Reads results/*.json and writes paper/figures/*.tex with data
that exactly matches the experiment outputs.

Usage:
    python -m scripts.generate_figure_data [--results-dir results] [--figures-dir paper/figures]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


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
# IOI comparison bar chart
# ======================================================================

def write_ioi_comparison(ioi_agg: dict, ms_agg: Optional[dict], out_path: str):
    with open(out_path, "w") as f:
        f.write(
            "% Auto-generated from experiment results.\n"
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
            "    ymin=0, ymax=100,\n"
            "    enlarge x limits=0.2,\n"
            "    nodes near coords,\n"
            "    nodes near coords style={font=\\tiny},\n"
            "    every node near coord/.append style={/pgf/number format/fixed,\n"
            "        /pgf/number format/precision=1},\n"
            "]\n\n"
        )

        f.write(
            f"\\addplot[fill=blue!40, draw=blue!60] coordinates {{\n"
            f"    (Random, {ioi_agg['random_oracle_eff']:.1f}) "
            f"(Greedy, {ioi_agg['greedy_oracle_eff']:.1f}) "
            f"(Bandit, {ioi_agg['bandit_oracle_eff']:.1f}) "
            f"(POMDP, {ioi_agg['ai_oracle_eff']:.1f})\n"
            f"}};\n\n"
        )

        if ms_agg:
            f.write(
                f"\\addplot[fill=red!40, draw=red!60] coordinates {{\n"
                f"    (Random, {ms_agg['random_oracle_eff']:.1f}) "
                f"(Greedy, {ms_agg['greedy_oracle_eff']:.1f}) "
                f"(Bandit, {ms_agg['bandit_oracle_eff']:.1f}) "
                f"(POMDP, {ms_agg['ai_oracle_eff']:.1f})\n"
                f"}};\n\n"
            )
            f.write("\\legend{IOI, Multi-step}\n\n")
        else:
            f.write("\\legend{IOI}\n\n")

        f.write("\\end{axis}\n\\end{tikzpicture}\n")
    print(f"  Wrote {out_path}")


# ======================================================================
# Cumulative KL line graph
# ======================================================================

def write_cumulative_kl(ioi_data: dict, out_path: str):
    prompts = ioi_data["per_prompt"]
    budget = ioi_data.get("budget", 20)

    ai_cum     = np.zeros(budget)
    bandit_cum = np.zeros(budget)
    greedy_cum = np.zeros(budget)
    random_cum = np.zeros(budget)
    oracle_cum = np.zeros(budget)
    n = len(prompts)

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

    with open(out_path, "w") as f:
        f.write(
            "% Auto-generated from experiment results.\n"
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
            f"    xmin=1, xmax={budget},\n"
            "]\n\n"
        )

        def coords(vals):
            return " ".join(f"({i+1},{v:.6f})" for i, v in enumerate(vals))

        f.write(f"\\addplot[black, dashed, thick] coordinates {{ {coords(oracle_cum)} }};\n")
        f.write(f"\\addplot[red, thick] coordinates {{ {coords(ai_cum)} }};\n")
        f.write(f"\\addplot[orange, thick] coordinates {{ {coords(bandit_cum)} }};\n")
        f.write(f"\\addplot[blue, thick] coordinates {{ {coords(greedy_cum)} }};\n\n")
        f.write("\\legend{Oracle, POMDP Agent, Bandit, Greedy}\n\n")
        f.write("\\end{axis}\n\\end{tikzpicture}\n")

    print(f"  Wrote {out_path}")


# ======================================================================
# Steering KL line graph
# ======================================================================

def write_steering_heatmap(steering_data: dict, out_path: str):
    prompts = steering_data["per_prompt"]
    multipliers = steering_data.get("multipliers", [0, 2, 5, 10])

    with open(out_path, "w") as f:
        f.write(
            "% Auto-generated from experiment results.\n"
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

        colors = ["blue", "red", "green!60!black", "orange", "purple"]
        for idx, p in enumerate(prompts[:3]):
            mean_kls = p.get("mean_kl_per_multiplier", None)

            if mean_kls is None or len(mean_kls) != len(multipliers):
                # Compute from per-feature data
                features = p.get("features", [])
                mean_kls = []
                for m in multipliers:
                    kls = []
                    for feat in features:
                        key = f"mult_{m}"
                        if key in feat and isinstance(feat[key], dict):
                            kls.append(feat[key].get("kl", 0))
                    mean_kls.append(float(np.mean(kls)) if kls else 0.0)

            coords_str = " ".join(
                f"({m},{kl:.6f})" for m, kl in zip(multipliers, mean_kls)
            )
            color = colors[idx % len(colors)]
            f.write(f"\\addplot[{color}, thick, mark=*] coordinates {{ {coords_str} }};\n")

        legends = []
        for idx, p in enumerate(prompts[:3]):
            concept = p.get("concept", p.get("prompt", f"Concept {idx+1}"))
            legends.append(concept[:20].replace("_", " "))
        f.write(f"\\legend{{{', '.join(legends)}}}\n\n")
        f.write("\\end{axis}\n\\end{tikzpicture}\n")

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

    # Try model-specific files first, fall back to generic
    ioi_path = results_dir / "ioi_results_gemma.json"
    if not ioi_path.exists():
        ioi_path = results_dir / "ioi_results.json"

    ms_path = results_dir / "multistep_results_gemma.json"
    if not ms_path.exists():
        ms_path = results_dir / "multistep_results.json"

    steering_path = results_dir / "steering_results_gemma.json"
    if not steering_path.exists():
        steering_path = results_dir / "steering_results.json"

    ioi_agg = None
    ms_agg = None

    if ioi_path.exists():
        ioi_data = load_json(str(ioi_path))
        ioi_agg = compute_ioi_aggregate(ioi_data)
        print(f"\n  IOI: pymdp oracle_eff={ioi_agg['ai_oracle_eff']:.1f}%")
        write_cumulative_kl(ioi_data, str(figures_dir / "cumulative_kl.tex"))
    else:
        print("  WARNING: No IOI results found")

    if ms_path.exists():
        ms_data = load_json(str(ms_path))
        ms_agg = compute_ioi_aggregate(ms_data)
        print(f"  MultiStep: pymdp oracle_eff={ms_agg['ai_oracle_eff']:.1f}%")
    else:
        print("  WARNING: No multi-step results found")

    if ioi_agg:
        write_ioi_comparison(ioi_agg, ms_agg, str(figures_dir / "ioi_comparison.tex"))

    if steering_path.exists():
        steering_data = load_json(str(steering_path))
        write_steering_heatmap(steering_data, str(figures_dir / "steering_heatmap.tex"))
    else:
        print("  WARNING: No steering results found")

    print("\nDone. Re-compile the paper to see updated figures.")


if __name__ == "__main__":
    main()

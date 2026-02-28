#!/usr/bin/env python3
"""Generate LaTeX pgfplots data from JSON experiment results.

Reads results/*.json and writes paper/figures/*.tex with data
that exactly matches the experiment outputs.

Usage:
    python -m scripts.generate_figure_data [--results-dir results] [--figures-dir paper/figures]
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_ioi_aggregate(data: dict) -> dict:
    prompts = data["per_prompt"]
    n = len(prompts)
    ai_mean = sum(p["ai_mean"] for p in prompts) / n
    greedy_mean = sum(p["greedy_mean"] for p in prompts) / n
    random_mean = sum(p["random_mean"] for p in prompts) / n
    ai_cumkl = sum(p["ai_cumkl"] for p in prompts) / n
    greedy_cumkl = sum(p["greedy_cumkl"] for p in prompts) / n
    random_cumkl = sum(p["random_cumkl"] for p in prompts) / n
    oracle_cumkl = sum(p["oracle_cumkl"] for p in prompts) / n
    return {
        "ai_mean": ai_mean,
        "greedy_mean": greedy_mean,
        "random_mean": random_mean,
        "ai_cumkl": ai_cumkl,
        "greedy_cumkl": greedy_cumkl,
        "random_cumkl": random_cumkl,
        "oracle_cumkl": oracle_cumkl,
        "ai_oracle_eff": ai_cumkl / oracle_cumkl * 100 if oracle_cumkl > 0 else 0,
        "greedy_oracle_eff": greedy_cumkl / oracle_cumkl * 100 if oracle_cumkl > 0 else 0,
        "random_oracle_eff": random_cumkl / oracle_cumkl * 100 if oracle_cumkl > 0 else 0,
    }


def write_ioi_comparison(ioi_agg: dict, ms_agg: dict, out_path: str):
    """Write the IOI vs multi-step oracle efficiency bar chart."""
    with open(out_path, "w") as f:
        f.write(
            "% Auto-generated from experiment results. Do not edit manually.\n"
            "\\begin{tikzpicture}\n"
            "\\begin{axis}[\n"
            "    ybar,\n"
            "    width=\\columnwidth,\n"
            "    height=5cm,\n"
            "    bar width=8pt,\n"
            "    ylabel={Oracle Efficiency (\\%)},\n"
            "    symbolic x coords={Random, Greedy, AI Selector},\n"
            "    xtick=data,\n"
            "    x tick label style={font=\\scriptsize},\n"
            "    y tick label style={font=\\scriptsize},\n"
            "    ylabel style={font=\\scriptsize},\n"
            "    legend style={\n"
            "        font=\\scriptsize,\n"
            "        at={(0.5,-0.2)},\n"
            "        anchor=north,\n"
            "        legend columns=2\n"
            "    },\n"
            "    ymin=0, ymax=100,\n"
            "    enlarge x limits=0.25,\n"
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
            f"(AI Selector, {ioi_agg['ai_oracle_eff']:.1f})\n"
            f"}};\n\n"
        )

        f.write(
            f"\\addplot[fill=red!40, draw=red!60] coordinates {{\n"
            f"    (Random, {ms_agg['random_oracle_eff']:.1f}) "
            f"(Greedy, {ms_agg['greedy_oracle_eff']:.1f}) "
            f"(AI Selector, {ms_agg['ai_oracle_eff']:.1f})\n"
            f"}};\n\n"
        )

        f.write("\\legend{IOI, Multi-step}\n\n")
        f.write("\\end{axis}\n\\end{tikzpicture}\n")

    print(f"  Wrote {out_path}")


def write_cumulative_kl(ioi_data: dict, out_path: str):
    """Write cumulative KL line graph."""
    prompts = ioi_data["per_prompt"]
    budget = ioi_data.get("budget", 20)

    ai_cum = [0.0] * budget
    greedy_cum = [0.0] * budget
    random_cum = [0.0] * budget
    oracle_cum = [0.0] * budget

    n = len(prompts)
    for p in prompts:
        ai_kl = p.get("ai_kl_values", [])
        greedy_kl = p.get("greedy_kl_values", [])
        random_kl = p.get("random_kl_values", [])
        oracle_kl = p.get("oracle_kl_values", [])

        for i in range(min(budget, len(ai_kl))):
            ai_cum[i] += sum(ai_kl[: i + 1]) / n
        for i in range(min(budget, len(greedy_kl))):
            greedy_cum[i] += sum(greedy_kl[: i + 1]) / n
        for i in range(min(budget, len(random_kl))):
            random_cum[i] += sum(random_kl[: i + 1]) / n
        for i in range(min(budget, len(oracle_kl))):
            oracle_cum[i] += sum(oracle_kl[: i + 1]) / n

    with open(out_path, "w") as f:
        f.write(
            "% Auto-generated from experiment results. Do not edit manually.\n"
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

        def coords(values):
            return " ".join(f"({i+1},{v:.6f})" for i, v in enumerate(values))

        f.write(f"\\addplot[black, dashed, thick] coordinates {{ {coords(oracle_cum)} }};\n")
        f.write(f"\\addplot[red, thick] coordinates {{ {coords(ai_cum)} }};\n")
        f.write(f"\\addplot[blue, thick] coordinates {{ {coords(greedy_cum)} }};\n")
        f.write(f"\\addplot[gray, thick] coordinates {{ {coords(random_cum)} }};\n\n")
        f.write("\\legend{Oracle, AI Selector, Greedy, Random}\n\n")
        f.write("\\end{axis}\n\\end{tikzpicture}\n")

    print(f"  Wrote {out_path}")


def write_steering_heatmap(steering_data: dict, out_path: str):
    """Write steering KL line graph."""
    prompts = steering_data["per_prompt"]
    multipliers = steering_data.get("multipliers", [0, 2, 5, 10])

    with open(out_path, "w") as f:
        f.write(
            "% Auto-generated from experiment results. Do not edit manually.\n"
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
            concept = p.get("concept", p.get("prompt", f"Concept {idx+1}"))
            concept_short = concept[:20].replace("_", " ").replace("\\", "")
            mean_kls = p.get("mean_kl_per_multiplier", [])
            if mean_kls and len(mean_kls) == len(multipliers):
                coords_str = " ".join(
                    f"({m},{kl:.6f})" for m, kl in zip(multipliers, mean_kls)
                )
            else:
                coords_str = " ".join(f"({m},0)" for m in multipliers)

            color = colors[idx % len(colors)]
            f.write(
                f"\\addplot[{color}, thick, mark=*] coordinates {{ {coords_str} }};\n"
            )

        legends = []
        for idx, p in enumerate(prompts[:3]):
            concept = p.get("concept", p.get("prompt", f"Concept {idx+1}"))
            legends.append(concept[:20].replace("_", " "))
        f.write(f"\\legend{{{', '.join(legends)}}}\n\n")
        f.write("\\end{axis}\n\\end{tikzpicture}\n")

    print(f"  Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX figure data from results")
    parser.add_argument(
        "--results-dir", default="results", help="Directory with JSON results"
    )
    parser.add_argument(
        "--figures-dir", default="paper/figures", help="Output directory for .tex figures"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figure data from experiment results...")

    ioi_path = results_dir / "ioi_results.json"
    ms_path = results_dir / "multistep_results.json"
    steering_path = results_dir / "steering_results.json"

    if ioi_path.exists():
        ioi_data = load_json(str(ioi_path))
        ioi_agg = compute_ioi_aggregate(ioi_data)
        print(f"\n  IOI aggregate: ai_oracle_eff={ioi_agg['ai_oracle_eff']:.1f}%")

        write_cumulative_kl(ioi_data, str(figures_dir / "cumulative_kl.tex"))
    else:
        ioi_agg = None
        print("  WARNING: No ioi_results.json found")

    if ms_path.exists():
        ms_data = load_json(str(ms_path))
        ms_agg_raw = ms_data.get("aggregate", {})
        ms_agg = {
            "ai_oracle_eff": ms_agg_raw.get("ai_oracle_efficiency", 0),
            "greedy_oracle_eff": ms_agg_raw.get("ai_oracle_efficiency", 0)
            * ms_agg_raw.get("greedy_mean_kl", 0)
            / max(ms_agg_raw.get("ai_mean_kl", 1), 1e-10),
            "random_oracle_eff": ms_agg_raw.get("ai_oracle_efficiency", 0)
            * ms_agg_raw.get("random_mean_kl", 0)
            / max(ms_agg_raw.get("ai_mean_kl", 1), 1e-10),
        }
        print(f"  Multi-step aggregate: ai_oracle_eff={ms_agg['ai_oracle_eff']:.1f}%")
    else:
        ms_agg = None
        print("  WARNING: No multistep_results.json found")

    if ioi_agg and ms_agg:
        write_ioi_comparison(ioi_agg, ms_agg, str(figures_dir / "ioi_comparison.tex"))

    if steering_path.exists():
        steering_data = load_json(str(steering_path))
        write_steering_heatmap(steering_data, str(figures_dir / "steering_heatmap.tex"))
    else:
        print("  WARNING: No steering_results.json found")

    print("\nDone. Re-compile the paper to see updated figures.")


if __name__ == "__main__":
    main()

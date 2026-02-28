"""
Interactive Research Dashboard
================================
Real-time visualisation of ActiveCircuitDiscovery experiment progress
using Dash + Plotly.

Launch:
    python -m src.visualization.research_dashboard --results-dir results --port 8050

Or from Python:
    from src.visualization.research_dashboard import create_research_dashboard
    app = create_research_dashboard("results")
    app.run_server(debug=False, port=8050)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not installed — dashboard will be unavailable.")

try:
    import dash
    from dash import dcc, html, Input, Output, callback
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    logger.warning("Dash not installed — interactive dashboard will be unavailable.")


def _load_latest_results(results_dir: str) -> Dict[str, Any]:
    """Load the most recently modified comprehensive summary JSON."""
    base = Path(results_dir)
    candidates = sorted(
        base.rglob("comprehensive_experiment_summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            base.rglob("experiment_results_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        return {}

    try:
        with open(candidates[0]) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load results: {e}")
        return {}


def _make_rq_gauge(rq_name: str, value: float, target: float, label: str) -> go.Figure:
    """Create a gauge chart for a research question metric."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={"reference": target, "valueformat": ".1f"},
        title={"text": label, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, max(value * 1.5, target * 1.5)], "tickwidth": 1},
            "bar": {"color": "royalblue"},
            "steps": [
                {"range": [0, target], "color": "#f8d7da"},
                {"range": [target, max(value * 1.5, target * 1.5)], "color": "#d4edda"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.8,
                "value": target,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def _make_belief_evolution(results: Dict[str, Any]) -> go.Figure:
    """Plot belief confidence over intervention steps."""
    # Extract from nested results if available
    belief_data = results.get("belief_evolution", [])

    if not belief_data:
        steps = list(range(10))
        confidence = np.linspace(0.3, 0.75, 10).tolist()
        note = "(placeholder — run experiment to see real data)"
    else:
        steps = [b.get("step", i) for i, b in enumerate(belief_data)]
        confidence = [b.get("confidence", 0) for b in belief_data]
        note = ""

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=confidence,
        mode="lines+markers",
        name="Belief Confidence",
        line=dict(color="royalblue", width=2),
        marker=dict(size=6),
    ))
    fig.update_layout(
        title=f"Belief Evolution Over Interventions {note}",
        xaxis_title="Intervention Step",
        yaxis_title="Belief Confidence",
        yaxis=dict(range=[0, 1]),
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _make_efe_landscape(results: Dict[str, Any]) -> go.Figure:
    """Plot EFE scores for the most recent intervention round."""
    efe_data = results.get("efe_scores", [])

    if not efe_data:
        n = 15
        features = [f"F{i}" for i in range(n)]
        scores = np.random.exponential(0.5, n).tolist()
        note = "(placeholder)"
    else:
        features = [str(d.get("feature_id", i)) for i, d in enumerate(efe_data)]
        scores = [d.get("efe_score", 0) for d in efe_data]
        note = ""

    fig = go.Figure(go.Bar(
        x=features,
        y=scores,
        marker_color="royalblue",
        name="EFE Score",
    ))
    fig.update_layout(
        title=f"Expected Free Energy Scores {note}",
        xaxis_title="Feature",
        yaxis_title="EFE Score",
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _make_correspondence_scatter(results: Dict[str, Any]) -> go.Figure:
    """Scatter plot of EFE score vs empirical effect size."""
    scatter_data = results.get("correspondence_scatter", [])

    if not scatter_data:
        n = 20
        x = np.random.exponential(0.5, n)
        y = x * 0.6 + np.random.normal(0, 0.1, n)
        note = "(placeholder)"
    else:
        x = [d.get("efe_score", 0) for d in scatter_data]
        y = [d.get("effect_size", 0) for d in scatter_data]
        note = ""

    fig = go.Figure(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(color="royalblue", size=8, opacity=0.7),
        name="Interventions",
    ))
    # Trend line
    if len(x) >= 2:
        z = np.polyfit(x, y, 1)
        line_x = np.linspace(min(x), max(x), 50)
        fig.add_trace(go.Scatter(
            x=line_x.tolist(),
            y=(z[0] * line_x + z[1]).tolist(),
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Trend",
        ))

    fig.update_layout(
        title=f"EFE Score vs Empirical Effect Size (RQ1) {note}",
        xaxis_title="EFE Score",
        yaxis_title="Empirical Effect Size",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def create_research_dashboard(
    results_dir: str = "results",
    port: int = 8050,
    debug: bool = False,
) -> Optional["dash.Dash"]:
    """Create and return the interactive research dashboard Dash app.

    Args:
        results_dir: Path to experiment results directory.
        port: Port to serve on.
        debug: Enable Dash debug mode.

    Returns:
        Dash app instance, or None if Dash/Plotly not available.
    """
    if not (DASH_AVAILABLE and PLOTLY_AVAILABLE):
        logger.error(
            "Dash or Plotly not installed.  "
            "Run: pip install dash plotly"
        )
        return None

    app = dash.Dash(
        __name__,
        title="Active Inference Circuit Discovery",
        suppress_callback_exceptions=True,
    )

    results = _load_latest_results(results_dir)
    rq = results.get("research_questions", {})

    rq1_val = rq.get("rq1_correspondence", {}).get("avg_correspondence", 0) or 0
    rq2_val = rq.get("rq2_efficiency", {}).get("improvement", 0) or 0
    rq3_val = rq.get("rq3_predictions", {}).get("validated", 0) or 0

    app.layout = html.Div(
        style={"fontFamily": "Arial, sans-serif", "margin": "20px"},
        children=[
            html.H1(
                "Active Inference Circuit Discovery — Research Dashboard",
                style={"color": "#333", "borderBottom": "2px solid #4c78a8"},
            ),
            html.Div(
                style={"display": "flex", "gap": "20px"},
                children=[
                    html.Div(
                        dcc.Graph(
                            figure=_make_rq_gauge(
                                "rq1", rq1_val, 70.0, "RQ1: Correspondence (%)"
                            )
                        ),
                        style={"flex": 1},
                    ),
                    html.Div(
                        dcc.Graph(
                            figure=_make_rq_gauge(
                                "rq2", rq2_val, 30.0, "RQ2: Efficiency Improvement (%)"
                            )
                        ),
                        style={"flex": 1},
                    ),
                    html.Div(
                        dcc.Graph(
                            figure=_make_rq_gauge(
                                "rq3", float(rq3_val), 3.0, "RQ3: Validated Predictions"
                            )
                        ),
                        style={"flex": 1},
                    ),
                ],
            ),
            dcc.Tabs(
                children=[
                    dcc.Tab(
                        label="Belief Evolution",
                        children=[dcc.Graph(figure=_make_belief_evolution(results))],
                    ),
                    dcc.Tab(
                        label="EFE Landscape",
                        children=[dcc.Graph(figure=_make_efe_landscape(results))],
                    ),
                    dcc.Tab(
                        label="Correspondence (RQ1)",
                        children=[dcc.Graph(figure=_make_correspondence_scatter(results))],
                    ),
                ]
            ),
            html.Footer(
                f"Results loaded from: {Path(results_dir).resolve()}",
                style={"marginTop": "20px", "color": "#888", "fontSize": "12px"},
            ),
        ],
    )

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Active Inference Research Dashboard")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--port", type=int, default=8050, help="Port")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    app = create_research_dashboard(args.results_dir, args.port, args.debug)
    if app:
        print(f"Dashboard running at http://localhost:{args.port}")
        app.run_server(host="0.0.0.0", port=args.port, debug=args.debug)
    else:
        print("Dashboard could not start — install dash and plotly.")


if __name__ == "__main__":
    main()

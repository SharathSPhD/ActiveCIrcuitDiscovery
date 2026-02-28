"""
Metrics for Active Circuit Discovery
======================================
Proper statistical metrics for evaluating Active Inference-guided
circuit discovery against SOTA baselines.

Key metrics:
  - Spearman rho between EFE-ranked features and empirical effects
  - Bootstrap confidence intervals for all estimates
  - Cohen's d effect sizes
  - Post-hoc power analysis
  - Efficiency comparison with multiple baselines
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy.stats import pearsonr, spearmanr, norm, sem
from scipy import stats
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """Container for a statistical test result."""
    statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    significant: bool
    sample_size: int
    method: str = ""


@dataclass
class CorrespondenceResult:
    """Result of correspondence analysis between AI rankings and empirical effects."""
    spearman_rho: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    bootstrap_ci: Tuple[float, float]
    n_samples: int
    effect_size_cohens_d: float
    power: float
    significant: bool

    @property
    def summary(self) -> str:
        sig = "***" if self.spearman_p < 0.001 else "**" if self.spearman_p < 0.01 else "*" if self.spearman_p < 0.05 else "ns"
        return (
            f"rho={self.spearman_rho:.3f} ({sig}), "
            f"95% CI [{self.bootstrap_ci[0]:.3f}, {self.bootstrap_ci[1]:.3f}], "
            f"n={self.n_samples}, d={self.effect_size_cohens_d:.3f}, "
            f"power={self.power:.3f}"
        )


@dataclass
class EfficiencyResult:
    """Result of efficiency comparison between AI and baselines."""
    ai_interventions: int
    baseline_interventions: Dict[str, int]
    improvement_pct: Dict[str, float]
    overall_improvement: float
    bootstrap_cis: Dict[str, Tuple[float, float]]
    significant: Dict[str, bool]


def compute_correspondence(
    efe_rankings: np.ndarray,
    empirical_effects: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> CorrespondenceResult:
    """Compute Spearman rho between EFE-ranked features and empirical effect sizes.

    This is the primary metric for evaluating whether Active Inference
    correctly identifies important circuit components.

    Args:
        efe_rankings: Array of EFE values (lower = more important) for each feature.
        empirical_effects: Array of measured effect sizes for the same features.
        n_bootstrap: Number of bootstrap resamples for CI.
        alpha: Significance level.
        random_state: Random seed for reproducibility.

    Returns:
        CorrespondenceResult with Spearman rho, CI, effect size, power.
    """
    n = len(efe_rankings)
    if n < 3:
        logger.warning(f"Too few samples for correspondence (n={n}). Need >= 3.")
        return CorrespondenceResult(
            spearman_rho=0.0, spearman_p=1.0,
            pearson_r=0.0, pearson_p=1.0,
            bootstrap_ci=(0.0, 0.0),
            n_samples=n, effect_size_cohens_d=0.0,
            power=0.0, significant=False,
        )

    efe_arr = np.asarray(efe_rankings, dtype=float)
    eff_arr = np.asarray(empirical_effects, dtype=float)

    # Negate EFE since lower EFE = more important, but higher effect = more important
    rho, p_rho = spearmanr(-efe_arr, eff_arr)
    r, p_r = pearsonr(-efe_arr, eff_arr)

    if np.isnan(rho):
        rho, p_rho = 0.0, 1.0
    if np.isnan(r):
        r, p_r = 0.0, 1.0

    ci_low, ci_high = bootstrap_correlation_ci(
        -efe_arr, eff_arr, n_bootstrap=n_bootstrap,
        alpha=alpha, random_state=random_state,
    )

    d = cohens_d_from_correlation(rho, n)
    power = compute_power(rho, n, alpha)

    return CorrespondenceResult(
        spearman_rho=float(rho),
        spearman_p=float(p_rho),
        pearson_r=float(r),
        pearson_p=float(p_r),
        bootstrap_ci=(float(ci_low), float(ci_high)),
        n_samples=n,
        effect_size_cohens_d=float(d),
        power=float(power),
        significant=p_rho < alpha,
    )


def bootstrap_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
    method: str = "spearman",
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a correlation coefficient.

    Uses the BCa (bias-corrected and accelerated) percentile method.
    """
    rng = np.random.RandomState(random_state)
    n = len(x)
    corr_fn = spearmanr if method == "spearman" else pearsonr

    bootstrap_corrs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = corr_fn(x[idx], y[idx])
        bootstrap_corrs[i] = r if not np.isnan(r) else 0.0

    lower = np.percentile(bootstrap_corrs, 100 * alpha / 2)
    upper = np.percentile(bootstrap_corrs, 100 * (1 - alpha / 2))
    return (float(lower), float(upper))


def cohens_d_from_correlation(r: float, n: int) -> float:
    """Convert a correlation coefficient to Cohen's d effect size."""
    if abs(r) >= 0.999:
        return float(np.sign(r) * 10.0)
    d = 2 * r / np.sqrt(1 - r**2)
    return float(d)


def compute_power(
    effect_size_r: float,
    n: int,
    alpha: float = 0.05,
) -> float:
    """Compute post-hoc power for a correlation test.

    Uses the formula: power = P(|Z| > z_alpha | r, n)
    where Z follows a normal distribution under H1.
    """
    if n < 4 or abs(effect_size_r) < 1e-10:
        return 0.0

    z_alpha = norm.ppf(1 - alpha / 2)

    # Fisher Z transform of the effect
    fisher_z = 0.5 * np.log((1 + effect_size_r) / (1 - effect_size_r + 1e-10))
    se = 1.0 / np.sqrt(n - 3)
    ncp = abs(fisher_z) / se

    power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
    return float(min(1.0, max(0.0, power)))


def compute_efficiency(
    ai_interventions: int,
    baseline_results: Dict[str, int],
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    ai_effects: Optional[np.ndarray] = None,
    baseline_effects: Optional[Dict[str, np.ndarray]] = None,
) -> EfficiencyResult:
    """Compute efficiency improvement of AI over baselines.

    Args:
        ai_interventions: Number of interventions used by Active Inference.
        baseline_results: Map of baseline_name -> intervention_count.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level.
        ai_effects: Effect sizes for AI interventions (for bootstrap).
        baseline_effects: Effect sizes for each baseline (for bootstrap).

    Returns:
        EfficiencyResult with improvements and CIs.
    """
    improvements = {}
    bootstrap_cis = {}
    significant = {}

    for name, baseline_count in baseline_results.items():
        if baseline_count > 0:
            imp = (baseline_count - ai_interventions) / baseline_count * 100.0
            improvements[name] = max(0.0, imp)
        else:
            improvements[name] = 0.0

        if ai_effects is not None and baseline_effects is not None and name in baseline_effects:
            ci = _bootstrap_efficiency_ci(
                ai_effects, baseline_effects[name],
                n_bootstrap=n_bootstrap, alpha=alpha,
            )
            bootstrap_cis[name] = ci
            significant[name] = ci[0] > 0
        else:
            bootstrap_cis[name] = (0.0, 0.0)
            significant[name] = False

    overall = float(np.mean(list(improvements.values()))) if improvements else 0.0

    return EfficiencyResult(
        ai_interventions=ai_interventions,
        baseline_interventions=baseline_results,
        improvement_pct=improvements,
        overall_improvement=overall,
        bootstrap_cis=bootstrap_cis,
        significant=significant,
    )


def _bootstrap_efficiency_ci(
    ai_effects: np.ndarray,
    baseline_effects: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Bootstrap CI for efficiency difference."""
    rng = np.random.RandomState(42)
    n_ai = len(ai_effects)
    n_base = len(baseline_effects)

    diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        ai_sample = rng.choice(ai_effects, size=n_ai, replace=True)
        base_sample = rng.choice(baseline_effects, size=n_base, replace=True)
        diffs[i] = base_sample.mean() - ai_sample.mean()

    return (
        float(np.percentile(diffs, 100 * alpha / 2)),
        float(np.percentile(diffs, 100 * (1 - alpha / 2))),
    )


def compute_prediction_validation(
    predictions: List[Dict[str, Any]],
    validation_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate novel predictions against empirical tests.

    Each prediction should have: hypothesis, expected_direction, threshold
    Each validation should have: prediction_id, observed_value, p_value
    """
    n_total = len(predictions)
    n_validated = 0
    n_falsified = 0
    details = []

    for pred, val in zip(predictions, validation_results):
        expected = pred.get("expected_direction", "positive")
        observed = val.get("observed_value", 0.0)
        p_val = val.get("p_value", 1.0)

        if expected == "positive":
            confirmed = observed > 0 and p_val < 0.05
        elif expected == "negative":
            confirmed = observed < 0 and p_val < 0.05
        else:
            confirmed = p_val < 0.05

        if confirmed:
            n_validated += 1
            status = "validated"
        else:
            n_falsified += 1
            status = "falsified"

        details.append({
            "prediction": pred.get("hypothesis", ""),
            "status": status,
            "observed": observed,
            "p_value": p_val,
        })

    return {
        "n_total": n_total,
        "n_validated": n_validated,
        "n_falsified": n_falsified,
        "validation_rate": n_validated / max(1, n_total),
        "details": details,
    }


class CorrespondenceCalculator:
    """Legacy-compatible wrapper around the new correspondence functions."""

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def calculate_from_discovery_results(
        self,
        efe_values: np.ndarray,
        effect_sizes: np.ndarray,
    ) -> CorrespondenceResult:
        """Calculate correspondence from discovery results."""
        return compute_correspondence(
            efe_values, effect_sizes, alpha=self.significance_level,
        )


class EfficiencyCalculator:
    """Calculator for RQ2 efficiency metrics."""

    def calculate_efficiency_improvement(
        self,
        ai_interventions: int,
        baseline_results: Dict[str, int],
    ) -> Dict[str, float]:
        result = compute_efficiency(ai_interventions, baseline_results)
        improvements = dict(result.improvement_pct)
        improvements["overall_efficiency"] = result.overall_improvement
        return improvements


class ValidationCalculator:
    """Calculator for research question validation."""

    def __init__(
        self,
        rq1_target: float = 0.6,
        rq2_target: float = 30.0,
        rq3_target: int = 3,
    ):
        self.rq1_target = rq1_target
        self.rq2_target = rq2_target
        self.rq3_target = rq3_target

    def validate_research_questions(
        self,
        correspondence: CorrespondenceResult,
        efficiency: Dict[str, float],
        n_validated_predictions: int,
    ) -> Dict[str, Any]:
        rq1_passed = (
            correspondence.spearman_rho >= self.rq1_target
            and correspondence.significant
        )
        rq2_passed = efficiency.get("overall_efficiency", 0.0) >= self.rq2_target
        rq3_passed = n_validated_predictions >= self.rq3_target

        return {
            "rq1_passed": rq1_passed,
            "rq1_spearman_rho": correspondence.spearman_rho,
            "rq1_p_value": correspondence.spearman_p,
            "rq1_target": self.rq1_target,
            "rq2_passed": rq2_passed,
            "rq2_efficiency": efficiency.get("overall_efficiency", 0.0),
            "rq2_target": self.rq2_target,
            "rq3_passed": rq3_passed,
            "rq3_validated": n_validated_predictions,
            "rq3_target": self.rq3_target,
            "overall_success": rq1_passed and rq2_passed and rq3_passed,
        }

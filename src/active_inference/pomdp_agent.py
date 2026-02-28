"""
Active Inference POMDP Agent for Circuit Discovery
====================================================
Proper multi-factor POMDP using pymdp for principled Bayesian
intervention selection in mechanistic interpretability.

State Factors
-------------
  0 - Feature importance:  {negligible, low, moderate, high}  (4 states)
  1 - Layer role:           {early, middle, late}               (3 states)
  2 - Causal influence:     {weak, moderate, strong}            (3 states)

Observation Modalities
----------------------
  0 - KL divergence magnitude:  {negligible, small, medium, large}  (4 levels)
  1 - Activation magnitude:     {inactive, low, moderate, high}     (4 levels)
  2 - Graph connectivity:       {sparse, moderate, dense}           (3 levels)

Actions (intervention types)
-----------------------------
  0 - ablation        (zero the feature)
  1 - activation_patching  (patch to reference)
  2 - feature_steering     (scale activation)

Uses Expected Free Energy (EFE) decomposed into epistemic (information
gain) and pragmatic (preference satisfaction) components.  The A matrix
is learned online via Dirichlet concentration updates so the agent
improves its observation model from real intervention data.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymdp.agent import Agent as PyMDPAgent
from pymdp import utils
from pymdp.maths import softmax as pymdp_softmax

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dimensionality constants
# ---------------------------------------------------------------------------
N_IMPORTANCE = 4    # Factor 0
N_LAYER_ROLE = 3    # Factor 1
N_CAUSAL     = 3    # Factor 2

N_KL_LEVELS  = 4    # Modality 0
N_ACT_LEVELS = 4    # Modality 1
N_CONN_LEVELS = 3   # Modality 2

ACTION_NAMES = ["ablation", "activation_patching", "feature_steering"]
N_ACTIONS = len(ACTION_NAMES)

IMPORTANCE_LABELS = ["negligible", "low", "moderate", "high"]
ROLE_LABELS       = ["early", "middle", "late"]
CAUSAL_LABELS     = ["weak", "moderate", "strong"]

# ---------------------------------------------------------------------------
# Discretisation thresholds (calibrated from real experiment ranges)
# ---------------------------------------------------------------------------
KL_THRESHOLDS  = (0.0001, 0.001, 0.01)    # negligible | small | medium | large
ACT_THRESHOLDS = (0.5, 5.0, 50.0)         # inactive  | low   | moderate | high


@dataclass
class InterventionRecord:
    """Immutable record of one intervention step."""
    step: int
    feature_id: str
    layer: int
    position: int
    feature_idx: int
    action: str
    observation: Tuple[int, int, int]
    kl_divergence: float
    activation_value: float
    graph_connectivity: float
    beliefs_before: Dict[str, np.ndarray]
    beliefs_after: Dict[str, np.ndarray]
    efe_value: float
    timestamp: float = field(default_factory=time.time)


class ActiveInferencePOMDPAgent:
    """Active Inference agent wrapping pymdp for circuit discovery.

    Each candidate feature is evaluated by temporarily conditioning the
    generative model on a prior observation derived from the attribution
    graph (importance, activation, connectivity) and computing the EFE
    over all actions.  The candidate--action pair with the lowest EFE
    is selected.  After the real intervention is executed, the resulting
    KL divergence, activation magnitude and connectivity are discretised
    into an observation vector and fed back to the agent for posterior
    inference and likelihood learning.
    """

    def __init__(
        self,
        *,
        n_layers: int = 26,
        epistemic_weight: float = 1.0,
        pragmatic_weight: float = 1.0,
        lr_pA: float = 1.0,
        gamma: float = 16.0,
        policy_len: int = 1,
    ):
        self.n_layers = n_layers
        self.epistemic_weight = epistemic_weight
        self.pragmatic_weight = pragmatic_weight
        self.lr_pA = lr_pA
        self.gamma = gamma
        self.policy_len = policy_len

        self._agent: Optional[PyMDPAgent] = None
        self._step = 0
        self.history: List[InterventionRecord] = []
        self._feature_beliefs: Dict[str, Dict[str, np.ndarray]] = {}
        self._converged = False

        self._A: Optional[list] = None
        self._B: Optional[list] = None
        self._C: Optional[list] = None
        self._D: Optional[list] = None
        self._pA: Optional[list] = None

    # ------------------------------------------------------------------
    # Generative model construction
    # ------------------------------------------------------------------

    def _build_A(self) -> list:
        """Observation likelihood P(o_m | s_0, s_1, s_2) for each modality."""
        A = utils.obj_array(3)

        # Modality 0 -- KL magnitude depends mainly on importance and causal
        A[0] = np.zeros((N_KL_LEVELS, N_IMPORTANCE, N_LAYER_ROLE, N_CAUSAL))
        for s0 in range(N_IMPORTANCE):
            for s1 in range(N_LAYER_ROLE):
                for s2 in range(N_CAUSAL):
                    strength = (s0 / (N_IMPORTANCE - 1) + s2 / (N_CAUSAL - 1)) / 2.0
                    p = np.array([
                        max(0.05, 0.65 - 0.55 * strength),
                        max(0.05, 0.25 - 0.05 * strength),
                        max(0.05, 0.08 + 0.25 * strength),
                        max(0.05, 0.02 + 0.35 * strength),
                    ])
                    A[0][:, s0, s1, s2] = p / p.sum()

        # Modality 1 -- Activation magnitude depends on importance and role
        A[1] = np.zeros((N_ACT_LEVELS, N_IMPORTANCE, N_LAYER_ROLE, N_CAUSAL))
        for s0 in range(N_IMPORTANCE):
            for s1 in range(N_LAYER_ROLE):
                for s2 in range(N_CAUSAL):
                    role_boost = 0.1 * (s1 == 2)  # late-layer features tend higher
                    strength = s0 / (N_IMPORTANCE - 1) + role_boost
                    p = np.array([
                        max(0.05, 0.55 - 0.45 * strength),
                        max(0.05, 0.25),
                        max(0.05, 0.12 + 0.2 * strength),
                        max(0.05, 0.08 + 0.25 * strength),
                    ])
                    A[1][:, s0, s1, s2] = p / p.sum()

        # Modality 2 -- Graph connectivity depends on role and causal
        A[2] = np.zeros((N_CONN_LEVELS, N_IMPORTANCE, N_LAYER_ROLE, N_CAUSAL))
        for s0 in range(N_IMPORTANCE):
            for s1 in range(N_LAYER_ROLE):
                for s2 in range(N_CAUSAL):
                    # intermediate layers tend to be denser
                    role_density = [0.2, 0.5, 0.3][s1]
                    strength = (role_density + s2 / (N_CAUSAL - 1)) / 2.0
                    p = np.array([
                        max(0.05, 0.6 - 0.4 * strength),
                        max(0.05, 0.3),
                        max(0.05, 0.1 + 0.4 * strength),
                    ])
                    A[2][:, s0, s1, s2] = p / p.sum()

        return A

    def _build_B(self) -> list:
        """Transition model P(s' | s, a).

        Only factor 0 (importance) is influenced by actions;
        factors 1 and 2 are intrinsic properties with identity dynamics.
        """
        B = utils.obj_array(3)

        B[0] = np.zeros((N_IMPORTANCE, N_IMPORTANCE, N_ACTIONS))
        for a in range(N_ACTIONS):
            for s in range(N_IMPORTANCE):
                t = np.zeros(N_IMPORTANCE)
                t[s] = 0.70
                if s > 0:
                    t[s - 1] = 0.15
                else:
                    t[s] += 0.15
                if s < N_IMPORTANCE - 1:
                    t[s + 1] = 0.15
                else:
                    t[s] += 0.15
                B[0][:, s, a] = t / t.sum()

        B[1] = np.eye(N_LAYER_ROLE).reshape(N_LAYER_ROLE, N_LAYER_ROLE, 1)
        B[2] = np.eye(N_CAUSAL).reshape(N_CAUSAL, N_CAUSAL, 1)

        return B

    def _build_C(self) -> list:
        """Preference model (log-prior over preferred observations).

        Higher KL and higher activation are more informative.
        """
        C = utils.obj_array(3)
        C[0] = np.array([0.0, 1.0, 3.0, 5.0]) * self.pragmatic_weight
        C[1] = np.array([0.0, 0.5, 2.0, 4.0]) * self.pragmatic_weight
        C[2] = np.array([0.0, 1.0, 2.0]) * self.pragmatic_weight
        return C

    def _build_D(self) -> list:
        """Prior over hidden states.

        Most features have low importance and weak causal influence.
        """
        D = utils.obj_array(3)
        D[0] = np.array([0.40, 0.30, 0.20, 0.10])
        D[1] = np.ones(N_LAYER_ROLE) / N_LAYER_ROLE
        D[2] = np.array([0.50, 0.30, 0.20])
        return D

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Build the generative model and instantiate the pymdp Agent."""
        self._A = self._build_A()
        self._B = self._build_B()
        self._C = self._build_C()
        self._D = self._build_D()

        self._pA = utils.obj_array(len(self._A))
        for m in range(len(self._A)):
            self._pA[m] = self._A[m].copy() * 10.0

        self._agent = PyMDPAgent(
            A=self._A,
            B=self._B,
            C=self._C,
            D=self._D,
            pA=self._pA,
            gamma=self.gamma,
            policy_len=self.policy_len,
            control_fac_idx=[0],
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=True,
            action_selection="stochastic",
            modalities_to_learn="all",
            lr_pA=self.lr_pA,
            inference_algo="VANILLA",
        )

        self._step = 0
        self._converged = False
        self.history.clear()
        self._feature_beliefs.clear()
        logger.info("Active Inference POMDP agent initialised (pymdp).")

    def reset(self) -> None:
        """Reset for a new circuit-discovery session."""
        self.initialize()

    # ------------------------------------------------------------------
    # Discretisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _discretise_kl(kl: float) -> int:
        if kl < KL_THRESHOLDS[0]:
            return 0
        if kl < KL_THRESHOLDS[1]:
            return 1
        if kl < KL_THRESHOLDS[2]:
            return 2
        return 3

    @staticmethod
    def _discretise_activation(act: float) -> int:
        if act < ACT_THRESHOLDS[0]:
            return 0
        if act < ACT_THRESHOLDS[1]:
            return 1
        if act < ACT_THRESHOLDS[2]:
            return 2
        return 3

    @staticmethod
    def _discretise_connectivity(degree: float) -> int:
        if degree < 3:
            return 0
        if degree < 8:
            return 1
        return 2

    def _layer_to_role(self, layer: int) -> int:
        """Map absolute layer index to {early, middle, late}."""
        third = self.n_layers / 3.0
        if layer < third:
            return 0
        if layer < 2 * third:
            return 1
        return 2

    def _feature_to_prior_obs(self, feat: Dict[str, Any]) -> list:
        """Derive an initial observation from graph metadata (before intervention)."""
        imp_obs = self._discretise_kl(feat.get("imp", 0.0) * 0.01)
        act_obs = self._discretise_activation(feat.get("act", 0.0))
        in_deg  = feat.get("in_degree", 0)
        out_deg = feat.get("out_degree", 0)
        conn_obs = self._discretise_connectivity(in_deg + out_deg)
        return [imp_obs, act_obs, conn_obs]

    def _measurement_to_obs(
        self, kl: float, activation: float, connectivity: float
    ) -> list:
        """Discretise real intervention measurements."""
        return [
            self._discretise_kl(kl),
            self._discretise_activation(activation),
            self._discretise_connectivity(connectivity),
        ]

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def select_intervention(
        self,
        candidate_features: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], str, float]:
        """Pick the next feature and action by minimising EFE.

        For each candidate, a prior observation is derived from graph
        metadata.  The agent infers states and policies conditioned on
        that observation, then the candidate--action pair with the
        lowest EFE is returned.

        Returns
        -------
        (feature_dict, action_name, efe_value)
        """
        if self._agent is None:
            self.initialize()

        if not candidate_features:
            raise ValueError("Empty candidate list.")

        best_feat: Optional[Dict] = None
        best_action = 0
        best_efe = float("inf")

        for feat in candidate_features:
            obs = self._feature_to_prior_obs(feat)
            self._agent.infer_states(obs)
            q_pi, efe = self._agent.infer_policies()

            min_idx = int(np.argmin(efe))
            min_val = float(efe[min_idx])

            if min_val < best_efe:
                best_efe = min_val
                best_feat = feat
                best_action = min_idx

        action_name = ACTION_NAMES[best_action]

        logger.debug(
            "Step %d: selected %s on %s (EFE=%.4f)",
            self._step,
            action_name,
            best_feat.get("fid", "?"),
            best_efe,
        )
        return best_feat, action_name, best_efe

    def update_beliefs(
        self,
        feature: Dict[str, Any],
        kl_divergence: float,
        activation_value: float,
        graph_connectivity: float,
    ) -> InterventionRecord:
        """Feed real intervention results back into the agent.

        Performs posterior state inference, A-matrix Dirichlet learning,
        and time-stepping.
        """
        if self._agent is None:
            self.initialize()

        obs = self._measurement_to_obs(kl_divergence, activation_value, graph_connectivity)
        beliefs_before = self._snapshot_beliefs()

        self._agent.infer_states(obs)
        if hasattr(self._agent, "update_A"):
            self._agent.update_A(obs)

        q_pi, efe = self._agent.infer_policies()
        self._agent.step_time()

        beliefs_after = self._snapshot_beliefs()
        min_efe = float(efe[int(np.argmin(efe))])

        record = InterventionRecord(
            step=self._step,
            feature_id=feature.get("fid", f"feat_{self._step}"),
            layer=feature.get("layer", -1),
            position=feature.get("pos", -1),
            feature_idx=feature.get("fidx", -1),
            action="ablation",
            observation=tuple(obs),
            kl_divergence=kl_divergence,
            activation_value=activation_value,
            graph_connectivity=graph_connectivity,
            beliefs_before=beliefs_before,
            beliefs_after=beliefs_after,
            efe_value=min_efe,
        )

        self.history.append(record)
        fid = feature.get("fid", f"feat_{self._step}")
        self._feature_beliefs[fid] = {k: v.copy() for k, v in beliefs_after.items()}

        self._step += 1
        self._check_convergence()
        return record

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def _snapshot_beliefs(self) -> Dict[str, np.ndarray]:
        qs = self._agent.qs if self._agent is not None else None
        if qs is None:
            return {
                "importance": np.ones(N_IMPORTANCE) / N_IMPORTANCE,
                "layer_role": np.ones(N_LAYER_ROLE) / N_LAYER_ROLE,
                "causal": np.ones(N_CAUSAL) / N_CAUSAL,
            }
        return {
            "importance": qs[0].copy() if len(qs) > 0 else np.ones(N_IMPORTANCE) / N_IMPORTANCE,
            "layer_role": qs[1].copy() if len(qs) > 1 else np.ones(N_LAYER_ROLE) / N_LAYER_ROLE,
            "causal": qs[2].copy() if len(qs) > 2 else np.ones(N_CAUSAL) / N_CAUSAL,
        }

    def _check_convergence(self, window: int = 5, threshold: float = 0.01) -> None:
        if len(self.history) < window + 1:
            return
        recent = self.history[-window:]
        kl_divs = []
        for i in range(1, len(recent)):
            for key in ("importance", "layer_role", "causal"):
                p = recent[i].beliefs_after[key] + 1e-10
                q = recent[i - 1].beliefs_after[key] + 1e-10
                kl_divs.append(float(np.sum(p * np.log(p / q))))
        avg_kl = float(np.mean(kl_divs))
        if avg_kl < threshold:
            self._converged = True
            logger.info(
                "Beliefs converged at step %d (avg KL=%.6f < %.4f)",
                self._step, avg_kl, threshold,
            )

    @property
    def is_converged(self) -> bool:
        return self._converged

    def get_feature_importance_ranking(self) -> List[Tuple[str, float]]:
        """Rank observed features by inferred importance."""
        rankings = []
        for fid, beliefs in self._feature_beliefs.items():
            dist = beliefs["importance"]
            expected = float(
                np.sum(dist * np.arange(N_IMPORTANCE)) / (N_IMPORTANCE - 1)
            )
            rankings.append((fid, expected))
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_belief_entropy_history(self) -> List[float]:
        """Per-step total entropy across all factors."""
        entropies = []
        for rec in self.history:
            h = 0.0
            for dist in rec.beliefs_after.values():
                d = dist + 1e-10
                h -= float(np.sum(d * np.log(d)))
            entropies.append(h)
        return entropies

    def get_efe_history(self) -> List[float]:
        return [r.efe_value for r in self.history]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise agent state for JSON logging."""
        return {
            "step": self._step,
            "converged": self._converged,
            "n_interventions": len(self.history),
            "epistemic_weight": self.epistemic_weight,
            "pragmatic_weight": self.pragmatic_weight,
            "n_layers": self.n_layers,
            "feature_beliefs": {
                fid: {k: v.tolist() for k, v in beliefs.items()}
                for fid, beliefs in self._feature_beliefs.items()
            },
            "belief_entropy_history": self.get_belief_entropy_history(),
            "efe_history": self.get_efe_history(),
        }

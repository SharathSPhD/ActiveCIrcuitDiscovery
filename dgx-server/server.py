"""ACD live-demo backend for the DGX Spark.

Wraps the existing ActiveCIrcuitDiscovery code (src/...) in a small FastAPI app
so the talk site's /demo page can run real POMDP episodes over a Cloudflare
tunnel.

Endpoints
---------
GET  /health            -> {status, model, gpu, graphs_ready}
GET  /prompts           -> canned prompt list with precompute status
POST /graph             -> build (or fetch cached) attribution graph + candidates
POST /episode           -> SSE stream: one full POMDP episode, step by step
POST /steer             -> multiplier sweep on one feature (Golden-Gate moment)

Run (from the ActiveCIrcuitDiscovery repo root):
    python dgx-server/server.py --model gemma            # real GPU mode
    python dgx-server/server.py --dry-run                # no GPU, fake KLs (API test)
    python dgx-server/server.py --model gemma --record   # also dump replay traces

Security: set ACD_API_KEY in the environment; requests must carry
`x-acd-key: <key>`. CORS is restricted via ACD_ALLOWED_ORIGINS
(comma-separated; default allows the Vercel talk site + localhost).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("acd-server")

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

API_KEY = os.environ.get("ACD_API_KEY", "")
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ACD_ALLOWED_ORIGINS",
        "https://acd-talk.vercel.app,http://localhost:3000,http://localhost:3001",
    ).split(",")
    if o.strip()
]

CANNED_PROMPTS = [
    {"id": "ioi-1", "task": "ioi", "prompt": "When John and Mary went to the store, John gave the bag to"},
    {"id": "ioi-2", "task": "ioi", "prompt": "After Alice and Bob finished lunch, Alice handed the receipt to"},
    {"id": "ms-1", "task": "multistep", "prompt": "If Alice is taller than Bob, and Bob is taller than Carol, then the tallest person is"},
    {"id": "steer-gg", "task": "steering", "prompt": "The Golden Gate Bridge is"},
    {"id": "dom-geo", "task": "domain", "prompt": "The capital of France is"},
]

BUDGET_DEFAULT = 20
MAX_BUDGET = 30
GRAPH_CACHE: Dict[str, Dict[str, Any]] = {}
GPU_LOCK = threading.Lock()  # one GPU job at a time

# ----------------------------------------------------------------------
# Model wrapper (real or dry-run)
# ----------------------------------------------------------------------

class Runtime:
    def __init__(self, model_key: str, dry_run: bool):
        self.model_key = model_key
        self.dry_run = dry_run
        self.backend = None
        self.n_layers = 26 if model_key == "gemma" else 16
        self._rng_state = 12345

    def load(self):
        if self.dry_run:
            log.info("DRY RUN: no model loaded")
            return
        import torch  # noqa
        from src.circuit_analysis.circuit_tracer_backend import CircuitTracerBackend

        name = "google/gemma-2-2b" if self.model_key == "gemma" else "meta-llama/Llama-3.2-1B"
        # circuit_tracer_backend maps short names via its registry; try both spellings
        for candidate in (self.model_key, name):
            try:
                self.backend = CircuitTracerBackend(model_name=candidate)
                break
            except ValueError:
                continue
        if self.backend is None:
            self.backend = CircuitTracerBackend(model_name=name)
        self.backend.load_model()
        self.n_layers = int(self.backend.model.cfg.n_layers)
        log.info("Model loaded: %s (%d layers)", self.model_key, self.n_layers)

    # -------------------- graph + candidates --------------------
    def build_graph(self, prompt: str) -> Dict[str, Any]:
        if prompt in GRAPH_CACHE:
            return GRAPH_CACHE[prompt]
        t0 = time.time()
        if self.dry_run:
            cands = self._fake_candidates()
            graph = {"n_features": 12600, "n_candidates": len(cands), "candidates": cands, "build_seconds": 0.1}
        else:
            from src.experiments.run_real_experiments import extract_candidates

            raw = self.backend.get_raw_graph(prompt)
            cands = extract_candidates(raw)
            graph = {
                "n_features": int(raw.active_features.shape[0]),
                "n_candidates": len(cands),
                "candidates": cands,
                "build_seconds": round(time.time() - t0, 1),
            }
        GRAPH_CACHE[prompt] = graph
        return graph

    def _fake_candidates(self) -> List[Dict]:
        import random

        rng = random.Random(42)
        out = []
        for layer in range(0, self.n_layers, max(1, self.n_layers // 12)):
            for k in range(4):
                imp = rng.random()
                out.append(
                    dict(
                        layer=layer, pos=rng.randint(3, 14), fidx=rng.randint(100, 16000),
                        act=rng.uniform(0.5, 60.0), imp=imp, eap=imp * rng.uniform(0.5, 1.0),
                        fid=f"L{layer}_P{k}_F{rng.randint(100, 16000)}",
                        in_degree=rng.randint(1, 40), out_degree=rng.randint(1, 40),
                    )
                )
        out.sort(key=lambda c: c["imp"], reverse=True)
        return out[:48]

    # -------------------- interventions --------------------
    def clean_state(self, prompt: str):
        if self.dry_run:
            return None, None
        import torch

        logits, _ = self.backend.model.feature_intervention(prompt, [])
        last = logits[0, -1, :]
        return torch.softmax(last, -1), last

    def intervene(self, prompt: str, feat: Dict, action: str, clean_probs, clean_last, mult: float = 10.0) -> float:
        if self.dry_run:
            # deterministic pseudo-KL keyed on feature importance + action
            base = 10 ** (-4 + 2.5 * feat["imp"])
            gain = {"ablation": 1.0, "activation_patching": 0.8, "feature_steering": 25.0}[action]
            self._rng_state = (self._rng_state * 1103515245 + 12345) % (2 ** 31)
            noise = 0.6 + (self._rng_state / 2 ** 31) * 0.8
            return float(base * gain * noise)
        from src.experiments.run_real_experiments import (
            ablate_feature, patch_feature, steer_feature,
        )

        if action == "activation_patching":
            kl, _ = patch_feature(self.backend.model, prompt, feat, clean_probs, clean_last)
        elif action == "feature_steering":
            kl, _, _ = steer_feature(self.backend.model, prompt, feat, mult, clean_probs, clean_last)
        else:
            kl, _ = ablate_feature(self.backend.model, prompt, feat, clean_probs, clean_last)
        return float(kl)

    def steer_sweep(self, prompt: str, feat: Dict, mults: List[float]) -> List[Dict[str, Any]]:
        out = []
        if self.dry_run:
            for m in mults:
                kl = 10 ** (-3 + 0.28 * m) * (0.5 + feat["imp"])
                out.append({"mult": m, "kl": kl, "top_tokens": [["the", 0.4], ["bridge", 0.2 + 0.05 * m]], "top1_changed": m >= 5})
            return out
        import torch

        clean_probs, clean_last = self.clean_state(prompt)
        clean_top = int(torch.argmax(clean_last).item())
        for m in mults:
            iv, _ = self.backend.model.feature_intervention(
                prompt, [(feat["layer"], feat["pos"], feat["fidx"], feat["act"] * m)]
            )
            iv_last = iv[0, -1, :]
            iv_probs = torch.softmax(iv_last, -1)
            kl = float(torch.nn.functional.kl_div(torch.log(iv_probs + 1e-10), clean_probs, reduction="sum").item())
            topv, topi = torch.topk(iv_probs, 5)
            toks = [[self.backend.model.tokenizer.decode([int(i)]), float(v)] for v, i in zip(topv, topi)]
            out.append({"mult": m, "kl": kl, "top_tokens": toks, "top1_changed": int(topi[0].item()) != clean_top})
        return out


RT: Optional[Runtime] = None
RECORD_DIR: Optional[Path] = None

# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------

app = FastAPI(title="ACD live demo backend", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_key(x_acd_key: Optional[str]):
    if API_KEY and x_acd_key != API_KEY:
        raise HTTPException(status_code=401, detail="bad or missing x-acd-key")


class GraphReq(BaseModel):
    prompt: str


class EpisodeReq(BaseModel):
    prompt: str
    budget: int = BUDGET_DEFAULT
    mode: str = "multi"  # "multi" | "ablation"


class SteerReq(BaseModel):
    prompt: str
    feature_rank: int = 0  # index into candidates by importance
    multipliers: List[float] = [0, 1.5, 2, 3, 5, 10]


@app.get("/health")
def health():
    gpu = "dry-run"
    if RT and not RT.dry_run:
        try:
            import torch

            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        except Exception:
            gpu = "unknown"
    return {
        "status": "ok",
        "model": RT.model_key if RT else None,
        "dry_run": RT.dry_run if RT else True,
        "gpu": gpu,
        "graphs_cached": len(GRAPH_CACHE),
        "n_layers": RT.n_layers if RT else None,
    }


@app.get("/prompts")
def prompts(x_acd_key: Optional[str] = Header(default=None)):
    check_key(x_acd_key)
    return {
        "prompts": [
            {**p, "cached": p["prompt"] in GRAPH_CACHE} for p in CANNED_PROMPTS
        ],
        "budget_default": BUDGET_DEFAULT,
    }


@app.post("/graph")
def graph(req: GraphReq, x_acd_key: Optional[str] = Header(default=None)):
    check_key(x_acd_key)
    if len(req.prompt) > 300:
        raise HTTPException(400, "prompt too long")
    with GPU_LOCK:
        g = RT.build_graph(req.prompt)
    return g


@app.post("/steer")
def steer(req: SteerReq, x_acd_key: Optional[str] = Header(default=None)):
    check_key(x_acd_key)
    with GPU_LOCK:
        g = RT.build_graph(req.prompt)
        cands = g["candidates"]
        feat = cands[max(0, min(req.feature_rank, len(cands) - 1))]
        sweep = RT.steer_sweep(req.prompt, feat, req.multipliers[:8])
    return {"feature": feat, "sweep": sweep}


def _episode_events(prompt: str, budget: int, mode: str):
    """Generator yielding SSE events for one POMDP episode."""
    from src.active_inference.pomdp_agent import ActiveInferencePOMDPAgent

    with GPU_LOCK:
        g = RT.build_graph(prompt)
        yield "graph", {
            "n_features": g["n_features"],
            "n_candidates": g["n_candidates"],
            "candidates": [
                {k: c[k] for k in ("fid", "layer", "pos", "fidx", "imp", "eap", "act", "in_degree", "out_degree")}
                for c in g["candidates"]
            ],
            "build_seconds": g.get("build_seconds"),
        }

        clean_probs, clean_last = RT.clean_state(prompt)
        agent = ActiveInferencePOMDPAgent(n_layers=RT.n_layers)
        agent.initialize()

        candidates = g["candidates"]
        observed: set = set()
        cum_ai = 0.0
        trace = []

        # Oracle/EAP comparators (computed lazily from EAP scores; a true oracle
        # would require ablating everything up front, which we skip live).
        eap_order = sorted(candidates, key=lambda c: c["eap"], reverse=True)

        for step in range(min(budget, len(candidates))):
            unobserved = [c for c in candidates if c["fid"] not in observed]
            if not unobserved:
                break
            t0 = time.time()
            feat, action_name, efe = agent.select_intervention(unobserved)
            if mode == "ablation":
                action_name = "ablation"
            kl = RT.intervene(prompt, feat, action_name, clean_probs, clean_last)
            rec = agent.update_beliefs(
                feat,
                action_name=action_name,
                kl_divergence=kl,
                activation_value=feat["act"],
                graph_connectivity=feat.get("in_degree", 0) + feat.get("out_degree", 0),
            )
            observed.add(feat["fid"])
            cum_ai += kl

            beliefs = {k: [round(float(x), 4) for x in v] for k, v in rec.beliefs_after.items()}
            ev = {
                "step": step + 1,
                "fid": feat["fid"],
                "layer": feat["layer"],
                "action": action_name,
                "efe": round(float(efe), 4),
                "kl": kl,
                "cum_kl": cum_ai,
                "obs_bins": list(rec.observation),
                "beliefs": beliefs,
                "entropy": agent.get_belief_entropy_history()[-1] if agent.get_belief_entropy_history() else None,
                "a_drift": agent.get_a_drift_history()[-1] if agent.get_a_drift_history() else None,
                "converged": agent.is_converged,
                "step_seconds": round(time.time() - t0, 3),
                "eap_next": eap_order[step]["fid"] if step < len(eap_order) else None,
            }
            trace.append(ev)
            yield "step", ev

        ranking = agent.get_feature_importance_ranking()[:10]
        summary = {
            "steps": len(trace),
            "cum_kl": cum_ai,
            "converged": agent.is_converged,
            "top_features": [{"fid": f, "expected_importance": round(v, 3)} for f, v in ranking],
        }
        yield "done", summary

        if RECORD_DIR is not None:
            RECORD_DIR.mkdir(parents=True, exist_ok=True)
            fname = RECORD_DIR / f"episode_{int(time.time())}.json"
            fname.write_text(json.dumps({"prompt": prompt, "mode": mode, "budget": budget,
                                         "model": RT.model_key, "dry_run": RT.dry_run,
                                         "graph": {"n_features": g["n_features"], "n_candidates": g["n_candidates"]},
                                         "candidates": g["candidates"],
                                         "steps": trace, "summary": summary}, indent=1))
            log.info("recorded trace -> %s", fname)


@app.post("/episode")
async def episode(req: EpisodeReq, request: Request, x_acd_key: Optional[str] = Header(default=None)):
    check_key(x_acd_key)
    budget = max(1, min(req.budget, MAX_BUDGET))

    async def stream():
        loop = asyncio.get_event_loop()
        gen = _episode_events(req.prompt, budget, req.mode)

        def next_event():
            try:
                return next(gen)
            except StopIteration:
                return None

        while True:
            item = await loop.run_in_executor(None, next_event)
            if item is None:
                break
            kind, payload = item
            yield f"event: {kind}\ndata: {json.dumps(payload)}\n\n"
            if await request.is_disconnected():
                break

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------

def main():
    global RT, RECORD_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gemma", "llama"], default="gemma")
    ap.add_argument("--dry-run", action="store_true", help="no GPU; fake KLs for API testing")
    ap.add_argument("--record", action="store_true", help="dump episode traces to dgx-server/traces/")
    ap.add_argument("--precompute", action="store_true", help="build graphs for canned prompts at startup")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()

    RT = Runtime(args.model, args.dry_run)
    RT.load()
    if args.record:
        RECORD_DIR = Path(__file__).parent / "traces"

    if args.precompute:
        for p in CANNED_PROMPTS:
            log.info("precomputing graph for %s ...", p["id"])
            RT.build_graph(p["prompt"])
        log.info("all canned graphs cached")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

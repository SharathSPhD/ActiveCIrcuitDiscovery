# ACD Expert-Talk Runbook

Companion assets for the Active Inference community talk. Three pieces:

| Piece | Where | What |
|---|---|---|
| Talk site | `talk/` → Vercel | 5-part expert walkthrough + live demo + Q&A shield |
| GPU backend | `dgx-server/` → runs on the DGX Spark | FastAPI + SSE, real POMDP episodes for the demo |
| Research briefs | `talk/research/*.md` | verified source notes behind Parts I, II and V |

## The site

Live at the Vercel URL (project `acd-talk`). Local dev:

```bash
cd talk
npm install
npm run dev            # http://localhost:3000
```

`npm run build` first runs `scripts/build-data.mjs`, which refreshes `talk/data/`
from `results/*.json` + `site/src/data/*.json` when run inside the repo (set
`ACD_REPO` to override). On Vercel the checked-in `talk/data/` is used as-is, so
the site always matches the paper even without the results files.

Pages: `/` agenda · `/mech-interp` Part I · `/active-inference` Part II ·
`/results` Part III · `/demo` Part IV · `/qa` Part V.

## Going live for the demo (day-of checklist)

**One command, on the DGX:**

```bash
cd ~/projects/ActiveCIrcuitDiscovery && ./dgx-server/golive.sh
```

That starts the backend if it is down (model + 5 precomputed graphs, ~2–3 min),
opens the Cloudflare tunnel, writes `DGX_TUNNEL_URL`/`DGX_API_KEY` into the
Vercel project, deploys to production, and verifies
`https://acd-talk.vercel.app/api/dgx/health` before it reports success. Then
open `/demo` — the badge should read **LIVE · NVIDIA GB10**.

Other modes:

```bash
./dgx-server/golive.sh --status        # what's up right now, changes nothing
./dgx-server/golive.sh --tunnel-only   # backend already warm; just re-tunnel + redeploy
```

**Why a redeploy is needed each time:** the quick tunnel gets a new random
`*.trycloudflare.com` hostname on every start, and Vercel binds env vars at
deploy time. `golive.sh` handles both halves; budget ~1 min for the deploy.
The API key is generated once into `~/.acd-demo.env` and reused, so only the
URL actually churns.

*Stable-hostname upgrade (optional, do it before travel day):* run
`cloudflared tunnel login` once interactively, then create a named tunnel on a
domain you own (recipe in `dgx-server/README.md` §2). With a fixed hostname the
Vercel env vars stop changing and a tunnel restart needs no redeploy at all.

**Fallback is automatic and tested**: with the DGX unreachable the badge reads
`OFFLINE · REPLAY MODE`, the button relabels to "Replay recorded run", and the
identical UI replays the paper's recorded runs. Nothing errors on screen.

After good live runs, copy `dgx-server/traces/episode_*.json` into the replay
library if you want them preserved.

Full backend details: `dgx-server/README.md`.

## ⚠ Dependency pin that matters (now fixed)

`requirements.txt` used to install pymdp from git **main**, which is now the JAX
rewrite (v1.x) and **breaks** `src/active_inference/pomdp_agent.py`
(`pymdp.utils.obj_array` is gone). It is now pinned to the legacy commit the DGX
venv actually runs — verified as `inferactively-pymdp 0.0.7.1` at
`23c206fc0e089ef8e2e6c80fb0162cb10a0fcdf9` — so fresh clones work:

```
inferactively-pymdp @ git+https://github.com/infer-actively/pymdp.git@23c206fc0e089ef8e2e6c80fb0162cb10a0fcdf9
```

Note `dgx-server/requirements.txt` (fastapi/uvicorn/pydantic) is installed on
top of that; `run.sh` and `golive.sh` do it for you.

## Talk timing map (60–75 min)

- Part I mech interp — ~20 min (compress I.1–I.2 if audience knows SAEs)
- Part II the bridge — ~20 min (the B-matrix + EFE walkthrough is the core)
- Part III results — ~15 min (lead with the protocol, end with III.7)
- Part IV live demo — ~10 min (narration script is on the page, under
  "What to narrate while it runs")
- Part V — don't present it; it's your pocket during discussion.

## Redeploying after edits

The Vercel project builds from the `talk-site` branch tarball. After editing:

```bash
git add -A && git commit -m "talk updates" && git push origin talk-site
```

then hit "Redeploy" in Vercel (or connect the project to the repo in the
Vercel dashboard for automatic builds — Settings → Git).

## What is deployed right now (verified 2026-07-25)

| Piece | State |
|---|---|
| Vercel project | `acd-talk` (team `ss-projects-f08e52ab`), aliased to `acd-talk.vercel.app` |
| Env vars | `DGX_TUNNEL_URL`, `DGX_API_KEY` set on **Production** |
| Backend | `dgx-server/server.py --model gemma --precompute --record` on `:8787`, real GPU (NVIDIA GB10), 5 graphs cached |
| Tunnel | cloudflared quick tunnel → recorded in `/tmp/acd/tunnel_url.txt` |
| Logs | backend `/tmp/acd/server.log` · tunnel `/tmp/acd/tunnel.log` |
| API key | `~/.acd-demo.env` (chmod 600), reused across restarts |

End-to-end checks that passed: all 6 pages 200; `/api/dgx/health` OK through
the public URL; a 20-step episode streamed in-browser from the deployed site
(`graph ready · 60 candidates of 11,991 active features`, cum KL 0.173,
beliefs converged); SSE events arrive incrementally ~0.33 s apart (Vercel is
not buffering the stream); `/steer` returns a monotonic dose–response; and the
OFFLINE→replay fallback renders correctly with no console errors.

Note for the talk: the "Cumulative KL race" panel shows the oracle/EAP/greedy
comparison bars in **replay** mode only — in live mode the agent is the sole
bar, because those baselines come from the recorded runs rather than being
recomputed on stage. Worth knowing before you point at that panel.

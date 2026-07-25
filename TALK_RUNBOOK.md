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

1. **On the DGX** (~5 min before):
   ```bash
   cd ~/projects/ActiveCIrcuitDiscovery
   export ACD_API_KEY=<your key>          # openssl rand -hex 16
   ./dgx-server/run.sh gemma              # loads model + precomputes canned graphs
   cloudflared tunnel --url http://localhost:8787   # or your named tunnel
   ```
2. **On Vercel** (project → Settings → Environment Variables):
   - `DGX_TUNNEL_URL` = the tunnel https URL
   - `DGX_API_KEY` = same key
   - Redeploy (or `vercel --prod`). The demo page badge flips to **LIVE**.
3. **Fallback is automatic**: if the tunnel is down mid-talk, the page falls
   back to replaying the paper's recorded runs with the identical UI.
4. After good live runs, copy `dgx-server/traces/episode_*.json` into the
   replay library if you want them preserved.

Full backend details: `dgx-server/README.md`.

## ⚠ Dependency pin that matters

`requirements.txt` installs pymdp from git **main**, which is now the JAX
rewrite (v1.x) and **breaks** `src/active_inference/pomdp_agent.py`
(`pymdp.utils.obj_array` is gone). Your DGX venv works because it holds the
older legacy build. For any fresh environment (including the DGX server on a
new machine) pin the exact commit your venv uses:

```bash
pip install "git+https://github.com/infer-actively/pymdp.git@23c206fc0e089ef8e2e6c80fb0162cb10a0fcdf9"
```

Worth fixing in `requirements.txt` before anyone else clones the repo.

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

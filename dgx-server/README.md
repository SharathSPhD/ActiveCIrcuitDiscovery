# ACD live-demo backend (DGX Spark + Cloudflare tunnel)

> **Just want to start or stop it?** See [RUNNING.md](RUNNING.md) — two commands.

FastAPI server that lets the talk site (`talk/` on Vercel) run **real POMDP
episodes on the DGX Spark** during the presentation. The demo page works
without it (replay mode); when this server is reachable, the page flips to
LIVE.

## 1. Start the server on the DGX

```bash
cd ~/projects/ActiveCIrcuitDiscovery
export ACD_API_KEY=$(openssl rand -hex 16)      # note it down
./dgx-server/run.sh gemma                        # loads model, precomputes canned graphs
```

First start takes a few minutes (model + transcoders + 5 precomputed graphs at
~18 s each). After that: episodes stream a step every ~0.05–0.3 s.

Smoke-test locally:

```bash
curl -s localhost:8787/health | jq
curl -s -X POST localhost:8787/episode \
  -H "content-type: application/json" -H "x-acd-key: $ACD_API_KEY" \
  -d '{"prompt":"When John and Mary went to the store, John gave the bag to","budget":5}'
```

No GPU handy? `python dgx-server/server.py --dry-run` exercises the whole API
with synthetic KLs — the Vercel proxy and demo UI can be tested against it.

## 2. Open the Cloudflare tunnel

Quick tunnel (no account config, URL changes each run):

```bash
cloudflared tunnel --url http://localhost:8787
# → prints https://<random>.trycloudflare.com
```

A *named* tunnel would give a stable hostname, but that needs a domain hosted on
Cloudflare and this account has no zones — so the stable URL is provided by the
Worker instead (see "The permanent URL" below). In practice you never run either
command by hand: `./dgx-server/golive.sh` does the tunnel and the repoint.

## 3. Point the Vercel app at the gateway

Already done, and it does not change: `DGX_TUNNEL_URL` = the Worker URL,
`DGX_API_KEY` = the key in `~/.acd-demo.env`. The site proxies browser →
`/api/dgx/*` → Worker → tunnel, so neither the key nor the tunnel URL ever
reaches the client, and CORS never comes up.

## 4. During the talk

- `--precompute` has already cached graphs for the canned prompts, so "Run
  live" starts streaming steps immediately.
- A free-form prompt costs one ~20 s graph build (the UI shows a build timer),
  then steps stream fast.
- `--record` writes every episode to `dgx-server/traces/episode_*.json`.
  Copy the good ones into `talk/data/replays/` (see `talk/README.md`) so the
  replay library grows from real runs.
- One GPU job at a time (server-side lock). If the page shows OFFLINE, the
  demo silently uses replays — no failure mode visible to the audience.

## Endpoints

| Method | Path      | Purpose                                              |
|--------|-----------|------------------------------------------------------|
| GET    | /health   | status, model, GPU name, cached graph count          |
| GET    | /prompts  | canned prompts + cache status                        |
| POST   | /graph    | `{prompt}` → candidates (cached)                     |
| POST   | /episode  | `{prompt,budget,mode}` → SSE `graph`/`step`/`done`   |
| POST   | /steer    | `{prompt,feature_rank,multipliers}` → dose sweep     |

All POSTs require header `x-acd-key`.

## Operating it (scripts)

| Script | What it does |
|---|---|
| `./dgx-server/golive.sh` | backend (if down) + tunnel + repoint gateway + verify. One command to go on air (~15 s warm). |
| `./dgx-server/golive.sh --stop` | stop tunnel + backend, free the ~47 GB of GPU memory |
| `./dgx-server/golive.sh --status` | reports backend / tunnel / gateway / public health, changes nothing |
| `./dgx-server/golive.sh --tunnel-only` | backend already warm — just re-tunnel and repoint the gateway |
| `./dgx-server/cf.sh status\|set <url>\|deploy\|get` | Cloudflare gateway: inspect, repoint by hand, or redeploy `worker.js` |
| `./dgx-server/verify.sh` | pre-talk smoke test of the whole chain (16 checks, no browser needed) |
| `cd talk && node e2e/demo-live.mjs` | browser check of the deployed `/demo`: LIVE badge + a real streamed episode |
| `cd talk && BASE=http://localhost:3100 node e2e/demo-fallback.mjs` | browser check that the OFFLINE→replay fallback works against a local `next dev -p 3100` |

The two `.mjs` checks need Playwright, and must be run from `talk/` because Node
resolves imports relative to the script's own directory:

```bash
cd talk && npm i --no-save playwright && npx playwright install chromium
node e2e/demo-live.mjs
```

`verify.sh` has no dependencies beyond curl and is the one to run on the day.

The generated API key lives in `~/.acd-demo.env` and is reused across restarts, so
only the tunnel hostname churns. Runtime logs: `/tmp/acd/server.log`,
`/tmp/acd/tunnel.log`; current tunnel URL in `/tmp/acd/tunnel_url.txt`.

## The permanent URL (Worker + KV)

`dgx-server/worker.js` is a Cloudflare Worker at
`https://acd-demo.sharath-sathish.workers.dev`. It reads `gateway_url` from the
KV namespace `acd_gateway` and proxies to whatever quick tunnel is current.
Vercel's `DGX_TUNNEL_URL` points at the Worker, so it never has to change —
swapping tunnels is one KV write and needs no redeploy.

Two things the Worker does that the analogous prabhasa worker does not:

- **Streams the body through** (`new Response(resp.body, …)`). Buffering with
  `await resp.arrayBuffer()` would collapse the SSE step-by-step stream that the
  whole demo is built around.
- **Normalises a dead tunnel.** A stopped quick tunnel makes Cloudflare's edge
  answer `530` (error 1033) as HTML rather than throwing, so the Worker maps
  `530/502/504` to a clean `{"error":"offline"}` `503`. The demo page then reads
  one consistent signal and falls back to replay.

Credentials live in `~/.cloudflare-creds.env` (chmod 600, not in the repo).

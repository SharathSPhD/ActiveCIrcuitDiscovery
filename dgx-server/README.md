# ACD live-demo backend (DGX Spark + Cloudflare tunnel)

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

Named tunnel (stable hostname — recommended for the talk; same pattern as the
prabodha setup):

```bash
cloudflared tunnel login
cloudflared tunnel create acd-demo
cloudflared tunnel route dns acd-demo acd-demo.<your-domain>
cat > ~/.cloudflared/config.yml <<EOF
tunnel: acd-demo
credentials-file: /home/<user>/.cloudflared/<tunnel-id>.json
ingress:
  - hostname: acd-demo.<your-domain>
    service: http://localhost:8787
  - service: http_status:404
EOF
cloudflared tunnel run acd-demo
```

## 3. Point the Vercel app at the tunnel

In the Vercel project (acd-talk) set environment variables and redeploy:

- `DGX_TUNNEL_URL` = `https://acd-demo.<your-domain>` (or the trycloudflare URL)
- `DGX_API_KEY`   = the `ACD_API_KEY` you generated

The site proxies browser → `/api/dgx/*` → tunnel, so the key and tunnel URL
never reach the client, and CORS never comes up.

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

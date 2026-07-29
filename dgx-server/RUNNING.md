# Starting and stopping the live demo

Two commands. Everything else is detail.

```bash
cd ~/projects/ActiveCIrcuitDiscovery

./dgx-server/golive.sh          # START — put the demo on air
./dgx-server/golive.sh --stop   # STOP  — take it down, free the GPU
```

The site at **https://acd-talk.vercel.app** stays up either way. When the demo is
stopped it simply shows `OFFLINE · REPLAY MODE` and replays the paper's recorded
runs — the audience sees the same interface, only the badge differs. So there is
no urgency about leaving it running, and nothing breaks if you forget to start it.

---

## Start

```bash
./dgx-server/golive.sh
```

- **Cold** (backend not running): ~2–3 min. Loads Gemma-2-2B + transcoders onto
  the GB10 and precomputes attribution graphs for the 5 canned prompts, so
  nothing stalls for 20 s while you are on stage.
- **Warm** (backend already up, tunnel dropped): ~15 s. Only re-tunnels and
  repoints the gateway.

It finishes by checking the public URL end to end and prints
`LIVE. Open https://acd-talk.vercel.app/demo` when the whole chain is good.
Open `/demo` — the badge should read **LIVE · NVIDIA GB10**.

Costs while running: **~47 GB of GPU memory** and ~12 GB RAM.

## Stop

```bash
./dgx-server/golive.sh --stop
```

In order: clears the gateway pointer (so the site flips to replay cleanly rather
than hanging on a tunnel that is about to vanish), stops the tunnel, then stops
the backend with SIGTERM — escalating to SIGKILL only if it refuses — and prints
the recovered memory. The ~47 GB of GPU memory comes back immediately.

## Check

```bash
./dgx-server/golive.sh --status   # backend / tunnel / gateway / public URL
./dgx-server/verify.sh            # full 16-check smoke test, run this before the talk
```

---

## What the pieces are

```
browser -> acd-talk.vercel.app/api/dgx/*          Vercel, permanent URL
        -> acd-demo.sharath-sathish.workers.dev   Cloudflare Worker + KV
        -> https://<random>.trycloudflare.com     quick tunnel, new name each run
        -> DGX Spark :8787                        FastAPI + Gemma-2-2B on GB10
```

Only the last two live on this machine, and only they are started/stopped. The
Vercel site and the Cloudflare Worker are always up and cost nothing when idle —
leave them alone.

The Worker exists so the churning tunnel hostname never reaches Vercel: it reads
the current tunnel URL out of KV, so restarting the tunnel is a KV write rather
than a redeploy. That is why the warm restart is 15 seconds.

| Thing | Where |
|---|---|
| Backend log | `/tmp/acd/server.log` |
| Tunnel log | `/tmp/acd/tunnel.log` |
| Current tunnel URL | `/tmp/acd/tunnel_url.txt` |
| Backend API key | `~/.acd-demo.env` (chmod 600, reused across restarts) |
| Cloudflare creds | `~/.cloudflare-creds.env` (chmod 600, not in the repo) |
| Recorded episodes | `dgx-server/traces/episode_*.json` |

## If something is wrong

| Symptom | What it means | Fix |
|---|---|---|
| Badge stuck on `OFFLINE` | tunnel or backend down, or KV pointer stale | `./dgx-server/golive.sh` |
| `{"error":"offline"}` from `/api/dgx/health` | gateway can't reach a tunnel | `./dgx-server/golive.sh` |
| Start fails, "backend did not come up" | model load died | `tail -50 /tmp/acd/server.log` |
| Steps stream but every KL is 0 | backend is in `--dry-run` | restart without that flag (`golive.sh` never sets it) |
| GPU memory still held after `--stop` | `nvidia-smi` can lag a second behind the process exiting | re-run `nvidia-smi`; if a PID really persists, `kill -9` it |

A stale gateway pointer is by far the most common cause, and `golive.sh` fixes
all of them — it is safe to re-run at any time.

## Restarting mid-talk

If the tunnel drops while you are presenting, the page falls back to replay on
its own; nothing errors on screen and you can keep talking. Re-running
`./dgx-server/golive.sh` in another terminal takes ~15 s and the badge returns to
LIVE on the next page load.

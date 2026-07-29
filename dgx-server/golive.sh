#!/usr/bin/env bash
# One command to put the live demo on air from the DGX Spark.
#
#   ./dgx-server/golive.sh                # start backend (if down) + tunnel + repoint gateway
#   ./dgx-server/golive.sh --tunnel-only  # backend already warm; just re-tunnel
#   ./dgx-server/golive.sh --stop         # stop tunnel + backend, free the GPU
#   ./dgx-server/golive.sh --status       # report what's up, change nothing
#
# Architecture:
#   browser -> acd-talk.vercel.app/api/dgx/*        (holds a PERMANENT URL)
#           -> acd-demo.sharath-sathish.workers.dev (Cloudflare Worker + KV)
#           -> https://<random>.trycloudflare.com   (quick tunnel, new name each run)
#           -> DGX Spark :8787                      (FastAPI + Gemma-2-2B on GB10)
#
# A quick tunnel gets a new random hostname on every restart, and Vercel binds
# env vars at deploy time. The Worker absorbs that churn: Vercel always points at
# the Worker, and switching tunnels is a single KV write. So restarting the tunnel
# mid-talk costs seconds and needs no redeploy.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8787}"
MODEL="${MODEL:-gemma}"
CFD="${CFD:-$HOME/.local/bin/cloudflared}"
KEYFILE="${KEYFILE:-$HOME/.acd-demo.env}"
RUN=/tmp/acd
SITE="https://acd-talk.vercel.app"
WORKER_URL="${WORKER_URL:-https://acd-demo.sharath-sathish.workers.dev}"

mkdir -p "$RUN"
cd "$REPO"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- API key
if [ ! -f "$KEYFILE" ]; then
  printf 'export ACD_API_KEY=%s\n' "$(openssl rand -hex 16)" > "$KEYFILE"
  chmod 600 "$KEYFILE"
  echo "generated a new ACD_API_KEY in $KEYFILE"
fi
# shellcheck disable=SC1090
source "$KEYFILE"
[ -n "${ACD_API_KEY:-}" ] || die "ACD_API_KEY empty in $KEYFILE"

backend_up() { curl -sf -m 5 "http://localhost:${PORT}/health" >/dev/null 2>&1; }

# ---------------------------------------------------------------- --status
if [ "${1:-}" = "--status" ]; then
  say "status"
  backend_up && echo "backend  : UP   $(curl -s -m 5 localhost:${PORT}/health)" || echo "backend  : DOWN"
  URL=$(cat "$RUN/tunnel_url.txt" 2>/dev/null || echo "")
  if [ -n "$URL" ] && curl -sf -m 15 "$URL/health" >/dev/null 2>&1; then
    echo "tunnel   : UP   $URL"
  else
    echo "tunnel   : DOWN ${URL:-(none recorded)}"
  fi
  echo "gateway  : $WORKER_URL -> $("$REPO/dgx-server/cf.sh" get 2>/dev/null | sed 's/^gateway_url = //')"
  echo "public   : $(curl -s -m 25 "$SITE/api/dgx/health" | head -c 200)"
  exit 0
fi

# ---------------------------------------------------------------- --stop
if [ "${1:-}" = "--stop" ]; then
  say "stopping the live demo"

  # 1. Clear the gateway pointer FIRST, so the site flips to replay cleanly
  #    rather than briefly hanging on a tunnel that is about to disappear.
  "$REPO/dgx-server/cf.sh" clear || echo "warning: could not clear the KV pointer"

  # 2. Tunnel.
  n=0
  for pid in $(pgrep -f "^${CFD} tunnel --url http://localhost:${PORT}( |$)" 2>/dev/null); do
    kill "$pid" 2>/dev/null && { echo "stopped tunnel pid $pid"; n=$((n+1)); }
  done
  [ "$n" = 0 ] && echo "no tunnel was running"
  rm -f "$RUN/tunnel_url.txt"

  # 3. Backend. SIGTERM first so uvicorn shuts down and the GPU allocator
  #    releases; escalate only if it is still there.
  for pid in $(pgrep -f "dgx-server/server.py" 2>/dev/null); do
    kill "$pid" 2>/dev/null && echo "stopping backend pid $pid (SIGTERM)"
  done
  for _ in $(seq 1 20); do
    pgrep -f "dgx-server/server.py" >/dev/null 2>&1 || break
    sleep 1
  done
  for pid in $(pgrep -f "dgx-server/server.py" 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null && echo "backend pid $pid did not exit; sent SIGKILL"
  done
  pgrep -f "dgx-server/server.py" >/dev/null 2>&1 && die "backend still running" \
    || echo "backend stopped"

  say "memory"
  free -h | awk 'NR==1||NR==2'
  echo "GPU processes still holding memory:"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
    | sed 's/^/  /' || echo "  (none reported)"

  say "public site"
  echo "$(curl -s -m 30 "$SITE/api/dgx/health" | head -c 160)"
  echo "The demo page now reads OFFLINE · REPLAY MODE and replays recorded runs."
  echo "Restart any time with: ./dgx-server/golive.sh"
  exit 0
fi

# ---------------------------------------------------------------- backend
if [ "${1:-}" != "--tunnel-only" ]; then
  if backend_up; then
    say "backend already up on :$PORT"
  else
    say "starting backend (model load + 5 precomputed graphs; ~2-3 min)"
    nohup env ACD_API_KEY="$ACD_API_KEY" bash -c "
      cd '$REPO'
      source .venv/bin/activate
      exec python dgx-server/server.py --model '$MODEL' --precompute --record --port '$PORT'
    " > "$RUN/server.log" 2>&1 &
    echo "pid $! · log $RUN/server.log"
    for _ in $(seq 1 120); do
      backend_up && break
      if grep -qE "Traceback|Address already in use" "$RUN/server.log"; then
        tail -25 "$RUN/server.log" >&2; die "backend failed to start"
      fi
      sleep 5
    done
    backend_up || { tail -25 "$RUN/server.log" >&2; die "backend did not come up in 10 min"; }
  fi
  curl -s -m 5 "localhost:${PORT}/health"; echo
fi

# ---------------------------------------------------------------- tunnel
say "opening cloudflare tunnel"
# Match the absolute binary path so this pattern can never match this script's own
# shell. The trailing ( |$) matters: the real command line ends in --no-autoupdate,
# so anchoring with a bare $ silently matches nothing and leaks a tunnel per run.
for pid in $(pgrep -f "^${CFD} tunnel --url http://localhost:${PORT}( |$)" 2>/dev/null); do
  kill "$pid" 2>/dev/null && echo "stopped old tunnel pid $pid"
done
sleep 1
: > "$RUN/tunnel.log"
nohup "$CFD" tunnel --url "http://localhost:${PORT}" --no-autoupdate >>"$RUN/tunnel.log" 2>&1 &
echo "cloudflared pid $!"

URL=""
for _ in $(seq 1 40); do
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$RUN/tunnel.log" | head -1)
  [ -n "$URL" ] && break
  sleep 2
done
[ -n "$URL" ] || { tail -20 "$RUN/tunnel.log" >&2; die "no tunnel URL after 80s"; }
echo "$URL" > "$RUN/tunnel_url.txt"
echo "tunnel: $URL"

# cloudflared prints the hostname before the edge has finished registering it, so
# poll rather than checking once -- a single check here reliably loses the race.
TUNNEL_OK=0
for _ in $(seq 1 20); do
  if curl -sf -m 15 "$URL/health" >/dev/null 2>&1; then TUNNEL_OK=1; break; fi
  sleep 3
done
[ "$TUNNEL_OK" = 1 ] || die "tunnel is up but /health never answered through it (60s)"
echo "tunnel reaches the backend ✓"

# ---------------------------------------------------------------- gateway
# Vercel points at the Cloudflare Worker, which is a permanent URL. Switching
# tunnels is therefore just a KV write -- no env var change, no redeploy, no
# downtime. (Vercel only needs touching if the Worker URL itself ever changes.)
say "pointing the gateway at the new tunnel"
"$REPO/dgx-server/cf.sh" set "$URL" || die "could not update the Cloudflare KV pointer"

for _ in $(seq 1 10); do
  G=$(curl -s -m 30 "$WORKER_URL/health")
  case "$G" in *'"status":"ok"'*) break;; esac
  sleep 3
done
case "$G" in
  *'"status":"ok"'*) echo "worker reaches the backend ✓";;
  *) die "gateway did not come good: $G";;
esac

# ---------------------------------------------------------------- verify
say "verifying the public chain"
for _ in $(seq 1 12); do
  OUT=$(curl -s -m 30 "$SITE/api/dgx/health")
  case "$OUT" in *'"status": "ok"'*|*'"status":"ok"'*) break;; esac
  sleep 5
done
echo "$SITE/api/dgx/health -> $OUT"
case "$OUT" in
  *'"status"'*'"ok"'*) printf '\n\033[32mLIVE. Open %s/demo — badge should read LIVE.\033[0m\n' "$SITE";;
  *) die "public health check failed: $OUT";;
esac

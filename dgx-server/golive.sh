#!/usr/bin/env bash
# One command to put the live demo on air from the DGX Spark.
#
#   ./dgx-server/golive.sh              # start backend (if down) + tunnel + point Vercel at it
#   ./dgx-server/golive.sh --tunnel-only  # backend already running; just re-tunnel + redeploy
#   ./dgx-server/golive.sh --status     # report what's up, change nothing
#
# Why this exists: a cloudflared *quick* tunnel gets a new random hostname every
# run, and Vercel binds env vars at deploy time — so a new tunnel always needs a
# new production deploy. This does both, then verifies the whole chain.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8787}"
MODEL="${MODEL:-gemma}"
CFD="${CFD:-$HOME/.local/bin/cloudflared}"
KEYFILE="${KEYFILE:-$HOME/.acd-demo.env}"
RUN=/tmp/acd
SITE="https://acd-talk.vercel.app"

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
  echo "public   : $(curl -s -m 25 "$SITE/api/dgx/health" | head -c 200)"
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
# Match the absolute binary path so this pattern can never match this script itself.
for pid in $(pgrep -f "^${CFD} tunnel --url http://localhost:${PORT}$" 2>/dev/null); do
  kill "$pid" 2>/dev/null && echo "stopped old tunnel pid $pid"
done
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

curl -sf -m 25 "$URL/health" >/dev/null || die "tunnel is up but /health did not answer through it"
echo "tunnel reaches the backend ✓"

# ---------------------------------------------------------------- vercel
say "pointing Vercel at the tunnel"
cd "$REPO/talk"
command -v vercel >/dev/null || die "vercel CLI not on PATH"
vercel env rm DGX_TUNNEL_URL production --yes >/dev/null 2>&1 || true
vercel env rm DGX_API_KEY   production --yes >/dev/null 2>&1 || true
printf '%s' "$URL"           | vercel env add DGX_TUNNEL_URL production >/dev/null 2>&1 || die "could not set DGX_TUNNEL_URL"
printf '%s' "$ACD_API_KEY"   | vercel env add DGX_API_KEY   production >/dev/null 2>&1 || die "could not set DGX_API_KEY"
echo "env vars set"

say "deploying to production (~1 min)"
vercel --prod --yes 2>&1 | tail -3

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

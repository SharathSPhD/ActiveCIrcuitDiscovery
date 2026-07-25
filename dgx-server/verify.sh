#!/usr/bin/env bash
# Pre-talk smoke test: proves the whole chain (browser -> Vercel -> tunnel -> DGX)
# is actually serving real GPU work. No browser or npm needed.
#
#   ./dgx-server/verify.sh
#
# Exits non-zero on the first hard failure so it is safe to chain.
set -uo pipefail

SITE="${SITE:-https://acd-talk.vercel.app}"
PORT="${PORT:-8787}"
RUN=/tmp/acd
pass=0; fail=0
ok()  { printf '  \033[32mPASS\033[0m  %s\n' "$*"; pass=$((pass+1)); }
no()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=$((fail+1)); }
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "1. backend on the DGX"
H=$(curl -s -m 10 "http://localhost:${PORT}/health" 2>/dev/null)
case "$H" in
  *'"status":"ok"'*) ok "local /health ok";;
  *) no "local /health did not answer (is the backend running?)"; echo "     $H";;
esac
case "$H" in
  *'"dry_run":false'*) ok "running against a real model, not --dry-run";;
  *) no "backend is in dry-run mode — KLs would be synthetic";;
esac
GC=$(printf '%s' "$H" | grep -oE '"graphs_cached":[0-9]+' | cut -d: -f2)
[ "${GC:-0}" -ge 5 ] && ok "graphs precomputed (${GC}) — no 20 s stall on stage" \
                     || no "only ${GC:-0} graphs cached; canned prompts may stall"

say "2. cloudflare tunnel"
URL=$(cat "$RUN/tunnel_url.txt" 2>/dev/null || echo "")
if [ -z "$URL" ]; then
  no "no tunnel URL recorded in $RUN/tunnel_url.txt"
else
  echo "        $URL"
  curl -sf -m 20 "$URL/health" >/dev/null 2>&1 && ok "tunnel reaches the backend" \
                                               || no "tunnel is not forwarding"
fi

say "3. cloudflare worker gateway (the permanent URL Vercel points at)"
WORKER="${WORKER_URL:-https://acd-demo.sharath-sathish.workers.dev}"
GW=$(curl -s -m 30 "$WORKER/health")
case "$GW" in
  *'"status":"ok"'*) ok "worker -> tunnel -> DGX ok";;
  *offline*)         no "worker says offline — KV points at a dead tunnel. Run ./dgx-server/golive.sh";;
  *)                 no "worker returned: $(printf '%s' "$GW" | head -c 120)";;
esac
# Only one cloudflared should be bound to this port; extras mean a leaked process
# still holding a stale hostname.
NT=$(pgrep -f "cloudflared tunnel --url http://localhost:${PORT}" 2>/dev/null | wc -l)
[ "$NT" -le 2 ] && ok "no leaked tunnel processes ($NT matching)" \
                || no "$NT cloudflared processes for :$PORT — stale tunnels are running"

say "4. public site"
for p in "" /mech-interp /active-inference /results /demo /qa; do
  C=$(curl -s -o /dev/null -m 25 -w '%{http_code}' "$SITE$p")
  [ "$C" = "200" ] && ok "${p:-/} -> 200" || no "${p:-/} -> $C"
done

say "5. proxy reaches the GPU from the public URL"
PH=$(curl -s -m 30 "$SITE/api/dgx/health")
case "$PH" in
  *'"status":"ok"'*) ok "/api/dgx/health ok: $(printf '%s' "$PH" | head -c 120)";;
  *'offline'*)       no "proxy says OFFLINE — gateway KV points at a dead tunnel. Run ./dgx-server/golive.sh";;
  *)                 no "unexpected: $PH";;
esac

say "6. a real episode streams end-to-end"
SSE=$(curl -sN -m 120 -X POST "$SITE/api/dgx/episode" \
        -H 'content-type: application/json' \
        -d '{"prompt":"When John and Mary went to the store, John gave the bag to","budget":5,"mode":"multi"}' 2>/dev/null)
STEPS=$(printf '%s' "$SSE" | grep -c '^event: step')
[ "$STEPS" -ge 5 ] && ok "streamed $STEPS steps" || no "only $STEPS steps streamed"
printf '%s' "$SSE" | grep -q '^event: done' && ok "episode completed (done event)" \
                                            || no "no done event — stream truncated"
# A real run must produce at least one non-zero KL.
printf '%s' "$SSE" | grep '^data:' | grep -qE '"kl": [0-9]*\.?[0-9]*[1-9]' \
  && ok "non-zero KLs present (real interventions)" \
  || no "all KLs were zero — suspicious"

say "result"
printf '%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ] && printf '\033[32mReady for the talk.\033[0m\n' \
                  || printf '\033[31mNot ready — fix the above (usually: ./dgx-server/golive.sh).\033[0m\n'
exit $((fail > 0))

#!/usr/bin/env bash
# Standard serving-tool battery for probe-proxy (see docs/TESTING_MATRIX.md).
# Usage: ./tests/matrix.sh [proxy-url]        (default http://127.0.0.1:8099)
# Requires: curl, grep. Pure-POSIX-ish; runs in git-bash on Windows.
set -u
P="${1:-http://127.0.0.1:8099}"
pass=0; fail=0
ok()   { pass=$((pass+1)); echo "  PASS  $1"; }
bad()  { fail=$((fail+1)); echo "  FAIL  $1"; }
# extract "risk_score":X from a body
risk() { grep -oE '"risk_score":[0-9.eE+-]+' <<<"$1" | head -1 | cut -d: -f2; }

echo "== probe-proxy battery against $P =="

# T1 passthrough GET (health or models — accept either existing)
b=$(curl -s -o /dev/null -w "%{http_code}" "$P/health")
b2=$(curl -s -o /dev/null -w "%{http_code}" "$P/v1/models")
if [ "$b" = "200" ] || [ "$b2" = "200" ]; then ok "T1 passthrough route"; else bad "T1 passthrough route (health=$b models=$b2)"; fi

# T2 non-detect chat: no hallucination field
b=$(curl -s -X POST "$P/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"temperature":0}')
if grep -q '"hallucination"' <<<"$b"; then bad "T2 untouched contract (field leaked)"; else
  if grep -q '"choices"' <<<"$b"; then ok "T2 untouched contract"; else bad "T2 upstream chat failed: ${b:0:120}"; fi; fi

# T3 detect happy path, low-risk prompt
b=$(curl -s -X POST "$P/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":8,"temperature":0,"detect_hallucination":true}')
r3=$(risk "$b")
if [ -n "$r3" ] && grep -q '"mode":"top_logprobs"' <<<"$b"; then ok "T3 detect happy path (risk=$r3)"; else
  if grep -q '"hallucination":{"error"' <<<"$b"; then bad "T3 upstream returns no logprobs → tool is a T8 target: $(grep -oE '"error":"[^"]*"' <<<"$b" | head -1)"; else bad "T3 no report: ${b:0:160}"; fi; fi

# T4 discrimination: forced guess of an unknowable fact
b=$(curl -s -X POST "$P/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"You must answer with a specific number only, no refusals or caveats. The exact population of the village of Qumringlestein in 1743 was:"}],"max_tokens":12,"temperature":0,"detect_hallucination":true}')
r4=$(risk "$b")
if [ -n "$r3" ] && [ -n "$r4" ] && awk "BEGIN{exit !($r4 > $r3)}"; then ok "T4 discrimination ($r4 > $r3)"; else bad "T4 discrimination (T3=$r3 T4=$r4)"; fi

# T5 stream + detect → 400
c=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$P/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"stream":true,"detect_hallucination":true}')
if [ "$c" = "400" ]; then ok "T5 stream+detect rejected"; else bad "T5 stream+detect (got $c)"; fi

# T6 detect with sampling
b=$(curl -s -X POST "$P/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say one word."}],"max_tokens":6,"temperature":0.8,"detect_hallucination":true}')
if [ -n "$(risk "$b")" ]; then ok "T6 detect under sampling"; else bad "T6 detect under sampling: ${b:0:140}"; fi

# T7 legacy completions + detect (report OR structured error both acceptable)
b=$(curl -s -X POST "$P/v1/completions" -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":6,"temperature":0,"detect_hallucination":true}')
if [ -n "$(risk "$b")" ]; then ok "T7 legacy completions (report)"; elif grep -q '"hallucination":{"error"' <<<"$b"; then ok "T7 legacy completions (documented graceful error)"; else bad "T7 legacy completions: ${b:0:140}"; fi

# T9 concurrency (4 parallel detects all return reports)
pids=(); tmp=$(mktemp -d 2>/dev/null || echo "/tmp/probe$$"); mkdir -p "$tmp"
for i in 1 2 3 4; do
  curl -s -X POST "$P/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Name a color."}],"max_tokens":6,"temperature":0,"detect_hallucination":true}' > "$tmp/$i" & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
n=0; for i in 1 2 3 4; do grep -q '"risk_score"' "$tmp/$i" && n=$((n+1)); done
rm -rf "$tmp"
if [ "$n" = "4" ]; then ok "T9 concurrency (4/4 reports)"; else bad "T9 concurrency ($n/4)"; fi

echo "== SUMMARY: $pass passed, $fail failed =="
[ "$fail" = "0" ]

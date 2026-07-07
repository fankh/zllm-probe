# llm-probe

White-box instrumentation for **any** LLM serving tool — hallucination /
uncertainty detection today, a portable instrumentation protocol next.

Works with **llama.cpp (`llama-server`), vLLM, zllm, and any OpenAI-compatible
server** — no server modifications, no plugins to install into the host:

```bash
# 1. run your server as usual (llama-server, vllm serve, zllm, ...)
# 2. put the proxy in front of it
PROBE_UPSTREAM=http://127.0.0.1:8080 probe-proxy     # listens on :8099

# 3. add one flag to any request
curl :8099/v1/chat/completions -d '{
  "messages": [{"role":"user","content":"What was the population of Qumringlestein in 1743?"}],
  "detect_hallucination": true
}'
```

```jsonc
// the response gains:
"hallucination": {
  "risk_score": 0.63,        // mean per-token uncertainty, 0..1
  "mean_entropy": 2.9,       // nats (lower-bound approximation in proxy mode)
  "risky_fraction": 0.71,    // tokens over the uncertainty thresholds
  "peak_token": 3,           // index of the most uncertain token
  "flagged": true,           // risk_score over the flag bar
  "mode": "top_logprobs"     // "logits" when an in-process adapter provides them
}
```

**How it works:** the proxy injects `logprobs`/`top_logprobs` into the upstream
request, feeds each generated token's distribution into `probe-core`'s
detector (predictive entropy, top-probability, top-1/2 margin), and attaches
the aggregated report. Everything else — every other route and every
non-detect request — is forwarded untouched.

**Honest framing:** these are *uncertainty proxies*, not calibrated
hallucination oracles. High uncertainty correlates with confabulation; treat
`risk_score` as a relative flag and `peak_token` as the pointer to the risky
span. In proxy mode the entropy is a documented lower bound (the tail beyond
the returned top-K is folded into one outcome). In-process adapters with full
logits get the exact numbers.

## Crates

| crate | what |
|---|---|
| [`probe-core`](crates/probe-core) | engine-agnostic detector: full-logits mode (exact) + top-logprobs mode (API-level) |
| [`probe-proxy`](proxy) | the OpenAI-compatible middleware binary |

## Build

```bash
cargo build --release            # → target/release/probe-proxy
cargo test                       # core unit tests
```

Configuration (env): `PROBE_UPSTREAM` (default `http://127.0.0.1:8080`),
`PROBE_PORT` (8099), `PROBE_TOP_LOGPROBS` (10), `PROBE_PER_TOKEN` (include
per-token detail in reports).

v1 limits: non-streaming requests only (`stream: true` + detect → 400);
chat + legacy completions endpoints.

## Roadmap

The proxy is delivery mode 1 of a capability-negotiated cross-engine
instrumentation protocol — see [`docs/INSTRUMENTATION_PROTOCOL.md`](docs/INSTRUMENTATION_PROTOCOL.md):

1. **Proxy** (this repo, works today): logprobs-level signals for any backend.
2. **vLLM adapter** (planned): in-process logits-processor plugin — exact
   logits, later hidden-state probes.
3. **Native / deep tier**: full per-layer hidden-state hooks — reference
   implementation lives in the [zllm](https://github.com/fankh/zllm) engine
   (activation steering, hook write-back, mid-layer probes).

## Origin

Extracted from the [zllm](https://github.com/fankh/zllm) white-box inference
engine, where the detector is live-verified (factual prompt: risk 0.32,
unflagged; confabulated fact: 0.69, flagged — Llama-3.2-1B). MIT.

# Serving-tool testing matrix

Goal: prove (or honestly bound) the claim "works with any OpenAI-compatible
server" — one standard battery, run against every tool, results recorded here.

## The standard battery (`tests/matrix.sh`)

Run the target server, put probe-proxy in front, then:

```bash
PROBE_UPSTREAM=http://127.0.0.1:<port> ./tests/matrix.sh http://127.0.0.1:8099
```

| # | test | pass criteria |
|---|---|---|
| T1 | passthrough: GET health/models route | upstream status + body returned unchanged |
| T2 | non-detect chat request | 200, **no** `hallucination` field (bit-untouched contract) |
| T3 | detect happy path ("What is 2+2?") | `hallucination` present; `mode:"top_logprobs"`; `0 ≤ risk_score ≤ 1`; `n_tokens > 0`; low risk (< 0.15) |
| T4 | discrimination (forced guess of an unknowable fact) | `risk_score(T4) > risk_score(T3)` — monotonic signal |
| T5 | `stream:true` + detect | clean 400 with an error message (never a hung stream) |
| T6 | detect + `temperature: 0.8` | report still present and sane (sampling doesn't break parsing) |
| T7 | legacy `/v1/completions` + detect | report present **or** structured `hallucination.error` (shape unsupported is acceptable, silence is not) |
| T8 | upstream without logprobs support | response usable + `hallucination: {error: "..."}` — graceful degradation, not 500 |
| T9 | 4 concurrent detect requests | all 200 with reports (no cross-request state) |

Recording rule: paste the script's summary line per tool below, with server
version + model. A tool "passes" when T1–T6 + T9 green and T7/T8 behave as
documented (some tools legitimately land in the T8 path).

## Tier 1 — this box, now (Windows / Strix Halo)

| tool | logprobs support (chat) | status | notes / quirks |
|---|---|---|---|
| **llama.cpp `llama-server`** | ✅ `logprobs:true` + `top_logprobs:N` (chat AND legacy shapes) | **VALIDATED 2026-07-07: battery 8/8** (Llama-3.2-1B-Q4_K_M): T3 risk 0.0098 vs T4 0.342 monotonic; T7 legacy-shape report; T9 4/4 concurrent | reference target; rerun on llama.cpp upgrades |
| **Ollama** | ⚠️ verify — OpenAI-compat layer gained logprobs only in recent versions; older ones omit the field | todo | if absent → this is the designated **T8 graceful-degradation** target. Windows install: `winget install Ollama.Ollama`; model `ollama pull llama3.2:1b`; port 11434 |
| **LM Studio** | ⚠️ verify — OpenAI-compat server, logprobs historically not exposed | todo | GUI install; local server port 1234. Expect T8 path until confirmed |
| **KoboldCpp** | ⚠️ verify | todo | single exe; OpenAI-compat on /v1 |
| **zllm** | ❌ does not return logprobs today | **battery run 2026-07-07: T1/T2/T5/T7 pass; T3/T4/T6/T9 correctly land in the T8 graceful path.** Bonus: T2 caught a real zllm API bug (`"hallucination": null` on every response) — fixed in zllm same day | action item: add `logprobs`/`top_logprobs` to zllm's chat API (it computes full logits on the sampling path already) — ironic gap, fix first |

## Tier 2 — WSL2 / Docker / remote Linux

| tool | logprobs support (chat) | status | notes / quirks |
|---|---|---|---|
| **vLLM** | ✅ `logprobs:true` + `top_logprobs:N`, **N capped at 20** | todo | won't run natively on Windows. Options: WSL2 CPU mode (slow, fine for correctness: `pip install vllm; vllm serve meta-llama/Llama-3.2-1B-Instruct --device cpu`), or a CUDA/ROCm box. Set `PROBE_TOP_LOGPROBS=20` max. The **priority Tier-2 target** (it's also the planned in-process adapter host) |
| **TGI** (HF text-generation-inference) | ✅ on `/v1/chat/completions` | todo | docker; verify its top_logprobs cap (5 in some versions) |
| **SGLang** | ✅ | todo | docker/pip, Linux |

## Tier 3 — hosted OpenAI-compatible APIs (optional, needs keys)

OpenAI, Groq, Together, Fireworks — all return `top_logprobs` (≤ 20; some ≤ 5).
Value: proves the proxy against production-grade API idioms (chunked responses,
rate-limit errors through passthrough). Run the battery with `PROBE_UPSTREAM`
pointing at the vendor base URL + an auth-forwarding tweak (the proxy currently
forwards no `Authorization` header on the instrumented path — **known gap**,
fix before Tier 3).

## Known per-tool quirks to encode in the proxy as they're confirmed

- `top_logprobs` caps: vLLM 20, TGI 5(?), OpenAI 20 → clamp `PROBE_TOP_LOGPROBS`
  per upstream or on 400-retry.
- Legacy `/v1/completions` logprobs shape differs (`token_logprobs` +
  `top_logprobs: [{tok: lp}]`) — parser handles both; T7 verifies per tool.
- Chat templates differ per tool for the same model — risk scores are
  comparable in ORDER (T4 > T3) but not in absolute value across tools;
  never assert absolute values cross-tool.
- Auth header forwarding for hosted APIs (Tier 3 gap above).
- Streaming: v1 rejects detect+stream; the eventual fix is SSE accumulation
  with the report in the final chunk (tracked in the roadmap).

## Regression cadence

- llama-server battery: rerun on every probe-proxy change (it's local + fast).
- Full Tier-1: before each release.
- Tier-2 vLLM: before each release once WSL2 target is set up.

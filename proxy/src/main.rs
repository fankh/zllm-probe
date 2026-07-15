//! probe-proxy: an OpenAI-compatible middleware that adds hallucination /
//! uncertainty reports to ANY backend that returns `logprobs` — llama.cpp's
//! `llama-server`, vLLM, zllm, OpenAI-compatible gateways, ...
//!
//! Sit it in front of your server:
//!
//! ```text
//! PROBE_UPSTREAM=http://127.0.0.1:8080 probe-proxy       # listens on :8099
//! curl :8099/v1/chat/completions -d '{..., "detect_hallucination": true}'
//! → response gains "hallucination": {risk_score, mean_entropy, flagged, ...}
//! ```
//!
//! How: when a request carries `detect_hallucination: true`, the proxy strips
//! the flag, injects `logprobs: true, top_logprobs: N` (chat) into the
//! upstream request, feeds each returned token's top-logprobs into
//! `probe_core::Detector`, and attaches the report. Entropy in this mode is a
//! documented lower-bound approximation (see probe-core). Everything else —
//! all other routes, non-detect requests — is forwarded untouched: client
//! headers (auth included) pass upstream minus hop-by-hop, and response
//! bodies stream through as they arrive (SSE stays incremental).
//!
//! v1 limits: non-streaming only (`stream: true` + detect → 400, like the
//! zllm reference implementation); chat + legacy completions endpoints.

use axum::body::Bytes;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use probe_core::{Detector, DetectorConfig};
use serde_json::{json, Value};

/// Headers the proxy must not blindly copy upstream: hop-by-hop headers
/// (RFC 9110 §7.6.1), transport framing (reqwest recomputes host /
/// content-length), and `accept-encoding` — the proxy streams bodies
/// verbatim without decompressing, so upstream must send identity.
fn skip_request_header(name: &str) -> bool {
    matches!(
        name,
        "host" | "content-length" | "connection" | "keep-alive" | "transfer-encoding"
            | "upgrade" | "proxy-authenticate" | "proxy-authorization" | "te" | "trailer"
            | "accept-encoding"
    )
}

/// Client headers → upstream headers (drops hop-by-hop; keeps auth etc.).
fn upstream_headers(h: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (name, value) in h {
        if !skip_request_header(name.as_str()) {
            out.append(name.clone(), value.clone());
        }
    }
    out
}

/// Forward an upstream response as a *streaming* body (SSE chunks pass
/// through as they arrive; a buffered `bytes().await` would stall
/// `stream: true` clients until generation finished).
fn stream_response(resp: reqwest::Response) -> Response {
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/octet-stream")
        .to_string();
    Response::builder()
        .status(status)
        .header("content-type", ct)
        .body(axum::body::Body::from_stream(resp.bytes_stream()))
        .unwrap()
}

#[derive(Clone)]
struct Ctx {
    upstream: String,
    client: reqwest::Client,
    top_logprobs: u32,
    per_token: bool,
}

#[tokio::main]
async fn main() {
    let upstream = std::env::var("PROBE_UPSTREAM").unwrap_or_else(|_| "http://127.0.0.1:8080".into());
    let port: u16 = std::env::var("PROBE_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8099);
    let ctx = Ctx {
        upstream: upstream.trim_end_matches('/').to_string(),
        client: reqwest::Client::new(),
        top_logprobs: std::env::var("PROBE_TOP_LOGPROBS").ok().and_then(|v| v.parse().ok()).unwrap_or(10),
        per_token: std::env::var("PROBE_PER_TOKEN").is_ok(),
    };
    let app = Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .fallback(passthrough)
        .with_state(ctx.clone());
    let addr = format!("0.0.0.0:{port}");
    eprintln!("probe-proxy listening on {addr} → upstream {}", ctx.upstream);
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    axum::serve(listener, app).await.expect("serve");
}

async fn chat(State(ctx): State<Ctx>, headers: HeaderMap, body: Bytes) -> Response {
    instrumented(ctx, "/v1/chat/completions", headers, body, Mode::Chat).await
}

async fn completions(State(ctx): State<Ctx>, headers: HeaderMap, body: Bytes) -> Response {
    instrumented(ctx, "/v1/completions", headers, body, Mode::Legacy).await
}

enum Mode {
    Chat,
    Legacy,
}

async fn instrumented(ctx: Ctx, path: &str, headers: HeaderMap, body: Bytes, mode: Mode) -> Response {
    let mut req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_REQUEST, &format!("invalid JSON body: {e}")),
    };
    let detect = req.get("detect_hallucination").and_then(Value::as_bool).unwrap_or(false);
    if let Some(o) = req.as_object_mut() {
        o.remove("detect_hallucination"); // never leak our extension upstream
    }
    let fwd_headers = upstream_headers(&headers);
    if !detect {
        return forward_json(&ctx, path, fwd_headers, &req).await;
    }
    if req.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        return err(StatusCode::BAD_REQUEST, "detect_hallucination is not supported with stream=true yet");
    }
    // Inject the logprobs request in the shape the endpoint expects.
    if let Some(o) = req.as_object_mut() {
        match mode {
            Mode::Chat => {
                o.insert("logprobs".into(), json!(true));
                o.insert("top_logprobs".into(), json!(ctx.top_logprobs));
            }
            Mode::Legacy => {
                o.insert("logprobs".into(), json!(ctx.top_logprobs));
            }
        }
    }
    let resp = match ctx
        .client
        .post(format!("{}{}", ctx.upstream, path))
        .headers(fwd_headers)
        .json(&req)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return err(StatusCode::BAD_GATEWAY, &format!("upstream: {e}")),
    };
    let status = resp.status();
    let mut out: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_GATEWAY, &format!("upstream body: {e}")),
    };
    if status.is_success() {
        match analyze(&out, ctx.per_token) {
            Ok(report) => {
                if let Some(o) = out.as_object_mut() {
                    o.insert("hallucination".into(), report);
                }
            }
            Err(msg) => {
                if let Some(o) = out.as_object_mut() {
                    o.insert("hallucination".into(), json!({ "error": msg }));
                }
            }
        }
    }
    (StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY), Json(out)).into_response()
}

/// Pull per-token logprobs out of either response shape and run the detector.
/// Chat: `choices[0].logprobs.content[i] = {logprob, top_logprobs:[{logprob}]}`.
/// Legacy: `choices[0].logprobs = {token_logprobs:[f], top_logprobs:[{tok: lp}]}`.
fn analyze(resp: &Value, per_token: bool) -> Result<Value, String> {
    let choice = resp.get("choices").and_then(|c| c.get(0)).ok_or("no choices in upstream response")?;
    let lp = choice.get("logprobs").filter(|v| !v.is_null())
        .ok_or("upstream returned no logprobs (server too old, or logprobs unsupported on this endpoint)")?;
    let mut det = Detector::new(DetectorConfig::default());
    if let Some(content) = lp.get("content").and_then(Value::as_array) {
        for tok in content {
            let chosen = tok.get("logprob").and_then(Value::as_f64).unwrap_or(f64::NEG_INFINITY) as f32;
            let tops: Vec<f32> = tok.get("top_logprobs").and_then(Value::as_array).map(|a| {
                a.iter().filter_map(|t| t.get("logprob").and_then(Value::as_f64)).map(|v| v as f32).collect()
            }).unwrap_or_default();
            det.observe_top_logprobs(chosen, &tops);
        }
    } else if let Some(tlp) = lp.get("token_logprobs").and_then(Value::as_array) {
        let tops_arr = lp.get("top_logprobs").and_then(Value::as_array);
        for (i, chosen) in tlp.iter().enumerate() {
            let chosen = chosen.as_f64().unwrap_or(f64::NEG_INFINITY) as f32;
            let tops: Vec<f32> = tops_arr.and_then(|a| a.get(i)).and_then(Value::as_object).map(|m| {
                m.values().filter_map(Value::as_f64).map(|v| v as f32).collect()
            }).unwrap_or_default();
            det.observe_top_logprobs(chosen, &tops);
        }
    } else {
        return Err("unrecognized logprobs shape".into());
    }
    if det.is_empty() {
        return Err("no tokens with logprobs in upstream response".into());
    }
    serde_json::to_value(det.report(per_token)).map_err(|e| e.to_string())
}

/// Forward a JSON body verbatim (detect off), preserving status, client
/// headers (auth), and streaming (SSE chunks relay as they arrive).
async fn forward_json(ctx: &Ctx, path: &str, headers: HeaderMap, req: &Value) -> Response {
    match ctx
        .client
        .post(format!("{}{}", ctx.upstream, path))
        .headers(headers)
        .json(req)
        .send()
        .await
    {
        Ok(r) => stream_response(r),
        Err(e) => err(StatusCode::BAD_GATEWAY, &format!("upstream: {e}")),
    }
}

/// Transparent reverse proxy for every other route (health, models, UI, ...).
async fn passthrough(State(ctx): State<Ctx>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let bytes = match axum::body::to_bytes(body, 64 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => return err(StatusCode::BAD_REQUEST, &format!("body: {e}")),
    };
    let uri: Uri = parts.uri;
    let pq = uri.path_and_query().map(|p| p.as_str()).unwrap_or("/");
    let method = reqwest::Method::from_bytes(parts.method.as_str().as_bytes()).unwrap_or(reqwest::Method::GET);
    let mut r = ctx
        .client
        .request(method, format!("{}{}", ctx.upstream, pq))
        .headers(upstream_headers(&parts.headers));
    if parts.method != Method::GET && !bytes.is_empty() {
        r = r.body(bytes.to_vec());
    }
    match r.send().await {
        Ok(resp) => stream_response(resp),
        Err(e) => err(StatusCode::BAD_GATEWAY, &format!("upstream: {e}")),
    }
}

fn err(status: StatusCode, msg: &str) -> Response {
    (status, Json(json!({ "error": msg }))).into_response()
}
